"""Competition registry — one entry per week.

Adding a week means adding an entry here and a `prepare()` implementation. Nothing in
the app knows about Week 1 specifically; the routes, scoring and leaderboard all read
this registry. That is the whole reason Week 1 is worth building carefully.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

JUDGE = Path(__file__).resolve().parent
REPO = JUDGE.parent
COMP_DATA = JUDGE / "data"          # generated artifacts; gitignored


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    fmt: str = "{:.5f}"
    higher_is_better: bool = True
    blurb: str = ""


@dataclass(frozen=True)
class Competition:
    week: int
    slug: str
    title: str
    tagline: str
    task: str                      # markdown, shown on the week page
    id_column: str
    target_column: str
    primary_metric: Metric
    secondary_metrics: list[Metric]
    public_frac: float = 0.30      # share of test rows scored on the public leaderboard
    max_upload_mb: int = 60
    max_daily_submissions: int = 20
    open: bool = True
    baselines: dict[str, str] = field(default_factory=dict)

    # Where the data lives once published. The public files are served by redirecting to
    # the Hub's CDN, which keeps 50 MB of downloads (and all of that bandwidth) off this
    # server; the answer key goes to a private repo and is fetched once per boot.
    hf_repo: str | None = None            # public: train / test / sample_submission
    hf_solution_repo: str | None = None   # private: solution.parquet
    hf_revision: str = "main"

    def remote_url(self, filename: str) -> str:
        """The *stable* resolve URL, not a signed CDN link.

        The Hub answers this with its own 302 to a time-limited CloudFront URL, so
        redirecting here keeps working indefinitely; caching the signed link would start
        serving expired URLs a few hours later.
        """
        return (f"https://huggingface.co/datasets/{self.hf_repo}"
                f"/resolve/{self.hf_revision}/{filename}")

    @property
    def dir(self) -> Path:
        return COMP_DATA / f"week{self.week:02d}"

    @property
    def train_file(self) -> Path:
        return self.dir / "train.csv.gz"

    @property
    def test_file(self) -> Path:
        return self.dir / "test.csv.gz"

    @property
    def sample_file(self) -> Path:
        return self.dir / "sample_submission.csv.gz"

    @property
    def solution_file(self) -> Path:
        """Hidden labels. NEVER served over HTTP — see judge/app.py download routes."""
        return self.dir / "solution.parquet"

    def resolve_solution(self) -> Path:
        """Path to the answer key, fetching it from the private repo if it isn't local.

        Prefers the local file so development and offline work are unaffected. On a host
        with no persistent disk the fallback pulls 1.9 MB from a private dataset repo into
        the container's ephemeral filesystem — cheap enough to redo on every cold start,
        and the reason this app no longer needs a paid volume.

        Cached per process: fetched once per boot, not once per submission.
        """
        if self.solution_file.exists():
            return self.solution_file

        cached = _SOLUTION_CACHE.get(self.week)
        if cached and cached.exists():
            return cached

        if not self.hf_solution_repo:
            raise FileNotFoundError(
                f"{self.solution_file} is missing and no hf_solution_repo is configured — "
                f"run `python -m judge.prepare_data --week {self.week}`")

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "HF_TOKEN is not set, so the answer key cannot be fetched from "
                f"{self.hf_solution_repo}. Set it as a platform secret, or prepare the "
                f"data locally with `python -m judge.prepare_data --week {self.week}`.")

        from huggingface_hub import hf_hub_download

        # One private repo holds every week's answer key, so the path is week-scoped.
        # Keep this in step with tools/publish_competition.py.
        p = Path(hf_hub_download(self.hf_solution_repo,
                                 f"week{self.week:02d}/solution.parquet",
                                 repo_type="dataset", revision=self.hf_revision,
                                 token=token))
        _SOLUTION_CACHE[self.week] = p
        return p

    @property
    def is_prepared(self) -> bool:
        """True when this competition can be served at all.

        The answer key may be local *or* reachable in the private repo; the public files
        may be local *or* published. Either combination is a working deployment.
        """
        has_public = all(p.exists() for p in
                         (self.train_file, self.test_file, self.sample_file))
        has_solution = self.solution_file.exists() or bool(self.hf_solution_repo)
        return (has_public or bool(self.hf_repo)) and has_solution


# Resolved answer-key paths, keyed by week. Populated on first use per process.
_SOLUTION_CACHE: dict[int, Path] = {}


# --------------------------------------------------------------------------------------
# Metrics. Note the primary is NOT AUC, deliberately — see the task text below.
# --------------------------------------------------------------------------------------
NE = Metric(
    "normalised_entropy", "NE", "{:.5f}", higher_is_better=False,
    blurb=("Log-loss divided by the log-loss of predicting the base rate. Exactly 1.0 for "
           "a constant predictor; below 1.0 means the model earns its keep. A proper "
           "scoring rule, so unlike AUC it sees calibration as well as ranking."),
)
AUC = Metric("auc", "AUC", "{:.5f}", True,
             "Ranking only. Invariant to any monotone rescaling of your predictions.")
LOGLOSS = Metric("log_loss", "log-loss", "{:.5f}", False, "Mean negative log-likelihood.")
CALIB = Metric("calibration_ratio", "calib.", "{:.4f}", True,
               "mean(prediction) / mean(label). 1.0 is unbiased on aggregate. At 1.15 "
               "every bid is 15% too high.")

WEEK1_TASK = """\
### The task

Predict, for each impression in the test set, the probability that the user converted
within 30 days.

The data is Criteo's *Attribution Modeling for Bidding* log. **The label is
impression-level**: `conversion = 1` means "the user who saw this impression converted
within 30 days", not "this impression caused the conversion". Every impression in a
converting journey carries a 1. Predicting it is a legitimate CVR task — just don't call
it attribution.

### Split

Time-ordered, exactly as `docs/eval-protocol.md` requires. Train is the earlier part of
the window; test is the later part. There is no overlap and no shuffling. A random split
on this data inflates AUC by an amount that looks like progress, which is why you are not
given the option.

### Ranked on NE, not AUC

The leaderboard sorts by **normalised entropy** (lower is better). This is deliberate and
it is the point of Week 1.

AUC is invariant to monotone rescaling: multiply every prediction by 10 and your AUC does
not move by a single digit. A model with a great AUC and a calibration ratio of 3.0 will
overbid every auction threefold. NE is a proper scoring rule — it sees ranking *and*
calibration, and it cannot be gamed by a rescaling.

AUC and calibration ratio are shown alongside, because the disagreement between them is
the interesting part. If your AUC climbs while your NE gets worse, you have built a better
ranker and a worse bidder, and you should be able to say which one you wanted.

### Two leaderboards

30% of the test rows are scored publicly and shown live. The other 70% are held back.
This is the Kaggle arrangement and it exists for the same reason: to make the difference
between "tuned a model" and "tuned the leaderboard" visible after the fact.

### Submission format

Gzipped or plain CSV, two columns, header required:

```
impression_id,prediction
0,0.0421
1,0.1337
```

One row per test impression, `prediction` in [0, 1]. Order does not matter; the scorer
joins on `impression_id`. Missing or extra ids are rejected with a message telling you
which.

### Beat the baselines

The leaderboard is seeded with a global base-rate predictor and a logistic regression on
hashed categoricals. Beating the base rate is the bar for the model existing at all;
beating the logistic regression is the bar for Week 1 being finished.
"""

WEEK1 = Competition(
    week=1,
    slug="week01-cvr-baseline",
    title="Week 01 — CVR baseline",
    tagline="Predict conversion from a Criteo impression log. Ranked on calibration-aware NE.",
    task=WEEK1_TASK,
    id_column="impression_id",
    target_column="conversion",
    primary_metric=NE,
    secondary_metrics=[AUC, LOGLOSS, CALIB],
    baselines={
        "base_rate": "Predict the training base rate for every row. NE is 1.0 by construction.",
        "logreg_hashed_2^18": "Logistic regression on cat1..cat9 hashed to 262k buckets.",
    },
    # Namespace is the Hugging Face username (aditagar), which is not the GitHub one.
    # Until `tools/publish_competition.py` has actually pushed these, the download route
    # probes the repo once and quietly falls back to serving the local files — so setting
    # this early is safe and publishing later needs no code change.
    hf_repo="aditagar/ads-ml-lab-week01",
    hf_solution_repo="aditagar/ads-ml-lab-solutions",
)

COMPETITIONS: dict[int, Competition] = {c.week: c for c in [WEEK1]}

# Weeks 2-12 exist in the repo but have no competition yet. The index page lists them as
# locked so the shape of the whole course is visible from the front page.
UPCOMING = {
    2: "Deep CVR — FM, Wide & Deep, DLRM",
    3: "Calibration",
    4: "Delayed feedback",
    5: "Missing labels and conversion modeling",
    6: "Multi-touch attribution",
    7: "Uplift and incrementality",
    8: "Auction simulation and bid shading",
    9: "Budget pacing and CPA control",
    10: "Privacy-constrained learning",
    11: "Sequence models for user journeys",
    12: "Capstone — privacy-era measurement stack",
}


def get(week: int) -> Competition | None:
    return COMPETITIONS.get(week)
