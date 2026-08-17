#!/usr/bin/env python3
"""Publish a week's competition files to the Hugging Face Hub.

Why: the judge was serving 58 MB of files off its own disk, which forced every deployment
to rent a persistent volume. All but the answer key is public, so it belongs on a CDN. After
publishing, the judge redirects downloads to the Hub and its own storage need drops to the
1.9 MB answer key plus a 48 KB database — small enough that hosts with no disk become
viable.

Two repos, deliberately separate:

    <ns>/ads-ml-lab-weekNN      PUBLIC   train.csv.gz, test.csv.gz, sample_submission.csv.gz
    <ns>/ads-ml-lab-solutions   PRIVATE  solution.parquet

Licence: the upstream Criteo attribution dataset is CC-BY-NC-SA 4.0, which permits
redistributing a derived subset provided the derived work credits Criteo, stays
non-commercial, and carries the same licence. The generated dataset card does all three.

Usage
-----
    python tools/publish_competition.py --week 1 --dry-run   # show the plan, upload nothing
    python tools/publish_competition.py --week 1
    python tools/publish_competition.py --week 1 --verify    # check what is actually up there

Requires a WRITE token: `hf auth login` with a token created as "Write", or HF_TOKEN in
the environment. The read-only token used to download the source data cannot create repos.
(`huggingface-cli` was removed in huggingface_hub 1.x; the CLI is `hf` now.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from judge.competitions import COMPETITIONS, Competition  # noqa: E402

# The one file that must never reach a public repo. Checked by name at upload time rather
# than trusted to the caller getting the argument order right.
SECRET_FILES = {"solution.parquet"}

PUBLIC_FILES = ["train.csv.gz", "test.csv.gz", "sample_submission.csv.gz",
                "facts.json", "preview.json"]

CARD = """\
---
license: cc-by-nc-sa-4.0
tags:
- advertising
- conversion-prediction
- criteo
- tabular
pretty_name: "ads-ml-lab — Week {week:02d}: {short}"
---

# ads-ml-lab — Week {week:02d}: {short}

The data for week {week:02d} of [ads-ml-lab](https://github.com/o0aditya0o/ads-ml-lab), a
twelve-week course that rebuilds the core model classes of ads measurement from scratch on
public data. This repo exists so the leaderboard can serve downloads from a CDN instead of
from a small VM.

**This is a derived dataset, not a copy.** It is built by
[`tools/prepare_data.py`](https://github.com/o0aditya0o/ads-ml-lab/blob/main/judge/prepare_data.py)
from [`criteo/criteo-attribution-dataset`](https://huggingface.co/datasets/criteo/criteo-attribution-dataset).

## The task

Predict, per impression, the probability that the user converted within 30 days.

`conversion` is an **impression-level** label meaning "the user who saw this impression
converted within 30 days" — not "this impression caused the conversion". Every impression
in a converting journey carries a 1. It is a legitimate CVR target; it is not attribution.

## Files

| file | rows | contents |
|---|---|---|
| `train.csv.gz` | {train_rows:,} | features **and** the `conversion` label |
| `test.csv.gz` | {test_rows:,} | features only |
| `sample_submission.csv.gz` | {test_rows:,} | required submission format, filled with the train base rate |

`impression_id` is assigned during preparation and has no meaning upstream. It is the join
key for scoring.

## How it was built

- **Time-ordered split.** The boundary is the 70th percentile of `timestamp`, taken
  **before** subsampling. Subsampling first would let a test row predate a train row. Train
  is strictly earlier than test; there is no overlap and no shuffling. A random split on
  this data inflates AUC by an amount that looks like progress.
- **Subsampled** to {train_rows:,} train and {test_rows:,} test rows so a submission is a
  few MB rather than fifty.
- **Leaky columns removed from the test file:** `conversion`, `conversion_timestamp`,
  `conversion_id`, `cpo`, and `attribution`. That last one is the subtle one — it is
  Criteo's own last-click flag and is nonzero only on converting journeys, so shipping it
  would hand over the label under a different name.
- The held-out labels are **not** in this repo.

## Evaluation

The leaderboard ranks on **normalised entropy** (log-loss relative to a base-rate
predictor), not AUC. AUC is invariant to monotone rescaling, so a model calibrated three
times too high scores identically on AUC and would overbid every auction threefold. NE is a
proper scoring rule and sees both ranking and calibration.

Base rate is about 4.9%, and it drifts slightly downward across the window — the test half
converts a little less often than the train half. That drift is part of the problem.

## Licence and attribution

CC-BY-NC-SA 4.0, inherited from the upstream dataset. Non-commercial use only, and derived
works must carry the same licence.

```bibtex
@inproceedings{{DiemertMeynet2017,
  author    = {{{{Diemert Eustache, Meynet Julien}} and Galland, Pierre and Lefortier, Damien}},
  title     = {{Attribution Modeling Increases Efficiency of Bidding in Display Advertising}},
  booktitle = {{Proceedings of the AdKDD and TargetAd Workshop, KDD, Halifax, NS, Canada, August, 14, 2017}},
  year      = {{2017}},
  publisher = {{ACM}}
}}
```
"""

SOLUTION_CARD = """\
---
license: cc-by-nc-sa-4.0
viewer: false
---

# ads-ml-lab — held-out labels

**Private by design.** These are the answer keys for the
[ads-ml-lab](https://github.com/o0aditya0o/ads-ml-lab) leaderboards. Making this repo
public would end the competition.

Each `solution.parquet` holds `impression_id`, the true label, and `is_public` — the
deterministic 30/70 public/private leaderboard split, keyed on a hash of the id so that
re-preparing the data never reshuffles which rows are public.

The judge fetches this at boot with a read token and keeps it in memory, which is why the
server itself needs no persistent disk.

Derived from `criteo/criteo-attribution-dataset`, CC-BY-NC-SA 4.0.
"""


def row_count(path: Path) -> int:
    import gzip
    with gzip.open(path, "rb") as fh:
        return sum(1 for _ in fh) - 1


def namespace(token: str | None) -> str:
    from huggingface_hub import HfApi
    return HfApi().whoami(token=token)["name"]


def check_write_token(token: str | None) -> str | None:
    """Return an error message if the token cannot create repos."""
    from huggingface_hub import HfApi
    try:
        me = HfApi().whoami(token=token)
    except Exception as e:
        return f"not logged in to Hugging Face ({type(e).__name__}: {e})"
    role = (me.get("auth") or {}).get("accessToken", {}).get("role")
    if role and role not in ("write", "admin"):
        return (f"the active token is '{role}', which cannot create repos or upload.\n"
                f"      Create a WRITE token at https://huggingface.co/settings/tokens\n"
                f"      then run:  hf auth login\n"
                f"      (or set HF_TOKEN to the write token for this command only)")
    return None


def publish(comp: Competition, token: str | None, dry_run: bool) -> int:
    from huggingface_hub import HfApi

    api = HfApi()
    pub_repo, sol_repo = comp.hf_repo, comp.hf_solution_repo
    if not pub_repo or not sol_repo:
        print(f"week {comp.week}: hf_repo / hf_solution_repo not set in competitions.py",
              file=sys.stderr)
        return 2

    missing = [f for f in PUBLIC_FILES if not (comp.dir / f).exists()]
    if missing or not comp.solution_file.exists():
        print(f"missing local files: {missing + ([] if comp.solution_file.exists() else ['solution.parquet'])}\n"
              f"run: python -m judge.prepare_data --week {comp.week}", file=sys.stderr)
        return 2

    train_rows = row_count(comp.train_file)
    test_rows = row_count(comp.test_file)
    short = comp.title.split("—")[-1].strip()

    print(f"PUBLIC  -> {pub_repo}")
    for f in PUBLIC_FILES:
        p = comp.dir / f
        print(f"           {f:32s} {p.stat().st_size / 1e6:7.2f} MB")
    print(f"PRIVATE -> {sol_repo}")
    print(f"           {'solution.parquet':32s} "
          f"{comp.solution_file.stat().st_size / 1e6:7.2f} MB")
    print(f"\n  train rows {train_rows:,} · test rows {test_rows:,}")

    # Belt and braces: the public list must not contain anything secret.
    leaked = SECRET_FILES & set(PUBLIC_FILES)
    if leaked:
        print(f"\nREFUSING: {leaked} is in the public upload list", file=sys.stderr)
        return 1

    if dry_run:
        print("\n(dry run — nothing uploaded)")
        return 0

    if err := check_write_token(token):
        print(f"\nCannot upload: {err}", file=sys.stderr)
        return 1

    # ---- public repo ----------------------------------------------------------------
    api.create_repo(pub_repo, repo_type="dataset", exist_ok=True, private=False,
                    token=token)
    card = CARD.format(week=comp.week, short=short,
                       train_rows=train_rows, test_rows=test_rows)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=pub_repo, repo_type="dataset", token=token)
    for f in PUBLIC_FILES:
        print(f"  uploading {f} ...", flush=True)
        api.upload_file(path_or_fileobj=str(comp.dir / f), path_in_repo=f,
                        repo_id=pub_repo, repo_type="dataset", token=token)

    # ---- private repo ---------------------------------------------------------------
    api.create_repo(sol_repo, repo_type="dataset", exist_ok=True, private=True, token=token)
    api.upload_file(path_or_fileobj=SOLUTION_CARD.encode(), path_in_repo="README.md",
                    repo_id=sol_repo, repo_type="dataset", token=token)
    print("  uploading solution.parquet (private) ...", flush=True)
    api.upload_file(path_or_fileobj=str(comp.solution_file),
                    path_in_repo=f"week{comp.week:02d}/solution.parquet",
                    repo_id=sol_repo, repo_type="dataset", token=token)

    return verify(comp, token)


def verify(comp: Competition, token: str | None) -> int:
    """Confirm what is actually on the Hub, and that the answer key did not leak."""
    from huggingface_hub import HfApi

    api = HfApi()
    ok = True
    try:
        public = api.list_repo_files(comp.hf_repo, repo_type="dataset", token=token)
    except Exception as e:
        print(f"  public repo not readable: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(f"\n  {comp.hf_repo} contains: {sorted(f for f in public if f != '.gitattributes')}")
    for f in PUBLIC_FILES:
        if f not in public:
            print(f"  MISSING from public repo: {f}", file=sys.stderr)
            ok = False

    # The check that matters.
    for bad in SECRET_FILES:
        if any(Path(f).name == bad for f in public):
            print(f"\n  *** {bad} IS IN THE PUBLIC REPO — delete it now ***", file=sys.stderr)
            ok = False
    if ok:
        print(f"  no answer key in the public repo ✓")

    try:
        info = api.repo_info(comp.hf_solution_repo, repo_type="dataset", token=token)
        print(f"  {comp.hf_solution_repo} private={info.private} ✓"
              if info.private else
              f"  *** {comp.hf_solution_repo} IS PUBLIC — make it private ***")
        ok = ok and bool(info.private)
    except Exception as e:
        print(f"  solution repo not readable: {type(e).__name__}: {e}", file=sys.stderr)
        ok = False

    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true", help="only check what is on the Hub")
    ap.add_argument("--token", default=None, help="write token; defaults to the logged-in one")
    args = ap.parse_args()

    comp = COMPETITIONS.get(args.week)
    if comp is None:
        print(f"week {args.week} has no competition defined", file=sys.stderr)
        return 2
    if args.verify:
        return verify(comp, args.token)
    return publish(comp, args.token, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
