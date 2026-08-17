# How ads-ml-lab works

A twelve-week course that rebuilds ads measurement from scratch, plus a Kaggle-style judge
that scores submissions. This document explains what every piece **is** and how they fit —
the repo's other docs explain how to *run* things.

## Three different people use this repo

They touch almost nothing in common, which is the main source of confusion.

| | **The entrant** | **The learner** | **The operator** |
|---|---|---|---|
| Wants to | climb the leaderboard | do the twelve weeks | run the thing |
| Installs | nothing | the full Python stack | the full stack |
| Downloads | 52 MB from the judge | 4.2 GB of raw Criteo data | both |
| Touches the notebooks | no | yes, they are the point | maybe |
| Sees the answer key | **never** | never | yes, it is on their machine |

Most confusion comes from mixing the entrant's world (two CSV files and a web page) with
the learner's world (4.2 GB of raw data and twelve notebooks). They barely overlap.

---

## Where the data comes from

Four hops, each producing something different:

```
[1] Criteo publishes the raw log on Hugging Face
      criteo/criteo-attribution-dataset · 16,468,027 impressions · 30 days · CC-BY-NC-SA
                    │
                    │  make data          (tools/fetch_datasets.py)
                    ▼
[2] Your Mac: data/raw/  ·  4.2 GB
      The source. Used by the notebooks, and to build the competition. Never deployed.
                    │
                    │  make judge-data    (judge/prepare_data.py)
                    ▼
[3] Your Mac: judge/data/week01/  ·  58 MB
      train.csv.gz · test.csv.gz · sample_submission.csv.gz · solution.parquet · facts.json
                    │
                    │  make judge-publish (tools/publish_competition.py)
                    ▼
[4] Hugging Face
      aditagar/ads-ml-lab-week01     PUBLIC   the three entrant files + facts.json
      aditagar/ads-ml-lab-solutions  PRIVATE  solution.parquet
```

**An entrant never touches hops 1–3.** They get two files from step 4, via the judge.

**A learner doing the notebooks needs hop 2** — the full 4.2 GB — because Week 4 needs
conversion timestamps, Week 6 needs user journeys and Week 11 needs sequences. None of that
survives into the competition files. `make data` pulls from Criteo's official Hugging Face
repos; the `go.criteo.net` links in older blog posts are dead.

---

## The five files, and what each one actually is

All built by `judge/prepare_data.py` from the raw log.

### `train.csv.gz` — 39 MB, 1,500,000 rows, public

What an entrant learns from.

```
impression_id  timestamp  cat1..cat9  click  click_pos  click_nb  cost  time_since_last_click  conversion
```

`conversion` is included on purpose — this is the half of the data where you are allowed to
see outcomes.

### `test.csv.gz` — 10 MB, 400,000 rows, public

Identical columns **minus `conversion`**. The entrant predicts that missing column.

A precision note, because the code contains a guard that looks like more than it is: the
feature set never included `attribution`, `conversion_timestamp`, `conversion_id` or `cpo`,
so those are not *stripped* from the test file — they were never in either file. The `leaky`
assertion in `prepare_data.py` exists to catch a **future** edit that adds one of them to
`FEATURES`. `attribution` is the one to watch: it is Criteo's own last-click flag and is
nonzero only on converting journeys, so it would hand over the label under another name.

### `sample_submission.csv.gz` — 0.9 MB

The exact format the scorer accepts, filled with the training base rate for every row.
Submitting it unmodified is a valid entry scoring NE ≈ 1.0. It exists so nobody has to guess
the format.

```
impression_id,prediction
0,0.049104
1,0.049104
```

### `solution.parquet` — 1.9 MB, **private**

**The answer sheet.** Three columns, one row per test impression:

```
 impression_id  conversion  is_public
             0           0      False
             1           1       True
             2           0      False
             3           0       True
```

- `impression_id` — joins to the entrant's submission.
- `conversion` — **the true label** removed from `test.csv.gz`. Outside the raw Criteo data,
  this file is the only place it exists.
- `is_public` — whether the row counts toward the *public* leaderboard (30%) or the
  *private* one (70%).

If this file were public, anyone could score perfectly by submitting the `conversion` column
verbatim. That is why it lives in a private repo and is the one file the judge never serves:
downloads resolve against a three-entry allow-list of filenames, so no URL reaches it.

### `facts.json` — under 1 KB, public

Row counts, base rate and file sizes, so the web page can show them without reading a 39 MB
gzip on every page view.

---

## How the answer sheet is built

Run once by the operator, via `make judge-data`. The whole recipe, with real numbers:

**1. Load the raw log** — 16,468,027 impressions spanning 30.9 days.

**2. Cut by time, at the 70th percentile of `timestamp`:**

```
cut = t 1,848,196s  (day 21.4)
    11,527,618 impressions before  →  train pool
     4,940,409 impressions after   →  test pool
```

The order matters and is easy to get backwards: **split first, subsample second.**
Subsampling first would let a test row predate a train row, leaking the future into
training. `prepare_data.py` asserts `train.timestamp.max() < test.timestamp.min()`.

**3. Subsample** to 1,500,000 train and 400,000 test rows, seeded, so a submission is a few
MB rather than fifty.

**4. Assign `impression_id`** — a plain 0..n-1 index. It means nothing upstream; it exists
only as the join key for scoring.

**5. Write the three public files,** and separately the answer sheet: `impression_id`, the
true `conversion`, and the public/private flag.

**6. Decide public vs private deterministically,** not randomly:

```python
h = blake2b(f"{slug}:{impression_id}").digest()
is_public = (h % 10_000) < 3_000        # 30%
```

Keying on a hash of the id rather than a coin flip means re-running `prepare_data` never
reshuffles which rows are public, so scores already on the board stay comparable. Result:
120,021 public rows (30.0%), 279,979 private.

The base rate barely moves between the halves — 4.86% public, 4.88% private — which is what
you want: the public leaderboard should be an unbiased preview of the private one.

The label also drifts across the window: train converts at 4.91%, test at 4.88%. That is
real, it is a mild distribution shift, and it is part of the problem rather than a bug.

---

## How the judge scores a submission

`judge/scoring.py`, on every upload.

**Validation first.** Each failure returns a specific message rather than "invalid file":

1. Parse CSV, detecting gzip by **magic bytes**, not the filename.
2. Require `impression_id` and `prediction` columns.
3. Predictions must be numeric, finite and within `[0, 1]` — if they look like logits, the
   error says so.
4. No duplicate ids.
5. The id set must match the test set **exactly**; missing and unknown ids are each reported
   with counts and examples.

**Then scoring:**

```
load solution.parquet          (once per boot, then cached)
merge the submission onto it by impression_id
split rows by is_public
  ├── public  (120,021 rows)  → the score shown on the leaderboard
  └── private (279,979 rows)  → recorded, not shown
compute on each half: NE, AUC, log-loss, PR-AUC, calibration ratio
```

The same `adslab.metrics` functions score the notebooks, so a local number and a leaderboard
number can never disagree about the metric — only about the split or the join.

### Why the leaderboard ranks on NE, not AUC

**Normalised entropy** is log-loss divided by the log-loss of always predicting the base
rate. Exactly 1.0 for a constant predictor; below 1.0 means the model earns its keep.

AUC measures only *ranking*, and is unchanged by any monotone rescaling — multiply every
prediction by 10 and AUC does not move. A model with excellent AUC and a calibration ratio
of 3.0 would overbid every auction threefold. NE is a proper scoring rule: it sees ranking
**and** calibration, and cannot be gamed by rescaling.

AUC and calibration ratio sit beside it on the board because the disagreement between them
is the interesting part — and the leaderboard chart plots exactly that.

### Why two leaderboards

The public 30% updates live. The private 70% stays hidden. Tune against the public score
long enough and you start fitting its noise; the private score is what reveals whether you
built a model or tuned a leaderboard. Same arrangement as Kaggle, same reason.

---

## User journeys

### An entrant, start to finish

1. Opens the judge, sees Week 1.
2. **Task tab** — reads what to predict and how it is scored.
3. **Data tab** — clicks Download. The judge replies `307 Temporary Redirect` and the
   browser fetches from Hugging Face's CDN. *The 39 MB never passes through the judge* — the
   server transfers about 200 bytes.
4. Trains anything they like, anywhere. No install, and no account needed to download.
5. **Signs up** — email, display name, password. The password is hashed with argon2id and
   the plaintext is never stored. An account exists only to attribute a leaderboard row.
6. **Submit tab** — drops in a CSV of `impression_id,prediction`. Scored in seconds.
7. Sees their public NE, their rank, and how far above the base-rate baseline they are.
8. Up to 20 *scored* submissions per 24h. Rejected files do not count — the cap exists to
   stop leaderboard overfitting, not to punish a bad header.

### A learner doing the twelve weeks

```bash
make setup     # install what the system python is missing
make data      # 4.2 GB of raw Criteo data — the real thing, with timestamps and journeys
make papers    # 44 PDFs into the week folders
make verify    # asserts the splits are leak-free; prints a profile of every dataset
make lab       # open the notebooks
```

The notebooks are **scaffolded, not solved**: the harness, splits and evaluation are done;
the modelling is the exercise. Optionally submit a Week 1 model to the judge to see it
ranked against everyone else.

### The operator

```bash
make data           # 4.2 GB source
make judge-data     # build the competition + answer sheet
make judge-publish  # push public files and the private answer key to Hugging Face
make judge          # run locally at :8000
make judge-test     # 58-check end-to-end smoke test
```

**Adding Week 2:** define a `Competition` in `judge/competitions.py`, write
`prepare_week2()` in `judge/prepare_data.py`, publish. Routes, scoring, leaderboard and
templates need no changes — that generality is why Week 1 was worth building carefully.

---

## What is secret, and what happens if it leaks

| Secret | Where | If it leaked |
|---|---|---|
| `solution.parquet` | private HF repo | **The competition is over.** Anyone scores perfectly. |
| `JUDGE_SECRET_KEY` | host env var | Sessions can be forged. Rotate it; everyone is signed out. |
| `HF_TOKEN` | host env var | Read access to private repos → the answer key. Revoke it. |
| Password hashes | `judge.db` | argon2id with a per-user salt — passwords are not recoverable. |

The judge stores no payment data, no personal data beyond an email address, and never sees a
plaintext password after the moment of signup.

---

## Glossary of the confusing names

| Name | What it is |
|---|---|
| impression | One ad shown once. The row unit of the whole dataset. |
| `conversion` | Whether the user who saw it converted within 30 days. **Impression-level** — every impression in a converting journey carries a 1. A CVR target, not attribution. |
| `impression_id` | A join key invented during preparation. Means nothing upstream. |
| `solution.parquet` | The answer sheet: true labels plus the public/private flag. Private. |
| `is_public` | Whether a test row counts toward the visible leaderboard (30%) or the hidden one (70%). |
| NE | Normalised entropy. Log-loss ÷ base-rate log-loss. Lower is better; 1.0 means no better than guessing the average. |
| calibration ratio | mean(prediction) ÷ mean(label). 1.0 is unbiased; 1.15 means every bid is 15% too high. |
| `judge.db` | SQLite: accounts and submissions. The only thing that changes at runtime, and the only thing needing backup. |
| `adslab` | The shared Python package. The same code scores the notebooks and the leaderboard. |
| baseline | A seeded leaderboard entry to beat: the base rate (NE 1.00001) and a logistic regression (NE 0.80457). |

---

See also: [`deployment.md`](deployment.md) for where all of this runs and what it costs,
and [`eval-protocol.md`](eval-protocol.md) for the rules every week's evaluation follows.
