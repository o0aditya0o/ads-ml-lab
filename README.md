# ads-ml-lab

Twelve weeks of building the core model classes of ads measurement from scratch, on public
Criteo data, ending in a capstone that assembles them into one privacy-era measurement
stack.

Every model is built against a dumb baseline, evaluated on a time-ordered split, and
scored with the same metric bundle from week 1 to week 12 — so the numbers in week 9 are
comparable to the numbers in week 1, which is the whole reason for the shared harness.

---

## Quick start

```bash
make setup     # installs the handful of packages the system python lacks, registers the kernel
make data      # ~4.2 GB of Criteo data into data/raw/
make papers    # 44 PDFs into weekNN_*/papers/
make verify    # end-to-end check; prints a profile of every dataset
make test      # harness contract tests
make lab       # JupyterLab
```

`make verify` is the one that matters. It loads every dataset, asserts the splits are
leak-free, and prints the row counts, base rates, cardinalities and delay distribution
you need before modelling anything.

## Layout

```
adslab/          shared harness — the only code that must stay identical across 12 weeks
  data.py          loaders + the field semantics that actually matter (read the docstrings)
  split.py         time_split, user_grouped_time_split, check_no_leakage
  metrics.py       AUC / log-loss / NE / calibration ratio; ECE and Qini are exercises
  calibration.py   Platt, isotonic, sampling-rate correction — all Week 3 exercises
  encoders.py      feature hashing with a collision report
  plots.py         consistent figures: reliability, lift, delay, prediction histogram
  registry.py      append-only results.jsonl → the comparison table
weekNN_*/        one folder per week
  *.ipynb          the workbook — scaffolded, with the modelling left to do
  papers/          that week's reading, downloaded, with an index
  figures/         saved plots
  README.md        the write-up: what I built / what the numbers say / what surprised me
tools/           fetchers, notebook generator, setup verifier
tests/           contract tests for the harness
results/         results.jsonl — every run, including the losing ones
data/            gitignored; rebuild with `make data`
```

## The harness contract

Three things every week does identically:

```python
from adslab import data, metrics, plots, split, registry

df = data.add_attribution_derived(data.load_attribution())
sp = split.time_split(df, "timestamp")      # never random
split.check_no_leakage(df, sp)              # assertion, not convention

registry.log_result(week=1, model="lightgbm_hashed_2^18",
                    metrics=metrics.evaluate(y_test, p_test),
                    notes="the one sentence you would say out loud about this run")
```

### What is deliberately unfinished

`metrics.ece`, `metrics.reliability_curve`, `metrics.qini_auc` and the three calibrators
raise `TodoError`. They are the Week 3 and Week 7 exercises. Their **signatures and
documented requirements are fixed now** so that everything written before them still
slots in afterwards, and `tests/test_harness.py` carries an `xfail` test for each. Today
`make test` is green (11 passed, 6 xfailed); when you implement one correctly its test
reports `XPASS` and you delete the marker.

`metrics.evaluate` returns `None` for unimplemented metrics rather than raising, so a
Week 1 notebook runs end to end before Week 3 exists.

## Datasets

All from Criteo's official Hugging Face org. The `go.criteo.net/...` links that circulate
in blog posts are dead (404); `tools/fetch_datasets.py` documents what replaced them.

| dataset | size | rows | used in |
|---|---|---|---|
| `criteo/criteo-attribution-dataset` | 623 MB | 16.5M impressions, 30 days, 6.1M users, 675 campaigns | 1, 4, 6, 8, 11 |
| `criteo/criteo-uplift` | 297 MB | 13.98M rows, 85% treated | 7 |
| `criteo/CriteoPrivateAd` | 3.1 GB | 10.3M rows, days 1–14, features tagged by privacy bucket | 1, 10 |
| `criteo/FairJob` | 182 MB | 1.07M rows, protected attribute (bonus) | 10 |

`criteo/CriteoClickLogs` (276 GB) is deliberately not fetched; see the script for how to
pull a few parts if Week 2 wants more scale.

### Three traps found while building this, all of them silent

1. **The uplift CSV is grouped by treatment.** The first 300k rows are 100% treated and
   so are the last 300k. Any `nrows` head-read hands you an experiment with no control
   arm and uplift estimates of `nan`. `load_uplift` therefore always subsamples randomly,
   and `verify_setup` asserts the control group survived.
2. **CriteoPrivateAd has zero-row `part-00000` files** in most day partitions. Taking
   parts in lexical order downloads empty frames. Parts are selected by blob size.
3. **`conversion` on the attribution dataset is impression-level**, meaning "this user
   converted within 30 days" — not "this impression caused a conversion". Every
   impression in a converting journey carries `conversion=1`. Predicting it is a fine CVR
   task; calling it attribution is not.

## Reading

44 of 45 papers download automatically into the week that uses them. The one gap is
Chapelle 2014 (Week 4): the author's domain has lapsed and now hosts an unrelated content
farm, and the Wayback mirror was returning 503. `week04_delayed_feedback/papers/README.md`
explains how to get it and why you are not blocked without it.

Papers are committed to the repo, ~56 MB, so a fresh clone is immediately readable
offline. They are author copies and preprints from publicly accessible sources, each
traceable to its origin URL in `tools/fetch_papers.py`, and they are **not** covered by
this repo's MIT licence — see [`LICENSE`](LICENSE). Rights-holders who would rather not be
mirrored here can open an issue; `make papers` will still fetch on demand.

## Progress

| week | topic | status |
|---|---|---|
| 01 | Data plumbing and a CVR baseline | not started |
| 02 | Deep CVR — FM, Wide & Deep, DLRM | not started |
| 03 | Calibration | not started |
| 04 | Delayed feedback | not started |
| 05 | Missing labels and conversion modeling | not started |
| 06 | Multi-touch attribution | not started |
| 07 | Uplift and incrementality | not started |
| 08 | Auction simulation and bid shading | not started |
| 09 | Budget pacing and CPA control | not started |
| 10 | Privacy-constrained learning | not started |
| 11 | Sequence models for user journeys | not started |
| 12 | Capstone — privacy-era measurement stack | not started |

Run `make table` for the live results across all weeks.

## Environment

Homebrew Python 3.10 at `/opt/homebrew/bin/python3.10`, which already carries torch 2.10
(MPS available), tensorflow, lightgbm, xgboost, scikit-learn and JupyterLab. No virtualenv
— `make setup` adds the four or five missing packages and registers an `ads-ml-lab`
kernel. Swap to a venv if you prefer; nothing here depends on the global install beyond
the interpreter path in the `Makefile`.
