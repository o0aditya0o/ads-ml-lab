#!/usr/bin/env python3
"""Generate the twelve week notebooks from one spec.

Why generated rather than hand-written: the twelve weeks share a spine (same imports,
same split, same eval, same logging call) and the only way that spine stays identical
across three months of work is if one file owns it. Change the template here and
regenerate; the per-week content lives in ``SPEC`` below.

    python tools/make_notebooks.py            # write any notebook that doesn't exist
    python tools/make_notebooks.py --force    # overwrite ALL of them (destroys your work)
    python tools/make_notebooks.py --week 5 --force

Safety: without ``--force`` an existing notebook is never touched, so this is safe to
re-run after you have started filling one in.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------------------
# cell helpers
# --------------------------------------------------------------------------------------
def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.rstrip().split("\n")}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.rstrip().split("\n")}


def _lines(cell: dict) -> dict:
    """nbformat wants each line to keep its newline except the last."""
    src = cell["source"]
    cell["source"] = [l + "\n" for l in src[:-1]] + [src[-1]] if src else []
    return cell


SETUP = """\
import sys, warnings
sys.path.insert(0, "..")
warnings.filterwarnings("ignore", category=FutureWarning)

%load_ext autoreload
%autoreload 2

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from adslab import data, metrics, plots, split, registry, encoders, calibration

plots.use_style()
pd.set_option("display.width", 140, "display.max_columns", 60)
print("harness ready")"""


def header(week: int, title: str, goal: str, deliverable: str, hours: str) -> list[dict]:
    return [
        md(f"""\
# Week {week:02d} — {title}

**Goal.** {goal}

**Deliverable.** {deliverable}

**Rough shape of the week.** {hours}

---
### Ground rules (they apply every week)

1. **Beat a dumb baseline or it didn't happen.** Logistic regression or the global mean.
   Log the baseline in the same table as the fancy model.
2. **Split by time, never at random.** `split.time_split` — and call
   `split.check_no_leakage` so the assertion, not your memory, enforces it.
3. **Log every run** with `registry.log_result(...)`, including the ones that lost.
   The losing runs are what make the write-up honest.
4. **Write the finding down** in this week's `README.md` while it is fresh."""),
        md("### Reading\n\nPDFs are in `papers/` next to this notebook — see `papers/README.md`."),
        code(SETUP),
    ]


def footer(week: int) -> list[dict]:
    return [
        md("""\
---
## Log the results

Every model you tried, including the baseline and including the failures. `notes` is the
one sentence you would say out loud about the run — future-you assembles the write-up
from these, so write it now while you still remember why the run mattered."""),
        code(f"""\
# registry.log_result(
#     week={week},
#     model="lightgbm_hashed_2^18",
#     metrics=metrics.evaluate(y_test, p_test),
#     dataset="attribution",
#     params=dict(n_bits=18, num_leaves=63, lr=0.05),
#     notes="beats LR by 0.011 AUC; most of the gain is from cat3 x cat7 interactions",
# )

print(registry.to_markdown(week={week}))"""),
        md(f"""\
---
## Write it up

Open `README.md` in this folder and fill in the three sections. Keep it to a page.

- **What I built** — one paragraph, no code.
- **What the numbers say** — paste the table above; say which comparison is the honest one.
- **What surprised me** — the part worth reading. If nothing surprised you, you probably
  did not stress the model hard enough.

Then commit:

```bash
git add week{week:02d}_* results/
git commit -m "week {week:02d}: <the finding, not the task>"
```"""),
    ]


# --------------------------------------------------------------------------------------
# per-week content: list of (markdown, code) pairs
# --------------------------------------------------------------------------------------
LOAD_ATTRIBUTION = """\
df = data.add_attribution_derived(data.load_attribution())
print(f"{len(df):,} rows, {df.timestamp.max()/86400:.1f} days")

sp = split.time_split(df, "timestamp", train_frac=0.7, val_frac=0.1)
split.check_no_leakage(df, sp)
print(sp)

train, val, test = sp.apply(df)"""

SPEC: dict[int, dict] = {
    1: dict(
        title="Data plumbing and a CVR baseline",
        goal=("Get both Criteo datasets loading, profile them honestly, and establish the "
              "baseline that the next eleven weeks are measured against."),
        deliverable="Repo scaffold, a logistic-regression and a LightGBM CVR baseline, metrics table in the README.",
        hours="2h reading (FTRL) · 5h building · 1h write-up.",
        cells=[
            ("""\
## 1. Profile before you model

You cannot make a good modelling decision about a dataset you have not looked at. Answer
these in the cells below and write the answers into the README — several of them will
change what you do in weeks 4, 6 and 11.

- How many rows, users, campaigns? Over what time window?
- What is the conversion base rate? The click rate?
- What are the cardinalities of `cat1..cat9`? Which are high enough that one-hot is
  impossible?
- What fraction of impressions belong to users with more than one impression? (That is
  your Week 6 and 11 population.)
- How does the conversion rate drift across the 30 days? Is the last day comparable to
  the first?

Note the trap documented in `data.load_attribution`: `conversion` is an *impression-level*
label meaning "this user converted within 30 days", not "this impression caused it".""",
             """\
df = data.add_attribution_derived(data.load_attribution())
print(f"rows={len(df):,}  users={df.uid.nunique():,}  campaigns={df.campaign.nunique():,}")
print(f"conversion={df.conversion.mean():.4%}  click={df.click.mean():.4%}")

df.head()"""),
            ("""\
### Cardinality and sparsity

The number that matters for Week 1 is *distinct values per categorical column*.

Expect a surprise: the `cat*` values are enormous integers, but their distinct counts are
small — around 59k across all nine columns, and only `cat7` is above 2k. **One-hot
encoding is entirely feasible on this dataset.** Do it, and treat the hashed version as
the comparison. Hashing is what production uses because the vocabulary is unknown and
drifting, not because 59k values are too many — and here you can measure exactly what the
collisions cost, which you never can at real scale.""",
             """\
card = pd.DataFrame({
    "distinct": [df[c].nunique() for c in data.CAT_FEATURES],
}, index=data.CAT_FEATURES)
card["pct_of_rows"] = card.distinct / len(df)
card"""),
            ("""\
### Drift across the window

Plot daily conversion rate and daily volume. If the level moves, a model trained on days
1–21 is already slightly wrong about day 30 before you add any other problem — and that
is the mildest version of the distribution shift the rest of the course is about.""",
             """\
daily = df.groupby("day").agg(rows=("conversion", "size"), cvr=("conversion", "mean"))

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
ax[0].plot(daily.index, daily.cvr); ax[0].set_ylabel("CVR")
ax[1].bar(daily.index, daily.rows);  ax[1].set_ylabel("impressions"); ax[1].set_xlabel("day")
fig.suptitle("Volume and conversion rate across the window")
print(plots.save(fig, 1, "daily_drift"))"""),
            ("""\
## 2. The split

Everything downstream depends on this being time-ordered. `check_no_leakage` turns that
from a convention into an assertion.

**Worth doing once, for the shock value:** build a random split too, train the same model
on both, and record the AUC gap. That gap is the size of the lie a random split tells you.
Log it as a separate row called `lr_hashed_RANDOM_SPLIT_do_not_trust`.""",
             LOAD_ATTRIBUTION),
            ("""\
## 3. Baseline 1 — logistic regression on hashed features

The classic. `HashingEncoder` is in the harness; `n_bits` is yours to sweep.

Use `sklearn.linear_model.SGDClassifier(loss="log_loss")` if the full matrix is heavy,
or `LogisticRegression(solver="liblinear")` on a subsample to start.

Look at `enc.collision_report(train)` before you interpret a bad result — at low
`n_bits` you are measuring the hash, not the model.""",
             """\
from sklearn.linear_model import LogisticRegression

enc = encoders.HashingEncoder(data.CAT_FEATURES, n_bits=18)
print(enc.collision_report(train))

# TODO: fit on train, tune on val, report on test.
# X_train = enc.transform(train); y_train = train.conversion.values
# ..."""),
            ("""\
## 4. Baseline 2 — LightGBM

Trees want the categoricals as `category` dtype, not as hashed sparse columns — give
LightGBM the raw integer codes and let it split on them. That difference in encoding is
itself part of the trees-vs-linear story you will finish in Week 2.

Watch for: LightGBM will happily overfit `uid`. Do not feed it the user id.""",
             """\
import lightgbm as lgb

feats = data.CAT_FEATURES + ["click_pos", "click_nb", "time_since_last_click", "hour_of_day"]

# TODO: build lgb.Dataset with categorical_feature=data.CAT_FEATURES, early-stop on val.
# Then predict on test and evaluate."""),
            ("""\
## 5. Compare

Both models, same test set, same metric bundle. Then answer in the README: **which
metric moved and which did not?** If AUC improved but `calibration_ratio` got worse, you
have a better ranker and a worse bidder — and you should be able to say which one the
business wanted.""",
             """\
# results = {}
# for name, p in [("lr_hashed_2^18", p_lr), ("lightgbm", p_lgb)]:
#     results[name] = metrics.evaluate(test.conversion.values, p)
# pd.DataFrame(results).T"""),
        ],
    ),
    2: dict(
        title="Deep CVR — factorization machines, Wide & Deep, DLRM",
        goal=("Implement the embedding-based lineage of production ads models and find out, "
              "with numbers, whether it beats trees at this scale."),
        deliverable="FM + a DLRM-lite in PyTorch, a comparison table against Week 1's LightGBM, and a one-page verdict.",
        hours="2h reading (DLRM, Wide & Deep) · 6h building · 1h write-up.",
        cells=[
            ("""\
## The question this week actually answers

Not "can I implement DLRM" — you can. The question is **at what point do embeddings plus
an MLP beat gradient-boosted trees on tabular ads data, and why**. The honest published
answer (see `papers/shwartzziv2021-tabular-dl-is-not-all-you-need.pdf`) is: often they
don't, until the categorical cardinality and the row count are both large.

So the deliverable is a *curve*, not a point. Train each model at 1%, 10%, and 100% of
the rows and plot AUC against training-set size. Where the lines cross — if they cross —
is the finding.""",
             LOAD_ATTRIBUTION),
            ("""\
## 1. Embedding tables

Map each `cat*` column to a contiguous index range, then to a learned vector. Two
decisions to make deliberately and record:

- **Vocabulary**: hash to a fixed size (like Week 1) or build an index of values seen in
  train? Hashing collides; indexing has to handle unseen values at test time. Ads systems
  hash. Try both and measure the difference.
- **Embedding dim**: 16 is a reasonable default for everything. Dimension-per-feature
  proportional to `log(cardinality)` is what large systems actually do.""",
             """\
import torch
import torch.nn as nn

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

class EmbeddingBag(nn.Module):
    \"\"\"One embedding table per categorical column, concatenated.\"\"\"
    def __init__(self, cardinalities, dim=16):
        super().__init__()
        # TODO
        raise NotImplementedError"""),
            ("""\
## 2. Factorization machine

FM = linear terms + all pairwise interactions, computed in O(nd) with the classic
identity

$$\\sum_{i<j}\\langle v_i, v_j\\rangle x_i x_j = \\tfrac12\\sum_f\\Big[\\big(\\sum_i v_{i,f}x_i\\big)^2 - \\sum_i v_{i,f}^2x_i^2\\Big]$$

Implement it with that identity, not with a double loop — the point of FM is that the
interaction term is linear-time, and writing the naive version teaches you nothing except
that it is slow.""",
             """\
class FM(nn.Module):
    def __init__(self, cardinalities, dim=16):
        super().__init__()
        # TODO: linear term + the O(nd) pairwise identity above
        raise NotImplementedError"""),
            ("""\
## 3. Wide & Deep / DLRM-lite

Wide part: the hashed sparse features straight into a linear unit (memorisation).
Deep part: embeddings → MLP (generalisation).
DLRM's variation: explicit pairwise dot products between embeddings before the MLP.

Build one model with a flag that switches the interaction style, so the comparison is
apples to apples.""",
             """\
class WideAndDeep(nn.Module):
    def __init__(self, cardinalities, dim=16, mlp=(256, 128, 64), interaction="concat"):
        super().__init__()
        # interaction: "concat" (Wide&Deep) | "dot" (DLRM)
        # TODO
        raise NotImplementedError"""),
            ("""\
## 4. Training loop

Things that will bite you, in the order they usually do:

- **Class imbalance.** At a sub-1% base rate, a network happily learns to predict the
  base rate and stop. Negative downsampling fixes training speed but *breaks
  calibration* — and un-breaking it is exactly Week 3's `SamplingRateCorrector`. If you
  downsample, record the rate.
- **Embedding LR.** Sparse embedding tables usually want a much higher learning rate
  than the dense MLP. One global LR is the most common reason a DLRM underperforms.
- **Early stopping on val log-loss, not AUC.** You are going to calibrate this thing.""",
             """\
def train_epoch(model, loader, opt, loss_fn):
    # TODO
    raise NotImplementedError"""),
            ("""\
## 5. The scaling curve

Train every model at several training-set sizes and plot AUC vs rows. This is the plot
that answers the week's question.""",
             """\
# fractions = [0.01, 0.1, 0.5, 1.0]
# rows: model x fraction -> auc, then plot
# fig, ax = plt.subplots(); ...
# print(plots.save(fig, 2, "auc_vs_training_size"))"""),
        ],
    ),
    3: dict(
        title="Calibration",
        goal=("Implement ECE and reliability diagrams from scratch, then break a model's "
              "calibration on purpose and repair it."),
        deliverable="A working `adslab.calibration` module, before/after reliability diagrams, and the sampling-rate correction derived by hand.",
        hours="2h reading (Guo, Kumar) · 5h building · 2h write-up.",
        cells=[
            ("""\
## Why this week is the highest-leverage one in the course

AUC cannot see calibration. A model that multiplies every prediction by 3 has *identical*
AUC and will overbid every auction by 3x. Ranking metrics are what get published;
calibration is what gets billed.

This week you fill in the stubs in `adslab/metrics.py` and `adslab/calibration.py`. The
contract is already written — `tests/test_harness.py` has an `xfail` test for each one.
Implement, run `python -m pytest -q`, and delete the `xfail` marker when it XPASSes.""",
             """\
!cd .. && python -m pytest tests -q"""),
            ("""\
## 1. Look at the predictions first

Before any metric: where do the predictions live? On a 0.2% base rate almost all of the
mass is below 0.02, which is why equal-width bins are useless here and why the default
`strategy` in `reliability_curve` is `"quantile"`.""",
             """\
# Load a model's predictions from Week 1 or 2 (or refit quickly here).
# y_val, p_val, y_test, p_test = ...

fig, ax = plt.subplots()
plots.prediction_histogram(p_val, ax=ax)
print(plots.save(fig, 3, "prediction_distribution"))"""),
            ("""\
## 2. Implement `reliability_curve`, `ece`, `max_calibration_error`

In `adslab/metrics.py`. Requirements are in the docstrings; the tests enforce them.

Then read `papers/kumar2019-verified-uncertainty-calibration.pdf` and answer in the
README: **is your ECE an underestimate, and by roughly how much?** Compute it at
`n_bins` in {10, 20, 50, 100, 200} and plot ECE against bin count. If the curve is
still rising at 200 bins, your ECE is a lower bound and you should report it as one.""",
             """\
# after implementing:
# for nb in (10, 20, 50, 100, 200):
#     print(nb, metrics.ece(y_test, p_test, n_bins=nb))"""),
            ("""\
## 3. Reliability diagram

`plots.reliability_diagram` starts working as soon as `reliability_curve` does.

Read it as a bidder: points below the diagonal mean you will overbid.""",
             """\
# fig, ax = plt.subplots()
# plots.reliability_diagram(y_test, p_test, label="uncalibrated", ax=ax)
# print(plots.save(fig, 3, "reliability_uncalibrated"))"""),
            ("""\
## 4. Platt and isotonic

Implement both in `adslab/calibration.py`. **Fit on validation, evaluate on test.**
Fitting a calibrator on training predictions corrects a distortion that does not exist
at serving time and makes test calibration worse — do it once deliberately and record
the number, because seeing it is how you remember it.

Then check what isotonic costs you in AUC. It produces ties, and ties destroy
fine-grained ranking. If AUC drops, you have found the real trade-off.""",
             """\
# platt = calibration.PlattScaler().fit(p_val, y_val)
# iso   = calibration.IsotonicCalibrator().fit(p_val, y_val)
# for name, cal in [("platt", platt), ("isotonic", iso)]:
#     pc = cal.transform(p_test)
#     print(name, metrics.evaluate(y_test, pc))"""),
            ("""\
## 5. Break it on purpose, then fix it in closed form

The stress test from the plan: **drop 50% of the positives** from training (simulating
conversion loss), retrain, and watch calibration collapse while AUC barely moves. That
divergence is the entire argument for this week.

Then fix it with a known sampling rate. Derive the correction yourself — the formula in
`SamplingRateCorrector`'s docstring is for dropped *negatives*, and the positive-dropping
case is **not** symmetric. Getting this derivation right is what makes Week 5 easy,
because privacy-driven conversion loss has exactly this shape.""",
             """\
# 1. subsample positives in train at rate r
# 2. retrain the same model
# 3. evaluate on the UNMODIFIED test set
# 4. compare auc (barely moves) vs calibration_ratio (blows up)
# 5. apply your correction, re-evaluate"""),
        ],
    ),
    4: dict(
        title="Delayed feedback",
        goal=("Model the conversion lag distribution and correct the bias it puts into a "
              "model trained on fresh traffic."),
        deliverable="Naive vs. Chapelle exponential-delay vs. a Weibull survival variant, plus a predicted-vs-actual lag plot.",
        hours="2h reading (Chapelle) · 6h building · 1h write-up.",
        cells=[
            ("""\
## The bias, stated precisely

Train a model today on clicks from the last 7 days, labelling "converted so far" as 1 and
everything else 0. A click from 6 days ago has had 6 days to convert; a click from 2
hours ago has had 2 hours. The recent clicks are labelled *wrong* — not noisily, but
systematically in one direction — and the model learns that recency predicts
non-conversion.

Chapelle's move: model two things jointly.

- $\\Pr(C=1\\mid x)$ — will this ever convert?
- $\\Pr(D=d\\mid C=1, x)$ — given it converts, how long does it take?

with the delay $D$ exponential, $\\lambda(x)=\\exp(w_d\\cdot x)$. An unconverted click at
elapsed time $e$ then contributes $\\Pr(C=0) + \\Pr(C=1)\\Pr(D>e)$ instead of a hard zero.""",
             LOAD_ATTRIBUTION),
            ("""\
## 1. The lag distribution

`add_attribution_derived` already gives you `conversion_delay_hours`. Plot it on a log
axis. Report: median delay, p90, and **the fraction of conversions arriving after 24
hours** — that last number is the size of the problem.

Then check whether the delay depends on features. If `lambda` is genuinely constant
across campaigns, the fancy model buys you nothing over a global correction, and knowing
that is worth an hour.""",
             """\
d = df.conversion_delay_hours.dropna()
print(f"n={len(d):,}  p50={d.median():.2f}h  p90={d.quantile(.9):.1f}h  "
      f">24h: {(d > 24).mean():.1%}  >7d: {(d > 168).mean():.1%}")

fig, ax = plt.subplots()
plots.conversion_delay_hist(d, ax=ax)
print(plots.save(fig, 4, "conversion_delay_distribution"))"""),
            ("""\
## 2. Build the biased world

Pick an observation cutoff `T` inside the window. Everything after `T` is unobservable
"future". Relabel training rows as `converted_by_T`, keeping the true 30-day label aside
for evaluation only.

This is the single most important cell of the week: **the true label must never touch
training**, only evaluation. If it leaks, every result this week is meaningless.""",
             """\
T = df.timestamp.quantile(0.7)

obs = df[df.timestamp < T].copy()
obs["y_observed"] = ((obs.conversion == 1) & (obs.conversion_timestamp < T)).astype(int)
obs["elapsed"] = T - obs.timestamp

print(f"true CVR={obs.conversion.mean():.4%}  observed-by-T CVR={obs.y_observed.mean():.4%}")
print(f"-> {1 - obs.y_observed.sum()/obs.conversion.sum():.1%} of true positives are still invisible at T")"""),
            ("""\
## 3. Model A — naive

Train on `y_observed`. Evaluate against the *true* label. Then break the evaluation down
by how fresh the impression was: bucket test rows by `elapsed` and plot
`calibration_ratio` per bucket. The naive model should be badly biased on the freshest
bucket and fine on the oldest. That plot is the deliverable.""",
             """\
# TODO: train naive model, then:
# obs["elapsed_bucket"] = pd.qcut(obs.elapsed, 10, labels=False)
# per-bucket calibration_ratio -> plot"""),
            ("""\
## 4. Model B — Chapelle's delayed-feedback model

Two parameter vectors, trained jointly by maximising

$$\\log L = \\sum_{\\text{converted}}\\big[\\log p(x) + \\log\\lambda(x) - \\lambda(x)d\\big]
        + \\sum_{\\text{not yet}}\\log\\big[1-p(x) + p(x)e^{-\\lambda(x)e}\\big]$$

Implement it in PyTorch (autograd handles the gradients; the paper's hand-derived
gradients are for a 2014 LR system). Two heads on shared features: `p` through a sigmoid,
`log_lambda` linear.

Numerical warning: $\\lambda$ wants to run to 0 or ∞ on segments with few conversions.
Clamp `log_lambda` to something like [-12, 4] and say so in the write-up.""",
             """\
import torch, torch.nn as nn

class DelayedFeedbackModel(nn.Module):
    \"\"\"Two heads: conversion probability p(x), and exponential delay rate lambda(x).\"\"\"
    def __init__(self, n_features):
        super().__init__()
        # TODO
        raise NotImplementedError

    def loss(self, x, y_observed, elapsed):
        # TODO: the log-likelihood above
        raise NotImplementedError"""),
            ("""\
## 5. Model C — Weibull

Exponential assumes a constant hazard: a click is as likely to convert in its 100th hour
as its 1st, given it hasn't yet. Look at your own lag histogram — is that true? Almost
certainly not; there is a spike in the first minutes.

Weibull adds a shape parameter $k$: $\\Pr(D>d) = e^{-(\\lambda d)^k}$. One extra parameter.
Does it actually help, or does it just fit the training lag better without improving the
CVR estimate? Report both.""",
             """\
# TODO: swap the survival term for Weibull, refit, compare"""),
            ("""\
## 6. Verdict

Three models, one table, evaluated on the true label. Then the plot that tells the story:
calibration ratio by elapsed-time bucket, all three models on one axis. The naive line
should be dramatically wrong on the left and converge on the right.""",
             """\
# fig, ax = plt.subplots()
# ... one line per model
# print(plots.save(fig, 4, "calibration_by_elapsed_time"))"""),
        ],
    ),
    5: dict(
        title="Missing labels and conversion modeling",
        goal=("Rebuild, on open data, the thing you owned at Google: estimating conversions "
              "that were never observed because of consent and signal loss."),
        deliverable="A two-model correction (observed CVR + gap model) compared against naive and oracle, written up as an explainer.",
        hours="2h reading (PU learning) · 5h building · 2h write-up — this one is the blog post.",
        cells=[
            ("""\
## The setup

Split users into a *consented* segment whose conversions you observe and a *consentless*
segment whose conversions are invisible. You keep the ground truth aside so you can
score yourself — which is the one luxury this simulation has and the real problem does
not.

Three models to compare:

| model | trained on | what it represents |
|---|---|---|
| naive | observed labels, all traffic | what you get if you ignore the problem |
| corrected | observed labels + a model of the gap | conversion modeling |
| oracle | true labels | the ceiling; unattainable in production |

The gap between **naive and oracle** is the size of the business problem. The gap between
**corrected and oracle** is how much of it you recovered. Report both as one number each
and the week has landed.""",
             LOAD_ATTRIBUTION),
            ("""\
## 1. Simulate the loss

Drop conversions for ~30% of users. Do it **by user, not by row** — consent is a property
of a person, and dropping rows at random creates a much easier, and fake, problem.

Then make it harder and more realistic: make consent *non-random*. If consent correlates
with a feature that also predicts conversion (mobile users consent less and convert
less, say), the missingness is MNAR and a naive reweighting will not save you. Do the
MCAR version first to get the pipeline working, then the MNAR version to get the finding.""",
             """\
rng = np.random.default_rng(0)
users = df.uid.unique()
consentless = set(rng.choice(users, size=int(0.3 * len(users)), replace=False))

df["consented"] = (~df.uid.isin(consentless)).astype(int)
df["y_observed"] = df.conversion * df.consented   # invisible conversions become zeros

print(f"true CVR={df.conversion.mean():.4%}   observed CVR={df.y_observed.mean():.4%}")
print(f"-> {1 - df.y_observed.sum()/df.conversion.sum():.1%} of conversions are invisible")"""),
            ("""\
## 2. Naive model

Train on `y_observed` across all traffic. Evaluate on the true label. Note what happens
to `calibration_ratio` — it should land near 0.7, i.e. you would systematically underbid
by 30%, which in a real account means losing the auctions you should have won and slowly
starving the campaign.""",
             """\
# TODO"""),
            ("""\
## 3. The correction

This is the interesting part, and there is more than one defensible design. Try at least
two and argue for one:

**(a) Scale-up.** Estimate the observation rate $r$ (share of conversions you see) and
divide. Trivial, and correct only if consent is independent of everything. Establish it
as the baseline for the *correction*, not just for the model.

**(b) Two-model / imputation.** Train the CVR model on the consented segment only, where
labels are clean. Apply it to consentless traffic to *impute* expected conversions. This
is closest to how modeled conversions actually work. The catch, and the thing to write
about: if consent is MNAR, the consented segment is a biased sample and the model you
transfer is fitted to the wrong population.

**(c) PU learning.** Treat it formally: positives are reliable, "negatives" are
unlabelled. `papers/kiryo2017-nnpu.pdf` gives you a non-negative risk estimator that does
not collapse. This is the principled version of (b).""",
             """\
# TODO: implement at least (a) and (b); (c) if the week allows"""),
            ("""\
## 4. Oracle and the scoreboard

Same architecture, true labels, upper bound.

Then the table. And the sentence that matters: *"conversion modeling recovered X% of the
conversions that privacy loss made invisible, at the cost of Y in AUC."* If you cannot
fill in X and Y, the week is not finished.""",
             """\
# rows = {}
# for name, p in [("naive", p_naive), ("scale_up", p_scaled), ("two_model", p_imputed), ("oracle", p_oracle)]:
#     rows[name] = metrics.evaluate(y_true_test, p)
# pd.DataFrame(rows).T"""),
            ("""\
## 5. Where it breaks

The honest failure mode, and the part an interviewer will push on: sweep the consentless
share from 10% to 70% and plot recovery quality against it. There is a point where the
consented segment is too small or too unrepresentative to transfer from. Find it, name
it, and say what you would do past it (aggregate-only measurement — which is Week 10).""",
             """\
# for share in [0.1, 0.2, 0.3, 0.5, 0.7]:
#     ... -> plot recovery vs share
# print(plots.save(fig, 5, "recovery_vs_consent_loss"))"""),
        ],
    ),
    6: dict(
        title="Multi-touch attribution",
        goal="Replace last-click with Markov removal effects and Shapley values, and connect the credit back to bid value.",
        deliverable="An attribution library with three methods, a credit-distribution comparison, and the bidding implication.",
        hours="2h reading (Diemert) · 6h building · 1h write-up.",
        cells=[
            ("""\
## Journeys, not impressions

This week the unit of analysis changes. Group by `conversion_id` (impressions competing
for one conversion) and by `uid` (a user's whole timeline). Use
`split.user_grouped_time_split` — a plain time split cuts a journey in half and leaks the
outcome through the user id.

The dataset ships Criteo's own `attribution` column: their last-click assignment. That is
your baseline and your sanity check, not ground truth. There is no ground truth in
attribution; that is what makes it hard and what makes Week 7 necessary.""",
             """\
df = data.add_attribution_derived(data.load_attribution())

sp = split.user_grouped_time_split(df, "timestamp", "uid")
split.check_no_leakage(df, sp)   # note: approximate under grouping — read the docstring
print(sp)

journeys = (df[df.conversion == 1]
            .sort_values("timestamp")
            .groupby("conversion_id")
            .agg(path=("campaign", list), n=("campaign", "size")))
print(f"{len(journeys):,} converting journeys, median length {journeys.n.median():.0f}, "
      f"{(journeys.n > 1).mean():.1%} are multi-touch")"""),
            ("""\
## 1. Last click

Trivial to implement and it is 80% of the industry. Reproduce Criteo's `attribution`
column with your own rule and check you agree — if you don't, you have misunderstood the
data, and better to find that out now.""",
             """\
# TODO: assign credit 1.0 to the last click before the conversion, 0 otherwise
# then: agreement rate with df.attribution"""),
            ("""\
## 2. Markov chain removal effect

Model journeys as paths through a Markov chain over channels/campaigns, with absorbing
`(conversion)` and `(null)` states. The credit of channel $c$ is the **removal effect**:
how much total conversion probability drops when you delete $c$ from the graph.

Implementation notes that save an evening:

- You need *non-converting* journeys too, or every path ends in conversion and every
  removal effect is meaningless.
- Higher-order chains (remembering the last 2 steps) fit better and explode
  combinatorially. First order first.
- Removal effects do not sum to 1. Normalise at the end and say that you did.""",
             """\
# TODO: build transition matrix, compute baseline conversion prob,
# then recompute with each channel removed"""),
            ("""\
## 3. Shapley value

The axiomatically fair allocation: average marginal contribution over all orderings.
Exact computation is $O(2^n)$ over channels, so:

- with few enough distinct campaigns, compute it exactly on the *coalition* form
  (value of a set = conversion rate of journeys containing exactly that set);
- otherwise sample permutations (Monte Carlo Shapley) and report a confidence interval.
  A Shapley value without an error bar, from a sampler, is a number you cannot defend.""",
             """\
# TODO: coalition value function + exact or sampled Shapley"""),
            ("""\
## 4. Compare the credit distributions

Three methods, one bar chart of credit per campaign. Then the questions worth answering:

- Which campaigns does last-click *systematically* under-credit? (Upper-funnel ones —
  can you show it?)
- How much does total credit move? Is the reallocation big enough to change a budget
  decision, or is this all statistically indistinguishable?""",
             """\
# fig, ax = plt.subplots(figsize=(9, 4.5))
# ... grouped bars: last-click / markov / shapley
# print(plots.save(fig, 6, "credit_by_method"))"""),
            ("""\
## 5. So what — the bidding link

Attribution is only interesting because it changes what an impression is worth. Take your
Week 1 CVR model, multiply by each attribution scheme's credit to get an expected value
per impression, and show how the bid distribution shifts.

The Criteo paper's whole point is that attribution changes bidding *efficiency*. You now
have the pieces to show it, and Week 8 will plug these values into an actual auction.""",
             """\
# TODO: value = p_conversion * credit_share; compare bid distributions across schemes"""),
        ],
    ),
    7: dict(
        title="Uplift and incrementality",
        goal="Learn to predict incremental conversion rather than conversion, and evaluate it with Qini curves you wrote yourself.",
        deliverable="T-learner, class-transformation and X-learner, compared on Qini.",
        hours="2h reading (Gutierrez) · 5h building · 2h write-up.",
        cells=[
            ("""\
## The distinction the whole industry gets wrong

A CVR model finds users **likely to convert**. An uplift model finds users **who convert
*because* you showed them the ad**. These are different people, and the overlap can be
small: your best CVR segment is often people who were going to buy anyway, where the ad's
incremental value is approximately zero and possibly negative.

This is a randomised experiment, so a random split is correct here — the one dataset in
the repo where that is true. Read the `load_uplift` docstring on why `exposure` is
poison as a feature.""",
             """\
# frac= takes a RANDOM subsample. The file is grouped by treatment, so reading its
# head gives you 100% treated rows and an experiment with no control arm.
up = data.load_uplift(frac=0.2, seed=0)   # 13.98M rows is more than a laptop needs
print(f"{len(up):,} rows, treated={up.treatment.mean():.1%}")

t = up[up.treatment == 1].conversion.mean()
c = up[up.treatment == 0].conversion.mean()
print(f"conversion: treated={t:.4%} control={c:.4%}  ATE={t-c:+.4%}  lift={t/c-1:+.1%}")

X, y, w = up[data.UPLIFT_FEATURES].values, up.conversion.values, up.treatment.values"""),
            ("""\
## 1. Implement Qini first

Before any model. You cannot tune what you cannot measure, and uplift's failure mode is
that everything looks fine on AUC.

Fill in `qini_curve` and `qini_auc` in `adslab/metrics.py`. The trap is in the docstring:
the control group must be rescaled by `n_treated / n_control`, and here that ratio is
about 5.7. Skip it and a null model looks brilliant.

`tests/test_harness.py::test_qini_handles_unequal_group_sizes` checks exactly that.""",
             """\
!cd .. && python -m pytest tests -q -k qini"""),
            ("""\
## 2. T-learner (two models)

One model on treated, one on control, uplift = difference of predictions. Simple, and it
has a real weakness worth stating: each model is fitted to minimise *its own* prediction
error, and the difference of two well-fitted models can be mostly noise when the true
uplift is small relative to the outcome. Here the ATE is a fraction of a percent, so this
matters a lot.""",
             """\
# TODO"""),
            ("""\
## 3. Class transformation

Define $Z = Y\\cdot\\mathbb{1}[W{=}1] + (1-Y)\\cdot\\mathbb{1}[W{=}0]$ and fit a single
model to $Z$. Under 50/50 randomisation, $2\\Pr(Z{=}1|x)-1$ estimates the uplift directly.

**Our split is not 50/50** — it is ~85/15. Work out the propensity-weighted version
before you use it; the textbook formula silently assumes balance and will be biased here.
Deriving that correction is the most valuable twenty minutes of the week.""",
             """\
# TODO: weighted class transformation with p(W=1) = up.treatment.mean()"""),
            ("""\
## 4. X-learner

Built for exactly this situation: imbalanced groups where one arm has far more data.
Impute the treatment effect for each unit using the other arm's model, fit models to
those imputed effects, and combine them weighted by the propensity score. See
`papers/kunzel2017-metalearners.pdf` §3.

Predict: X-learner should beat T-learner *here specifically* because of the 85/15 split.
Write the prediction down before you run it.""",
             """\
# TODO"""),
            ("""\
## 5. Qini comparison and the operating point

All learners on one Qini plot. Then the practical question: **at what fraction of the
population targeted is incremental value maximised?** That number, not the AUUC, is what
a campaign manager would act on.

Also worth checking, and it makes the best slide of the week: take your top decile by
*predicted conversion* and your top decile by *predicted uplift*, and measure how much
they overlap. If it's low, you have shown the point of the week in one number.""",
             """\
# fig, ax = plt.subplots()
# ... one Qini curve per learner + random diagonal
# print(plots.save(fig, 7, "qini_comparison"))"""),
        ],
    ),
    8: dict(
        title="Auction simulation and bid shading",
        goal="Build a first-price auction, bid your CVR model into it, and learn the shading policy that maximises surplus.",
        deliverable="An auction simulator, a shading model, and a surplus-vs-aggressiveness curve.",
        hours="2h reading (bid shading) · 6h building · 1h write-up.",
        cells=[
            ("""\
## Why shading exists

In a second-price auction, bidding your true value is optimal — you pay the runner-up's
bid. First-price auctions, which the industry moved to, have no such guarantee: bid your
value and your surplus is exactly zero every time you win. So you shade: bid $b < v$,
trading win rate for margin.

The optimisation is
$$\\max_b \\;(v - b)\\cdot \\Pr(\\text{win}\\mid b)$$
and everything hinges on estimating $\\Pr(\\text{win}\\mid b)$ — the win-rate curve — from
censored data. You only observe the winning price when you win. That censoring is the
real problem of the week.""",
             """\
df = data.add_attribution_derived(data.load_attribution())
cost = df.cost[df.cost > 0]
print(f"observed price paid: p10={cost.quantile(.1):.5f} p50={cost.median():.5f} "
      f"p90={cost.quantile(.9):.5f}")

fig, ax = plt.subplots()
ax.hist(np.log10(cost.sample(200_000, random_state=0)), bins=60)
ax.set_xlabel("log10(price paid)"); ax.set_ylabel("count")
ax.set_title("Win-price distribution (Criteo, observed wins only)")
print(plots.save(fig, 8, "win_price_distribution"))"""),
            ("""\
## 1. The simulator

Keep it honest and simple: N competitors per auction, each drawing a bid from a
distribution you control. Calibrate that distribution so the simulated win prices
resemble the Criteo `cost` histogram above — then you are testing your policy against
something with the right shape.

The design decision that matters: **your simulator knows the true competing bids, and
your bidder must not.** Enforce it in the interface, or you will accidentally write an
oracle and be delighted by your own results.""",
             """\
class FirstPriceAuction:
    def __init__(self, n_competitors=5, price_dist="lognormal", seed=0):
        # TODO
        raise NotImplementedError

    def run(self, our_bid):
        \"\"\"Return (won, price_paid, highest_competing_bid_IF_WE_WON_else_None).\"\"\"
        raise NotImplementedError"""),
            ("""\
## 2. Value-based bidding, no shading

`bid = p_conversion * value_per_conversion * margin`, using your Week 1 model. Run it
through the simulator and record win rate, spend, conversions, and surplus. This is the
baseline every shading policy has to beat.""",
             """\
# TODO"""),
            ("""\
## 3. Learn the win-rate curve from censored data

The core exercise. You observe the price only on wins, so a naive fit to observed prices
is biased upward — it is a survival problem, not a regression.

Two approaches, do at least one properly:
- **Parametric**: assume the highest competing bid is lognormal and fit by maximum
  likelihood with right-censoring on losses.
- **Non-parametric**: Kaplan–Meier on the censored bid landscape.

Then verify against the simulator's known truth. That verification is the whole reason
to have built a simulator instead of only using real data.""",
             """\
# TODO"""),
            ("""\
## 4. Optimise the shade

With $\\hat{\\Pr}(\\text{win}\\mid b)$ in hand, maximise $(v-b)\\Pr(\\text{win}\\mid b)$ per
impression. Compare against fixed-multiplier shading (bid $0.7v$, $0.8v$, ...).

The plot: surplus vs. shading aggressiveness, with the learned policy marked. If your
learned policy does not beat the best fixed multiplier, say so — it is a real and common
result, and it means the win-rate curve is not varying enough per impression to be worth
modelling.""",
             """\
# fig, ax = plt.subplots()
# ... surplus vs multiplier, learned policy as a horizontal line
# print(plots.save(fig, 8, "surplus_vs_shading"))"""),
            ("""\
## 5. Feed it a miscalibrated model

The link back to Week 3, and the best experiment in this notebook. Take your Week 3
uncalibrated model — the one with a `calibration_ratio` of 1.3 — and bid with it.
Quantify the overspend in currency.

This is the moment the calibration work stops being an academic metric and becomes money.""",
             """\
# TODO: rerun the auction with calibrated vs uncalibrated predictions, compare spend & CPA"""),
        ],
    ),
    9: dict(
        title="Budget pacing and CPA control",
        goal=("Build a controller that spends a budget smoothly and hits a target CPA, then "
              "break it with delayed conversion data and fix it."),
        deliverable="A pacing + target-CPA controller and the overspend-under-delayed-data analysis. This is the startup prototype.",
        hours="2h reading (Smart Pacing) · 6h building · 1h write-up.",
        cells=[
            ("""\
## The week that matters most to you

Everything here feeds the budget-control idea. The experiment in section 5 is the
technical heart of it: **a CPA controller fed delayed conversion data overspends, and it
overspends in a specific, predictable, correctable way.** You saw this from the inside at
Google. This notebook reproduces it from the outside on open data, which is what makes it
something you can show anyone.

Build on Week 8's auction simulator and Week 4's delay model.""",
             """\
df = data.add_attribution_derived(data.load_attribution())

hourly = df.groupby(df.hour_of_day).size()
fig, ax = plt.subplots()
ax.bar(hourly.index, hourly.values)
ax.set_xlabel("hour of day"); ax.set_ylabel("impressions")
ax.set_title("Traffic shape — what the pacer has to spend against")
print(plots.save(fig, 9, "traffic_by_hour"))"""),
            ("""\
## 1. Budget pacing

Spend a daily budget smoothly against non-uniform traffic. Two families:

- **Probabilistic throttling**: participate in a fraction $\\theta$ of auctions.
- **Bid modulation**: participate always, scale the bid by $\\mu$.

Smart Pacing argues for the second on quality grounds — throttling drops good and bad
impressions indiscriminately, while modulating keeps you in the auctions you value most.
Implement both and show the difference in *what you bought*, not just in spend curve
smoothness.

The controller itself: PID on the error between actual and target cumulative spend.
Start with P only; add I when you see steady-state offset; add D last, if ever.""",
             """\
class PacingController:
    def __init__(self, daily_budget, kp=0.5, ki=0.05, kd=0.0):
        # TODO
        raise NotImplementedError

    def step(self, spent_so_far, target_so_far, dt):
        \"\"\"Return the bid multiplier for the next interval.\"\"\"
        raise NotImplementedError"""),
            ("""\
## 2. Target CPA

Now the outer loop: adjust the bid multiplier to hit a target cost per acquisition.

$$\\text{CPA} = \\frac{\\text{spend}}{\\text{conversions}}$$

Bid up and you win more and pay more per win; the relationship is nonlinear and noisy at
low conversion volume. Tune on a simulated week and record the settling time.

Note the sample-size problem that makes this hard in reality: at a 0.2% CVR and a few
thousand impressions an hour, an hourly CPA estimate is built on single-digit
conversions. The controller is mostly reacting to noise. Show that — plot the hourly CPA
estimate with its confidence interval next to the controller's response.""",
             """\
class TargetCPAController:
    def __init__(self, target_cpa, kp=0.3, ki=0.02):
        # TODO
        raise NotImplementedError"""),
            ("""\
## 3. The good case

Instant conversion reporting. Run a simulated week and confirm the controller converges:
spend tracks target, CPA settles near the goal. Establish that it works before you break
it, or you will not know which failure is which.""",
             """\
# TODO"""),
            ("""\
## 4. Break it — delayed reporting

**The experiment.** Feed the controller conversions with the Week 4 lag distribution
attached instead of instantly.

What should happen: early in the day, observed conversions are far below eventual
conversions, so the measured CPA looks terrible, so the controller bids *down* — or, if
it is a spend-pacing controller with budget left over, it bids *up* to spend the budget
and buys traffic it should not. Either way it is steering on a signal that is
systematically wrong in a known direction.

Quantify it: overspend as a percentage of budget, and realised CPA vs target, as a
function of the delay distribution's median. Sweep the median from 0 to 48 hours and
plot. **That curve is the pitch.**""",
             """\
# TODO: inject delays, sweep median delay, plot overspend %"""),
            ("""\
## 5. Fix it

Use the Week 4 model to estimate *eventual* conversions from observed ones, and feed the
lag-corrected CPA to the controller. Re-run the sweep.

Then be honest about the residual: the correction has its own error, and a controller
that trusts a corrected estimate too much has a new failure mode. Show the corrected
curve next to the naive one and mark where correction stops helping.""",
             """\
# fig, ax = plt.subplots()
# ... overspend vs median delay: naive vs lag-corrected
# print(plots.save(fig, 9, "overspend_vs_delay"))"""),
        ],
    ),
    10: dict(
        title="Privacy-constrained learning",
        goal="Quantify what privacy costs a measurement system, in accuracy, at several points on the tradeoff.",
        deliverable="Privacy/utility tradeoff curves — the evidence base for any privacy-era measurement argument.",
        hours="2h reading (CriteoPrivateAd + ARA docs) · 6h building · 1h write-up.",
        cells=[
            ("""\
## The dataset was built for this

CriteoPrivateAd tags every feature with the privacy regime it survives in. That mapping
*is* the experiment design — train on progressively smaller unions of buckets and the
accuracy you lose is the accuracy privacy costs.

`features_not_available_*` are the third-party-cookie-era signals that are gone. Training
with them gives you the oracle; the gap to it is the price.""",
             """\
pa = data.load_privatead(days=list(range(1, 11)))
print(f"{len(pa):,} rows, days {sorted(pa.day_int.unique())}")

groups = data.privatead_feature_groups(pa)
for b, cols in groups.items():
    print(f"{len(cols):3d} cols  {b}")
    print(f"          {data.PRIVACY_BUCKETS[b]}")

for lab in ("is_clicked", "is_click_landed", "is_visit"):
    print(f"{lab:18s} {pa[lab].mean():.4%}")"""),
            ("""\
## 1. The feature-ablation ladder

Train the same model on each nested feature set:

1. contextual only — the floor, available to everyone forever
2. + non-constrained key-value
3. + bit-constrained (Protected-Audience-style budget)
4. + browser bit-constrained
5. + `features_not_available` — the third-party-cookie oracle

Split by `day_int`. Plot AUC against feature set. **That plot is the deliverable** and it
is the single most reusable artifact in this repo: it is the answer to "how much does
Privacy Sandbox actually cost?" backed by your own numbers.""",
             """\
sp = split.time_split(pa, "day_int", train_frac=0.7, val_frac=0.1)
split.check_no_leakage(pa, sp, time_col="day_int")
print(sp)

ladder = {
    "ctx_only":      groups["features_ctx_not_constrained"],
    "+kv_free":      groups["features_ctx_not_constrained"] + groups["features_kv_not_constrained"],
    "+kv_bits":      groups["features_ctx_not_constrained"] + groups["features_kv_not_constrained"] + groups["features_kv_bits_constrained"],
    "+browser_bits": groups["features_ctx_not_constrained"] + groups["features_kv_not_constrained"] + groups["features_kv_bits_constrained"] + groups["features_browser_bits_constrained"],
    "oracle_3pc":    sum(groups.values(), []),
}
{k: len(v) for k, v in ladder.items()}"""),
            ("""\
## 2. Noised aggregates

The Attribution Reporting API does not hand you rows — it hands you *noisy aggregates*
under a contribution budget. Simulate it: group conversions by some key (campaign x day),
add Laplace noise calibrated to $\\varepsilon$, and see what survives.

The finding to chase: utility depends brutally on **how many keys you split the budget
across**. Same $\\varepsilon$, ten times the keys, and each aggregate is drowned. Sweep
both $\\varepsilon$ and key cardinality and plot the surface. Almost everyone who
discusses this only sweeps $\\varepsilon$, and that is the less interesting axis.

Read `papers/attribution-reporting-api-AGGREGATE.md` for how the real budget works.""",
             """\
def laplace_mechanism(counts, epsilon, sensitivity=1.0, rng=None):
    rng = rng or np.random.default_rng(0)
    return counts + rng.laplace(0, sensitivity / epsilon, size=np.shape(counts))

# TODO: sweep epsilon x n_keys, measure relative error of the noised aggregate"""),
            ("""\
## 3. Learning from aggregates only

The hard version: you never see a per-impression label, only noisy group totals. Train a
model whose *predicted group sums* match the observed noisy sums.

This is a real technique (learning from label proportions) and it is what measurement
looks like when the row-level join is gone for good. Even a partial result here is worth
more than a polished version of section 2.""",
             """\
# TODO: loss on aggregate predicted-vs-observed sums per key"""),
            ("""\
## 4. The tradeoff curves

Assemble everything into the deliverable: accuracy against privacy strength, on the same
axes, for each mechanism. Mark where a real product decision would sit.

Then the paragraph that makes it useful: for each regime, *what measurement question can
still be answered, and which one can't?* Aggregate reporting can still tell you which
campaign won. It cannot tell you which user to bid on. Say so plainly.""",
             """\
# fig, ax = plt.subplots()
# ... AUC vs feature regime; relative error vs epsilon
# print(plots.save(fig, 10, "privacy_utility_tradeoff"))"""),
        ],
    ),
    11: dict(
        title="Sequence models for user journeys",
        goal="Find out whether sequence models beat well-engineered tabular features on ads data, and by how much.",
        deliverable="An LSTM and a small transformer over user timelines, plus an honest tabular-vs-sequence comparison.",
        hours="2h reading (DIN) · 6h building · 1h write-up.",
        cells=[
            ("""\
## Set the bar before you build

The interesting result here is quantitative, and it is often *negative*: sequence models
frequently fail to beat a good tabular model with hand-built recency and frequency
features, because those features already capture most of what the sequence contains.

So build the **strong tabular baseline first**: counts, time since last impression, time
since first, campaign diversity, session gaps. Get its number. Only then build the
sequence model. Doing it in that order is what makes the answer credible instead of
self-congratulatory.

Use `split.user_grouped_time_split` — a user must not appear in two folds.""",
             """\
df = data.add_attribution_derived(data.load_attribution())

seq = (df.sort_values(["uid", "timestamp"])
         .groupby("uid")
         .agg(n=("campaign", "size"), converted=("conversion", "max")))
print(f"{len(seq):,} users, median {seq.n.median():.0f} impressions, "
      f"{(seq.n >= 5).mean():.1%} have >=5")

sp = split.user_grouped_time_split(df, "timestamp", "uid")
print(sp)"""),
            ("""\
## 1. The tabular baseline to beat

Aggregate each user's history into features. Be generous — the point is to make this hard
to beat. Recency, frequency, campaign entropy, inter-arrival statistics, click ratio.

**Leakage warning, and it is subtle:** every aggregate must be computed from events
*strictly before* the impression being scored. Aggregating a user's whole timeline and
attaching it to every row of that timeline uses the future to predict the past, and it
produces a spectacular AUC that means nothing.""",
             """\
# TODO: expanding-window per-user features, strictly causal"""),
            ("""\
## 2. Sequences

Pad or truncate to a fixed length (start with 20, keep the most recent). Each step: a
campaign embedding plus the time delta since the previous event.

The time delta matters more than people expect — an ads sequence is irregularly sampled,
and a plain LSTM treats "three impressions in one minute" the same as "three impressions
over three weeks". Bucket the log-delta and embed it.""",
             """\
# TODO: build padded (n_users, seq_len) tensors of campaign ids + time-delta buckets"""),
            ("""\
## 3. LSTM

Packed sequences, so padding does not contribute to the hidden state. Getting that wrong
is quiet: the model still trains, it just learns partly from padding.""",
             """\
import torch, torch.nn as nn

class SequenceCVR(nn.Module):
    def __init__(self, n_campaigns, n_delta_buckets, dim=32, hidden=64):
        super().__init__()
        # TODO: embeddings -> LSTM -> head; use packed sequences to ignore padding
        raise NotImplementedError"""),
            ("""\
## 4. Transformer, and DIN-style attention

A 2-layer transformer with learned positional embeddings. Then the ads-specific variant
worth more than the vanilla one: **DIN's attention over history, keyed by the candidate
campaign** (`papers/zhou2018-deep-interest-network.pdf`).

DIN's insight is that relevance is not fixed — which parts of a user's history matter
depends on what you are about to show them. That is a genuinely ads-shaped idea rather
than an NLP import, and it is the reason to prefer it here.""",
             """\
# TODO"""),
            ("""\
## 5. The honest comparison

One table: tabular baseline, LSTM, transformer, DIN. Same users, same split, same metrics.

Then the analysis that makes it publishable rather than a leaderboard:

- **Where does the sequence model win?** Slice by history length. It almost certainly
  adds nothing for single-impression users and everything for long journeys. Plot AUC
  gain against history length.
- **What does it cost?** Training time, inference latency, parameter count. A +0.003 AUC
  for 40x the inference cost is a real answer, and in production it is usually "no".""",
             """\
# fig, ax = plt.subplots()
# ... AUC gain over tabular baseline, bucketed by user history length
# print(plots.save(fig, 11, "sequence_gain_by_history_length"))"""),
        ],
    ),
    12: dict(
        title="Capstone — a privacy-era measurement stack",
        goal="Wire eleven weeks into one system and tell the story end to end.",
        deliverable="A polished repo, a README that reads like an engineering post, and the blog post itself.",
        hours="No new reading. 8h assembling and writing.",
        cells=[
            ("""\
## The system

```
      impressions (simulated traffic, privacy-induced label loss)
                        |
        [W5/W10]  consent + signal loss applied
                        |
        [W4]  delayed-feedback-corrected CVR model
                        |
        [W3]  calibration layer
                        |
        [W6]  attribution -> value per impression
                        |
        [W8]  auction + bid shading
                        |
        [W9]  pacing + target-CPA controller
                        |
                  spend, conversions, realised CPA
```

Each stage exists in a previous week. This week is integration and narrative, not new
modelling. Resist the urge to improve a component — the value is in the seams.""",
             """\
import sys; sys.path.insert(0, "..")
from adslab import data, metrics, plots, split, registry
plots.use_style()

print(registry.to_markdown())   # everything you have built, in one table"""),
            ("""\
## 1. The pipeline object

Wrap each stage behind one interface so the whole thing can be run with components
switched on and off. The ablation is the experiment: run the stack with each correction
disabled and measure what the business metric does.

| configuration | realised CPA vs target | conversions measured vs true |
|---|---|---|
| everything off (naive) | | |
| + calibration | | |
| + delayed-feedback correction | | |
| + conversion modeling | | |
| full stack | | |

That table is the capstone's headline result.""",
             """\
class MeasurementStack:
    def __init__(self, cvr_model, calibrator=None, delay_model=None,
                 attribution=None, shader=None, pacer=None):
        # TODO: each stage optional so you can ablate
        raise NotImplementedError

    def run_day(self, traffic):
        raise NotImplementedError"""),
            ("""\
## 2. Simulated traffic with everything wrong at once

Real conditions, all at the same time: delayed conversions, 30% consent loss,
non-stationary traffic, and a competitive auction. Individually each was survivable.
The question is whether the corrections compose or interfere — and interference is a
finding, not a bug in your write-up.""",
             """\
# TODO"""),
            ("""\
## 3. The ablation

Run every configuration, fill in the table, plot the cumulative CPA error over a
simulated month per configuration.""",
             """\
# fig, ax = plt.subplots()
# ... one line per configuration
# print(plots.save(fig, 12, "ablation_cpa_error"))"""),
            ("""\
## 4. Write the post

Structure that works:

1. **The problem, in money.** Privacy loss makes conversions invisible; invisible
   conversions make bids wrong; wrong bids waste budget. One paragraph, no jargon.
2. **What I built and on what data.** Emphasise it is all public data — that is what
   makes it checkable, and checkable is the whole point.
3. **Four findings with plots.** Pick the four best from twelve weeks. Suggested:
   the calibration-breaks-under-label-loss plot (W3), the delayed-feedback bias by
   elapsed time (W4), the recovery-vs-consent-loss curve (W5), and the
   overspend-vs-delay curve (W9).
4. **What I got wrong.** The negative results — where the sequence model did not help,
   where correction stopped working. This section is what separates it from content
   marketing, and it is the section a good interviewer will want to talk about.
5. **What I would build next.**

Then the repo README: the same story, shorter, with the reproduction instructions that
actually work on a clean clone. Test that claim by following them yourself.""",
             """\
# print(registry.to_markdown())  # the full twelve-week table for the post"""),
        ],
    ),
}


def build(week: int) -> dict:
    s = SPEC[week]
    cells = header(week, s["title"], s["goal"], s["deliverable"], s["hours"])
    for m, c in s["cells"]:
        cells.append(md(m))
        cells.append(code(c))
    cells += footer(week)
    out = []
    for i, c in enumerate(cells):
        c["id"] = f"w{week:02d}-{i:02d}"   # stable ids keep notebook diffs readable
        out.append(_lines(c))
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "ads-ml-lab", "language": "python",
                           "name": "ads-ml-lab"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def week_dir(n: int) -> Path:
    hits = sorted(REPO.glob(f"week{n:02d}_*"))
    if not hits:
        raise FileNotFoundError(f"no week{n:02d}_* directory")
    return hits[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, action="append")
    ap.add_argument("--force", action="store_true", help="overwrite existing notebooks")
    args = ap.parse_args()

    weeks = args.week or sorted(SPEC)
    for w in weeks:
        d = week_dir(w)
        path = d / f"{d.name}.ipynb"
        if path.exists() and not args.force:
            print(f"skip   {path.relative_to(REPO)} (exists; --force to overwrite)")
            continue
        path.write_text(json.dumps(build(w), indent=1) + "\n")
        print(f"write  {path.relative_to(REPO)}  ({len(build(w)['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
