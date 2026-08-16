# The evaluation protocol

Twelve weeks of results are only comparable if they were produced the same way. This is
that way. Deviating is allowed — recording that you deviated is not optional.

## 1. Split by time, always

`split.time_split(df, "timestamp", 0.7, 0.1)` — oldest 70% train, next 10% validation,
newest 20% test. Then `split.check_no_leakage(df, sp)`, which asserts it rather than
trusting you.

Two exceptions, both principled:

- **Uplift (Week 7).** The Criteo uplift set is one randomised experiment with no
  timestamp. Rows are i.i.d. draws, so a random split is correct there.
- **Journey-level labels (Weeks 6, 11).** Use `user_grouped_time_split`. A plain time
  split cuts a user's journey across folds, and the outcome leaks through the user id.
  The cost is that fold sizes drift from the requested fractions — on the attribution
  set, 70/10/20 becomes roughly 83/7/11, because heavy users start early. Report the
  actual fractions.

**Do it wrong once, deliberately, in Week 1.** Train the same model on a random split and
record the AUC. The gap is the size of the lie, and having measured it yourself is worth
more than being told.

## 2. Always beat something dumb

Every table needs a baseline row. In order of preference: logistic regression on hashed
features, then the global base rate. A model that does not beat the base rate on
`normalised_entropy` is not worth its inference cost, and `normalised_entropy` says so
directly — it is exactly 1.0 for the constant predictor.

## 3. The metric bundle

`metrics.evaluate(y, p)` returns all of these. Report all of them; they disagree, and the
disagreements are the interesting part.

| metric | what it sees | what it is blind to |
|---|---|---|
| `auc` | ranking quality | any monotone rescaling — multiply every prediction by 10 and AUC is unchanged |
| `pr_auc` | ranking, weighted toward positives | same rescaling blindness |
| `log_loss` | ranking **and** calibration | hard to interpret in absolute terms |
| `normalised_entropy` | log-loss relative to the base rate | same |
| `calibration_ratio` | aggregate bias, `mean(p)/mean(y)` | per-segment error — can be 1.000 with every segment wrong |
| `ece` | per-bin calibration error | direction of the error; biased downward at low bin counts |

The pairing that matters: **AUC up and `calibration_ratio` away from 1.0 means a better
ranker and a worse bidder.** Know which one the week wanted.

## 4. Record everything, including losses

```python
registry.log_result(
    week=4, model="chapelle_exponential",
    metrics=metrics.evaluate(y_test, p_test),
    dataset="attribution", params=dict(clamp_log_lambda=(-12, 4)),
    notes="fixes the freshest-decile bias; costs 0.002 AUC vs naive",
)
```

Append-only. Reruns add rows; `to_markdown()` shows the newest per
`(week, model, dataset, split, label)`. Keeping the failures is the point — "deep learning
did not beat LightGBM until 5M rows" is only a claim you can make if the losing runs are
still on disk.

Write `notes` at the time of the run. The end-of-week write-up is assembled from these,
and reconstructing why a run mattered three days later does not work.

## 5. What "done" looks like for a week

- The notebook runs top to bottom on a fresh kernel.
- The baseline and the model are both in `results/results.jsonl`.
- At least one figure is saved to `weekNN_*/figures/` via `plots.save`.
- The README's three sections are filled in, including **What surprised me** — and a
  negative result counts fully.
- One commit, whose message states the finding rather than the task.

## 6. Numbers to sanity-check against

Measured during setup on the full attribution dataset — if your loaders disagree with
these, something is wrong before any modelling starts.

```
rows                16,468,027        users            6,142,256
window              30.9 days         campaigns              675
conversion rate         4.90%         click rate          36.12%
last-click attributed   2.69%
conversion delay        p50 89.9h · p75 287.3h · p90 503.5h · p99 695.4h
                        65.6% of conversions land >24h after the impression
cat cardinalities   9, 70, 1829, 21, 51, 30, 57196, 11, 30  (~59k total)
```

That delay distribution deserves a second look: the **median** conversion arrives about
90 hours — nearly four days — after the impression. Week 4 is not a marginal correction
on this data.
