# Week 04 — Delayed feedback

> **Status:** not started
> **Goal.** Model the conversion lag distribution and correct the bias it puts into a model trained on fresh traffic.
> **Deliverable.** Naive vs. Chapelle exponential-delay vs. a Weibull survival variant, plus a predicted-vs-actual lag plot.

Notebook: [`week04_delayed_feedback.ipynb`](week04_delayed_feedback.ipynb) · Reading: [`papers/`](papers/README.md)

---

## What I built

_One paragraph, no code. What exists now that did not exist on Monday._

## What the numbers say

_Paste the table from `registry.to_markdown(week=4)`. Then one sentence naming the
honest comparison — which two rows is the reader supposed to look at, and why._

| model | dataset | AUC | log-loss | NE | calib. ratio | ECE | notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

**Baseline beaten?** _yes / no — and by how much._

## What surprised me

_The part worth reading. A negative result belongs here and counts fully: "the sequence
model did not beat tabular features until histories exceeded 12 events" is a better
sentence than anything that could have been predicted in advance._

## Loose ends

_What you would do with another day. Feeds the capstone._
