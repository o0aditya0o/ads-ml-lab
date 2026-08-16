# Week 05 — Missing labels and conversion modeling

> **Status:** not started
> **Goal.** Rebuild, on open data, the thing you owned at Google: estimating conversions that were never observed because of consent and signal loss.
> **Deliverable.** A two-model correction (observed CVR + gap model) compared against naive and oracle, written up as an explainer.

Notebook: [`week05_conversion_modeling.ipynb`](week05_conversion_modeling.ipynb) · Reading: [`papers/`](papers/README.md)

---

## What I built

_One paragraph, no code. What exists now that did not exist on Monday._

## What the numbers say

_Paste the table from `registry.to_markdown(week=5)`. Then one sentence naming the
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
