#!/usr/bin/env python3
"""Write the per-week README skeletons (and only if they don't already exist).

Each one is a three-question template. The questions are the same every week on purpose:
twelve answers to the same three questions is a portfolio, twelve differently-shaped
documents is a pile.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from make_notebooks import SPEC, week_dir  # single source of truth for titles/goals

REPO = Path(__file__).resolve().parents[1]

TEMPLATE = """\
# Week {week:02d} — {title}

> **Status:** not started
> **Goal.** {goal}
> **Deliverable.** {deliverable}

Notebook: [`{nb}`]({nb}) · Reading: [`papers/`](papers/README.md)

---

## What I built

_One paragraph, no code. What exists now that did not exist on Monday._

## What the numbers say

_Paste the table from `registry.to_markdown(week={week})`. Then one sentence naming the
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
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for w, s in sorted(SPEC.items()):
        d = week_dir(w)
        p = d / "README.md"
        if p.exists() and not args.force:
            print(f"skip   {p.relative_to(REPO)}")
            continue
        p.write_text(TEMPLATE.format(
            week=w, title=s["title"], goal=s["goal"],
            deliverable=s["deliverable"], nb=f"{d.name}.ipynb"))
        print(f"write  {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
