"""Append-only results log, so week 9 can be compared against week 1 without archaeology.

One JSON object per experiment in ``results/results.jsonl``. Never edited in place --
if you rerun an experiment you get a second row with a later timestamp, and the table
renderer shows the newest per (week, model, dataset, split). Keeping the losing runs is
the point: the story of Week 2 is "deep did not beat trees until X", and that story is
only tellable if the failures are still on disk.
"""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import RESULTS

LOG = RESULTS / "results.jsonl"

# Columns that make up an experiment's identity; everything else is a measurement.
KEY_FIELDS = ("week", "model", "dataset", "split", "label")


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=RESULTS.parent,
        ).stdout.strip() or None
    except Exception:
        return None


def log_result(
    week: int,
    model: str,
    metrics: dict[str, Any],
    *,
    dataset: str = "attribution",
    split: str = "time_70_10_20",
    label: str = "conversion",
    params: dict[str, Any] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Record one evaluated model. Returns the row it wrote.

    Parameters
    ----------
    model
        A name you will still understand in eight weeks: ``"lightgbm_hashed_2^20"``
        beats ``"model_v3_final"``.
    metrics
        Usually the output of :func:`adslab.metrics.evaluate`.
    notes
        The one sentence you would say out loud about this run. Write it now; the
        write-up at the end of the week is assembled from these.
    """
    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "week": int(week),
        "model": model,
        "dataset": dataset,
        "split": split,
        "label": label,
        **{k: v for k, v in metrics.items()},
        "params": params or {},
        "notes": notes,
        "git": _git_sha(),
        "python": platform.python_version(),
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def load_results(week: int | None = None, latest_only: bool = True):
    """Read the log into a DataFrame, newest run per experiment by default."""
    import pandas as pd

    if not LOG.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    if week is not None:
        df = df[df["week"] == week]
    if latest_only and len(df):
        keys = [k for k in KEY_FIELDS if k in df.columns]
        df = df.sort_values("ts").drop_duplicates(subset=keys, keep="last")
    return df.reset_index(drop=True)


DISPLAY = ["week", "model", "dataset", "auc", "log_loss", "normalised_entropy",
           "calibration_ratio", "ece", "n", "base_rate", "notes"]


def to_markdown(week: int | None = None, columns: list[str] | None = None) -> str:
    """Render the results table as markdown for pasting into a week's README."""
    df = load_results(week)
    if df.empty:
        return "_No results logged yet._"
    cols = [c for c in (columns or DISPLAY) if c in df.columns]
    df = df[cols].sort_values(["week", "auc"], ascending=[True, False])
    fmt = {c: "{:.4f}" for c in ("auc", "log_loss", "normalised_entropy", "calibration_ratio", "ece")}
    for c, f in fmt.items():
        if c in df.columns:
            df[c] = df[c].map(lambda v: f.format(v) if isinstance(v, (int, float)) else "--")
    return df.to_markdown(index=False)
