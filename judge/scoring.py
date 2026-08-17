"""Score a submission against the hidden solution.

Scoring calls ``adslab.metrics`` — the same functions that score the notebooks. If the
leaderboard and your local evaluation ever disagree, it is the split or the join, never
the metric.

Rejection messages are written to be actionable. "Invalid submission" wastes a person's
evening; "missing 12 ids, first few: 4, 91, 233" does not.
"""
from __future__ import annotations

import gzip
import io
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from adslab import metrics as M
from judge import reference_metrics as R
from judge.competitions import Competition

MAX_PREVIEW = 6


class Rejected(Exception):
    """Submission is not scoreable. The message goes straight to the entrant."""


@dataclass
class Score:
    public: float
    private: float
    public_metrics: dict
    private_metrics: dict
    n_rows: int

    def as_json(self) -> tuple[str, str]:
        return json.dumps(self.public_metrics), json.dumps(self.private_metrics)


def _read_csv(raw: bytes) -> pd.DataFrame:
    """Accept plain or gzipped CSV; detect by magic bytes, not by filename."""
    if raw[:2] == b"\x1f\x8b":
        try:
            raw = gzip.decompress(raw)
        except OSError as e:
            raise Rejected(f"File looks gzipped but could not be decompressed: {e}")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise Rejected(f"Could not parse as CSV: {type(e).__name__}: {e}")
    if df.empty:
        raise Rejected("The file parsed but contains no rows.")
    return df


def _preview(values) -> str:
    v = list(values)[:MAX_PREVIEW]
    more = "" if len(values) <= MAX_PREVIEW else f", ... (+{len(values) - MAX_PREVIEW} more)"
    return ", ".join(str(x) for x in v) + more


# Bin count is fixed and never tuned: binned ECE shrinks as bins are added, so a
# leaderboard has to pick one number and keep it.
ECE_BINS = 20


def evaluate(y, p) -> dict:
    """``adslab.metrics.evaluate`` plus a working ECE.

    adslab leaves ``ece`` unimplemented on purpose — it is the Week 3 exercise — so it
    comes back as ``None``. The judge fills it in from its own reference implementation,
    which means every leaderboard shows calibration error, not just week 3's.
    """
    out = M.evaluate(y, p)
    if out.get("ece") is None:
        out["ece"] = R.ece(y, p, n_bins=ECE_BINS)
    return out


def score_submission(raw: bytes, comp: Competition) -> Score:
    """Validate and score. Raises :class:`Rejected` with a specific reason."""
    sub = _read_csv(raw)

    id_col, pred_col = comp.id_column, "prediction"
    cols = {c.strip().lower(): c for c in sub.columns}
    if id_col not in cols:
        raise Rejected(
            f"Missing the '{id_col}' column. Found: {', '.join(map(str, sub.columns[:8]))}. "
            f"The header row is required.")
    if pred_col not in cols:
        raise Rejected(
            f"Missing the 'prediction' column. Found: {', '.join(map(str, sub.columns[:8]))}.")
    sub = sub.rename(columns={cols[id_col]: id_col, cols[pred_col]: pred_col})
    sub = sub[[id_col, pred_col]]

    try:
        sub[id_col] = sub[id_col].astype("int64")
    except Exception:
        raise Rejected(f"'{id_col}' must be integers.")
    sub[pred_col] = pd.to_numeric(sub[pred_col], errors="coerce")

    if sub[pred_col].isna().any():
        bad = sub.index[sub[pred_col].isna()][:MAX_PREVIEW].tolist()
        raise Rejected(f"{int(sub[pred_col].isna().sum())} prediction(s) are missing or "
                       f"non-numeric, at row(s): {_preview(bad)}.")
    if not np.isfinite(sub[pred_col]).all():
        raise Rejected("Predictions contain inf or -inf.")

    lo, hi = float(sub[pred_col].min()), float(sub[pred_col].max())
    if lo < 0 or hi > 1:
        raise Rejected(f"Predictions must lie in [0, 1]; yours span [{lo:.4g}, {hi:.4g}]. "
                       f"If these are logits, apply a sigmoid first.")

    dupes = sub[id_col].duplicated()
    if dupes.any():
        raise Rejected(f"{int(dupes.sum())} duplicate {id_col} value(s): "
                       f"{_preview(sub.loc[dupes, id_col].unique())}.")

    sol = pd.read_parquet(comp.resolve_solution())
    need, got = set(sol[id_col]), set(sub[id_col])

    missing, extra = need - got, got - need
    if missing:
        raise Rejected(f"Missing {len(missing):,} of {len(need):,} required "
                       f"{id_col} values: {_preview(sorted(missing))}.")
    if extra:
        raise Rejected(f"{len(extra):,} {id_col} value(s) are not in the test set: "
                       f"{_preview(sorted(extra))}.")

    merged = sol.merge(sub, on=id_col, how="left", validate="one_to_one")
    y = merged[comp.target_column].to_numpy()
    p = merged[pred_col].to_numpy()
    pub = merged["is_public"].to_numpy(dtype=bool)

    if pub.sum() == 0 or (~pub).sum() == 0:
        raise Rejected("Internal: the solution file has an empty public or private split.")

    pub_m = evaluate(y[pub], p[pub])
    prv_m = evaluate(y[~pub], p[~pub])
    key = comp.primary_metric.key

    return Score(
        public=float(pub_m[key]),
        private=float(prv_m[key]),
        public_metrics={k: v for k, v in pub_m.items()},
        private_metrics={k: v for k, v in prv_m.items()},
        n_rows=len(sub),
    )
