"""Time-ordered splitting.

The rule for this repo: **never** split ads data randomly. A random split lets the model
see the future of the same campaign, the same user and the same auction dynamics that it
is being scored on, and inflates AUC by an amount that looks like progress. Every number
in ``results/`` is only comparable because every week splits the same way.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    """Boolean masks over the original frame, plus the boundaries that produced them."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_end: float
    val_end: float
    time_col: str

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return df[self.train], df[self.val], df[self.test]

    def __repr__(self) -> str:  # noqa: D105
        n = len(self.train)
        return (f"Split(on={self.time_col!r}, "
                f"train={self.train.sum():,} ({self.train.mean():.0%}) < {self.train_end:g} "
                f"<= val={self.val.sum():,} ({self.val.mean():.0%}) < {self.val_end:g} "
                f"<= test={self.test.sum():,} ({self.test.mean():.0%}) of {n:,})")


def time_split(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    train_frac: float = 0.7,
    val_frac: float = 0.1,
) -> Split:
    """Split by quantiles of ``time_col`` -- train is oldest, test is newest.

    Quantiles rather than fixed dates so the same call works on a 30-day log and on a
    3-day sample. ``test_frac`` is whatever is left over.
    """
    if not 0 < train_frac < 1 or not 0 <= val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError(f"bad fractions: train={train_frac}, val={val_frac}")
    t = df[time_col].to_numpy()
    train_end = float(np.quantile(t, train_frac))
    val_end = float(np.quantile(t, train_frac + val_frac))
    return Split(
        train=t < train_end,
        val=(t >= train_end) & (t < val_end),
        test=t >= val_end,
        train_end=train_end,
        val_end=val_end,
        time_col=time_col,
    )


def user_grouped_time_split(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    user_col: str = "uid",
    train_frac: float = 0.7,
    val_frac: float = 0.1,
) -> Split:
    """Time split that additionally keeps each user wholly inside one fold.

    Use this whenever the label is defined over a user's whole journey -- attribution
    (Week 6) and sequence models (Week 11). A plain time split cuts journeys in half, so
    the model sees the first three impressions of a converting user in train and is then
    asked about the fourth in test, which leaks the outcome through the user id.

    Users are assigned by the time of their *first* event, so the fold boundaries stay
    chronological; the cost is that fold sizes drift from the requested fractions.
    """
    first = df.groupby(user_col)[time_col].min()
    train_end = float(np.quantile(first.to_numpy(), train_frac))
    val_end = float(np.quantile(first.to_numpy(), train_frac + val_frac))

    fold = pd.Series(
        np.where(first < train_end, 0, np.where(first < val_end, 1, 2)), index=first.index
    )
    assigned = df[user_col].map(fold).to_numpy()
    return Split(
        train=assigned == 0,
        val=assigned == 1,
        test=assigned == 2,
        train_end=train_end,
        val_end=val_end,
        time_col=f"{time_col} (grouped by {user_col})",
    )


def check_no_leakage(df: pd.DataFrame, split: Split, time_col: str = "timestamp") -> None:
    """Assert the folds really are ordered in time. Call it once per notebook; it is cheap."""
    tr, va, te = df[split.train][time_col], df[split.val][time_col], df[split.test][time_col]
    if len(tr) and len(va):
        assert tr.max() <= va.min(), f"train overlaps val: {tr.max()} > {va.min()}"
    if len(va) and len(te):
        assert va.max() <= te.min(), f"val overlaps test: {va.max()} > {te.min()}"
    if len(tr) and len(te) and not len(va):
        assert tr.max() <= te.min(), "train overlaps test"
