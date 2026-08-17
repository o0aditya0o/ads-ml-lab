"""Metrics the judge needs to score weeks 3 and 7.

**Why this file exists at all.** ``adslab.metrics.ece`` and ``adslab.metrics.qini_auc``
raise ``TodoError`` on purpose — implementing them is the Week 3 and Week 7 exercise, and
the contract tests in ``tests/test_harness.py`` are written against that. But the judge has
to score submissions today, so it needs working versions.

Keeping them here rather than filling in the stubs means the exercise survives: the file a
learner edits still has the hole in it. If you are doing the course, do not read this file
until you have written your own — that is the whole point.

These are also deliberately the *straightforward* implementations. Week 3 asks you to think
about the bias in binned ECE (see ``papers/kumar2019-verified-uncertainty-calibration.pdf``);
this file does not, because a leaderboard needs one fixed, boring, reproducible definition
rather than the best available estimator.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-15


def _clean(y_true, y_prob) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.clip(np.asarray(y_prob, dtype=float).ravel(), EPS, 1 - EPS)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y{y.shape} vs p{p.shape}")
    return y, p


# --------------------------------------------------------------------------------------
# calibration — week 3
# --------------------------------------------------------------------------------------
def reliability_curve(y_true, y_prob, n_bins: int = 20, strategy: str = "quantile"):
    """Return ``(mean_pred, mean_label, count)`` per bin, empty bins dropped.

    Quantile binning by default: on a 4.9% base rate the predictions crowd into a narrow
    band, and equal-width bins would leave most of them empty.
    """
    y, p = _clean(y_true, y_prob)

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)          # ties collapse; fewer bins is correct here
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    n_actual = len(edges) - 1

    counts = np.bincount(idx, minlength=n_actual).astype(float)
    sum_p = np.bincount(idx, weights=p, minlength=n_actual)
    sum_y = np.bincount(idx, weights=y, minlength=n_actual)

    keep = counts > 0
    return sum_p[keep] / counts[keep], sum_y[keep] / counts[keep], counts[keep]


def ece(y_true, y_prob, n_bins: int = 20, strategy: str = "quantile") -> float:
    """Expected calibration error: count-weighted mean |mean_pred - mean_label| per bin."""
    mean_p, mean_y, counts = reliability_curve(y_true, y_prob, n_bins, strategy)
    if counts.sum() == 0:
        return float("nan")
    return float(np.sum(counts * np.abs(mean_p - mean_y)) / counts.sum())


def max_calibration_error(y_true, y_prob, n_bins: int = 20, strategy: str = "quantile") -> float:
    """Worst per-bin gap — the tail risk that ECE's averaging hides."""
    mean_p, mean_y, _ = reliability_curve(y_true, y_prob, n_bins, strategy)
    return float(np.max(np.abs(mean_p - mean_y))) if len(mean_p) else float("nan")


# --------------------------------------------------------------------------------------
# uplift — week 7
# --------------------------------------------------------------------------------------
def qini_curve(y_true, uplift_score, treatment):
    """Return ``(fraction_targeted, incremental_conversions)``.

    Sort by predicted uplift descending; at each prefix,

        incremental = conversions_treated - conversions_control * (n_treated / n_control)

    The control term is rescaled because the arms are different sizes — on the Criteo
    uplift data roughly 85/15. Omitting that rescaling is the classic Qini bug and makes
    every model look excellent.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    s = np.asarray(uplift_score, dtype=float).ravel()
    w = np.asarray(treatment, dtype=float).ravel()
    if not (y.shape == s.shape == w.shape):
        raise ValueError("y, uplift_score and treatment must be the same length")

    order = np.argsort(-s, kind="stable")
    y, w = y[order], w[order]

    n_t = np.cumsum(w)                      # treated seen so far
    n_c = np.cumsum(1.0 - w)                # control seen so far
    y_t = np.cumsum(y * w)                  # conversions among treated
    y_c = np.cumsum(y * (1.0 - w))          # conversions among control

    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = np.where(n_c > 0, y_c * (n_t / np.maximum(n_c, 1.0)), 0.0)
    inc = y_t - scaled

    frac = np.arange(1, len(y) + 1, dtype=float) / len(y)
    return np.concatenate([[0.0], frac]), np.concatenate([[0.0], inc])


def qini_auc(y_true, uplift_score, treatment) -> float:
    """Area between the Qini curve and the random-targeting diagonal, normalised.

    Normalised by the area of the *perfect* ordering (every truly-incremental unit first),
    so the number is comparable across subsamples. 0 means no better than random targeting;
    1 means the theoretical maximum. Negative means actively worse than random.
    """
    frac, inc = qini_curve(y_true, uplift_score, treatment)
    total = inc[-1]                                    # incremental at full population
    area = np.trapezoid(inc, frac) if hasattr(np, "trapezoid") else np.trapz(inc, frac)
    random_area = total / 2.0                          # the diagonal
    y = np.asarray(y_true, dtype=float).ravel()
    w = np.asarray(treatment, dtype=float).ravel()

    # Perfect ordering: all incremental conversions first, then the rest.
    perfect_frac, perfect_inc = qini_curve(y, _oracle_score(y, w), w)
    perfect_area = (np.trapezoid(perfect_inc, perfect_frac)
                    if hasattr(np, "trapezoid") else np.trapz(perfect_inc, perfect_frac))
    denom = perfect_area - random_area
    if abs(denom) < 1e-12:
        return 0.0
    return float((area - random_area) / denom)


def _oracle_score(y, w):
    """A ranking that puts converting-treated first and converting-control last.

    Only used to normalise the Qini score. It is not achievable by any real model — it
    peeks at the outcome — which is exactly what makes it the right upper bound.
    """
    return np.where((w == 1) & (y == 1), 2.0, np.where((w == 0) & (y == 1), 0.0, 1.0))
