"""The metric contract every week is scored against.

Some functions here are deliberately unimplemented -- they raise :class:`TodoError` and
are the exercise for the week named in the docstring. The *signatures* are fixed now so
that Week 3's calibration work and Week 7's uplift work drop straight into the same
results table as everything else. ``tests/test_metrics.py`` encodes what a correct
implementation must satisfy; the tests for unimplemented pieces are marked ``xfail``, so
a green test run today turns into an ``XPASS`` the moment you finish one.

Wrappers around scikit-learn are wrappers on purpose: there is nothing to learn from
re-deriving ROC AUC, and everything to learn from deriving ECE and Qini.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, log_loss as _sk_log_loss, roc_auc_score

EPS = 1e-15


class TodoError(NotImplementedError):
    """Raised by the metrics you implement yourself. Carries the week that owns it."""

    def __init__(self, week: int, what: str) -> None:
        super().__init__(f"{what} is the Week {week} exercise -- implement it in adslab/metrics.py")
        self.week = week


def _clean(y_true, y_prob) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(float).ravel()
    p = np.clip(np.asarray(y_prob).astype(float).ravel(), EPS, 1 - EPS)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y{y.shape} vs p{p.shape}")
    return y, p


# --------------------------------------------------------------------------------------
# Ranking / likelihood -- provided
# --------------------------------------------------------------------------------------
def auc(y_true, y_prob) -> float:
    """ROC AUC. Pure ranking: invariant to any monotone rescaling of ``y_prob``.

    Which is exactly why it cannot be your only metric. A model that multiplies every
    prediction by 10 has identical AUC and is useless for bidding.
    """
    y, p = _clean(y_true, y_prob)
    return float(roc_auc_score(y, p))


def log_loss(y_true, y_prob) -> float:
    """Mean negative log-likelihood. Proper scoring rule, so it *does* punish miscalibration."""
    y, p = _clean(y_true, y_prob)
    return float(_sk_log_loss(y, p, labels=[0, 1]))


def pr_auc(y_true, y_prob) -> float:
    """Average precision. More informative than ROC AUC when positives are <1% of rows."""
    y, p = _clean(y_true, y_prob)
    return float(average_precision_score(y, p))


def normalised_entropy(y_true, y_prob) -> float:
    """Log-loss divided by the log-loss of predicting the base rate everywhere.

    The Facebook "practical lessons" paper's headline metric. <1 is better than the
    constant baseline; >=1 means your model is not worth its inference cost.
    """
    y, p = _clean(y_true, y_prob)
    base = float(y.mean())
    denom = -(base * np.log(base) + (1 - base) * np.log(1 - base))
    return log_loss(y, p) / denom


def calibration_ratio(y_true, y_prob) -> float:
    """mean(prediction) / mean(label) -- the aggregate bias ratio.

    1.0 is perfect on aggregate. This is the number an ads business actually feels: at
    1.15 every bid is 15% too high and the campaign overspends, whatever the AUC says.
    It is a *necessary* not sufficient condition -- a model can be perfect on aggregate
    and badly wrong in every segment, which is what :func:`ece` is for.
    """
    y, p = _clean(y_true, y_prob)
    return float(p.mean() / max(y.mean(), EPS))


# --------------------------------------------------------------------------------------
# Calibration -- Week 3 exercise
# --------------------------------------------------------------------------------------
def reliability_curve(y_true, y_prob, n_bins: int = 20, strategy: str = "quantile"):
    """Bin predictions and return ``(bin_mean_pred, bin_mean_label, bin_count)``.

    Week 3. Requirements a correct implementation must meet:

    - ``strategy="uniform"``: bins are equal-width over [0, 1].
    - ``strategy="quantile"``: bins hold equal *counts*. Ads predictions crowd into a
      narrow band near the base rate (4.9% on the attribution set, 0.7% on FairJob), so
      uniform bins leave most bins empty and the resulting diagram is noise -- hence
      quantile as the default here.
    - Empty bins must be dropped, not returned as zeros.
    - The three arrays must have equal length and ``bin_count`` must sum to ``len(y)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    raise TodoError(3, "reliability_curve")


def ece(y_true, y_prob, n_bins: int = 20, strategy: str = "quantile") -> float:
    """Expected Calibration Error: count-weighted mean ``|mean_pred - mean_label|`` per bin.

    Week 3. Build it on top of :func:`reliability_curve`.

    Read ``papers/kumar2019-verified-uncertainty-calibration.pdf`` before you trust the
    number you get: binned ECE is a *biased downward* estimator of true calibration
    error, and the bias shrinks as you add bins -- so ECE at 10 bins and ECE at 100 bins
    are not comparable quantities. Pick ``n_bins`` once, record it, never tune it.
    """
    raise TodoError(3, "ece")


def max_calibration_error(y_true, y_prob, n_bins: int = 20, strategy: str = "quantile") -> float:
    """Worst per-bin calibration gap. The tail risk that ECE's averaging hides."""
    raise TodoError(3, "max_calibration_error")


# --------------------------------------------------------------------------------------
# Uplift -- Week 7 exercise
# --------------------------------------------------------------------------------------
def qini_curve(y_true, uplift_score, treatment):
    """Return ``(fraction_targeted, incremental_conversions)`` for the Qini curve.

    Week 7. Sort by ``uplift_score`` descending; at each prefix of the population,

        incremental = conversions_treated - conversions_control * (n_treated / n_control)

    The control term is rescaled because the treated and control groups are different
    sizes (on the Criteo set, wildly so -- ~85/15). Forgetting that rescaling is the
    single most common Qini bug and it makes every model look brilliant.
    """
    raise TodoError(7, "qini_curve")


def qini_auc(y_true, uplift_score, treatment) -> float:
    """Area between the Qini curve and the random-targeting diagonal, normalised.

    Week 7. Normalise by the area of the perfect-targeting curve so the number is
    comparable across subsamples.
    """
    raise TodoError(7, "qini_auc")


# --------------------------------------------------------------------------------------
# The standard bundle
# --------------------------------------------------------------------------------------
CORE_METRICS = {
    "auc": auc,
    "log_loss": log_loss,
    "pr_auc": pr_auc,
    "normalised_entropy": normalised_entropy,
    "calibration_ratio": calibration_ratio,
    "ece": ece,
}


def evaluate(y_true, y_prob, metrics: dict | None = None) -> dict[str, float | None]:
    """Score a prediction with the standard bundle.

    Metrics that are still exercises come back as ``None`` rather than raising, so a
    Week 1 notebook runs end to end before Week 3 exists. Once you implement ``ece`` the
    column populates itself everywhere, including in rows recorded earlier -- rerun the
    week to backfill.
    """
    out: dict[str, float | None] = {}
    for name, fn in (metrics or CORE_METRICS).items():
        try:
            out[name] = fn(y_true, y_prob)
        except TodoError:
            out[name] = None
    out["n"] = int(len(np.asarray(y_true).ravel()))
    out["base_rate"] = float(np.asarray(y_true).astype(float).mean())
    return out
