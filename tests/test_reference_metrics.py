"""Tests for the judge's scoring metrics.

These cover ``judge/reference_metrics.py`` — the working ECE and Qini the leaderboard
needs. They are separate from ``tests/test_harness.py``, whose equivalents are marked
``xfail`` because implementing them is the Week 3 and Week 7 exercise.

The thresholds here were calibrated against measured behaviour rather than guessed; where
a bound looks loose, the comment says why.
"""
from __future__ import annotations

import numpy as np
import pytest

from judge import reference_metrics as R

RNG = np.random.default_rng(0)
N = 100_000


@pytest.fixture(scope="module")
def calibrated():
    p = RNG.beta(1.5, 30, N)          # ads-shaped: small probabilities, long tail
    return RNG.binomial(1, p), p


# --------------------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------------------
def test_ece_near_zero_when_calibrated(calibrated):
    y, p = calibrated
    assert R.ece(y, p) < 0.006


def test_ece_detects_inflation(calibrated):
    y, p = calibrated
    assert R.ece(y, np.clip(p * 3, 0, 1)) > 10 * R.ece(y, p)


def test_reliability_bins_are_wellformed(calibrated):
    y, p = calibrated
    mean_p, mean_y, counts = R.reliability_curve(y, p, n_bins=20)
    assert len(mean_p) == len(mean_y) == len(counts)
    assert counts.sum() == len(y), "every row must land in exactly one bin"
    assert (counts > 0).all(), "empty bins must be dropped"
    assert np.all(np.diff(mean_p) >= 0), "bins should ascend in predicted probability"


def test_quantile_binning_beats_uniform_on_skewed_predictions(calibrated):
    """The reason quantile is the default: uniform bins mostly come back empty here."""
    y, p = calibrated
    _, _, q = R.reliability_curve(y, p, n_bins=20, strategy="quantile")
    _, _, u = R.reliability_curve(y, p, n_bins=20, strategy="uniform")
    assert len(q) > len(u), "quantile should retain more usable bins"


def test_max_error_at_least_mean_error(calibrated):
    y, p = calibrated
    assert R.max_calibration_error(y, p) >= R.ece(y, p)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        R.ece([0, 1, 1], [0.1, 0.2])


# --------------------------------------------------------------------------------------
# uplift
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def uplift_data():
    """An 85/15 treated split with real uplift confined to a 30% segment."""
    rng = np.random.default_rng(7)
    w = rng.binomial(1, 0.85, N)
    seg = rng.random(N) < 0.30
    y = rng.binomial(1, np.where(seg & (w == 1), 0.10, 0.02))
    return y, w, seg


def test_null_uplift_random_score_is_about_zero():
    """The check that catches a missing control-arm rescaling."""
    rng = np.random.default_rng(1)
    w = rng.binomial(1, 0.85, N)              # deliberately imbalanced
    y = rng.binomial(1, 0.02, N)              # no uplift at all
    assert abs(R.qini_auc(y, rng.random(N), w)) < 0.05


def test_informative_score_beats_random(uplift_data):
    y, w, seg = uplift_data
    rng = np.random.default_rng(2)
    good = R.qini_auc(y, seg.astype(float), w)
    rand = R.qini_auc(y, rng.random(N), w)
    assert good > 0.15 and abs(rand) < 0.05 and good > rand


def test_inverted_score_is_negative(uplift_data):
    y, w, seg = uplift_data
    assert R.qini_auc(y, -seg.astype(float), w) < -0.15


def test_qini_is_monotone_in_model_quality(uplift_data):
    """Degrading a good score with noise must degrade Qini.

    Only compared across clearly different quality levels: once the segment signal is
    fully captured, extra precision changes nothing and the remaining differences are
    tie-breaking noise (measured at ~0.008 sd, versus the ~0.04 steps asserted here).
    """
    y, w, seg = uplift_data
    rng = np.random.default_rng(3)
    base = seg.astype(float)
    scores = [R.qini_auc(y, base + rng.normal(0, sd, N), w) for sd in (2.0, 1.0, 0.5, 0.2)]
    assert scores == sorted(scores), f"not monotone: {scores}"


def test_oracle_scores_one(uplift_data):
    y, w, _ = uplift_data
    assert R.qini_auc(y, R._oracle_score(y, w), w) == pytest.approx(1.0, abs=1e-6)


def test_curve_starts_at_origin(uplift_data):
    y, w, seg = uplift_data
    frac, inc = R.qini_curve(y, seg.astype(float), w)
    assert frac[0] == 0.0 and inc[0] == 0.0
    assert frac[-1] == pytest.approx(1.0)
