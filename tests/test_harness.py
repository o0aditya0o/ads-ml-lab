"""Contract tests for the shared harness.

Tests for pieces that are still exercises are marked ``xfail``. That means:

* today, ``pytest`` is green -- the repo is not shouting at you about unfinished work;
* the moment you implement one correctly, its test reports ``XPASS``, which is your
  signal to delete the marker.

Run with:  ``python -m pytest -q``
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adslab import calibration, encoders, metrics, split
from adslab.metrics import TodoError

RNG = np.random.default_rng(0)


@pytest.fixture(scope="module")
def toy():
    """A small, well-calibrated-by-construction binary problem."""
    n = 20_000
    p = RNG.beta(1.5, 60, size=n)            # ads-shaped: tiny probabilities, long tail
    y = RNG.binomial(1, p)
    return y, p


@pytest.fixture(scope="module")
def toy_frame():
    n = 5_000
    return pd.DataFrame({
        "timestamp": np.sort(RNG.integers(0, 30 * 86400, n)),
        "uid": RNG.integers(0, 400, n),
        "cat1": RNG.integers(0, 10_000, n),
        "cat2": RNG.integers(0, 10_000, n),
        "y": RNG.binomial(1, 0.03, n),
    })


# --------------------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------------------
def test_time_split_is_ordered_and_covers_everything(toy_frame):
    sp = split.time_split(toy_frame, "timestamp", 0.7, 0.1)
    split.check_no_leakage(toy_frame, sp)
    assert (sp.train | sp.val | sp.test).all()
    assert not (sp.train & sp.val).any()
    assert abs(sp.train.mean() - 0.7) < 0.02
    assert abs(sp.val.mean() - 0.1) < 0.02


def test_time_split_rejects_bad_fractions(toy_frame):
    with pytest.raises(ValueError):
        split.time_split(toy_frame, "timestamp", 0.9, 0.2)


def test_user_grouped_split_keeps_users_whole(toy_frame):
    sp = split.user_grouped_time_split(toy_frame, "timestamp", "uid")
    folds = pd.DataFrame({
        "uid": toy_frame["uid"],
        "fold": np.select([sp.train, sp.val, sp.test], [0, 1, 2], default=-1),
    })
    assert (folds.groupby("uid")["fold"].nunique() == 1).all(), "a user spans two folds"


# --------------------------------------------------------------------------------------
# metrics that are provided
# --------------------------------------------------------------------------------------
def test_auc_is_ranking_only(toy):
    y, p = toy
    assert metrics.auc(y, p) == pytest.approx(metrics.auc(y, p * 0.5), abs=1e-9)


def test_calibration_ratio_detects_inflation(toy):
    y, p = toy
    assert metrics.calibration_ratio(y, p) == pytest.approx(1.0, abs=0.06)
    assert metrics.calibration_ratio(y, np.clip(p * 1.5, 0, 1)) > 1.3


def test_normalised_entropy_beats_one_for_a_real_model(toy):
    y, p = toy
    assert metrics.normalised_entropy(y, p) < 1.0
    base = np.full_like(p, y.mean())
    assert metrics.normalised_entropy(y, base) == pytest.approx(1.0, abs=1e-6)


def test_evaluate_survives_unimplemented_metrics(toy):
    y, p = toy
    out = metrics.evaluate(y, p)
    assert out["auc"] > 0.5 and out["n"] == len(y)
    assert out["ece"] is None, "ece should report None until Week 3 implements it"


def test_shape_mismatch_is_an_error():
    with pytest.raises(ValueError):
        metrics.auc([0, 1, 1], [0.1, 0.2])


# --------------------------------------------------------------------------------------
# encoders
# --------------------------------------------------------------------------------------
def test_hashing_encoder_shape_and_density(toy_frame):
    enc = encoders.HashingEncoder(["cat1", "cat2"], n_bits=14)
    X = enc.transform(toy_frame)
    assert X.shape == (len(toy_frame), 1 << 14)
    assert X.nnz <= 2 * len(toy_frame)          # <= because two features may collide
    assert X.sum() == pytest.approx(2 * len(toy_frame))


def test_hashing_is_deterministic_within_a_process(toy_frame):
    enc = encoders.HashingEncoder(["cat1"], n_bits=12)
    assert (enc.transform(toy_frame) != enc.transform(toy_frame)).nnz == 0


def test_same_value_in_different_columns_hashes_apart():
    df = pd.DataFrame({"cat1": [7] * 100, "cat2": [7] * 100})
    enc = encoders.HashingEncoder(["cat1", "cat2"], n_bits=16)
    assert enc.transform(df).nnz == 200, "columns are not salted independently"


# --------------------------------------------------------------------------------------
# Week 3 exercises -- remove the xfail marker as you implement each one
# --------------------------------------------------------------------------------------
@pytest.mark.xfail(raises=TodoError, reason="Week 3 exercise")
def test_reliability_curve_bins_are_wellformed(toy):
    y, p = toy
    mp, mt, cnt = metrics.reliability_curve(y, p, n_bins=20, strategy="quantile")
    assert len(mp) == len(mt) == len(cnt)
    assert cnt.sum() == len(y), "every row must land in exactly one bin"
    assert (cnt > 0).all(), "empty bins must be dropped, not returned"
    assert np.all(np.diff(mp) >= 0), "bins should come back in ascending prediction order"


@pytest.mark.xfail(raises=TodoError, reason="Week 3 exercise")
def test_ece_is_near_zero_for_a_calibrated_model_and_large_when_inflated(toy):
    y, p = toy
    assert metrics.ece(y, p) < 0.01
    assert metrics.ece(y, np.clip(p * 3, 0, 1)) > metrics.ece(y, p)


@pytest.mark.xfail(raises=TodoError, reason="Week 3 exercise")
def test_platt_scaler_recovers_calibration(toy):
    y, p = toy
    bad = np.clip(p * 2.5, 1e-9, 1 - 1e-9)
    fixed = calibration.PlattScaler().fit_transform(bad, y)
    assert abs(metrics.calibration_ratio(y, fixed) - 1) < abs(metrics.calibration_ratio(y, bad) - 1)


@pytest.mark.xfail(raises=TodoError, reason="Week 3 exercise")
def test_isotonic_preserves_monotonicity(toy):
    y, p = toy
    out = calibration.IsotonicCalibrator().fit_transform(p, y)
    order = np.argsort(p)
    assert np.all(np.diff(out[order]) >= -1e-12)


@pytest.mark.xfail(raises=TodoError, reason="Week 3 exercise")
def test_sampling_rate_corrector_inverts_downsampling():
    w = 0.1
    q = np.array([0.01, 0.1, 0.5, 0.9])
    p = calibration.SamplingRateCorrector(w).transform(q)
    assert np.all(p < q), "correcting for dropped negatives must lower the prediction"
    assert p[0] == pytest.approx(q[0] / (q[0] + (1 - q[0]) / w))


# --------------------------------------------------------------------------------------
# Week 7 exercises
# --------------------------------------------------------------------------------------
@pytest.mark.xfail(raises=TodoError, reason="Week 7 exercise")
def test_qini_handles_unequal_group_sizes():
    """The treated/control imbalance must be rescaled away, or a null model 'wins'."""
    n = 20_000
    t = RNG.binomial(1, 0.85, n)                    # Criteo-like 85/15 imbalance
    y = RNG.binomial(1, 0.02 + 0.0 * t)             # zero true uplift
    score = RNG.random(n)                           # random ranking
    assert abs(metrics.qini_auc(y, score, t)) < 0.05, "no uplift + random score => ~0 Qini"
