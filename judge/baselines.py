"""Seeded leaderboard baselines.

Two entries, matching the repo's first ground rule: something dumb, and something
respectable. Beating the base rate is the bar for the model existing at all; beating the
logistic regression is the bar for the week being finished.

Predictions are cached to disk, because retraining a logistic regression on 1.5M rows at
every server start would make restarts unpleasant.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from adslab import data as adsdata
from adslab.encoders import HashingEncoder
from judge.competitions import Competition


def _cache_path(comp: Competition, label: str):
    return comp.dir / f"baseline_{label.replace('^', '')}.csv.gz"


def build_baseline_predictions(comp: Competition, label: str) -> bytes:
    """Return a submission file (gzipped CSV bytes) for the named baseline."""
    cache = _cache_path(comp, label)
    if cache.exists():
        return cache.read_bytes()

    train = pd.read_csv(comp.train_file)
    test = pd.read_csv(comp.test_file)
    y = train[comp.target_column].to_numpy()

    if label == "base_rate":
        pred = np.full(len(test), float(y.mean()))

    elif label.startswith("logreg_hashed"):
        from sklearn.linear_model import LogisticRegression

        enc = HashingEncoder(adsdata.CAT_FEATURES, n_bits=18)
        # Subsample for a sane startup time; this is a baseline, not an entry.
        n = min(400_000, len(train))
        idx = np.random.default_rng(0).choice(len(train), n, replace=False)
        Xtr = enc.transform(train.iloc[idx])
        model = LogisticRegression(solver="liblinear", C=1.0, max_iter=200)
        model.fit(Xtr, y[idx])
        pred = model.predict_proba(enc.transform(test))[:, 1]

    elif label == "raw_uncalibrated":
        # The predictions exactly as handed out. The bar every entry must clear, and a
        # useful sanity check: if a submission scores worse than this, the calibrator
        # made things worse, which happens more often than people expect.
        pred = test["raw_prediction"].to_numpy()

    elif label == "global_scalar":
        # One multiplicative constant fitted on train — the obvious first move, and the
        # one that a known negative-downsampling rate would give you in closed form.
        scale = train[comp.target_column].mean() / train["raw_prediction"].mean()
        pred = np.clip(test["raw_prediction"].to_numpy() * scale, 1e-9, 1 - 1e-9)

    else:
        raise ValueError(f"unknown baseline {label!r}")

    out = pd.DataFrame({comp.id_column: test[comp.id_column], "prediction": pred})
    buf = io.BytesIO()
    out.to_csv(buf, index=False, compression="gzip")
    raw = buf.getvalue()
    cache.write_bytes(raw)
    return raw
