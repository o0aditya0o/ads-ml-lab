#!/usr/bin/env python3
"""Build the downloadable competition files for a week from ``data/raw``.

Run once per week before the judge can serve it:

    python -m judge.prepare_data --week 1

What it guarantees, and why each one matters:

* **The split is time-ordered.** Train is strictly earlier than test. This is the same
  rule ``docs/eval-protocol.md`` imposes on the notebooks; a competition that let people
  shuffle would teach the opposite of the repo.
* **The solution file never enters the served directory tree.** It lands next to the
  public files but every download route in ``judge/app.py`` resolves against an explicit
  allow-list of three filenames, so it cannot be requested.
* **Public/private assignment is deterministic** — a hash of the impression id, not a
  random draw — so re-running does not silently reshuffle which rows are public and
  invalidate every score already on the board.
"""
from __future__ import annotations

import argparse
import hashlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from adslab import data as adsdata  # noqa: E402
from judge.competitions import COMPETITIONS, Competition  # noqa: E402

# Kept small enough that a submission is a few MB rather than fifty, and that the whole
# thing fits comfortably on a small deployment volume.
TRAIN_ROWS = 1_500_000
TEST_ROWS = 400_000

# click_pos and click_nb are NOT here, and the omission is the important part: in this
# dataset both are -1 on every non-converting impression and >=1 on every converting one,
# so each encodes the label exactly. Measured, not assumed — the column documentation
# describes them as click-sequence position and per-user click count, which is what they
# mean upstream but not what survives into this file.
FEATURES = (adsdata.CAT_FEATURES + ["click", "cost", "time_since_last_click"])

# Anything here is refused in a served file. Kept as an explicit list so a future edit that
# reintroduces one fails loudly instead of quietly handing out the answer.
LEAKY = {"conversion", "conversion_timestamp", "conversion_id", "attribution", "cpo",
         "click_pos", "click_nb", "conversion_delay", "conversion_delay_hours"}


def _public_mask(ids: np.ndarray, frac: float, salt: str) -> np.ndarray:
    """Deterministic public/private assignment keyed on the id, not on chance."""
    h = np.array([
        int.from_bytes(hashlib.blake2b(f"{salt}:{i}".encode(), digest_size=8).digest(), "big")
        for i in ids
    ], dtype=np.uint64)
    return (h % np.uint64(10_000)) < np.uint64(int(frac * 10_000))



def _assert_no_feature_leaks(df, target: str, cols, max_auc: float = 0.99) -> None:
    """Fail if any single served column predicts the label almost perfectly.

    A blocklist of column names only catches leaks you already know about. This catches
    them by *behaviour*, which is how ``click_pos`` and ``click_nb`` were eventually
    found: both are -1 on every non-converting impression and >=1 on every converting one,
    so each reproduced the label exactly while looking like an ordinary count.

    Run on every week before its files are written. Cheap, and it turns a silent,
    competition-ending bug into a failed build.
    """
    from adslab import metrics as M

    y = df[target].to_numpy()
    if len(np.unique(y)) < 2:
        return
    offenders = []
    for c in cols:
        if c == target:
            continue
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(v).any():
            continue
        v = np.nan_to_num(v, nan=float(np.nanmedian(v)))
        span = v.max() - v.min()
        if span <= 0:
            continue
        a = M.auc(y, (v - v.min()) / span)
        if max(a, 1 - a) > max_auc:
            offenders.append((c, a))
    if offenders:
        detail = ", ".join(f"{c} (AUC {a:.4f})" for c, a in offenders)
        raise AssertionError(
            f"these columns predict {target!r} almost perfectly and must not be served: "
            f"{detail}. Add them to LEAKY and drop them from FEATURES.")


def _write_sidecars(comp, train, test, train_cols, test_cols, n_preview: int = 8) -> None:
    """Write facts.json and preview.json — the small files the web pages read.

    Both exist so a page never has to open a multi-MB gzip, and so an instance with no
    local copy of the data can still render. Shared by every week's preparer.
    """
    import json

    def clean(v):
        if pd.isna(v):
            return None
        return float(f"{v:.6g}") if isinstance(v, float) else v

    def block(df, cols):
        h = df[cols].head(n_preview)
        return {"columns": list(h.columns),
                "rows": [[clean(v) for v in rec] for rec in h.itertuples(index=False, name=None)]}

    sample = pd.read_csv(comp.sample_file, nrows=n_preview)
    (comp.dir / "preview.json").write_text(json.dumps({
        "train": block(train, train_cols),
        "test": block(test, test_cols),
        "sample": block(sample, list(sample.columns)),
    }, indent=2, default=str))

    (comp.dir / "facts.json").write_text(json.dumps({
        "train_rows": len(train),
        "test_rows": len(test),
        "base_rate": float(test[comp.target_column].mean()),
        "train_size_mb": round(comp.train_file.stat().st_size / 1e6, 1),
        "test_size_mb": round(comp.test_file.stat().st_size / 1e6, 1),
        "sample_size_mb": round(comp.sample_file.stat().st_size / 1e6, 1),
    }, indent=2))


def prepare_week1(comp: Competition, seed: int = 0) -> None:
    print("loading the attribution dataset ...", flush=True)
    df = adsdata.add_attribution_derived(adsdata.load_attribution())
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    print(f"  {len(df):,} rows over {df.timestamp.max()/86400:.1f} days")

    # Time-ordered boundary first, subsample within each side second. Subsampling before
    # splitting would let a test row predate a train row.
    cut = df.timestamp.quantile(0.70)
    train_pool = df[df.timestamp < cut]
    test_pool = df[df.timestamp >= cut]
    print(f"  boundary at t={cut:.0f} ({cut/86400:.1f}d): "
          f"{len(train_pool):,} before / {len(test_pool):,} after")

    rng = np.random.default_rng(seed)
    train = train_pool.iloc[np.sort(rng.choice(len(train_pool), min(TRAIN_ROWS, len(train_pool)), replace=False))]
    test = test_pool.iloc[np.sort(rng.choice(len(test_pool), min(TEST_ROWS, len(test_pool)), replace=False))]

    train = train.reset_index(drop=True).copy()
    test = test.reset_index(drop=True).copy()
    train[comp.id_column] = np.arange(len(train), dtype=np.int64)
    test[comp.id_column] = np.arange(len(test), dtype=np.int64)

    assert train.timestamp.max() < test.timestamp.min(), "train/test overlap in time"

    _assert_no_feature_leaks(train, comp.target_column, FEATURES)

    comp.dir.mkdir(parents=True, exist_ok=True)

    # ---- public: train with labels -----------------------------------------------
    train_cols = [comp.id_column, "timestamp"] + FEATURES + [comp.target_column]
    train[train_cols].to_csv(comp.train_file, index=False, compression="gzip")

    # ---- public: test WITHOUT labels ---------------------------------------------
    # Drop every column that leaks the outcome. conversion_timestamp and conversion_id
    # are the obvious ones; `attribution` is the subtle one — it is Criteo's last-click
    # flag and it is nonzero only on converting journeys, so it is a perfect giveaway.
    test_cols = [c for c in [comp.id_column, "timestamp"] + FEATURES if c not in LEAKY]
    assert not (set(test_cols) & LEAKY), "leaky column survived into the test file"
    assert not (set(train_cols) - {comp.target_column}) & LEAKY, \
        "leaky column survived into the train file"
    test[test_cols].to_csv(comp.test_file, index=False, compression="gzip")

    # ---- public: sample submission -----------------------------------------------
    pd.DataFrame({
        comp.id_column: test[comp.id_column],
        "prediction": float(train[comp.target_column].mean()),
    }).to_csv(comp.sample_file, index=False, compression="gzip")

    # ---- private: the solution ----------------------------------------------------
    sol = pd.DataFrame({
        comp.id_column: test[comp.id_column],
        comp.target_column: test[comp.target_column].astype("int8"),
        "is_public": _public_mask(test[comp.id_column].to_numpy(), comp.public_frac,
                                  salt=comp.slug),
    })
    sol.to_parquet(comp.solution_file, index=False)

    _write_sidecars(comp, train, test, train_cols, test_cols)

    print(f"\n  train  {len(train):>9,} rows  {comp.train_file.stat().st_size/1e6:6.1f} MB  "
          f"base rate {train[comp.target_column].mean():.4%}")
    print(f"  test   {len(test):>9,} rows  {comp.test_file.stat().st_size/1e6:6.1f} MB  "
          f"base rate {test[comp.target_column].mean():.4%}  (hidden)")
    print(f"  public {sol.is_public.sum():>9,} rows ({sol.is_public.mean():.1%}) / "
          f"private {(~sol.is_public).sum():,}")
    print(f"  solution -> {comp.solution_file.name} (never served)")

    drift = test[comp.target_column].mean() / train[comp.target_column].mean() - 1
    print(f"\n  note: test base rate is {drift:+.1%} vs train — the label drifts across the "
          f"window,\n        which is a real part of the problem and not a bug in the split.")


def prepare_week3(comp: Competition, seed: int = 0) -> None:
    """Week 3: hand out deliberately miscalibrated predictions to be corrected.

    The miscalibration is produced the way production systems produce it — by training on
    negatively downsampled data. That inflates every predicted probability, and because
    the model is non-linear the inflation is not a clean constant, so a single scalar
    correction leaves error on the table. That gap is the competition.

    The entrant never sees the sampling rate, and never sees the model.
    """
    import lightgbm as lgb

    from adslab.encoders import HashingEncoder

    NEG_KEEP = 0.05          # kept secret from entrants; the source of the inflation
    SEGMENTS = ["cat1", "cat6", "cat8"]

    print("loading the attribution dataset ...", flush=True)
    df = adsdata.add_attribution_derived(adsdata.load_attribution())
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    cut = df.timestamp.quantile(0.70)
    train_pool, test_pool = df[df.timestamp < cut], df[df.timestamp >= cut]
    rng = np.random.default_rng(seed)
    train = train_pool.iloc[np.sort(rng.choice(len(train_pool), min(TRAIN_ROWS, len(train_pool)), replace=False))].reset_index(drop=True)
    test = test_pool.iloc[np.sort(rng.choice(len(test_pool), min(TEST_ROWS, len(test_pool)), replace=False))].reset_index(drop=True)
    assert train.timestamp.max() < test.timestamp.min(), "train/test overlap in time"
    train[comp.id_column] = np.arange(len(train), dtype=np.int64)
    test[comp.id_column] = np.arange(len(test), dtype=np.int64)

    # --- fit the miscalibrated model on downsampled negatives -----------------------
    pos = train[train[comp.target_column] == 1]
    neg = train[train[comp.target_column] == 0].sample(frac=NEG_KEEP, random_state=seed)
    fit = pd.concat([pos, neg]).sample(frac=1.0, random_state=seed)
    print(f"  model trained on {len(fit):,} rows "
          f"({len(pos):,} pos + {len(neg):,} of {len(train)-len(pos):,} neg, keep={NEG_KEEP})")

    feats = adsdata.CAT_FEATURES + ["click", "cost", "time_since_last_click"]
    # The hashed ids are fed as plain numerics rather than declared categorical: with
    # cat7 at 57k levels, LightGBM's categorical handling sorts every level at every split
    # and the run takes half an hour. This model only has to be plausible and badly
    # calibrated, not good, so numeric splits on hashed ids are entirely adequate.
    model = lgb.train(
        {"objective": "binary", "learning_rate": 0.1, "num_leaves": 31,
         "verbose": -1, "seed": seed, "num_threads": 4},
        lgb.Dataset(fit[feats], label=fit[comp.target_column]),
        num_boost_round=100,
    )
    print("  scoring train/test ...", flush=True)
    train["raw_prediction"] = model.predict(train[feats])
    test["raw_prediction"] = model.predict(test[feats])

    inflation = train.raw_prediction.mean() / train[comp.target_column].mean()
    print(f"  raw predictions are inflated {inflation:.1f}x on train "
          f"(mean {train.raw_prediction.mean():.4f} vs base rate "
          f"{train[comp.target_column].mean():.4f})")

    cols = [comp.id_column, "raw_prediction"] + SEGMENTS
    # raw_prediction is *meant* to correlate with the label; it is the thing being
    # calibrated. Everything else served must not.
    _assert_no_feature_leaks(train, comp.target_column, SEGMENTS)

    comp.dir.mkdir(parents=True, exist_ok=True)
    train[cols + [comp.target_column]].to_csv(comp.train_file, index=False, compression="gzip")
    test[cols].to_csv(comp.test_file, index=False, compression="gzip")
    pd.DataFrame({comp.id_column: test[comp.id_column],
                  "prediction": test.raw_prediction}).to_csv(
        comp.sample_file, index=False, compression="gzip")

    sol = pd.DataFrame({
        comp.id_column: test[comp.id_column],
        comp.target_column: test[comp.target_column].astype("int8"),
        "is_public": _public_mask(test[comp.id_column].to_numpy(), comp.public_frac,
                                  salt=comp.slug),
    })
    sol.to_parquet(comp.solution_file, index=False)
    _write_sidecars(comp, train, test, cols + [comp.target_column], cols)

    from adslab import metrics as M
    from judge import reference_metrics as R

    y, p = test[comp.target_column].to_numpy(), test.raw_prediction.to_numpy()
    rescaled = np.clip(p * y.mean() / p.mean(), 1e-9, 1 - 1e-9)
    print(f"\n  handed out on test : ECE {R.ece(y, p):.5f}  AUC {M.auc(y, p):.5f}  "
          f"mean {p.mean():.4f} vs base rate {y.mean():.4f}")
    print(f"  after a global rescale: ECE {R.ece(y, rescaled):.5f}  AUC {M.auc(y, rescaled):.5f}")
    print("  -> the gap between those two ECEs is what a shape-aware calibrator can win")


PREPARERS = {1: prepare_week1, 3: prepare_week3}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    comp = COMPETITIONS.get(args.week)
    if comp is None:
        print(f"week {args.week} has no competition defined in judge/competitions.py",
              file=sys.stderr)
        return 2
    if comp.is_prepared and not args.force:
        print(f"{comp.dir} already prepared; --force to rebuild")
        return 0
    PREPARERS[args.week](comp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
