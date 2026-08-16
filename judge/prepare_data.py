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

FEATURES = (adsdata.CAT_FEATURES
            + ["click", "click_pos", "click_nb", "cost", "time_since_last_click"])


def _public_mask(ids: np.ndarray, frac: float, salt: str) -> np.ndarray:
    """Deterministic public/private assignment keyed on the id, not on chance."""
    h = np.array([
        int.from_bytes(hashlib.blake2b(f"{salt}:{i}".encode(), digest_size=8).digest(), "big")
        for i in ids
    ], dtype=np.uint64)
    return (h % np.uint64(10_000)) < np.uint64(int(frac * 10_000))


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

    comp.dir.mkdir(parents=True, exist_ok=True)

    # ---- public: train with labels -----------------------------------------------
    train_cols = [comp.id_column, "timestamp"] + FEATURES + [comp.target_column]
    train[train_cols].to_csv(comp.train_file, index=False, compression="gzip")

    # ---- public: test WITHOUT labels ---------------------------------------------
    # Drop every column that leaks the outcome. conversion_timestamp and conversion_id
    # are the obvious ones; `attribution` is the subtle one — it is Criteo's last-click
    # flag and it is nonzero only on converting journeys, so it is a perfect giveaway.
    leaky = {comp.target_column, "conversion_timestamp", "conversion_id", "attribution",
             "conversion_delay", "conversion_delay_hours", "cpo"}
    test_cols = [c for c in [comp.id_column, "timestamp"] + FEATURES if c not in leaky]
    assert not (set(test_cols) & leaky), "leaky column survived into the test file"
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


PREPARERS = {1: prepare_week1}


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
