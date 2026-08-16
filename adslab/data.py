"""Loaders for the four datasets, with the field semantics that actually matter.

Every loader returns a pandas DataFrame and caches an interim parquet copy, because
re-parsing a 650 MB gzipped TSV on every notebook restart is the fastest way to stop
running experiments.

The docstrings here are the dataset documentation -- read them before modelling. The
single most expensive mistake available in this repo is misreading ``conversion`` on the
attribution dataset (see :func:`load_attribution`).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import INTERIM, RAW

# --------------------------------------------------------------------------------------
# Criteo Attribution Modeling for Bidding
# --------------------------------------------------------------------------------------
ATTRIBUTION_COLS = {
    "timestamp": "seconds since the start of the 30-day window; the ONLY valid sort key",
    "uid": "hashed user id -- gives you per-user journeys (weeks 6 and 11)",
    "campaign": "hashed campaign id",
    "conversion": "1 if this impression's user converted within 30 days of it",
    "conversion_timestamp": "absolute time of that conversion, -1 if none. Week 4 lives here",
    "conversion_id": "groups the impressions that competed for one conversion",
    "attribution": "1 if Criteo's own last-click rule gave THIS impression the credit",
    "click": "1 if the impression was clicked",
    "click_pos": "index of this click within the user's click sequence, -1 if not clicked",
    "click_nb": "number of clicks the user made in the window",
    "cost": "price paid for the impression -- Week 8 uses this for win-price distributions",
    "cpo": "cost per order at campaign level",
    "time_since_last_click": "seconds since this user's previous click, -1 if none",
    **{f"cat{i}": "hashed contextual/user feature" for i in range(1, 10)},
}
CAT_FEATURES = [f"cat{i}" for i in range(1, 10)]


def load_attribution(
    nrows: int | None = None,
    columns: list[str] | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """Criteo Attribution Modeling for Bidding: 16.5M impressions over 30 days.

    Two traps, both of which silently produce a great-looking, meaningless model:

    1. ``conversion`` is an *impression-level* label meaning "the user converted within
       30 days", not "this impression caused a conversion". Every impression in a
       converting journey carries ``conversion=1``. Predicting it is a legitimate CVR
       task, but it is not attribution -- that is ``attribution`` (last-click) and it is
       what Week 6 replaces with better rules.
    2. ``conversion_timestamp`` is *absolute*, not a delay. The delay Week 4 needs is
       ``conversion_timestamp - timestamp``, and only where ``conversion == 1``.

    A third, subtler one: because the label looks 30 days into the future, the last 30
    days of any window are censored. On this dataset the whole window is 30 days, so the
    tail is progressively under-labelled -- which is exactly the bias Week 4 exists to
    model, and exactly why the time-ordered split in :mod:`adslab.split` matters.
    """
    src = RAW / "attribution" / "criteo_attribution_dataset.tsv.gz"
    cached = INTERIM / "attribution.parquet"

    if cache and cached.exists() and nrows is None:
        df = pd.read_parquet(cached, columns=columns)
    else:
        if not src.exists():
            raise FileNotFoundError(f"{src} missing -- run `python tools/fetch_datasets.py attribution`")
        df = pd.read_csv(src, sep="\t", nrows=nrows, compression="gzip")
        if cache and nrows is None:
            df.to_parquet(cached, index=False)
        if columns:
            df = df[columns]

    return df


def add_attribution_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add the columns every week ends up recomputing. Cheap, vectorised, non-destructive."""
    out = df.copy()
    if {"conversion", "conversion_timestamp", "timestamp"} <= set(out.columns):
        delay = out["conversion_timestamp"] - out["timestamp"]
        # -1 sentinel and any non-positive delay are "no observed conversion"
        out["conversion_delay"] = np.where(out["conversion"] == 1, delay, np.nan)
        out.loc[out["conversion_delay"] < 0, "conversion_delay"] = np.nan
        out["conversion_delay_hours"] = out["conversion_delay"] / 3600.0
    if "timestamp" in out.columns:
        out["day"] = (out["timestamp"] // 86400).astype("int32")
        out["hour_of_day"] = ((out["timestamp"] % 86400) // 3600).astype("int8")
    return out


# --------------------------------------------------------------------------------------
# Criteo Uplift
# --------------------------------------------------------------------------------------
def load_uplift(
    frac: float | None = None,
    nrows: int | None = None,
    seed: int = 0,
    cache: bool = True,
) -> pd.DataFrame:
    """Criteo Uplift v2.1: 13.98M rows from a randomised ad experiment.

    Columns: ``f0..f11`` (dense, already anonymised/scaled), ``treatment`` (1 = the user
    was eligible to be shown the ad), and three outcomes: ``visit``, ``conversion``,
    ``exposure``.

    **The file is grouped by treatment.** The first 300k rows are 100% treated and so are
    the last 300k; the 85/15 split only appears if you read the whole thing. So
    ``nrows``/``frac`` here mean *random subsample*, taken after a full read -- they do
    not stream the head of the file, which would silently hand you an experiment with no
    control group and uplift estimates of ``nan``. (This repo learned that the hard way;
    ``tools/verify_setup.py`` still checks for it.)

    Two more things to internalise before Week 7:

    - Treatment is ~85% of rows, so the control group is the scarce resource and it caps
      the precision of every uplift estimate you can make here.
    - ``exposure`` is *post-treatment* -- it says the user actually saw the ad.
      Conditioning or splitting on it breaks randomisation and reintroduces exactly the
      selection bias uplift modelling exists to avoid. Use ``treatment`` as the
      intervention and leave ``exposure`` as a descriptive column.

    There is no timestamp: this is the one dataset in the repo where a *random* split is
    correct, because the rows are i.i.d. draws from one experiment.
    """
    src = RAW / "uplift" / "criteo-research-uplift-v2.1.csv.gz"
    cached = INTERIM / "uplift.parquet"

    if cache and cached.exists():
        df = pd.read_parquet(cached)
    else:
        if not src.exists():
            raise FileNotFoundError(f"{src} missing -- run `python tools/fetch_datasets.py uplift`")
        df = pd.read_csv(src, compression="gzip")
        if cache:
            df.to_parquet(cached, index=False)

    if frac is not None:
        df = df.sample(frac=frac, random_state=seed)
    elif nrows is not None:
        df = df.sample(n=min(nrows, len(df)), random_state=seed)
    else:
        return df
    return df.reset_index(drop=True)


UPLIFT_FEATURES = [f"f{i}" for i in range(12)]


# --------------------------------------------------------------------------------------
# CriteoPrivateAd
# --------------------------------------------------------------------------------------
PRIVACY_BUCKETS = {
    "features_kv_bits_constrained": (
        "survives inside a Protected-Audience-style bit budget: the key-value signals a "
        "buyer may carry into an on-device auction"
    ),
    "features_browser_bits_constrained": (
        "browser-provided signals under the same bit budget"
    ),
    "features_kv_not_constrained": "key-value signals available without a bit budget",
    "features_ctx_not_constrained": "contextual signals -- always available, never restricted",
    "features_not_available": (
        "third-party-cookie-era signals that are GONE in the privacy-preserving setting. "
        "Training with these is the upper-bound oracle; the gap to it is the cost of privacy"
    ),
}


def privatead_feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Map each privacy bucket to the columns of ``df`` that belong to it.

    This mapping *is* Week 10's experiment design: train on progressively smaller unions
    of these buckets and the accuracy you lose is the accuracy privacy costs.
    """
    return {
        bucket: sorted(c for c in df.columns if c.startswith(bucket + "_"))
        for bucket in PRIVACY_BUCKETS
    }


def load_privatead(days: list[int] | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    """CriteoPrivateAd: a bidding log whose columns are tagged by privacy bucket.

    Labels: ``is_clicked``, ``is_click_landed``, ``is_visit``, ``nb_sales``. The
    ``*_delay_after_display_array`` columns hold per-event delays, so this dataset also
    supports delayed-feedback work with real privacy structure attached.

    ``day_int`` is recovered from the partition path and added as a column -- it is the
    time axis for splitting. Note the download only keeps a subset of days (see
    ``tools/fetch_datasets.py``), and that Spark wrote a zero-row ``part-00000`` into
    most days, so parts are selected by size upstream.
    """
    parts = sorted(glob.glob(str(RAW / "privatead" / "data" / "day_int=*" / "*.parquet")))
    if not parts:
        raise FileNotFoundError("no privatead parts -- run `python tools/fetch_datasets.py privatead`")

    frames = []
    for p in parts:
        day = int(Path(p).parent.name.split("=")[1])
        if days is not None and day not in days:
            continue
        f = pd.read_parquet(p, columns=columns)
        if len(f) == 0:
            continue
        f["day_int"] = np.int16(day)
        frames.append(f)
    if not frames:
        raise ValueError(f"no non-empty parts matched days={days}")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------
# FairJob
# --------------------------------------------------------------------------------------
def load_fairjob() -> pd.DataFrame:
    """FairJob: job-ad clicks with a binary ``protected_attribute`` and a ``senior`` label.

    ``displayrandom`` marks impressions from a randomised bucket -- that subset is an
    unbiased slice you can use as a pseudo-holdout, which is rare and useful.
    """
    src = RAW / "fairjob" / "fairjob.csv.gz"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing -- run `python tools/fetch_datasets.py fairjob`")
    return pd.read_csv(src, compression="gzip")
