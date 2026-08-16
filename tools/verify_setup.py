#!/usr/bin/env python3
"""End-to-end check that the repo is usable: data loads, splits hold, harness imports.

Run this first thing after cloning, and any time a notebook behaves strangely. It is
deliberately noisy -- the printout doubles as a profile of each dataset (row counts,
base rates, cardinalities, delay distribution) which is genuinely useful to eyeball
before modelling.

    python tools/verify_setup.py            # everything
    python tools/verify_setup.py --quick    # skip the 650 MB attribution parse
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback

import numpy as np

from adslab import data, encoders, metrics, split

OK, BAD = "  ok  ", " FAIL "


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def check(name: str, fn):
    t0 = time.time()
    try:
        out = fn()
        print(f"[{OK}] {name}  ({time.time() - t0:.1f}s)")
        return out
    except Exception as e:
        print(f"[{BAD}] {name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    failures = 0

    section("attribution (Criteo Attribution Modeling for Bidding)")
    if args.quick:
        df = check("load_attribution(nrows=200k)",
                   lambda: data.load_attribution(nrows=200_000, cache=False))
    else:
        df = check("load_attribution() [full 16.5M rows; first run parses 650 MB gzip]",
                   data.load_attribution)
    if df is None:
        failures += 1
    else:
        df = data.add_attribution_derived(df)
        print(f"       rows={len(df):,}  cols={len(df.columns)}")
        print(f"       window={df.timestamp.max() / 86400:.1f} days, "
              f"users={df.uid.nunique():,}, campaigns={df.campaign.nunique():,}")
        print(f"       conversion rate={df.conversion.mean():.4%}  "
              f"click rate={df.click.mean():.4%}  "
              f"last-click attributed={df.attribution.mean():.4%}")
        d = df.conversion_delay_hours.dropna()
        if len(d):
            q = np.percentile(d, [50, 75, 90, 99])
            print(f"       conversion delay hours: p50={q[0]:.2f} p75={q[1]:.2f} "
                  f"p90={q[2]:.1f} p99={q[3]:.1f}  max={d.max():.1f}")
            print(f"       -> {(d > 24).mean():.1%} of conversions land more than a day "
                  f"after the impression. That fraction is Week 4's whole problem.")
        print(f"       cat feature cardinalities: "
              f"{ {c: int(df[c].nunique()) for c in data.CAT_FEATURES[:4]} }")

        sp = check("time_split + leakage assertion",
                   lambda: (lambda s: (split.check_no_leakage(df, s), s)[1])(
                       split.time_split(df, "timestamp")))
        if sp is None:
            failures += 1
        else:
            print(f"       {sp}")

        gsp = check("user_grouped_time_split",
                    lambda: split.user_grouped_time_split(df, "timestamp", "uid"))
        if gsp is not None:
            print(f"       {gsp}")

        def _enc():
            enc = encoders.HashingEncoder(data.CAT_FEATURES, n_bits=18)
            sub = df.head(200_000)
            X = enc.transform(sub)
            print(f"       X={X.shape} nnz={X.nnz:,}  {enc.collision_report(sub)}")
            return X
        if check("HashingEncoder on 200k rows", _enc) is None:
            failures += 1

    section("uplift (Criteo Uplift v2.1)")
    up = check("load_uplift(nrows=300k) [random subsample; head-reads are all-treated]",
               lambda: data.load_uplift(nrows=300_000))
    if up is None:
        failures += 1
    else:
        print(f"       rows={len(up):,}  treated={up.treatment.mean():.1%}")
        if up.treatment.mean() > 0.95:
            print("       [ FAIL ] no control group -- the subsample is not random")
            failures += 1
        for out in ("visit", "conversion", "exposure"):
            t = up.loc[up.treatment == 1, out].mean()
            c = up.loc[up.treatment == 0, out].mean()
            print(f"       {out:11s} treated={t:.4%}  control={c:.4%}  "
                  f"naive lift={t - c:+.4%}")

    section("privatead (CriteoPrivateAd)")
    pa = check("load_privatead(days=[1,2])", lambda: data.load_privatead(days=[1, 2]))
    if pa is None:
        failures += 1
    else:
        print(f"       rows={len(pa):,}  cols={len(pa.columns)}  days={sorted(pa.day_int.unique())}")
        for b, cols in data.privatead_feature_groups(pa).items():
            print(f"       {len(cols):3d} cols  {b}")
        for lab in ("is_clicked", "is_click_landed", "is_visit"):
            if lab in pa.columns:
                print(f"       {lab:16s} rate={pa[lab].mean():.4%}")

    section("fairjob (bonus)")
    fj = check("load_fairjob()", data.load_fairjob)
    if fj is not None:
        print(f"       rows={len(fj):,}  click rate={fj.click.mean():.4%}  "
              f"randomised slice={fj.displayrandom.mean():.1%}")

    section("metrics harness")
    rng = np.random.default_rng(0)
    p = rng.beta(1.5, 60, 50_000)
    y = rng.binomial(1, p)
    m = metrics.evaluate(y, p)
    print("       " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in m.items()))
    todo = [k for k, v in m.items() if v is None]
    print(f"       still unimplemented (by design): {todo}")

    print(f"\n{'=' * 72}\n{'SETUP OK' if not failures else f'{failures} CHECK(S) FAILED'}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
