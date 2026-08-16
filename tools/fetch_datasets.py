#!/usr/bin/env python3
"""Download the datasets used across the 12 weeks into ``data/raw/``.

Everything is pulled from Criteo's *official* Hugging Face org (``criteo/*``), which is
the only source still serving these files -- the old ``go.criteo.net/...`` direct links
that most blog posts cite are dead (404 as of 2026-08).

Nothing here is committed to git; ``data/`` is ignored. Re-running is cheap: the
Hugging Face cache is content-addressed, so completed files are not re-downloaded.

Usage
-----
    python tools/fetch_datasets.py                # the default set (~3.7 GB)
    python tools/fetch_datasets.py --list         # show the catalogue and exit
    python tools/fetch_datasets.py attribution    # just one
    python tools/fetch_datasets.py --privatead-days 0 1 2   # widen the PrivateAd slice
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"

# --------------------------------------------------------------------------------------
# Catalogue.  ``weeks`` is documentation only -- it drives the printed summary and the
# generated data/README.md so you can tell at a glance what a week needs.
# --------------------------------------------------------------------------------------
CATALOGUE = {
    "attribution": dict(
        repo="criteo/criteo-attribution-dataset",
        files=["criteo_attribution_dataset.tsv.gz"],
        approx_gb=0.65,
        weeks=[1, 4, 6, 8, 11],
        note=(
            "16.5M impressions, 30 days, per-user timelines with click AND conversion "
            "timestamps plus the price paid. The workhorse of this repo: it is the only "
            "public set that supports attribution, delayed feedback, auction prices and "
            "user sequences at once."
        ),
    ),
    "uplift": dict(
        repo="criteo/criteo-uplift",
        files=["criteo-research-uplift-v2.1.csv.gz"],
        approx_gb=0.31,
        weeks=[7],
        note=("13.98M rows from a randomised treatment/control ad experiment, 85% treated. "
              "(Often cited as 25M -- that is the v2.0 figure; v2.1 is smaller.) "
              "Week 7 (Qini). Note the file is grouped by treatment: never read its head."),
    ),
    "fairjob": dict(
        repo="criteo/FairJob",
        files=["fairjob.csv.gz"],
        approx_gb=0.19,
        weeks=[10],
        note=(
            "Job-ad click log with a protected attribute. Not in the original plan -- "
            "added as a bonus for Week 10, because 'privacy-constrained' and "
            "'fairness-constrained' learning are the same shape of problem and this is "
            "the only public ads dataset built for it."
        ),
    ),
    "privatead": dict(
        repo="criteo/CriteoPrivateAd",
        # Filled in at runtime from --privatead-days; the full set is 34 GB / 291 files.
        files=None,
        approx_gb=2.5,
        weeks=[1, 10],
        note=(
            "Bidding log designed for Privacy Sandbox research: features are tagged by "
            "which privacy bucket they survive in (cross-domain, single-domain, "
            "user-level, aggregate). The full repo is 34 GB, so we take a per-day slice."
        ),
    ),
}

# CriteoClickLogs is deliberately NOT downloadable by default: 276 GB.
CLICKLOGS_NOTE = """\
criteo/CriteoClickLogs is 276 GB (6029 parquet parts) and is NOT fetched by default.
If Week 2 needs more scale than the attribution set gives, pull a few parts by hand:

    from huggingface_hub import hf_hub_download
    hf_hub_download("criteo/CriteoClickLogs",
                    "data/day=2015-02-21/part-00415-99c339d5-fbac-4110-9dcf-75453a61a5c1.c000.snappy.parquet",
                    repo_type="dataset", local_dir="data/raw/clicklogs")
"""


def human(gb: float) -> str:
    return f"{gb * 1000:.0f} MB" if gb < 1 else f"{gb:.2f} GB"


def free_gb() -> float:
    return shutil.disk_usage(REPO).free / 1e9


def privatead_files(days: list[int], parts_per_day: int = 1) -> list[str]:
    """List the parquet parts belonging to the requested ``day_int`` partitions.

    The repo is partitioned ``day_int=1..30`` with ~10 parts per day. Parts within a day
    are interchangeable (same schema, same period), so one part per day across a *span*
    of days beats many parts from one day: it is the day axis that makes a time-ordered
    split possible.

    Parts are chosen by blob size, largest first. This is not a preference for big files
    -- it is a correctness fix. Spark wrote a zero-row ``part-00000`` into most days, so
    picking parts in lexical order silently yields empty frames.
    """
    from huggingface_hub import HfApi

    info = HfApi().repo_info("criteo/CriteoPrivateAd", repo_type="dataset", files_metadata=True)
    sizes = {s.rfilename: (s.size or 0) for s in info.siblings}
    wanted = []
    for d in days:
        parts = [f for f in sizes if f.startswith(f"data/day_int={d}/")]
        parts = sorted((p for p in parts if sizes[p] > 1_000_000), key=lambda p: -sizes[p])
        if not parts:
            print(f"  ! no non-empty parts found for day_int={d}", file=sys.stderr)
        wanted.extend(parts[:parts_per_day])
    return wanted


def fetch(name: str, spec: dict) -> list[Path]:
    from huggingface_hub import hf_hub_download

    dest = RAW / name
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for i, fn in enumerate(spec["files"], 1):
        print(f"  [{i}/{len(spec['files'])}] {fn}", flush=True)
        p = hf_hub_download(
            spec["repo"], fn, repo_type="dataset", local_dir=str(dest)
        )
        out.append(Path(p))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", default=None, help="subset of the catalogue; default = all")
    ap.add_argument("--list", action="store_true", help="print the catalogue and exit")
    ap.add_argument("--privatead-days", type=int, nargs="+", default=list(range(1, 15)),
                    help="day_int partitions of CriteoPrivateAd to pull (valid: 1-30; default: 1-14)")
    ap.add_argument("--privatead-parts-per-day", type=int, default=1,
                    help="parts to take from each day (default 1, ~130 MB each)")
    args = ap.parse_args()

    if args.list:
        for n, s in CATALOGUE.items():
            print(f"{n:14s} {human(s['approx_gb']):>9}  weeks {s['weeks']}\n{' ' * 16}{s['note']}\n")
        print(CLICKLOGS_NOTE)
        return 0

    names = args.names or list(CATALOGUE)
    unknown = [n for n in names if n not in CATALOGUE]
    if unknown:
        print(f"unknown dataset(s): {unknown}; known: {list(CATALOGUE)}", file=sys.stderr)
        return 2

    if "privatead" in names:
        print("resolving CriteoPrivateAd parts ...", flush=True)
        CATALOGUE["privatead"]["files"] = privatead_files(
            args.privatead_days, args.privatead_parts_per_day)
        CATALOGUE["privatead"]["approx_gb"] = 0.13 * len(CATALOGUE["privatead"]["files"])

    need = sum(CATALOGUE[n]["approx_gb"] for n in names)
    print(f"about to download ~{human(need)}; {free_gb():.1f} GB free\n")
    if free_gb() < need + 5:
        print("refusing to start: want at least 5 GB of headroom after the download", file=sys.stderr)
        return 1

    ok, failed = [], []
    for n in names:
        print(f"=== {n}  ({CATALOGUE[n]['repo']})", flush=True)
        try:
            fetch(n, CATALOGUE[n])
            ok.append(n)
        except Exception as e:  # keep going; one dead repo shouldn't sink the batch
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            failed.append(n)

    print(f"\ndone. ok={ok} failed={failed}  ({free_gb():.1f} GB free)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
