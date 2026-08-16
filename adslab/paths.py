"""Filesystem layout. Import this instead of hard-coding paths in a notebook."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DATA = REPO / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"      # cached intermediate frames (parquet), safe to delete
PROCESSED = DATA / "processed"  # model-ready matrices
RESULTS = REPO / "results"      # results.jsonl + generated tables; committed

for _d in (RAW, INTERIM, PROCESSED, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)


def week_dir(n: int) -> Path:
    """``week_dir(3)`` -> the week03_* folder, whatever its descriptive suffix is."""
    hits = sorted(REPO.glob(f"week{n:02d}_*"))
    if not hits:
        raise FileNotFoundError(f"no week{n:02d}_* directory in {REPO}")
    return hits[0]


def figures(n: int) -> Path:
    d = week_dir(n) / "figures"
    d.mkdir(exist_ok=True)
    return d
