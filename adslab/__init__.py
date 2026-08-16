"""adslab -- shared plumbing for the 12-week ads measurement curriculum.

The deal this package makes with the notebooks: it owns everything that must be
*identical* across twelve weeks (how data loads, how time splits, how a result is
scored and recorded), and it owns nothing that is worth learning by writing. The
metrics and calibrators that are the point of a given week ship as documented stubs
raising :class:`adslab.metrics.TodoError`.

Typical opening of a notebook::

    from adslab import data, metrics, plots, split, registry
    plots.use_style()

    df = data.add_attribution_derived(data.load_attribution())
    sp = split.time_split(df, "timestamp")
    split.check_no_leakage(df, sp)
"""
from __future__ import annotations

__version__ = "0.1.0"

from . import calibration, data, encoders, metrics, paths, plots, registry, split  # noqa: F401

__all__ = [
    "calibration", "data", "encoders", "metrics", "paths", "plots", "registry", "split",
]
