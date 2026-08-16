"""Plot helpers with consistent styling, so figures from week 1 and week 12 sit together.

Every function takes an optional ``ax`` and returns it, so they compose into subplots.
Nothing here calls ``plt.show()``; notebooks do that.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .paths import figures as _figdir

# A restrained palette that survives greyscale printing and colour-blind readers.
PALETTE = ["#3b6ea5", "#c2603d", "#4f8a5b", "#8a6bab", "#b8973f", "#7a7a7a"]


def use_style() -> None:
    """Call once at the top of a notebook."""
    plt.rcParams.update({
        "figure.figsize": (7, 4.5),
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "font.size": 10,
        "legend.frameon": False,
    })


def save(fig, week: int, name: str) -> str:
    """Save into ``weekNN_*/figures/`` and return the path (print it in the notebook)."""
    p = _figdir(week) / (name if name.endswith(".png") else name + ".png")
    fig.savefig(p)
    return str(p)


def reliability_diagram(y_true, y_prob, n_bins: int = 20, strategy: str = "quantile",
                        label: str | None = None, ax=None):
    """Plot mean label against mean prediction per bin, with the diagonal for reference.

    Depends on :func:`adslab.metrics.reliability_curve`, so it starts working the moment
    you finish the Week 3 exercise -- until then it raises ``TodoError``, on purpose.

    Reading it: points *below* the diagonal are over-prediction (you will overbid),
    above is under-prediction (you will lose winnable auctions). The marker sizes show
    bin counts, because a dramatic deviation in a bin holding 40 rows is not a finding.
    """
    from .metrics import reliability_curve

    mean_pred, mean_true, counts = reliability_curve(y_true, y_prob, n_bins, strategy)
    ax = ax or plt.gca()
    lim = max(float(np.max(mean_pred)), float(np.max(mean_true))) * 1.08
    ax.plot([0, lim], [0, lim], ls="--", lw=1, color="#999", zorder=1,
            label="perfect calibration")
    sizes = 20 + 120 * (counts / counts.max())
    ax.scatter(mean_pred, mean_true, s=sizes, alpha=0.85, zorder=3, label=label)
    ax.plot(mean_pred, mean_true, lw=1, alpha=0.5, zorder=2)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Reliability diagram")
    ax.legend()
    return ax


def prediction_histogram(y_prob, bins: int = 60, log_y: bool = True, ax=None):
    """Where the predictions actually live. Run this before any calibration work.

    Predictions cluster tightly around the base rate -- 4.9% on the attribution set,
    0.7% on FairJob -- so almost the whole distribution occupies a sliver of [0, 1].
    That is why equal-width calibration bins are useless here, and why a reliability
    diagram drawn to a full 0-1 axis looks like a single dot near the origin.
    """
    ax = ax or plt.gca()
    ax.hist(np.asarray(y_prob).ravel(), bins=bins, color=PALETTE[0], alpha=0.85)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("count (log)" if log_y else "count")
    ax.set_title("Prediction distribution")
    return ax


def lift_by_decile(y_true, y_prob, n_groups: int = 10, ax=None):
    """Observed rate per predicted-score decile -- the plot a business stakeholder reads."""
    import pandas as pd

    d = pd.DataFrame({"y": np.asarray(y_true).ravel(), "p": np.asarray(y_prob).ravel()})
    d["grp"] = pd.qcut(d["p"].rank(method="first"), n_groups, labels=False)
    g = d.groupby("grp").agg(observed=("y", "mean"), predicted=("p", "mean"))
    ax = ax or plt.gca()
    x = np.arange(len(g))
    ax.bar(x - 0.2, g["observed"], width=0.4, label="observed", color=PALETTE[0])
    ax.bar(x + 0.2, g["predicted"], width=0.4, label="predicted", color=PALETTE[1])
    ax.axhline(d["y"].mean(), ls="--", lw=1, color="#999", label="base rate")
    ax.set_xlabel(f"predicted-score group (1 = lowest, {n_groups} = highest)")
    ax.set_ylabel("rate")
    ax.set_title("Lift by decile")
    ax.legend()
    return ax


def conversion_delay_hist(delays_hours, ax=None, max_hours: float = 24 * 30):
    """Histogram of conversion delay in hours -- the picture Week 4 is built around.

    The shape to look for is a huge spike inside the first hour and a long, fat tail over
    days. That tail is the reason a model trained on "converted so far" is biased: at any
    training cutoff, a large share of the eventual positives have not happened yet.
    """
    ax = ax or plt.gca()
    d = np.asarray(delays_hours, dtype=float)
    d = d[np.isfinite(d) & (d >= 0) & (d <= max_hours)]
    ax.hist(d, bins=np.logspace(-2, np.log10(max_hours), 60), color=PALETTE[2], alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("conversion delay (hours, log)")
    ax.set_ylabel("count")
    ax.set_title("Conversion delay distribution")
    return ax
