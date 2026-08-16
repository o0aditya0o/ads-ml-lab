"""Server-rendered inline SVG for the leaderboard.

No charting library and no client JS: the data is a handful of points already loaded
for the table, and an inline SVG inherits the page's CSS custom properties, so it
themes itself in light and dark without a second palette to maintain.

The palette is the validated default from the dataviz reference — slot 1 blue and
slot 2 orange, which clear the all-pairs CVD and normal-vision floors in both modes
against these surfaces. Baselines are deliberately *not* a third categorical hue:
they are reference marks, so they get neutral ink and a square marker, which keeps
the categorical set at two and leaves identity carried by shape as well as colour.
"""
from __future__ import annotations

import html
from dataclasses import dataclass

# Plot geometry
W, H = 760, 340
PAD_L, PAD_R, PAD_T, PAD_B = 64, 132, 28, 52


@dataclass
class Point:
    x: float
    y: float
    name: str
    label: str
    kind: str          # "entrant" | "you" | "baseline"


def _scale(v, lo, hi, out_lo, out_hi):
    if hi - lo < 1e-12:
        return (out_lo + out_hi) / 2
    return out_lo + (v - lo) / (hi - lo) * (out_hi - out_lo)


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Round tick values covering [lo, hi]."""
    import math

    if hi - lo < 1e-9:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=mag)
    start = math.floor(lo / step) * step
    out, v = [], start
    while v <= hi + step * 0.5:
        if v >= lo - step * 0.5:
            out.append(round(v, 10))
        v += step
    return out


def calibration_scatter(rows: list[dict], you: str | None = None) -> str:
    """AUC (x) against calibration ratio (y) for every leaderboard entry.

    This is the chart that makes the site argue its own case. A point far to the
    right ranks well; a point far from the y=1 line is mispriced. The two are
    independent, and seeing an entry that is excellent on one axis and wrong on the
    other is the entire reason this leaderboard does not rank on AUC.
    """
    pts: list[Point] = []
    for r in rows:
        m = r.get("metrics") or {}
        auc, cal = m.get("auc"), m.get("calibration_ratio")
        if auc is None or cal is None:
            continue
        kind = ("baseline" if r["is_baseline"]
                else "you" if you and r["name"] == you else "entrant")
        pts.append(Point(float(auc), float(cal), r["name"], r["label"] or r["name"], kind))

    if len(pts) < 2:
        return ""

    xs = [p.x for p in pts]
    ys = [p.y for p in pts] + [1.0]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    xm, ym = max((x_hi - x_lo) * 0.18, 0.01), max((y_hi - y_lo) * 0.28, 0.02)
    x_lo, x_hi = x_lo - xm, x_hi + xm
    y_lo, y_hi = y_lo - ym, y_hi + ym

    px = lambda v: _scale(v, x_lo, x_hi, PAD_L, W - PAD_R)          # noqa: E731
    py = lambda v: _scale(v, y_lo, y_hi, H - PAD_B, PAD_T)          # noqa: E731

    o: list[str] = [
        f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" width="100%" '
        f'aria-label="Scatter of AUC against calibration ratio for every leaderboard entry">',
        '<title>AUC against calibration ratio</title>',
    ]

    # ---- grid + axes (recessive) -----------------------------------------------
    for t in _nice_ticks(y_lo, y_hi):
        y = py(t)
        o.append(f'<line class="grid" x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}"/>')
        o.append(f'<text class="tick" x="{PAD_L-10}" y="{y+4:.1f}" text-anchor="end">{t:g}</text>')
    for t in _nice_ticks(x_lo, x_hi):
        x = px(t)
        o.append(f'<line class="grid" x1="{x:.1f}" y1="{PAD_T}" x2="{x:.1f}" y2="{H-PAD_B}"/>')
        o.append(f'<text class="tick" x="{x:.1f}" y="{H-PAD_B+20}" text-anchor="middle">{t:g}</text>')

    # ---- the reference that matters: perfectly calibrated ----------------------
    if y_lo < 1.0 < y_hi:
        y1 = py(1.0)
        o.append(f'<line class="ref" x1="{PAD_L}" y1="{y1:.1f}" x2="{W-PAD_R}" y2="{y1:.1f}"/>')
        o.append(f'<text class="ref-label" x="{W-PAD_R-6}" y="{y1-8:.1f}" text-anchor="end">'
                 f'perfectly calibrated</text>')

    # ---- marks ------------------------------------------------------------------
    # 2px surface ring on every mark so overlapping points stay separable.
    for p in sorted(pts, key=lambda p: {"baseline": 0, "entrant": 1, "you": 2}[p.kind]):
        x, y = px(p.x), py(p.y)
        tip = (f"{p.name} — {p.label}\nAUC {p.x:.5f}\ncalibration ratio {p.y:.4f}"
               f"\n{'over' if p.y > 1 else 'under'}-predicting by "
               f"{abs(p.y - 1) * 100:.1f}%")
        o.append(f'<g class="mark {p.kind}"><title>{html.escape(tip)}</title>')
        if p.kind == "baseline":
            o.append(f'<rect x="{x-5:.1f}" y="{y-5:.1f}" width="10" height="10" rx="2"/>')
        else:
            o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5"/>')
        o.append("</g>")

    # ---- selective direct labels: baselines and you, never every point ---------
    for p in pts:
        if p.kind == "entrant":
            continue
        x, y = px(p.x), py(p.y)
        anchor, dx = ("end", -11) if x > (W - PAD_R + PAD_L) / 2 else ("start", 11)
        o.append(f'<text class="point-label {p.kind}" x="{x+dx:.1f}" y="{y+4:.1f}" '
                 f'text-anchor="{anchor}">{html.escape(p.label[:26])}</text>')

    # ---- axis titles -------------------------------------------------------------
    o.append(f'<text class="axis-title" x="{(PAD_L + W - PAD_R)/2:.0f}" y="{H-10}" '
             f'text-anchor="middle">AUC — ranking quality →</text>')
    o.append(f'<text class="axis-title" transform="translate(16,{(PAD_T + H - PAD_B)/2:.0f}) '
             f'rotate(-90)" text-anchor="middle">calibration ratio</text>')

    # ---- legend (always present for two series) ---------------------------------
    lx, ly = W - PAD_R + 18, PAD_T + 8
    o.append(f'<g class="legend"><circle class="sw entrant" cx="{lx}" cy="{ly}" r="5"/>'
             f'<text x="{lx+14}" y="{ly+4}">entrant</text>')
    if you:
        o.append(f'<circle class="sw you" cx="{lx}" cy="{ly+22}" r="5"/>'
                 f'<text x="{lx+14}" y="{ly+26}">you</text>')
        ly += 22
    o.append(f'<rect class="sw baseline" x="{lx-5}" y="{ly+17}" width="10" height="10" rx="2"/>'
             f'<text x="{lx+14}" y="{ly+26}">baseline</text></g>')

    o.append("</svg>")
    return "".join(o)


def score_bar_width(value: float, best: float, worst: float, higher_is_better: bool) -> float:
    """Percentage width for the inline bar behind a leaderboard score.

    Scaled across the observed range rather than from zero: every entry on a CVR
    leaderboard sits between 0.78 and 1.0 NE, and a from-zero bar would render them
    as twelve identical full-width blocks. The bar encodes *relative standing in
    this field*, which is what the column is for, and the number beside it remains
    the ground truth.
    """
    if abs(best - worst) < 1e-12:
        return 100.0
    frac = (value - worst) / (best - worst)
    if not higher_is_better:
        frac = (worst - value) / (worst - best)
    return max(4.0, min(100.0, frac * 100.0))
