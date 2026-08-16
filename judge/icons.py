"""Line-art marks for each of the twelve weeks, plus the accent hue that goes with them.

Each icon draws the *idea* of its week rather than a generic glyph — week 3 is a
reliability diagram, week 7 is two diverging arms with the gap between them marked,
week 9 is a damped oscillation settling onto a target. Someone who knows the material
should be able to name the week from the mark alone.

Rules they all follow, so twelve marks read as one set:
  · 24x24 box, stroke-only, no fills except deliberate solid dots
  · stroke-width 1.5, round caps and joins
  · currentColor throughout, so the mark inherits whatever colour the context sets
  · content stays inside a 3..21 box, leaving optical padding at every edge
"""
from __future__ import annotations

# Per-week accent. Mid-lightness, mid-chroma so the same hex reads on both the light
# paper and the dark surface; used only for tints, rules and the mark itself — never to
# encode data, which is what the validated dataviz palette is for.
WEEK_ACCENT: dict[int, str] = {
    1:  "#3b7fc4",   # blue        — foundations
    2:  "#6a5acd",   # violet      — deep nets
    3:  "#2f9e8f",   # teal        — calibration
    4:  "#c98a2e",   # amber       — delayed feedback
    5:  "#c2607a",   # rose        — missing labels
    6:  "#5b8f3a",   # olive       — attribution
    7:  "#8a5fc7",   # purple      — uplift
    8:  "#cc6b45",   # terracotta  — auctions
    9:  "#2d8fb3",   # cyan        — pacing
    10: "#7a7fd0",   # periwinkle  — privacy
    11: "#b5763a",   # bronze      — sequences
    12: "#4a8f7b",   # jade        — capstone
}

# fmt: off
PATHS: dict[int, str] = {
    # 1 — Foundations: axes and a fitted curve rising off a flat baseline.
    1: """
      <path d="M4 4v16h16"/>
      <path d="M4 16.5c4 0 5.5-1.5 8-5.5S18 5.5 20 5.5" />
      <circle cx="20" cy="5.5" r="1.6"/>
      <path d="M4 18.5h16" stroke-dasharray="1.5 2.5" opacity=".55"/>
    """,
    # 2 — Deep CVR: a small MLP, two inputs to three hidden units to one output.
    2: """
      <circle cx="4.5" cy="8"  r="1.5"/><circle cx="4.5" cy="16" r="1.5"/>
      <circle cx="12" cy="5"  r="1.5"/><circle cx="12" cy="12" r="1.5"/><circle cx="12" cy="19" r="1.5"/>
      <circle cx="19.5" cy="12" r="1.5"/>
      <path d="M6 8l4.4-2.6M6 8l4.6 3.4M6 8l4.5 10.4M6 16l4.4-10.4M6 16l4.6-3.4M6 16l4.5 2.6"
            opacity=".5"/>
      <path d="M13.5 5.4l4.6 5.5M13.5 12h4.4M13.5 18.6l4.6-5.5" opacity=".5"/>
    """,
    # 3 — Calibration: the perfect diagonal, with observations sagging below it.
    3: """
      <path d="M4 4v16h16" opacity=".4"/>
      <path d="M5.5 18.5L18.5 5.5" stroke-dasharray="2 2.5"/>
      <circle cx="9" cy="17" r="1.35"/><circle cx="12.5" cy="14.5" r="1.35"/>
      <circle cx="16" cy="10.5" r="1.35"/>
    """,
    # 4 — Delayed feedback: the impression, a long lag, the conversion arriving late.
    4: """
      <path d="M4.5 5v14"/>
      <circle cx="4.5" cy="12" r="1.5"/>
      <path d="M7 12h9.5" stroke-dasharray="2 2.5"/>
      <circle cx="19" cy="12" r="2.2"/>
      <path d="M19 10.4V12l1.1.9" stroke-width="1.2"/>
    """,
    # 5 — Missing labels: a lattice where some outcomes were never observed.
    5: """
      <circle cx="6" cy="6" r="1.7"/><circle cx="12" cy="6" r="1.7" stroke-dasharray="1.6 1.6" opacity=".55"/>
      <circle cx="18" cy="6" r="1.7"/>
      <circle cx="6" cy="12" r="1.7" stroke-dasharray="1.6 1.6" opacity=".55"/>
      <circle cx="12" cy="12" r="1.7"/>
      <circle cx="18" cy="12" r="1.7" stroke-dasharray="1.6 1.6" opacity=".55"/>
      <circle cx="6" cy="18" r="1.7"/><circle cx="12" cy="18" r="1.7"/>
      <circle cx="18" cy="18" r="1.7" stroke-dasharray="1.6 1.6" opacity=".55"/>
    """,
    # 6 — Attribution: several touchpoints, one conversion, credit to divide.
    6: """
      <circle cx="4.5" cy="5.5" r="1.4"/><circle cx="4.5" cy="12" r="1.4"/><circle cx="4.5" cy="18.5" r="1.4"/>
      <path d="M6.2 6.2c5 1.4 7.4 3.2 9.4 5.4M6 12h9.4M6.2 17.8c5-1.4 7.4-3.2 9.4-5.4"/>
      <circle cx="18" cy="12" r="2.6"/>
    """,
    # 7 — Uplift: treated and control arms diverging; the gap between them is the effect.
    7: """
      <path d="M3.5 16c3.6 0 6.2-1.7 8.4-4.3S15.9 7.4 17 6.6"/>
      <path d="M3.5 16c3.6 0 6.2.6 8.4 1.1s3.6.8 5.1.9" opacity=".6" stroke-dasharray="2 2"/>
      <path d="M20.2 8v9.6" stroke-width="1.2"/>
      <path d="M19 9.1l1.2-1.3 1.2 1.3M19 16.5l1.2 1.3 1.2-1.3" stroke-width="1.2"/>
    """,
    # 8 — Auctions: sealed bids, one of them clears.
    8: """
      <path d="M4 20h16"/>
      <path d="M6.5 20v-5M11 20v-9M15.5 20v-6.5M20 20v-3"/>
      <path d="M11 8.4L9.4 6.4h3.2z"/>
    """,
    # 9 — Pacing: a controller overshooting, then settling onto its target.
    9: """
      <path d="M3.5 12h17" stroke-dasharray="2 2.5" opacity=".55"/>
      <path d="M3.5 12Q4.8 4 6.9 12T10.6 12Q12.3 7.2 13.9 12T17 12Q18.2 10 19.6 12"/>
      <circle cx="20.4" cy="12" r="1.4"/>
    """,
    # 10 — Privacy: an aggregate released only behind a boundary, with noise added.
    10: """
      <path d="M12 3.5l7 2.6v5.4c0 4.4-3 7.4-7 9-4-1.6-7-4.6-7-9V6.1z"/>
      <circle cx="9.4" cy="11" r="1.1"/><circle cx="14.6" cy="10" r="1.1"/>
      <circle cx="12" cy="14.4" r="1.1"/>
    """,
    # 11 — Sequences: a journey of events, each weighted by how much it bears on the
    #      candidate at the end — the bars above are attention, growing toward it.
    11: """
      <circle cx="4.5" cy="17.5" r="1.4"/><circle cx="10" cy="17.5" r="1.4"/>
      <circle cx="15.5" cy="17.5" r="1.4"/><circle cx="20.5" cy="17.5" r="2.2"/>
      <path d="M5.9 17.5h2.7M11.4 17.5h2.7M17 17.5h1.3"/>
      <path d="M4.5 14.3v-2.6M10 14.3V9.2M15.5 14.3V5.8" opacity=".5"/>
    """,
    # 12 — Capstone: the stack, assembled.
    12: """
      <path d="M12 3.5l8 4-8 4-8-4z"/>
      <path d="M4 12l8 4 8-4" opacity=".7"/>
      <path d="M4 16.5l8 4 8-4" opacity=".45"/>
    """,
}
# fmt: on


def week_icon(week: int, size: int = 24, cls: str = "wk-icon") -> str:
    """Return the week's mark as an inline SVG string, or an empty string if unknown."""
    body = PATHS.get(week)
    if not body:
        return ""
    return (
        f'<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true" focusable="false">'
        f"{body}</svg>"
    )


def accent(week: int) -> str:
    return WEEK_ACCENT.get(week, "#3b7fc4")
