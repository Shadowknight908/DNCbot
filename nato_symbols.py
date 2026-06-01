"""Simplified APP-6 / NATO tactical symbology for matplotlib overlays.

This is a *pragmatic* subset of APP-6 (MIL-STD-2525) chosen to read clearly at
map scale rather than to be standards-compliant. Four symbol kinds are drawn:

  - ``unit``      — an affiliation-framed icon (shape + standard color encode
                    friend/hostile/neutral/unknown), with an echelon marker
                    above (I / II / X / XX …), a branch glyph inside
                    (infantry ✕, armor ⬭, artillery ●, …), an optional command
                    "staff" for headquarters, and a text label below.
  - ``frontline`` — a bold polyline (forward line of own troops) in geo coords.
  - ``arrow``     — an axis-of-advance / attack / withdrawal arrow in geo coords.
  - ``objective`` — a starred map point with a label.

Design note — sizing: unit icons are drawn through ``AnnotationBbox`` +
``DrawingArea`` so they keep a *constant screen size* no matter how far the
theater map is zoomed. Frontlines, arrows and objectives are genuine geographic
features, so they are drawn in data (lon/lat) coordinates and scale with the map.

The renderer is intentionally pure: it consumes a plain ``symbol`` dict and never
reaches back into game state. Affiliation drives color/shape; the owning nation
tag is shown only as label text.
"""
from __future__ import annotations

import logging
from typing import Optional

from matplotlib.offsetbox import (
    AnnotationBbox,
    DrawingArea,
    TextArea,
    VPacker,
)
from matplotlib.patches import (
    Arc,
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)
from matplotlib.lines import Line2D

log = logging.getLogger("dnc.nato_symbols")

# ---------------------------------------------------------------------------
# Style tables
# ---------------------------------------------------------------------------

# Affiliation → frame shape + standard APP-6 colors. ``shape`` selects the
# frame geometry; ``edge`` is the frame/icon line color; ``face`` the fill.
AFFIL_STYLE: dict[str, dict] = {
    "friendly": {"edge": "#1c4fd4", "face": "#a9c4ff", "shape": "rect"},
    "hostile":  {"edge": "#bf1722", "face": "#ff9b9b", "shape": "diamond"},
    "neutral":  {"edge": "#1f7a1f", "face": "#9fe39b", "shape": "square"},
    "unknown":  {"edge": "#b8860b", "face": "#ffe08a", "shape": "quatrefoil"},
}

# Echelon → marker drawn above the frame (centered, touching the top edge).
ECHELON_MARKS: dict[str, str] = {
    "team": "Ø", "crew": "Ø",
    "squad": "•",
    "section": "••",
    "platoon": "•••", "detachment": "•••",
    "company": "I", "battery": "I", "troop": "I",
    "battalion": "II", "squadron": "II",
    "regiment": "III", "group": "III",
    "brigade": "X",
    "division": "XX",
    "corps": "XXX",
    "army": "XXXX",
    "army_group": "XXXXX", "front": "XXXXX",
    "theater": "XXXXXX", "region": "XXXXXX",
}

# Arrow kind → line style for axis-of-advance overlays.
_ARROW_STYLE: dict[str, dict] = {
    "advance":  {"linestyle": "-",  "lw": 2.6},
    "attack":   {"linestyle": "-",  "lw": 3.2},
    "withdraw": {"linestyle": (0, (5, 3)), "lw": 2.4},
    "retreat":  {"linestyle": (0, (5, 3)), "lw": 2.4},
}

_FRONTLINE_COLOR = "#ffd166"   # warm contrast when no side color is supplied
_NEUTRAL_LINE = "#c7cdd1"
_LABEL_COLOR = "#f7f9fa"
_LABEL_HALO = "#10191f"

# Icon frame geometry, in points (constant screen size).
_FRAME_W = 32.0
_FRAME_H = 22.0
_HQ_STAFF = 11.0       # length of the command "staff" below an HQ frame
_INSET = 4.0           # branch-glyph inset inside the frame

# z-order layering for the overlay (above all base map layers).
_Z_FRONTLINE = 7.0
_Z_ARROW = 8.0
_Z_OBJECTIVE = 9.0
_Z_UNIT = 12.0


def _text_outline(width: float = 2.0):
    from matplotlib import patheffects as pe
    return [pe.withStroke(linewidth=width, foreground=_LABEL_HALO)]


# ---------------------------------------------------------------------------
# Branch glyphs — each draws into the DrawingArea within interior box (gx0..gx1, gy0..gy1)
# ---------------------------------------------------------------------------

def _glyph_infantry(da, gx0, gy0, gx1, gy1, color):
    da.add_artist(Line2D([gx0, gx1], [gy0, gy1], color=color, lw=1.7,
                          solid_capstyle="round"))
    da.add_artist(Line2D([gx0, gx1], [gy1, gy0], color=color, lw=1.7,
                          solid_capstyle="round"))


def _glyph_armor(da, gx0, gy0, gx1, gy1, color):
    cx, cy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
    da.add_artist(Ellipse((cx, cy), width=(gx1 - gx0) * 0.96,
                          height=(gy1 - gy0) * 0.66, fill=False,
                          edgecolor=color, lw=1.7))


def _glyph_mechanized(da, gx0, gy0, gx1, gy1, color):
    _glyph_armor(da, gx0, gy0, gx1, gy1, color)
    # smaller infantry cross inside the track oval
    mx0 = gx0 + (gx1 - gx0) * 0.28
    mx1 = gx1 - (gx1 - gx0) * 0.28
    my0 = gy0 + (gy1 - gy0) * 0.22
    my1 = gy1 - (gy1 - gy0) * 0.22
    _glyph_infantry(da, mx0, my0, mx1, my1, color)


def _glyph_artillery(da, gx0, gy0, gx1, gy1, color):
    cx, cy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
    r = min(gx1 - gx0, gy1 - gy0) * 0.22
    da.add_artist(Circle((cx, cy), r, color=color))


def _glyph_recon(da, gx0, gy0, gx1, gy1, color):
    # single diagonal slash (cavalry / reconnaissance)
    da.add_artist(Line2D([gx0, gx1], [gy0, gy1], color=color, lw=1.8,
                          solid_capstyle="round"))


def _glyph_airborne(da, gx0, gy0, gx1, gy1, color):
    _glyph_infantry(da, gx0, gy0, gx1, gy1, color)
    cx = (gx0 + gx1) / 2
    da.add_artist(Arc((cx, gy1), width=(gx1 - gx0) * 0.85,
                      height=(gy1 - gy0) * 0.7, angle=0, theta1=0, theta2=180,
                      edgecolor=color, lw=1.5))


def _glyph_aviation(da, gx0, gy0, gx1, gy1, color):
    cx = (gx0 + gx1) / 2
    da.add_artist(Arc((cx, gy0), width=(gx1 - gx0) * 0.95,
                      height=(gy1 - gy0) * 1.3, angle=0, theta1=25, theta2=155,
                      edgecolor=color, lw=1.7))


def _glyph_text(letters: str):
    def _draw(da, gx0, gy0, gx1, gy1, color):
        from matplotlib.text import Text
        cx, cy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
        t = Text(cx, cy, letters[:3].upper(), color=color, ha="center",
                 va="center", fontsize=8, fontweight="bold")
        da.add_artist(t)
    return _draw


_BRANCH_DRAWERS = {
    "infantry": _glyph_infantry,
    "armor": _glyph_armor, "armour": _glyph_armor, "tank": _glyph_armor,
    "mechanized": _glyph_mechanized, "mechanised": _glyph_mechanized,
    "motorized": _glyph_mechanized,
    "artillery": _glyph_artillery, "fires": _glyph_artillery,
    "recon": _glyph_recon, "reconnaissance": _glyph_recon, "cavalry": _glyph_recon,
    "airborne": _glyph_airborne, "paratrooper": _glyph_airborne,
    "aviation": _glyph_aviation, "air": _glyph_aviation, "rotary": _glyph_aviation,
    "engineer": _glyph_text("E"),
    "naval": _glyph_text("NV"), "navy": _glyph_text("NV"), "marine": _glyph_text("MR"),
    "supply": _glyph_text("S"), "logistics": _glyph_text("S"), "sustainment": _glyph_text("S"),
    "signal": _glyph_text("SI"), "medical": _glyph_text("MD"),
    "air_defense": _glyph_text("AD"), "air_defence": _glyph_text("AD"),
    "headquarters": None, "hq": None, "command": None,
}


def _norm_branch(branch: Optional[str]) -> str:
    return (branch or "infantry").strip().lower().replace("-", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
# Frame geometry
# ---------------------------------------------------------------------------

def _build_frame(da, shape: str, fb: float, style: dict):
    """Add the affiliation frame to the DrawingArea; return interior glyph box.

    ``fb`` is the frame's bottom y in the DrawingArea (above any HQ staff).
    Returns (gx0, gy0, gx1, gy1) — the box a branch glyph should draw inside.
    """
    edge, face = style["edge"], style["face"]
    top = fb + _FRAME_H
    if shape == "rect":  # friendly — wide rectangle
        da.add_artist(Rectangle((0, fb), _FRAME_W, _FRAME_H, facecolor=face,
                                edgecolor=edge, lw=1.8))
        return (_INSET, fb + _INSET, _FRAME_W - _INSET, top - _INSET)
    if shape == "square":  # neutral — upright square
        side = _FRAME_H
        sx0 = (_FRAME_W - side) / 2
        da.add_artist(Rectangle((sx0, fb), side, side, facecolor=face,
                                edgecolor=edge, lw=1.8))
        return (sx0 + _INSET, fb + _INSET, sx0 + side - _INSET, top - _INSET)
    if shape == "diamond":  # hostile — square rotated 45°
        cx, cy = _FRAME_W / 2, fb + _FRAME_H / 2
        hw, hh = _FRAME_W / 2, _FRAME_H / 2
        da.add_artist(Polygon([(cx, fb), (cx + hw, cy), (cx, top), (cx - hw, cy)],
                              closed=True, facecolor=face, edgecolor=edge, lw=1.8))
        # inscribe a centered box (~55%) for the glyph
        return (cx - hw * 0.5, cy - hh * 0.5, cx + hw * 0.5, cy + hh * 0.5)
    # quatrefoil (unknown) — approximated by a rounded square
    side = _FRAME_H
    sx0 = (_FRAME_W - side) / 2
    da.add_artist(FancyBboxPatch((sx0, fb), side, side,
                                 boxstyle="round,pad=0,rounding_size=5",
                                 facecolor=face, edgecolor=edge, lw=1.8))
    return (sx0 + _INSET, fb + _INSET, sx0 + side - _INSET, top - _INSET)


# ---------------------------------------------------------------------------
# Public draw functions
# ---------------------------------------------------------------------------

def draw_unit(ax, symbol: dict) -> None:
    """Place one affiliation-framed unit icon at the symbol's (lon, lat)."""
    lon, lat = symbol.get("lon"), symbol.get("lat")
    if lon is None or lat is None:
        return
    affil = (symbol.get("affiliation") or "unknown").strip().lower()
    style = AFFIL_STYLE.get(affil, AFFIL_STYLE["unknown"])
    branch = _norm_branch(symbol.get("branch"))
    is_hq = bool(symbol.get("hq")) or branch in ("headquarters", "hq", "command")

    staff = _HQ_STAFF if is_hq else 0.0
    da = DrawingArea(_FRAME_W, _FRAME_H + staff, 0, 0, clip=False)

    fb = staff  # frame sits above the staff
    glyph_box = _build_frame(da, style["shape"], fb, style)

    if is_hq:
        # command "staff" dropping from the frame's lower-left corner
        da.add_artist(Line2D([1.2, 1.2], [0, fb], color=style["edge"], lw=1.8,
                             solid_capstyle="round"))

    drawer = _BRANCH_DRAWERS.get(branch, _glyph_text(branch[:3]))
    if drawer is not None:
        try:
            drawer(da, *glyph_box, style["edge"])
        except Exception:
            log.debug("branch glyph %r failed", branch, exc_info=True)

    children = []
    ech = ECHELON_MARKS.get((symbol.get("echelon") or "").strip().lower())
    if ech:
        children.append(TextArea(ech, textprops=dict(
            color=_LABEL_COLOR, fontsize=8, fontweight="bold",
            path_effects=_text_outline(1.6))))
    children.append(da)
    label = (symbol.get("label") or symbol.get("tag") or "").strip()
    if label:
        children.append(TextArea(label[:22], textprops=dict(
            color=_LABEL_COLOR, fontsize=7.5, fontweight="bold",
            path_effects=_text_outline(1.8))))

    box = VPacker(children=children, align="center", pad=0, sep=1.5) \
        if len(children) > 1 else da

    ab = AnnotationBbox(box, (lon, lat), xycoords="data", frameon=False,
                        pad=0.0, box_alignment=(0.5, 0.5), zorder=_Z_UNIT,
                        annotation_clip=True)
    ax.add_artist(ab)


def draw_frontline(ax, symbol: dict) -> None:
    """Bold polyline marking a forward line of troops, in geo coords."""
    pts = symbol.get("points") or []
    if len(pts) < 2:
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    color = symbol.get("color") or _FRONTLINE_COLOR
    ax.plot(xs, ys, color=color, lw=2.6, solid_capstyle="round",
            solid_joinstyle="round", zorder=_Z_FRONTLINE,
            path_effects=_text_outline(4.0))


def draw_arrow(ax, symbol: dict) -> None:
    """Axis-of-advance / attack / withdrawal arrow between two geo points."""
    frm, to = symbol.get("from"), symbol.get("to")
    if not frm or not to:
        return
    atype = (symbol.get("arrow_type") or "advance").strip().lower()
    sstyle = _ARROW_STYLE.get(atype, _ARROW_STYLE["advance"])
    affil = (symbol.get("affiliation") or "").strip().lower()
    color = symbol.get("color") or AFFIL_STYLE.get(affil, {}).get("edge", _NEUTRAL_LINE)
    arr = FancyArrowPatch(
        (frm[0], frm[1]), (to[0], to[1]),
        arrowstyle="-|>", mutation_scale=22,
        linewidth=sstyle["lw"], linestyle=sstyle["linestyle"],
        color=color, zorder=_Z_ARROW, shrinkA=0, shrinkB=2,
        path_effects=_text_outline(3.0),
    )
    ax.add_patch(arr)


def draw_objective(ax, symbol: dict) -> None:
    """Starred objective point with a label, in geo coords."""
    lon, lat = symbol.get("lon"), symbol.get("lat")
    if lon is None or lat is None:
        return
    ax.scatter([lon], [lat], marker="*", s=240, color="#ffd166",
               edgecolors=_LABEL_HALO, linewidths=0.8, zorder=_Z_OBJECTIVE)
    label = (symbol.get("label") or "").strip()
    if label:
        ax.annotate(f"OBJ {label[:18]}", xy=(lon, lat), xytext=(0, 9),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=_LABEL_COLOR,
                    zorder=_Z_OBJECTIVE, path_effects=_text_outline(2.0))


_DRAW_DISPATCH = {
    "unit": draw_unit,
    "frontline": draw_frontline,
    "arrow": draw_arrow,
    "objective": draw_objective,
}


def overlay_symbols(ax, symbols: list[dict]) -> int:
    """Draw every symbol onto ``ax``. One bad symbol never kills the render.

    Returns the count of symbols successfully drawn.
    """
    drawn = 0
    for sym in symbols or []:
        kind = (sym.get("kind") or "unit").strip().lower()
        fn = _DRAW_DISPATCH.get(kind)
        if fn is None:
            continue
        try:
            fn(ax, sym)
            drawn += 1
        except Exception:
            log.warning("failed to draw symbol %r", sym.get("id"), exc_info=True)
    return drawn
