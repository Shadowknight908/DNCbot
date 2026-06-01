"""Tool-driven LLM frontend for the war tactical map (NATO symbology).

Unlike ``map_llm`` (which accumulates *proposals* an admin then confirms), the
war map updates the battlefield directly: each tool call mutates a working set
of symbols seeded from the war's current map state. The caller persists the
final set after the session. This matches the fast war-adjudication loop, where
the map is re-editable move-by-move and a reroll/revise rolls it back via a
snapshot.

Symbol kinds (consumed by ``nato_symbols``):
    unit       — affiliation-framed icon at (lon, lat)
    frontline  — bold polyline (forward line of own troops)
    arrow      — axis of advance / attack / withdrawal
    objective  — starred map point

Realism (``check_realism``) is advisory by design: it reports violations
(over-extension, teleporting, fighting deep in enemy/neutral territory, bad
coordinates) but never blocks a placement. The system prompt requires the model
to acknowledge and, where sensible, correct them — but the GM's narrative wins.

Geometry tools (``lookup_province`` / ``provinces_in_radius`` /
``provinces_in_country`` / ``provinces_bordering``) are reused from ``map_llm``
so the model can resolve named places to real coordinates.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

import map_geometry
import nato_symbols
from map_llm import GEOMETRY_TOOLS

log = logging.getLogger("dnc.war_map_llm")

# Geometry tools the war-map model may call (read-only place resolution). These
# need no province_context, so they dispatch straight to map_geometry.
_GEOMETRY_NAMES = {
    "lookup_province",
    "provinces_in_radius",
    "provinces_in_country",
    "provinces_bordering",
}
_GEOMETRY_TOOLS = [t for t in GEOMETRY_TOOLS if t["function"]["name"] in _GEOMETRY_NAMES]

# Enums kept in sync with nato_symbols.
_AFFILIATIONS = ["friendly", "hostile", "neutral", "unknown"]
_ECHELONS = [
    "team", "squad", "section", "platoon", "company", "battalion",
    "regiment", "brigade", "division", "corps", "army", "army_group", "front",
]
_BRANCHES = [
    "infantry", "armor", "mechanized", "motorized", "artillery", "air_defense",
    "recon", "cavalry", "airborne", "aviation", "engineer", "naval", "marine",
    "supply", "signal", "medical", "headquarters",
]
_ARROW_TYPES = ["advance", "attack", "withdraw"]


# ── War-map tool schemas ─────────────────────────────────────────────────────

_LONLAT = {
    "lon": {"type": "number", "description": "Longitude (decimal degrees, -180..180)."},
    "lat": {"type": "number", "description": "Latitude (decimal degrees, -90..90)."},
}

WAR_MAP_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_map_state",
            "description": (
                "Return the current battlefield symbols (units, frontlines, "
                "arrows, objectives) with their ids and positions. Call this "
                "FIRST so you update/move existing units instead of duplicating "
                "them."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_unit",
            "description": (
                "Create a new unit, or update an existing one if you pass its "
                "id. Position with lon/lat — resolve named places via "
                "lookup_province / provinces_in_radius first. Set affiliation "
                "from the unit's allegiance relative to the friendly side."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Existing unit id to update; omit to create."},
                    "tag": {"type": "string", "description": "Owning nation game tag (e.g. 'USA', 'PRK')."},
                    "affiliation": {"type": "string", "enum": _AFFILIATIONS},
                    "echelon": {"type": "string", "enum": _ECHELONS},
                    "branch": {"type": "string", "enum": _BRANCHES},
                    "label": {"type": "string", "description": "Short unit name shown under the icon (e.g. '3rd Armored')."},
                    "hq": {"type": "boolean", "description": "True for a headquarters/command post (adds a command staff)."},
                    **_LONLAT,
                },
                "required": ["affiliation", "lon", "lat"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_unit",
            "description": (
                "Move an existing unit to a new position. Optionally draw an "
                "axis-of-advance arrow from its old position to the new one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Unit id to move (from get_map_state)."},
                    "draw_arrow": {"type": "boolean", "description": "Also add an advance/withdraw arrow along the move."},
                    "arrow_type": {"type": "string", "enum": _ARROW_TYPES},
                    **_LONLAT,
                },
                "required": ["id", "lon", "lat"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draw_frontline",
            "description": (
                "Create or update a frontline (forward line of troops) as an "
                "ordered polyline of [lon, lat] points. Pass an id to update an "
                "existing frontline."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "points": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Ordered list of [lon, lat] pairs (>= 2).",
                    },
                    "label": {"type": "string"},
                },
                "required": ["points"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draw_arrow",
            "description": "Create or update a standalone axis-of-advance / attack / withdrawal arrow.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "affiliation": {"type": "string", "enum": _AFFILIATIONS},
                    "arrow_type": {"type": "string", "enum": _ARROW_TYPES},
                    "from": {"type": "array", "items": {"type": "number"}, "description": "[lon, lat] start."},
                    "to": {"type": "array", "items": {"type": "number"}, "description": "[lon, lat] end."},
                },
                "required": ["from", "to"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_objective",
            "description": "Create or update a starred objective point.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    **_LONLAT,
                },
                "required": ["label", "lon", "lat"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_symbol",
            "description": "Delete a symbol (unit, frontline, arrow, objective) by its id.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_focus",
            "description": (
                "Frame the theater map. For most wars use mode 'auto' (the map "
                "auto-fits to the units you place). For a DISTANT or expeditionary "
                "war — e.g. a power invading an island or country far from its "
                "homeland — use mode 'place' with the theater's name (e.g. 'Cuba') "
                "so the map zooms to the fighting, not the empty ocean between the "
                "belligerents. Use 'tags' to frame on whole countries by game tag."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["auto", "place", "tags"]},
                    "place": {"type": "string", "description": "Place/country/region name for mode 'place' (resolved to coordinates)."},
                    "radius_km": {"type": "number", "description": "Half-extent for mode 'place'. Default 500."},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Game tags for mode 'tags'."},
                },
                "required": ["mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_realism",
            "description": (
                "Validate the current battlefield against geography and the "
                "previous map state. Returns violations: units that advanced "
                "implausibly far this move, units fighting deep in enemy or "
                "non-belligerent territory, or bad coordinates. Call this AFTER "
                "your placements. Acknowledge and correct what you reasonably "
                "can — but you are not blocked from leaving a flagged placement."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


_SYSTEM_PROMPT = """\
You are the operations cartographer for a 1970s Cold War wargame. You maintain a \
tactical map of one war using NATO/APP-6 military symbology. Given the latest \
move and its adjudicated outcome (or a direct GM map instruction), you update \
the battlefield: place and move units, draw frontlines and axes of advance, and \
mark objectives.

Symbology you control:
  * units — each has an AFFILIATION relative to the friendly side: 'friendly' \
(blue rectangle), 'hostile' (red diamond), 'neutral' (green square), 'unknown' \
(amber). Set echelon (team…front) and branch (infantry, armor, artillery, \
mechanized, airborne, aviation, recon, engineer, naval, headquarters, …). \
Headquarters units: set hq=true.
  * frontlines — ordered [lon, lat] polylines marking the forward line of troops.
  * arrows — axis of advance ('advance'/'attack') or 'withdraw'.
  * objectives — starred points (cities, passes, river crossings).

Theater framing: by default the map auto-fits to the units you place. If this \
war is fought far from a belligerent's homeland (an island landing, an \
expeditionary or overseas campaign — e.g. one power invading Cuba), call \
set_focus(mode='place', place='<theater>') so the map zooms to the actual \
fighting instead of the empty ocean between the belligerents.

Workflow for EVERY update:
  1. Call get_map_state to see existing symbols and their ids.
  2. Resolve every named place to real coordinates with lookup_province / \
provinces_in_radius / provinces_in_country / provinces_bordering. NEVER guess \
coordinates from memory — always resolve them with a tool.
  3. MOVE existing units (by id) rather than creating duplicates. Only \
place_unit with a new id for forces that genuinely appear this move. Remove \
units that are destroyed or fully withdrawn.
  4. Update the frontline(s) to reflect the new forward line of troops.
  5. Add arrows for the advances/attacks/withdrawals described in the move.
  6. Call check_realism, then acknowledge any violations and correct what you \
reasonably can (e.g. pull an over-extended spearhead back toward its start, or \
keep it but note it is unsupported). Do not invent forces the war log/lore \
don't support.

Hard rules:
  * Only place forces consistent with the war log, the state digest, and the \
acting nation's lore. No fictional armies.
  * Ground every position in tool-resolved coordinates.
  * Keep the map faithful to the adjudicated OUTCOME — a failed assault does \
not advance the frontline; a rout pulls it back.

When done, reply with a 1-2 sentence summary of what changed on the map and \
note any realism caveats you left standing.
"""


# ── Backend ──────────────────────────────────────────────────────────────────

class _WarMapBackend:
    """Maintains a mutable working set of symbols and runs realism checks."""

    def __init__(
        self,
        prior_symbols: list[dict],
        *,
        max_advance_km: float,
        focus: Optional[dict] = None,
        owner_of: Optional[Callable[[str, str], Optional[str]]] = None,
        belligerents: Optional[dict[str, str]] = None,
        tavily: Any = None,
    ):
        self.prior: dict[str, dict] = {
            s["id"]: dict(s) for s in prior_symbols if s.get("id")
        }
        self.symbols: list[dict] = [dict(s) for s in prior_symbols]
        self.focus: dict = dict(focus) if focus else {"mode": "auto"}
        self.max_advance_km = float(max_advance_km)
        self.owner_of = owner_of or (lambda pid, iso: None)
        self.belligerents = {k.upper(): v for k, v in (belligerents or {}).items()}
        self.tavily = tavily
        self.calls_made: list[str] = []
        self.violations: list[str] = []
        self._id_counter = 0

    # -- helpers ----------------------------------------------------------
    def _find(self, sid: str) -> Optional[dict]:
        for s in self.symbols:
            if s.get("id") == sid:
                return s
        return None

    def _new_id(self, prefix: str) -> str:
        existing = {s.get("id") for s in self.symbols}
        while True:
            self._id_counter += 1
            cand = f"{prefix}{self._id_counter}"
            if cand not in existing:
                return cand

    def _nearest_province(self, lon: float, lat: float) -> Optional[dict]:
        for radius in (150, 400, 900):
            hits = map_geometry.provinces_in_radius(lat, lon, radius, unit="km", max_results=1)
            if hits:
                return hits[0]
        return None

    def _side_of(self, sym: dict) -> Optional[str]:
        tag = (sym.get("tag") or "").upper()
        if tag and tag in self.belligerents:
            return self.belligerents[tag]
        return sym.get("side")

    # -- realism ----------------------------------------------------------
    def _run_realism(self) -> list[str]:
        violations: list[str] = []
        for s in self.symbols:
            if (s.get("kind") or "unit") != "unit":
                continue
            lon, lat = s.get("lon"), s.get("lat")
            label = s.get("label") or s.get("tag") or s.get("id") or "unit"
            if lon is None or lat is None:
                continue
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                violations.append(f"{label}: coordinates out of range ({lon}, {lat}).")
                continue
            # Over-extension since the previous battlefield.
            prev = self.prior.get(s.get("id"))
            if prev and prev.get("lon") is not None and prev.get("lat") is not None:
                d = map_geometry.haversine_km(prev["lat"], prev["lon"], lat, lon)
                if d > self.max_advance_km:
                    violations.append(
                        f"{label}: advanced ~{int(d)} km this move, beyond the "
                        f"~{int(self.max_advance_km)} km plausible per move — verify "
                        f"this is motorized/airlifted or pull it back."
                    )
            # Territory check: whose ground is the unit standing on?
            prov = self._nearest_province(lon, lat)
            if prov:
                owner = None
                try:
                    owner = self.owner_of(prov.get("province_id", ""), prov.get("iso_cc", ""))
                except Exception:
                    owner = None
                my_side = self._side_of(s)
                if owner and my_side:
                    owner_side = self.belligerents.get(owner.upper())
                    if owner_side and owner_side != my_side:
                        violations.append(
                            f"{label}: positioned in {owner}-held territory "
                            f"(near {prov.get('name')}) — confirm this is a "
                            f"breakthrough/encirclement with supply, not a teleport."
                        )
        self.violations = violations
        return violations

    # -- dispatch ---------------------------------------------------------
    async def execute(self, name: str, args: dict) -> str:
        self.calls_made.append(name)
        try:
            if name in _GEOMETRY_NAMES:
                return self._geometry(name, args)
            if name == "get_map_state":
                return json.dumps({"count": len(self.symbols), "symbols": self.symbols})
            if name == "place_unit":
                return self._place_unit(args)
            if name == "move_unit":
                return self._move_unit(args)
            if name == "draw_frontline":
                return self._upsert(args, kind="frontline", prefix="f",
                                    fields=["points", "label"])
            if name == "draw_arrow":
                return self._upsert(args, kind="arrow", prefix="a",
                                    fields=["from", "to", "affiliation", "arrow_type"])
            if name == "set_objective":
                return self._upsert(args, kind="objective", prefix="o",
                                    fields=["label", "lon", "lat"])
            if name == "remove_symbol":
                sid = (args.get("id") or "").strip()
                before = len(self.symbols)
                self.symbols = [s for s in self.symbols if s.get("id") != sid]
                ok = len(self.symbols) < before
                return json.dumps({"removed": ok, "id": sid})
            if name == "set_focus":
                return self._set_focus(args)
            if name == "check_realism":
                v = self._run_realism()
                return json.dumps({"violations": v, "ok": not v})
            if name == "web_search" and self.tavily is not None:
                return await self.tavily.execute(name, args)
            return json.dumps({"error": f"unknown tool: {name}"})
        except Exception as e:
            log.exception("war-map tool %s failed", name)
            return json.dumps({"error": str(e)})

    def _geometry(self, name: str, args: dict) -> str:
        if name == "lookup_province":
            return json.dumps({"results": map_geometry.lookup_province(
                args.get("query", ""), max_results=int(args.get("max_results", 5) or 5))})
        if name == "provinces_in_radius":
            res = map_geometry.provinces_in_radius(
                float(args["lat"]), float(args["lon"]), float(args["distance"]),
                unit=args.get("unit", "mi"), country_filter=args.get("country_filter"))
            return json.dumps({"count": len(res), "provinces": res})
        if name == "provinces_in_country":
            res = map_geometry.provinces_in_country(args["country"], region_hint=args.get("region_hint"))
            return json.dumps({"count": len(res), "provinces": res})
        if name == "provinces_bordering":
            res = map_geometry.provinces_bordering(args["province_id"])
            return json.dumps({"count": len(res), "provinces": res})
        return json.dumps({"error": f"unknown geometry tool: {name}"})

    def _place_unit(self, args: dict) -> str:
        lon, lat = args.get("lon"), args.get("lat")
        if lon is None or lat is None:
            return json.dumps({"error": "lon and lat are required"})
        sid = (args.get("id") or "").strip()
        existing = self._find(sid) if sid else None
        sym = existing if existing else {"id": self._new_id("u"), "kind": "unit"}
        for k in ("tag", "affiliation", "echelon", "branch", "label", "hq"):
            if args.get(k) is not None:
                sym[k] = args[k] if k != "tag" else str(args[k]).upper()
        sym["lon"], sym["lat"] = float(lon), float(lat)
        if existing is None:
            self.symbols.append(sym)
        return json.dumps({"id": sym["id"], "updated": existing is not None})

    def _move_unit(self, args: dict) -> str:
        sid = (args.get("id") or "").strip()
        sym = self._find(sid)
        if not sym:
            return json.dumps({"error": f"no symbol with id {sid!r}"})
        old = [sym.get("lon"), sym.get("lat")]
        sym["lon"], sym["lat"] = float(args["lon"]), float(args["lat"])
        out = {"id": sid, "moved": True}
        if args.get("draw_arrow") and old[0] is not None and old[1] is not None:
            arrow = {
                "id": self._new_id("a"), "kind": "arrow",
                "arrow_type": args.get("arrow_type", "advance"),
                "affiliation": sym.get("affiliation"),
                "from": [float(old[0]), float(old[1])],
                "to": [sym["lon"], sym["lat"]],
            }
            self.symbols.append(arrow)
            out["arrow_id"] = arrow["id"]
        return json.dumps(out)

    def _set_focus(self, args: dict) -> str:
        mode = (args.get("mode") or "auto").lower()
        if mode == "auto":
            self.focus = {"mode": "auto"}
        elif mode == "tags":
            tags = [str(t).upper() for t in (args.get("tags") or []) if t]
            if not tags:
                return json.dumps({"error": "mode 'tags' needs a non-empty tags list"})
            self.focus = {"mode": "tags", "tags": tags}
        elif mode == "place":
            place = (args.get("place") or "").strip()
            if not place:
                return json.dumps({"error": "mode 'place' needs a place name"})
            res = map_geometry.lookup_province(place, max_results=1)
            if not res:
                return json.dumps({"error": f"could not resolve place {place!r}"})
            r0 = res[0]
            if r0.get("kind") == "country" and r0.get("bbox"):
                self.focus = {"mode": "bbox", "bbox": r0["bbox"]}
            elif r0.get("centroid"):
                lon, lat = r0["centroid"]
                self.focus = {"mode": "place", "lon": lon, "lat": lat,
                              "radius_km": float(args.get("radius_km", 500) or 500)}
            else:
                return json.dumps({"error": f"could not frame {place!r}"})
        else:
            return json.dumps({"error": f"unknown focus mode {mode!r}"})
        return json.dumps({"focus": self.focus})

    def _upsert(self, args: dict, *, kind: str, prefix: str, fields: list[str]) -> str:
        sid = (args.get("id") or "").strip()
        existing = self._find(sid) if sid else None
        sym = existing if existing else {"id": self._new_id(prefix), "kind": kind}
        for k in fields:
            if args.get(k) is not None:
                sym[k] = args[k]
        if existing is None:
            self.symbols.append(sym)
        return json.dumps({"id": sym["id"], "updated": existing is not None})


# ── Public API ───────────────────────────────────────────────────────────────

async def update_war_map(
    *,
    current_symbols: list[dict],
    current_focus: Optional[dict] = None,
    context_blob: str,
    instructions: str,
    client: Any,
    model: Optional[str] = None,
    thinking_budget: Any = None,
    max_tool_depth: int = 8,
    max_advance_km: float = 250.0,
    owner_of: Optional[Callable[[str, str], Optional[str]]] = None,
    belligerents: Optional[dict[str, str]] = None,
    tavily: Any = None,
) -> tuple[list[dict], dict, list[str], list[str]]:
    """Run a war-map tool session.

    Returns ``(new_symbols, focus, violations, tool_calls)``. The new symbol set
    and focus should be persisted by the caller; violations are surfaced to the
    GM. A final realism pass is always run so violations reflect the committed
    battlefield even if the model forgot to call check_realism.
    """
    backend = _WarMapBackend(
        current_symbols, max_advance_km=max_advance_km, focus=current_focus,
        owner_of=owner_of, belligerents=belligerents, tavily=tavily,
    )
    tools = list(_GEOMETRY_TOOLS) + list(WAR_MAP_TOOLS)
    if tavily is not None:
        tools.append(tavily.tool_definition())

    user_message = f"{context_blob}\n\n---\nUpdate the tactical map for:\n{instructions}"
    kwargs: dict = {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "model": model,
        "tools": tools,
        "tool_executor": backend.execute,
        "temperature": 0.2,
    }
    if thinking_budget is not None and thinking_budget != "":
        kwargs["thinking_budget"] = thinking_budget
    try:
        await client.chat_messages(max_tool_depth=max_tool_depth, **kwargs)
    except Exception:
        log.exception("war-map LLM session failed")

    # Always reconcile realism against the final committed set.
    backend._run_realism()
    log.info("war-map tool calls: %s", backend.calls_made)
    return backend.symbols, backend.focus, backend.violations, backend.calls_made


def summarize_map(symbols: list[dict]) -> str:
    """Short human-readable inventory of the battlefield, for the map post."""
    units = [s for s in symbols if (s.get("kind") or "unit") == "unit"]
    fronts = [s for s in symbols if s.get("kind") == "frontline"]
    arrows = [s for s in symbols if s.get("kind") == "arrow"]
    objs = [s for s in symbols if s.get("kind") == "objective"]
    by_affil: dict[str, int] = {}
    for u in units:
        a = (u.get("affiliation") or "unknown").lower()
        by_affil[a] = by_affil.get(a, 0) + 1
    parts = []
    if by_affil:
        parts.append(", ".join(f"{n} {a}" for a, n in sorted(by_affil.items())) + " units")
    if fronts:
        parts.append(f"{len(fronts)} frontline(s)")
    if arrows:
        parts.append(f"{len(arrows)} axis-of-advance arrow(s)")
    if objs:
        parts.append(f"{len(objs)} objective(s)")
    return "; ".join(parts) if parts else "empty battlefield"
