"""LLM frontend for province ownership changes (tool-driven, v2).

Two entry points, both share the same geometry tool set:

    parse_post_for_changes(post_text, ...)
        Reads a roleplay post and proposes territorial changes implied by
        the narrative.

    process_map_instructions(instructions, ...)
        Executes direct GM commands like "Russia invades northern China
        450 miles around Xinjiang".

Both return a list of proposal dicts the caller can present for admin
confirmation before they take effect.

Geometry tools (live in map_geometry):
    - lookup_province           name/alias → province IDs (or country meta)
    - provinces_in_radius       centroid disc query
    - provinces_in_country      country + optional region filter
    - provinces_bordering       shared-boundary adjacency

Action tool:
    - apply_changes             submit a batch of assign/release/transfer ops

Optional tool when configured:
    - web_search (Tavily)       general web search
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

import map_geometry

log = logging.getLogger("dnc.map_llm")


# ── Tool schemas ─────────────────────────────────────────────────────────────

GEOMETRY_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "provinces_by_owner",
            "description": (
                "Return all provinces currently owned by a given game tag, along "
                "with each province's ISO-2 country code. Use this when you need "
                "to enumerate what a tag actually holds in the game — for example, "
                "before releasing non-core territory or dissolving an empire. "
                "The iso_cc field lets you filter by real-world country boundaries "
                "(e.g. release everything owned by USSR where iso_cc != 'RU')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "player_tag": {
                        "type": "string",
                        "description": "Game tag to query (e.g. 'USSR', 'CHN').",
                    },
                },
                "required": ["player_tag"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_province",
            "description": (
                "Resolve a free-text place name to one or more province records. "
                "Use this whenever the instruction names a region, city, state, "
                "or historical name and you need the actual province ID and "
                "coordinates. May return a 'country' meta record (with bbox + "
                "game_tags) when the name refers to a whole country — follow up "
                "with provinces_in_country if you need its provinces."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Name to resolve. Examples: 'Xinjiang', 'Bavaria', "
                            "'Inner Mongolia', 'Persia', 'the Caucasus'."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Cap on candidates returned. Default 5.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "provinces_in_radius",
            "description": (
                "Return every province whose centroid lies within `distance` of "
                "(lat, lon). Use this for any 'within N miles/km of X' instruction. "
                "Distance is measured along the great circle. Pass country_filter "
                "to restrict to specific countries (ISO2 codes OR game tags)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number", "description": "Latitude of centre point."},
                    "lon": {"type": "number", "description": "Longitude of centre point."},
                    "distance": {"type": "number", "description": "Radius value."},
                    "unit": {
                        "type": "string",
                        "enum": ["mi", "km"],
                        "description": "Unit (mi or km). Default mi.",
                    },
                    "country_filter": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional list of ISO2 codes or game tags to restrict "
                            "the search to, e.g. ['CN', 'MN'] or ['CHN', 'MPR']."
                        ),
                    },
                },
                "required": ["lat", "lon", "distance"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "provinces_in_country",
            "description": (
                "List provinces in a country, optionally filtered to a "
                "directional region computed from each province's centroid "
                "relative to the country bbox."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {
                        "type": "string",
                        "description": "Game tag or ISO2 code (e.g. 'CHN' or 'CN').",
                    },
                    "region_hint": {
                        "type": "string",
                        "description": (
                            "Optional spatial filter. Accepted: north, south, "
                            "east, west, northern, southern, eastern, western, "
                            "northeast, northwest, southeast, southwest, central."
                        ),
                    },
                },
                "required": ["country"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "provinces_bordering",
            "description": (
                "List provinces sharing a land/coastal boundary with a given "
                "province. Use when the instruction describes adjacency, "
                "spreading, or a frontier."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "province_id": {"type": "string"},
                },
                "required": ["province_id"],
            },
        },
    },
]

APPLY_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "apply_changes",
        "description": (
            "Submit a batch of ownership changes for admin confirmation. "
            "Call this ONCE after you've resolved every province ID. "
            "Use the action names exactly: 'assign', 'release', 'transfer'. "
            "player_tag / from_tag / to_tag must be GAME TAGS (e.g. 'USSR', "
            "'CHN', 'FRG'), not ISO2 codes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "operations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["assign", "release", "transfer"],
                            },
                            "province_id": {"type": "string"},
                            "player_tag": {
                                "type": "string",
                                "description": (
                                    "Required for assign. Game tag of the new owner."
                                ),
                            },
                            "from_tag": {
                                "type": "string",
                                "description": "Required for transfer.",
                            },
                            "to_tag": {
                                "type": "string",
                                "description": "Required for transfer.",
                            },
                            "reason": {
                                "type": "string",
                                "description": "Short rationale drawn from the instruction.",
                            },
                        },
                        "required": ["action", "province_id"],
                    },
                }
            },
            "required": ["operations"],
        },
    },
}


# ── System prompts ───────────────────────────────────────────────────────────

_INSTRUCTION_SYSTEM_PROMPT = """\
You are the map administrator for a 1970s Cold War roleplay game. You receive \
plain-English commands from a game master describing territorial changes and \
your job is to translate them into precise province-level operations.

You have geometry tools — USE THEM. Never guess province IDs from training data.

Workflow for every command:
  1. Identify the actors: who is taking territory, who is losing it, and the \
target region (a country, a directional slice of a country, a radius around a \
place, an adjacency front, etc.).
  2. If the command involves releasing, collapsing, or reducing a tag's \
holdings, call `provinces_by_owner` first to get every province that tag \
currently holds, including each province's iso_cc (geographic country code). \
This is the only reliable way to enumerate non-standard assignments.
  3. Call `lookup_province` to resolve any named places to province IDs and \
coordinates. If you get a 'country' meta record back, follow up with \
`provinces_in_country`.
  4. Call `provinces_in_radius`, `provinces_in_country`, and/or \
`provinces_bordering` to enumerate every affected province. Combine results \
across calls when needed.
  5. Call `apply_changes` ONCE with the full batch of operations.

Rules:
  - Only use province IDs returned by tools. Never invent IDs.
  - player_tag, from_tag, to_tag must be GAME TAGS (e.g. 'USSR', 'CHN', 'FRG'), \
not ISO2 codes. The country tag directory in the user message lists valid tags.
  - For divided countries (Germany FRG/GDR, Korea KOR/PRK, Vietnam DRV/RVN, \
Yemen YAR/PDY), pick the tag that matches the province's actual location.
  - If the instruction mentions a radius in miles or km, USE provinces_in_radius. \
Do not eyeball geography.
  - If the radius excludes part of an intended region, expand by 10-20% or also \
call provinces_in_country with a region_hint to fill in.
  - If something is ambiguous, do your best interpretation but explain in the \
reason field of each operation.

Example — territory collapse:
  Command: "The Soviet Union falls; keep only Russian Federation territory."
  Step 1: provinces_by_owner(player_tag="USSR") → list with iso_cc per province.
  Step 2: For every province where iso_cc != "RU", add a release operation.
  Step 3: apply_changes(operations=[{action:"release", province_id:"UA-01", \
reason:"Soviet collapse — non-Russian territory released"}, ...])

Example — invasion:
  Command: "Northern China is invaded by Russia 450 miles in every direction \
from Xinjiang."
  Step 1: lookup_province(query="Xinjiang") → CN-65 centred at ~(85.3, 41.1).
  Step 2: provinces_in_radius(lat=41.1, lon=85.3, distance=450, unit="mi", \
country_filter=["CN"]) → list of affected Chinese provinces.
  Step 3: apply_changes(operations=[{action:"assign", province_id:"CN-65", \
player_tag:"USSR", reason:"Russian incursion within 450mi of Xinjiang"}, ...])
"""

_POST_PARSE_SYSTEM_PROMPT = """\
You are a territorial analyst for a 1970s Cold War roleplay game. You read a \
roleplay post and identify any territorial changes it describes — captures, \
occupations, cessions, liberations, or transfers.

You have geometry tools — USE THEM to resolve named places and to enumerate \
affected regions. Do not guess province IDs from training data.

Workflow:
  1. Read the post. Identify each territorial change: who took what, from whom, \
and where.
  2. If the change involves releasing or reducing a tag's territory, call \
`provinces_by_owner` to get every province that tag currently holds (with \
iso_cc), then filter as appropriate.
  3. Resolve named places with `lookup_province`. Use `provinces_in_radius`, \
`provinces_in_country`, or `provinces_bordering` to enumerate the affected \
provinces.
  4. Call `apply_changes` ONCE with all proposed operations.

Rules:
  - If the post does NOT describe any territorial changes, do not call \
apply_changes — reply with a brief plain-text explanation instead.
  - Only use province IDs returned by tools. Never invent IDs.
  - player_tag / from_tag / to_tag are GAME TAGS, not ISO2.
  - Be conservative. If a post says "we plan to invade" or "we threaten", \
do NOT propose changes — only confirmed/executed actions.
"""


# ── Executor ─────────────────────────────────────────────────────────────────

class _ToolBackend:
    """Holds proposal accumulator + dispatch logic across geometry tools."""

    def __init__(self, tavily: Any = None, province_context: dict | None = None):
        self.proposals: list[dict] = []
        self.tavily = tavily
        self.calls_made: list[str] = []
        self.province_context: dict = province_context or {}

    def _record_operation(self, op: dict) -> dict:
        """Normalise an LLM-emitted operation to the internal proposal shape."""
        action_raw = (op.get("action") or "").strip().lower()
        action_map = {
            "assign": "assign_province",
            "assign_province": "assign_province",
            "release": "release_province",
            "release_province": "release_province",
            "transfer": "transfer_province",
            "transfer_province": "transfer_province",
        }
        action = action_map.get(action_raw)
        if not action:
            return {"status": "rejected", "reason": f"unknown action: {action_raw!r}"}
        pid = (op.get("province_id") or "").strip().upper()
        if not pid:
            return {"status": "rejected", "reason": "missing province_id"}
        proposal: dict = {"action": action, "province_id": pid}
        if action == "assign_province":
            tag = (op.get("player_tag") or op.get("to_tag") or "").strip().upper()
            if not tag:
                return {"status": "rejected", "reason": "assign needs player_tag"}
            proposal["player_tag"] = tag
        elif action == "transfer_province":
            from_tag = (op.get("from_tag") or "").strip().upper()
            to_tag = (op.get("to_tag") or op.get("player_tag") or "").strip().upper()
            if not from_tag or not to_tag:
                return {"status": "rejected", "reason": "transfer needs from_tag and to_tag"}
            proposal["from_tag"] = from_tag
            proposal["to_tag"] = to_tag
        if op.get("reason"):
            proposal["reason"] = str(op["reason"]).strip()
        self.proposals.append(proposal)
        return {"status": "recorded", "province_id": pid}

    async def execute(self, name: str, args: dict) -> str:
        self.calls_made.append(name)
        try:
            if name == "provinces_by_owner":
                tag = (args.get("player_tag") or "").strip().upper()
                if not tag:
                    return json.dumps({"error": "player_tag is required"})
                import arcgis_provinces as _ap
                all_provinces = _ap.get_all()
                result = []
                for pid, info in self.province_context.items():
                    if (info.get("owner") or "").upper() != tag:
                        continue
                    prov = all_provinces.get(pid)
                    result.append({
                        "province_id": pid,
                        "display_name": info.get("display_name", pid),
                        "iso_cc": prov.iso_cc if prov else None,
                    })
                return json.dumps({"count": len(result), "provinces": result})

            if name == "lookup_province":
                res = map_geometry.lookup_province(
                    args.get("query", ""),
                    max_results=int(args.get("max_results", 5) or 5),
                )
                return json.dumps({"results": res})

            if name == "provinces_in_radius":
                res = map_geometry.provinces_in_radius(
                    float(args["lat"]),
                    float(args["lon"]),
                    float(args["distance"]),
                    unit=args.get("unit", "mi"),
                    country_filter=args.get("country_filter"),
                )
                return json.dumps({"count": len(res), "provinces": res})

            if name == "provinces_in_country":
                res = map_geometry.provinces_in_country(
                    args["country"],
                    region_hint=args.get("region_hint"),
                )
                return json.dumps({"count": len(res), "provinces": res})

            if name == "provinces_bordering":
                res = map_geometry.provinces_bordering(args["province_id"])
                return json.dumps({"count": len(res), "provinces": res})

            if name == "apply_changes":
                ops = args.get("operations") or []
                if not isinstance(ops, list):
                    return json.dumps({"error": "operations must be a list"})
                statuses = [self._record_operation(o) for o in ops]
                recorded = sum(1 for s in statuses if s.get("status") == "recorded")
                return json.dumps({
                    "recorded": recorded,
                    "rejected": len(ops) - recorded,
                    "statuses": statuses,
                })

            if name == "web_search" and self.tavily is not None:
                return await self.tavily.execute(name, args)

            return json.dumps({"error": f"unknown tool: {name}"})
        except Exception as e:
            log.exception("Tool %s failed", name)
            return json.dumps({"error": str(e)})


def _build_tag_directory_blob() -> str:
    """Compact reference block listing valid game tags grouped by ISO2."""
    directory = map_geometry.country_tag_directory()
    lines = ["Country tag directory (ISO2 → valid game tags):"]
    for iso2 in sorted(directory):
        tags = directory[iso2]
        lines.append(f"  {iso2}: {', '.join(tags)}")
    return "\n".join(lines)


def _ownership_summary(province_context: dict) -> str:
    """Optional short summary of currently-owned provinces for the LLM."""
    if not province_context:
        return "(No provinces are currently owned.)"
    by_tag: dict[str, list[str]] = {}
    for pid, info in province_context.items():
        owner = info.get("owner")
        if not owner:
            continue
        by_tag.setdefault(str(owner), []).append(pid)
    if not by_tag:
        return "(No provinces are currently owned.)"
    lines = ["Currently owned provinces (by owner):"]
    for owner, pids in sorted(by_tag.items()):
        if len(pids) <= 8:
            lines.append(f"  {owner}: {', '.join(sorted(pids))}")
        else:
            shown = ", ".join(sorted(pids)[:8])
            lines.append(f"  {owner}: {shown}, +{len(pids) - 8} more")
    return "\n".join(lines)


# ── Public API ───────────────────────────────────────────────────────────────

async def _run_tool_session(
    system_prompt: str,
    user_message: str,
    client: Any,
    model: Optional[str],
    tavily: Any,
    thinking_budget: Any,
    max_tool_depth: int,
    province_context: dict | None = None,
) -> _ToolBackend:
    backend = _ToolBackend(tavily=tavily, province_context=province_context)
    tools = list(GEOMETRY_TOOLS) + [APPLY_TOOL]
    if tavily is not None:
        tools.append(tavily.tool_definition())

    kwargs: dict = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "model": model,
        "tools": tools,
        "tool_executor": backend.execute,
        "temperature": 0.0,
    }
    if thinking_budget is not None and thinking_budget != "":
        kwargs["thinking_budget"] = thinking_budget
    # Plumb tool-call recursion ceiling when the client supports it.
    try:
        await client.chat_messages(max_tool_depth=max_tool_depth, **kwargs)
    except TypeError:
        await client.chat_messages(**kwargs)
    return backend


async def process_map_instructions(
    instructions: str,
    province_context: dict,
    client: Any,
    model: Optional[str] = None,
    tavily: Any = None,
    thinking_budget: Any = None,
    max_tool_depth: int = 6,
) -> list[dict]:
    """LLM executes a direct GM instruction and returns proposed changes.

    Changes are NOT applied — they are returned for admin confirmation.
    """
    user_blob = (
        f"{_build_tag_directory_blob()}\n\n"
        f"{_ownership_summary(province_context)}\n\n"
        f"---\n"
        f"Map change instructions:\n{instructions}"
    )
    backend = await _run_tool_session(
        _INSTRUCTION_SYSTEM_PROMPT, user_blob, client, model, tavily,
        thinking_budget, max_tool_depth, province_context=province_context,
    )
    log.info("mapchange LLM tool calls: %s", backend.calls_made)
    return backend.proposals


async def parse_post_for_changes(
    post_text: str,
    province_context: dict,
    client: Any,
    model: Optional[str] = None,
    tavily: Any = None,
    thinking_budget: Any = None,
    max_tool_depth: int = 6,
) -> list[dict]:
    """LLM reads a roleplay post and returns proposed changes (if any)."""
    user_blob = (
        f"{_build_tag_directory_blob()}\n\n"
        f"{_ownership_summary(province_context)}\n\n"
        f"---\n"
        f"Roleplay post to analyze:\n{post_text}"
    )
    backend = await _run_tool_session(
        _POST_PARSE_SYSTEM_PROMPT, user_blob, client, model, tavily,
        thinking_budget, max_tool_depth, province_context=province_context,
    )
    log.info("mapparse LLM tool calls: %s", backend.calls_made)
    return backend.proposals


# ── Validation + summary ─────────────────────────────────────────────────────

def validate_proposals(proposals: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split proposals into (valid, invalid) with reasons attached to invalids.

    A proposal is invalid when:
      - province_id doesn't exist in the ArcGIS dataset
      - assign/transfer specifies an unknown player_tag (not in TAG_TO_ISO2 and
        not in TAG_TO_NAME)
    """
    import arcgis_provinces as _ap
    from map_renderer import TAG_TO_ISO2, TAG_TO_NAME
    known_tags = set(TAG_TO_ISO2.keys()) | set(TAG_TO_NAME.keys())
    all_provinces = _ap.get_all()
    valid: list[dict] = []
    invalid: list[dict] = []
    for p in proposals:
        pid = (p.get("province_id") or "").upper()
        if pid not in all_provinces:
            invalid.append({**p, "invalid_reason": f"unknown province_id {pid!r}"})
            continue
        action = p.get("action")
        if action == "assign_province":
            tag = (p.get("player_tag") or "").upper()
            if not tag:
                invalid.append({**p, "invalid_reason": "missing player_tag"})
                continue
            if tag not in known_tags:
                invalid.append({**p, "invalid_reason": f"unknown player_tag {tag!r}"})
                continue
        elif action == "transfer_province":
            from_tag = (p.get("from_tag") or "").upper()
            to_tag = (p.get("to_tag") or "").upper()
            if not from_tag or not to_tag:
                invalid.append({**p, "invalid_reason": "transfer needs from_tag/to_tag"})
                continue
            if to_tag not in known_tags:
                invalid.append({**p, "invalid_reason": f"unknown to_tag {to_tag!r}"})
                continue
        valid.append(p)
    return valid, invalid


def format_change_summary(
    changes: list[dict],
    province_store: Any,
    invalids: Optional[list[dict]] = None,
) -> str:
    """Markdown summary of proposed changes, grouped for readability."""
    invalids = invalids or []
    if not changes and not invalids:
        return "No territorial changes detected."

    lines: list[str] = []
    if changes:
        # Group assigns by destination tag for compactness.
        assigns: dict[str, list[tuple[str, str]]] = {}
        releases: list[tuple[str, str]] = []
        transfers: list[tuple[str, str, str, str]] = []
        for ch in changes:
            pid = ch.get("province_id", "?")
            prov = province_store.get_province(pid) if province_store else None
            name = prov.name if prov else pid
            reason = ch.get("reason", "")
            if ch.get("action") == "assign_province":
                tag = ch.get("player_tag", "?")
                assigns.setdefault(tag, []).append((pid, name))
            elif ch.get("action") == "release_province":
                releases.append((pid, name))
            elif ch.get("action") == "transfer_province":
                transfers.append((pid, name, ch.get("from_tag", "?"), ch.get("to_tag", "?")))
            else:
                lines.append(f"• `{ch.get('action','?')}` `{pid}` ({name})  *{reason}*")

        lines.insert(0, f"**{len(changes)} proposed change(s):**")
        for tag, items in sorted(assigns.items()):
            if len(items) == 1:
                pid, name = items[0]
                lines.append(f"• **Assign** `{pid}` ({name}) → **{tag}**")
            else:
                ids = ", ".join(f"`{p}`" for p, _ in items[:12])
                more = f", +{len(items) - 12} more" if len(items) > 12 else ""
                lines.append(f"• **Assign** {len(items)} provinces → **{tag}**: {ids}{more}")
        if releases:
            ids = ", ".join(f"`{p}`" for p, _ in releases[:12])
            more = f", +{len(releases) - 12} more" if len(releases) > 12 else ""
            lines.append(f"• **Release** {len(releases)} province(s): {ids}{more}")
        for pid, name, frm, to in transfers:
            lines.append(f"• **Transfer** `{pid}` ({name}): **{frm}** → **{to}**")

    if invalids:
        lines.append("")
        lines.append(f"⚠️ **{len(invalids)} invalid proposal(s) — will be skipped:**")
        for inv in invalids[:8]:
            lines.append(
                f"• `{inv.get('province_id','?')}` ({inv.get('action','?')}): {inv.get('invalid_reason','?')}"
            )
        if len(invalids) > 8:
            lines.append(f"• …+{len(invalids) - 8} more")

    if changes:
        lines.append("\nReact ✅ to apply, ❌ to cancel.")
    return "\n".join(lines)
