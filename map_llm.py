"""LLM frontend for province ownership changes.

Given the text of a roleplay post, the LLM is asked to identify any territorial
changes (captures, cessions, liberations) and emit structured tool calls. The
tool executor does NOT apply changes — it records proposals so the caller can
present them for admin confirmation before committing.

Public API:
    parse_post_for_changes(post_text, province_context, client, model) → list[dict]
    format_change_summary(changes, province_store) → str
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

log = logging.getLogger("dnc.map_llm")

_SYSTEM_PROMPT = """\
You are a territorial analyst for a 1970s historical roleplay game. \
You are given a roleplay post and a list of currently defined map provinces. \
Your job is to identify any territorial changes described in the post — \
captures, occupations, cessions, liberations, or transfers of specific regions.

For each territorial change you detect, call the appropriate tool:
- assign_province: a player/nation takes control of a province
- release_province: a province becomes uncontrolled (abandoned, liberated without new owner)
- transfer_province: a province changes hands between two players

Only emit tool calls for provinces that appear in the provided province list. \
Do not invent province IDs. If the post does not describe any territorial changes, \
do not call any tools and reply with a brief explanation.\
"""

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "assign_province",
            "description": "A player/nation takes control of a province.",
            "parameters": {
                "type": "object",
                "properties": {
                    "province_id": {
                        "type": "string",
                        "description": "Province ID, e.g. USSR-03",
                    },
                    "player_tag": {
                        "type": "string",
                        "description": "Country tag of the controlling player, e.g. USSR",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Short explanation drawn from the post text",
                    },
                },
                "required": ["province_id", "player_tag"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "release_province",
            "description": "A province becomes uncontrolled (abandoned or liberated with no new owner).",
            "parameters": {
                "type": "object",
                "properties": {
                    "province_id": {
                        "type": "string",
                        "description": "Province ID, e.g. USA-07",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Short explanation drawn from the post text",
                    },
                },
                "required": ["province_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_province",
            "description": "A province changes hands from one player to another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "province_id": {
                        "type": "string",
                        "description": "Province ID, e.g. DEW-02",
                    },
                    "from_tag": {
                        "type": "string",
                        "description": "Current owner's country tag",
                    },
                    "to_tag": {
                        "type": "string",
                        "description": "New owner's country tag",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Short explanation drawn from the post text",
                    },
                },
                "required": ["province_id", "from_tag", "to_tag"],
            },
        },
    },
]


def _build_province_context_text(province_context: dict) -> str:
    """Build a compact province listing for the LLM's user message."""
    if not province_context:
        return "(No provinces are currently defined on the map.)"
    lines = ["Current province definitions:"]
    by_parent: dict[str, list] = {}
    for pid, info in sorted(province_context.items()):
        parent = info.get("parent_tag", pid.split("-")[0])
        by_parent.setdefault(parent, []).append((pid, info))
    for parent in sorted(by_parent):
        lines.append(f"\n  {parent}:")
        for pid, info in by_parent[parent]:
            owner = info.get("owner", "uncontrolled")
            lines.append(f"    {pid} — {owner}")
    return "\n".join(lines)


async def parse_post_for_changes(
    post_text: str,
    province_context: dict,
    client: Any,
    model: Optional[str] = None,
    tavily: Any = None,
) -> list[dict]:
    """Call the LLM to identify territorial changes in post_text.

    province_context: {province_id: {owner: str|None, parent_tag: str, name: str}}

    Returns a list of change dicts:
        {action: "assign"|"release"|"transfer", province_id: str, ...args, reason: str}
    The changes are NOT applied — they are returned for admin confirmation.
    """
    proposals: list[dict] = []

    async def _collect(name: str, args: dict) -> str:
        if name == "web_search" and tavily is not None:
            return await tavily.execute(name, args)
        proposal = {"action": name, **args}
        proposals.append(proposal)
        log.info("Map LLM proposed: %s %s", name, args.get("province_id", ""))
        return json.dumps({"status": "recorded"})

    tools = list(TOOL_DEFINITIONS)
    if tavily is not None:
        tools.append(tavily.tool_definition())

    province_text = _build_province_context_text(province_context)
    user_message = f"{province_text}\n\n---\nRoleplay post to analyze:\n{post_text}"

    try:
        await client.chat_messages(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            model=model,
            tools=tools,
            tool_executor=_collect,
            temperature=0.0,
        )
    except Exception as e:
        log.error("map_llm: LLM call failed: %s", e, exc_info=True)
        raise

    return proposals


_INSTRUCTION_SYSTEM_PROMPT = """\
You are a map administrator for a 1970s Cold War roleplay game. \
You receive direct instructions from a game master describing territorial changes to apply. \
Execute the instructions exactly as given by calling the appropriate tool for each province affected:
- assign_province: set a province's owner to a specific player/nation tag
- release_province: make a province uncontrolled (no owner)
- transfer_province: move a province from one owner to another

You have been given a complete list of province IDs grouped by country. \
Only use IDs that appear in that list — do not invent province IDs. \
If the instructions name a region or state (e.g. "Bavaria", "Alaska"), find the matching province ID in the list. \
If an instruction covers multiple provinces, call the tool for each one individually. \
Include a short reason for each change drawn from the instructions.\
"""


async def process_map_instructions(
    instructions: str,
    province_context: dict,
    client: Any,
    model: Optional[str] = None,
    tavily: Any = None,
) -> list[dict]:
    """Call the LLM to execute direct map-change instructions.

    Unlike parse_post_for_changes (which infers changes from RP narrative),
    this accepts explicit administrator commands such as
    "Assign Bavaria to FRG and release Hamburg."

    Returns proposed changes for admin confirmation — NOT applied yet.
    """
    proposals: list[dict] = []

    async def _collect(name: str, args: dict) -> str:
        if name == "web_search" and tavily is not None:
            return await tavily.execute(name, args)
        proposals.append({"action": name, **args})
        log.info("Map instruction LLM proposed: %s %s", name, args.get("province_id", ""))
        return json.dumps({"status": "recorded"})

    tools = list(TOOL_DEFINITIONS)
    if tavily is not None:
        tools.append(tavily.tool_definition())

    province_text = _build_province_context_text(province_context)
    user_message = f"{province_text}\n\n---\nMap change instructions:\n{instructions}"

    try:
        await client.chat_messages(
            messages=[
                {"role": "system", "content": _INSTRUCTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            model=model,
            tools=tools,
            tool_executor=_collect,
            temperature=0.0,
        )
    except Exception as e:
        log.error("map_llm.process_map_instructions: LLM call failed: %s", e, exc_info=True)
        raise

    return proposals


def format_change_summary(changes: list[dict], province_store: Any) -> str:
    """Return a human-readable markdown summary of proposed changes."""
    if not changes:
        return "No territorial changes detected in this post."

    lines = [f"**{len(changes)} proposed territorial change(s):**\n"]
    for ch in changes:
        action = ch.get("action", "?")
        pid = ch.get("province_id", "?")
        reason = ch.get("reason", "")
        prov = province_store.get_province(pid)
        prov_label = f"`{pid}`" + (f" ({prov.name})" if prov else " *(unknown)*")

        if action == "assign_province":
            player = ch.get("player_tag", "?")
            lines.append(f"• **Assign** {prov_label} → **{player}**")
        elif action == "release_province":
            lines.append(f"• **Release** {prov_label} (becomes uncontrolled)")
        elif action == "transfer_province":
            frm = ch.get("from_tag", "?")
            to = ch.get("to_tag", "?")
            lines.append(f"• **Transfer** {prov_label}: **{frm}** → **{to}**")
        else:
            lines.append(f"• **{action}** {prov_label}")

        if reason:
            lines.append(f"  *{reason}*")

    lines.append("\nReact ✅ to apply all changes, ❌ to cancel.")
    return "\n".join(lines)
