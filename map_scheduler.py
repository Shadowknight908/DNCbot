"""Nickname scanning utilities for the spatial mapping feature."""
from __future__ import annotations

import logging

import discord

from nickname_parser import parse_tag

log = logging.getLogger("dnc.map_scheduler")


def scan_members(members: list[discord.Member]) -> dict[str, tuple[str, str]]:
    """Return {tag: (user_id, display_name)} for all members with a valid tag.

    If two members share a tag, the first encountered is kept and a warning
    is logged — duplicate tags should be resolved by server admins.
    """
    assignments: dict[str, tuple[str, str]] = {}
    for member in members:
        tag = parse_tag(member.nick or member.name)
        if not tag:
            continue
        if tag in assignments:
            log.warning(
                "Duplicate tag %s: held by both %s and %s — keeping first",
                tag, assignments[tag][1], member.display_name,
            )
            continue
        assignments[tag] = (str(member.id), member.display_name)
    return assignments
