"""Persistent faction membership loaded from a human-editable file.

File format (default path: factions):

    [NATO]
    color = #457b9d
    members = USA, GBR, FRA

    [Warsaw Pact]
    color = #e63946
    members =
        RUS
        POL
        GDR

Faction sections are ordered. If a country is listed in multiple sections,
the first faction supplies the base fill and later factions are rendered as
hatches.
"""
from __future__ import annotations

import configparser
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from map_colors import DEFAULT_PALETTE

log = logging.getLogger("dnc.faction_store")

TAG_RE = re.compile(r"^[A-Z]{2,4}$")

DEFAULT_TEMPLATE = """# DNC faction memberships.
# Edit this file while the bot is running; !DNC mapfaction reloads it each time.
#
# Format:
# [Faction Name]
# color = #457b9d
# members = USA, GBR, FRA
#
# Countries can appear in multiple factions. The faction map will render the
# first listed faction as the base fill and later factions as hatch overlays.

[Example Faction]
color = #457b9d
members =
    USA
    GBR
"""


@dataclass(frozen=True)
class FactionMembership:
    name: str
    color: str


@dataclass(frozen=True)
class Faction:
    name: str
    color: str
    members: tuple[str, ...]


class FactionStore:
    def __init__(self, path: str = "factions", palette: Optional[list[str]] = None):
        self._path = path
        self._palette = palette or DEFAULT_PALETTE
        self._factions: list[Faction] = []
        self._warnings: list[str] = []
        self._ensure_file()
        self.reload()

    @property
    def path(self) -> str:
        return self._path

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    def _ensure_file(self) -> None:
        if os.path.exists(self._path):
            return
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                f.write(DEFAULT_TEMPLATE)
            log.info("Created default factions file at %s", self._path)
        except OSError:
            log.warning("Could not create default factions file at %s", self._path)

    def reload(self) -> None:
        parser = configparser.ConfigParser(
            interpolation=None,
            inline_comment_prefixes=(";",),
        )
        parser.optionxform = str
        read_files = parser.read(self._path, encoding="utf-8")
        if not read_files:
            self._factions = []
            self._warnings = [f"Could not read {self._path}"]
            return

        factions: list[Faction] = []
        warnings: list[str] = []
        for idx, section in enumerate(parser.sections()):
            color = parser.get(section, "color", fallback="").strip()
            if not color:
                color = self._palette[idx % len(self._palette)]
            if not re.match(r"^#[0-9a-fA-F]{6}$", color):
                warnings.append(f"{section}: invalid color '{color}', using palette fallback")
                color = self._palette[idx % len(self._palette)]

            raw_members = parser.get(section, "members", fallback="")
            members: list[str] = []
            for token in re.split(r"[\s,]+", raw_members.upper()):
                tag = token.strip()
                if not tag:
                    continue
                if not TAG_RE.match(tag):
                    warnings.append(f"{section}: ignored invalid country tag '{tag}'")
                    continue
                if tag not in members:
                    members.append(tag)

            factions.append(Faction(section.strip(), color.lower(), tuple(members)))

        self._factions = factions
        self._warnings = warnings
        log.info("Loaded %d factions from %s", len(factions), self._path)

    def get_factions(self) -> list[Faction]:
        return list(self._factions)

    def get_memberships(self) -> dict[str, list[FactionMembership]]:
        memberships: dict[str, list[FactionMembership]] = {}
        for faction in self._factions:
            membership = FactionMembership(faction.name, faction.color)
            for tag in faction.members:
                memberships.setdefault(tag, []).append(membership)
        return memberships
