"""Runtime occupation state: which player holds which ISO tag.

Occupation is rebuilt from Discord nicknames on startup and kept in sync
via on_member_update events. Colors are delegated to MapColorStore and
persist on disk across restarts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from map_colors import MapColorStore

log = logging.getLogger("dnc.map_store")


@dataclass
class OccupiedNation:
    tag: str
    user_id: str
    display_name: str
    color: str
    since: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MapStore:
    def __init__(self, color_store: MapColorStore):
        self._colors = color_store
        self._occupation: dict[str, OccupiedNation] = {}

    def set_occupation(self, tag: str, user_id: str, display_name: str) -> str:
        """Register or update a tag→player assignment. Returns hex color."""
        tag = tag.upper()
        color = self._colors.assign(tag)
        self._occupation[tag] = OccupiedNation(
            tag=tag, user_id=user_id, display_name=display_name, color=color
        )
        return color

    def clear_occupation(self, tag: str) -> bool:
        """Vacate a tag and release its color. Returns True if it existed."""
        tag = tag.upper()
        if tag not in self._occupation:
            return False
        del self._occupation[tag]
        self._colors.release(tag)
        return True

    def rebuild(self, assignments: dict[str, tuple[str, str]]) -> None:
        """Atomically replace occupation state from {tag: (user_id, display_name)}.

        Releases colors for tags that are no longer held by anyone.
        """
        new_tags = {t.upper() for t in assignments}
        for vacated in set(self._occupation) - new_tags:
            del self._occupation[vacated]
            self._colors.release(vacated)
        for tag, (uid, name) in assignments.items():
            self.set_occupation(tag, uid, name)

    def get_occupation(self) -> dict[str, OccupiedNation]:
        return dict(self._occupation)
