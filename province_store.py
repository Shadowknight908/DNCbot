"""Province ownership state.

Province *definitions* come from arcgis_provinces (real ArcGIS admin divisions,
always loaded). This module only manages who owns each province.

Province ownership persists to province_ownership.json (atomic JSON dict).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import arcgis_provinces
from arcgis_provinces import Province
from map_colors import MapColorStore

log = logging.getLogger("dnc.province_store")


@dataclass
class ProvinceOwnership:
    province_id: str
    player_tag: str
    user_id: str
    display_name: str
    color: str
    since: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def _atomic_json(path: str, data: object) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


class ProvinceStore:
    """Manages per-province ownership; definitions come from ArcGIS data."""

    def __init__(
        self,
        color_store: MapColorStore,
        ownership_path: str = "province_ownership.json",
        **_,  # absorb legacy kwargs (definitions_path, etc.)
    ):
        self._colors = color_store
        self._own_path = ownership_path
        self._lock = threading.Lock()
        self._ownership: dict[str, ProvinceOwnership] = {}
        self._load_ownership()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_ownership(self) -> None:
        if not os.path.exists(self._own_path):
            return
        with open(self._own_path, encoding="utf-8") as f:
            raw: dict = json.load(f)
        provinces = arcgis_provinces.get_all()
        loaded = 0
        for pid, d in raw.items():
            if pid not in provinces:
                log.debug("Dropping stale ownership for unknown province %s", pid)
                continue
            try:
                ownership = ProvinceOwnership(**d)
            except TypeError as e:
                log.warning("Malformed ownership record %s: %s", pid, e)
                continue
            # Reconcile color: always key by player_tag, not province_id.
            # This migrates any legacy per-province-ID color entries.
            tag = ownership.player_tag.upper()
            stored = self._colors.get(tag)
            if stored:
                ownership.color = stored
            else:
                ownership.color = self._colors.assign(tag)
            self._ownership[pid] = ownership
            loaded += 1
        log.info("ProvinceStore: loaded %d ownership records", loaded)

    def _save_ownership(self) -> None:
        _atomic_json(self._own_path, {pid: vars(o) for pid, o in self._ownership.items()})

    # ── Ownership management ──────────────────────────────────────────────────

    def assign(
        self,
        province_id: str,
        player_tag: str,
        user_id: str,
        display_name: str,
    ) -> str:
        """Assign province_id to a player. Returns the assigned hex color."""
        province_id = province_id.upper()
        player_tag = player_tag.upper()
        with self._lock:
            if province_id not in arcgis_provinces.get_all():
                raise ValueError(
                    f"Province '{province_id}' not found in ArcGIS data. "
                    "Province IDs look like 'US-AK' or 'DE-BB'."
                )
            # If province already owned, check whether the old tag loses its last province
            if province_id in self._ownership:
                old_tag = self._ownership[province_id].player_tag
                if old_tag != player_tag:
                    others = [
                        pid for pid, o in self._ownership.items()
                        if o.player_tag == old_tag and pid != province_id
                    ]
                    if not others:
                        self._colors.release(old_tag)
            # Color is stable per nation tag — assigned once, reused for all its provinces
            color = self._colors.assign(player_tag)
            self._ownership[province_id] = ProvinceOwnership(
                province_id=province_id,
                player_tag=player_tag,
                user_id=user_id,
                display_name=display_name,
                color=color,
                since=datetime.now(timezone.utc).isoformat(),
            )
            self._save_ownership()
        return color

    def release(self, province_id: str) -> bool:
        """Release province ownership. Returns True if it existed."""
        province_id = province_id.upper()
        with self._lock:
            if province_id not in self._ownership:
                return False
            old_tag = self._ownership[province_id].player_tag
            del self._ownership[province_id]
            # Release the nation's color only if it no longer holds any province
            still_holds = any(o.player_tag == old_tag for o in self._ownership.values())
            if not still_holds:
                self._colors.release(old_tag)
            self._save_ownership()
        return True

    def release_all_for_tag(self, player_tag: str) -> int:
        """Release all provinces owned by a given player tag. Returns count released."""
        player_tag = player_tag.upper()
        with self._lock:
            owned = [pid for pid, o in self._ownership.items() if o.player_tag == player_tag]
            for pid in owned:
                del self._ownership[pid]
            if owned:
                self._colors.release(player_tag)  # release once for the tag
                self._save_ownership()
        return len(owned)

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_ownership(self) -> dict[str, ProvinceOwnership]:
        with self._lock:
            return dict(self._ownership)

    def list_by_country(self, tag: str) -> list[Province]:
        """Return provinces for the game tag, using the TAG_TO_ISO2 mapping."""
        from map_renderer import TAG_TO_ISO2
        iso2 = TAG_TO_ISO2.get(tag.upper())
        if not iso2:
            return []
        return sorted(arcgis_provinces.get_by_iso2(iso2), key=lambda p: p.name)

    def all_definitions(self) -> dict[str, Province]:
        return arcgis_provinces.get_all()

    def get_province(self, province_id: str) -> Optional[Province]:
        return arcgis_provinces.get_province(province_id)
