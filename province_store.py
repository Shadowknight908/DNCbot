"""Province definitions and ownership state.

Province definitions persist to map_data/provinces.json (GeoJSON FeatureCollection).
Province ownership persists to province_ownership.json (atomic JSON dict).

A country is either held whole (via MapStore) or subdivided into provinces
(via ProvinceStore). The two states coexist: map_renderer draws province cells
on top of the country base layer for subdivided nations.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from map_colors import MapColorStore
from province_generator import Province, merge_and_subdivide

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
    """Manages province definitions (generated via subdivision) and per-province ownership."""

    def __init__(
        self,
        color_store: MapColorStore,
        definitions_path: str = "map_data/provinces.json",
        ownership_path: str = "province_ownership.json",
    ):
        self._colors = color_store
        self._defs_path = definitions_path
        self._own_path = ownership_path
        self._lock = threading.Lock()

        self._definitions: dict[str, Province] = {}       # province_id → Province
        self._ownership: dict[str, ProvinceOwnership] = {}  # province_id → ProvinceOwnership
        self._subdivided_tags: set[str] = set()

        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(self._defs_path):
            with open(self._defs_path, encoding="utf-8") as f:
                fc = json.load(f)
            for feat in fc.get("features", []):
                props = feat["properties"]
                pid = props["province_id"]
                parent = props["parent_tag"]
                p = Province(
                    id=pid,
                    name=props["name"],
                    parent_tag=parent,
                    geometry=feat["geometry"],
                    centroid=tuple(props["centroid"]),
                )
                self._definitions[pid] = p
                self._subdivided_tags.add(parent)
            log.info("ProvinceStore: loaded %d province definitions", len(self._definitions))

        if os.path.exists(self._own_path):
            with open(self._own_path, encoding="utf-8") as f:
                raw: dict = json.load(f)
            loaded = 0
            for pid, d in raw.items():
                if pid in self._definitions:
                    self._ownership[pid] = ProvinceOwnership(**d)
                    loaded += 1
            log.info("ProvinceStore: loaded %d ownership records", loaded)

    def _save_definitions(self) -> None:
        features = []
        for p in self._definitions.values():
            features.append({
                "type": "Feature",
                "geometry": p.geometry,
                "properties": {
                    "province_id": p.id,
                    "name": p.name,
                    "parent_tag": p.parent_tag,
                    "centroid": list(p.centroid),
                },
            })
        _atomic_json(self._defs_path, {"type": "FeatureCollection", "features": features})

    def _save_ownership(self) -> None:
        _atomic_json(self._own_path, {pid: vars(o) for pid, o in self._ownership.items()})

    # ── Province color key ────────────────────────────────────────────────────

    @staticmethod
    def _color_key(province_id: str) -> str:
        # Province color keys are stored as-is in MapColorStore (already uppercase).
        # The hyphen distinguishes them from bare country tags ("USA-01" vs "USA").
        return province_id.upper()

    # ── Subdivision management ────────────────────────────────────────────────

    def subdivide(
        self,
        tag: str,
        n: int,
        geojson_features: list[dict],
    ) -> list[Province]:
        """Subdivide country TAG into N auto-numbered provinces.

        geojson_features: GeoJSON feature dicts from world_1970.geojson matching
        this country (may be multiple for island nations; they are merged first).
        Clears any existing subdivision and ownership for TAG before creating new cells.
        """
        tag = tag.upper()
        provinces = merge_and_subdivide(geojson_features, tag, n)

        with self._lock:
            self._remove_tag_definitions(tag)
            for p in provinces:
                self._definitions[p.id] = p
            self._subdivided_tags.add(tag)
            self._save_definitions()
            self._save_ownership()

        log.info("ProvinceStore: subdivided %s into %d provinces", tag, len(provinces))
        return provinces

    def merge(self, tag: str) -> bool:
        """Remove subdivision for TAG, releasing all province ownership. Returns True if existed."""
        tag = tag.upper()
        with self._lock:
            if tag not in self._subdivided_tags:
                return False
            self._remove_tag_definitions(tag)
            self._subdivided_tags.discard(tag)
            self._save_definitions()
            self._save_ownership()
        log.info("ProvinceStore: merged %s back to country level", tag)
        return True

    def _remove_tag_definitions(self, tag: str) -> None:
        """Remove definitions and release colors for all provinces of TAG. Caller holds lock."""
        pids = [pid for pid, p in self._definitions.items() if p.parent_tag == tag]
        for pid in pids:
            if pid in self._ownership:
                self._colors.release(self._color_key(pid))
                del self._ownership[pid]
            del self._definitions[pid]

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
        with self._lock:
            if province_id not in self._definitions:
                raise ValueError(
                    f"Province '{province_id}' is not defined. "
                    "Run `!DNC mapdivide TAG N` first."
                )
            key = self._color_key(province_id)
            if province_id in self._ownership:
                self._colors.release(key)
            color = self._colors.assign(key)
            self._ownership[province_id] = ProvinceOwnership(
                province_id=province_id,
                player_tag=player_tag.upper(),
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
            self._colors.release(self._color_key(province_id))
            del self._ownership[province_id]
            self._save_ownership()
        return True

    def release_all_for_country(self, tag: str) -> int:
        """Release all province ownership for a country tag. Returns count released."""
        tag = tag.upper()
        with self._lock:
            owned_pids = [
                pid for pid, o in self._ownership.items()
                if self._definitions.get(pid) is not None
                and self._definitions[pid].parent_tag == tag
            ]
            for pid in owned_pids:
                self._colors.release(self._color_key(pid))
                del self._ownership[pid]
            if owned_pids:
                self._save_ownership()
        return len(owned_pids)

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_ownership(self) -> dict[str, ProvinceOwnership]:
        with self._lock:
            return dict(self._ownership)

    def list_by_country(self, tag: str) -> list[Province]:
        tag = tag.upper()
        with self._lock:
            return sorted(
                [p for p in self._definitions.values() if p.parent_tag == tag],
                key=lambda p: p.id,
            )

    def all_definitions(self) -> dict[str, Province]:
        with self._lock:
            return dict(self._definitions)

    def is_subdivided(self, tag: str) -> bool:
        return tag.upper() in self._subdivided_tags

    def get_province(self, province_id: str) -> Optional[Province]:
        return self._definitions.get(province_id.upper())

    def subdivided_tags(self) -> set[str]:
        with self._lock:
            return set(self._subdivided_tags)
