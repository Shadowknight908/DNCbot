"""Persistent ISO tag → hex color assignments.

File format (map_colors.txt):
    USA #e63946
    GBR #457b9d
    # lines starting with # are comments, blank lines ignored

Colors are assigned from a palette when a new tag is first occupied.
Colors are released (deleted from file) when a tag becomes vacant.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

log = logging.getLogger("dnc.map_colors")

DEFAULT_PALETTE: list[str] = [
    "#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261",
    "#6a4c93", "#1982c4", "#8ac926", "#ff595e", "#ffca3a",
    "#06d6a0", "#118ab2", "#ef476f", "#ffd166", "#3a0ca3",
    "#0d232a", "#f72585", "#7209b7", "#560bad", "#480ca8",
]


class MapColorStore:
    def __init__(self, path: str = "map_colors.txt", palette: Optional[list[str]] = None):
        self._path = path
        self._palette = palette or DEFAULT_PALETTE
        self._lock = threading.Lock()
        self._colors: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._path):
            self._colors = {}
            return
        colors: dict[str, str] = {}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        colors[parts[0].upper()] = parts[1]
        except OSError:
            log.warning("Could not read %s; starting fresh", self._path)
        self._colors = colors
        log.info("Map colors loaded: %d tags", len(self._colors))

    def _save(self) -> None:
        tmp = self._path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                for tag in sorted(self._colors):
                    f.write(f"{tag} {self._colors[tag]}\n")
            os.replace(tmp, self._path)
        except OSError:
            log.error("Failed to save %s", self._path)

    def assign(self, tag: str) -> str:
        """Return existing color for tag, or pick the next unused palette color."""
        tag = tag.upper()
        with self._lock:
            if tag in self._colors:
                return self._colors[tag]
            used = set(self._colors.values())
            for color in self._palette:
                if color not in used:
                    self._colors[tag] = color
                    self._save()
                    log.info("Assigned color %s to tag %s", color, tag)
                    return color
            # All palette colors in use — cycle deterministically
            color = self._palette[len(self._colors) % len(self._palette)]
            self._colors[tag] = color
            self._save()
            log.info("Palette exhausted; cycled color %s to tag %s", color, tag)
            return color

    def release(self, tag: str) -> bool:
        """Remove the color assignment for a vacated tag. Returns True if removed."""
        tag = tag.upper()
        with self._lock:
            if tag not in self._colors:
                return False
            del self._colors[tag]
            self._save()
            log.info("Released color for vacated tag %s", tag)
            return True

    def get(self, tag: str) -> Optional[str]:
        return self._colors.get(tag.upper())

    def all_colors(self) -> dict[str, str]:
        with self._lock:
            return dict(self._colors)
