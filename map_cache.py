"""In-memory + on-disk cache for the rendered world map.

`render_world_map` is the bot's most expensive operation (matplotlib in a
thread executor — multi-second). The output is fully determined by:
  - occupation:        {tag: OccupiedNation(..., color, ...)}
  - province_ownership:{pid: ProvinceOwnership(..., player_tag, color, ...)}
  - year:              baked into the map title

Other fields on those records (display_name, user_id, since) do NOT affect
the rendered pixels. We hash only the render-affecting fields into a
deterministic state-key; if the state-key matches the cached entry, the
PNG is reused without invoking matplotlib.

A double-checked-lock pattern collapses concurrent misses into a single
render. Optionally, the (key, png) pair is persisted to disk so a bot
restart on unchanged state can serve the previous render immediately.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional, Tuple

log = logging.getLogger("dnc.map_cache")

# Bump when the renderer's output semantics change — busts old disk caches.
CACHE_VERSION = 3


def _atomic_write_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _atomic_write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


class MapCache:
    """Cache one rendered world-map PNG, keyed by a deterministic state hash.

    Persistence: when `persist_path` is set, the rendered PNG is mirrored to
    `<persist_path>.png` and a small sidecar JSON `<persist_path>.json`
    holding {key, v, rendered_at}. On instantiation we attempt to load the
    sidecar; if its `v` matches CACHE_VERSION the cache is warm from disk.
    """

    def __init__(self, *, persist_path: Optional[str] = None):
        self._key: Optional[str] = None
        self._png: Optional[bytes] = None
        self._lock = asyncio.Lock()
        self._persist_path = persist_path
        if persist_path:
            self._load_from_disk()

    # ── State key ────────────────────────────────────────────────────────────

    @staticmethod
    def compute_key(occupation: dict, province_ownership: dict, year: int) -> str:
        payload = {
            "v": CACHE_VERSION,
            "year": year,
            "occ": sorted((tag, getattr(n, "color", "")) for tag, n in occupation.items()),
            "prov": sorted(
                (pid, getattr(o, "player_tag", ""), getattr(o, "color", ""))
                for pid, o in province_ownership.items()
            ),
        }
        blob = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    # ── Public API ───────────────────────────────────────────────────────────

    async def get_or_render(
        self,
        occupation: dict,
        province_ownership: dict,
        year: int,
        renderer: Callable[[], Awaitable[bytes]],
    ) -> Tuple[bytes, str, bool]:
        """Return (png, state_key, was_hit).

        If state-key matches the cached entry, the cached PNG is returned
        without invoking `renderer`. Otherwise `renderer` is awaited under
        an asyncio lock (double-checked) and the result is stored.
        """
        key = self.compute_key(occupation, province_ownership, year)

        # Fast path: lock-free read.
        png = self._png
        if png is not None and self._key == key:
            return png, key, True

        async with self._lock:
            # Re-check after acquiring the lock — another coroutine may have
            # just rendered while we were waiting.
            png = self._png
            if png is not None and self._key == key:
                return png, key, True

            rendered = await renderer()
            self._png = rendered
            self._key = key
            if self._persist_path:
                try:
                    self._persist(key, rendered)
                except Exception:
                    log.exception("MapCache: failed to persist to %s", self._persist_path)
            return rendered, key, False

    def peek(self) -> Optional[Tuple[str, bytes]]:
        """Return (key, png) currently cached, or None."""
        if self._png is None or self._key is None:
            return None
        return self._key, self._png

    def invalidate(self) -> None:
        """Drop the in-memory entry. Disk copy is untouched."""
        self._key = None
        self._png = None

    # ── Persistence ──────────────────────────────────────────────────────────

    def _png_path(self) -> str:
        return f"{self._persist_path}.png"

    def _json_path(self) -> str:
        return f"{self._persist_path}.json"

    def _persist(self, key: str, png: bytes) -> None:
        _atomic_write_bytes(self._png_path(), png)
        _atomic_write_json(
            self._json_path(),
            {
                "v": CACHE_VERSION,
                "key": key,
                "rendered_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        log.info("MapCache: persisted PNG (%d bytes, key=%s) to %s",
                 len(png), key[:8], self._png_path())

    def _load_from_disk(self) -> None:
        if not self._persist_path:
            return
        png_path, json_path = self._png_path(), self._json_path()
        if not (os.path.exists(png_path) and os.path.exists(json_path)):
            return
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if int(meta.get("v", 0)) != CACHE_VERSION:
                log.info("MapCache: disk cache v=%s ≠ current v=%d, ignoring",
                         meta.get("v"), CACHE_VERSION)
                return
            key = str(meta.get("key", "")).strip()
            if not key:
                return
            with open(png_path, "rb") as f:
                png = f.read()
            if not png:
                return
            self._key = key
            self._png = png
            log.info("MapCache: loaded from disk key=%s (%d bytes)", key[:8], len(png))
        except Exception:
            log.exception("MapCache: failed to load %s / %s — starting cold",
                          png_path, json_path)
