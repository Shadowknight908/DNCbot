"""Persistent runtime state.

Atomic writes via unique-temp-then-rename. Each write uses a uniquely
named temp file so concurrent writes can't clobber each other.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict

log = logging.getLogger("dnc.state")


DEFAULT_STATE: Dict[str, Any] = {
    "current_year": 1900,
    "last_rollover_at": None,
    "started_at": None,
    "stats": {
        "messages_seen": 0,
        "messages_filtered_local": 0,
        "messages_sent_to_archivist": 0,
        "messages_archived": 0,
        "messages_filtered_non_lore": 0,
        "queries_answered": 0,
        "chats_answered": 0,
        "chatuc_answered": 0,
        "gm_rulings_made": 0,
        "gm_rulings_revised": 0,
        "wars_started": 0,
        "war_moves_adjudicated": 0,
        "war_npc_moves": 0,
        "wars_ended": 0,
        "voids_executed": 0,
        "unvoids_executed": 0,
        "archivist_prompt_tokens": 0,
        "archivist_completion_tokens": 0,
        "query_prompt_tokens": 0,
        "query_completion_tokens": 0,
        "chat_prompt_tokens": 0,
        "chat_completion_tokens": 0,
        "chatuc_prompt_tokens": 0,
        "chatuc_completion_tokens": 0,
        "gm_prompt_tokens": 0,
        "gm_completion_tokens": 0,
        "war_prompt_tokens": 0,
        "war_completion_tokens": 0,
        "vision_prompt_tokens": 0,
        "vision_completion_tokens": 0,
        "embedding_tokens": 0,
        "stats_reset_at": None,
    },
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateStore:
    def __init__(self, path: str = "state.json"):
        self._path = path
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {}
        self._cleanup_orphan_tmps()
        self._load()

    def _cleanup_orphan_tmps(self) -> None:
        pattern = self._path + ".tmp.*"
        for orphan in glob.glob(pattern):
            try:
                os.unlink(orphan)
                log.info("Cleaned up orphan temp file: %s", orphan)
            except OSError:
                pass

    def _load(self) -> None:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                log.warning("state.json unreadable; starting fresh")
                self._data = {}
        self._merge_defaults(self._data, DEFAULT_STATE)
        self._save()

    @staticmethod
    def _merge_defaults(target: Dict[str, Any], defaults: Dict[str, Any]) -> None:
        for k, v in defaults.items():
            if k not in target:
                target[k] = v if not isinstance(v, dict) else dict(v)
            elif isinstance(v, dict) and isinstance(target[k], dict):
                StateStore._merge_defaults(target[k], v)

    def _save(self) -> None:
        tmp = f"{self._path}.tmp.{os.getpid()}.{id(self)}.{time.time_ns()}"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
            os.replace(tmp, self._path)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            except OSError:
                pass
            raise

    @property
    def current_year(self) -> int:
        return int(self._data.get("current_year", 1900))

    def set_year(self, year: int, mark_rollover: bool = False) -> None:
        with self._lock:
            self._data["current_year"] = int(year)
            if mark_rollover:
                self._data["last_rollover_at"] = _now_iso()
            self._save()

    def reset_rollover_timer(self) -> None:
        with self._lock:
            self._data["last_rollover_at"] = _now_iso()
            self._save()

    @property
    def last_rollover_at(self) -> str:
        return self._data.get("last_rollover_at") or ""

    def mark_started(self) -> None:
        with self._lock:
            self._data["started_at"] = _now_iso()
            self._save()

    @property
    def started_at(self) -> str:
        return self._data.get("started_at") or ""

    def bump(self, key: str, by: int = 1) -> None:
        with self._lock:
            stats = self._data.setdefault("stats", {})
            stats[key] = int(stats.get(key, 0)) + int(by)
            self._save()

    def add_usage(self, kind: str, usage: Dict[str, int]) -> None:
        with self._lock:
            stats = self._data.setdefault("stats", {})
            if kind == "embedding":
                stats["embedding_tokens"] = int(stats.get("embedding_tokens", 0)) + int(
                    usage.get("total_tokens", 0)
                )
            else:
                p = f"{kind}_prompt_tokens"
                c = f"{kind}_completion_tokens"
                stats[p] = int(stats.get(p, 0)) + int(usage.get("prompt_tokens", 0))
                stats[c] = int(stats.get(c, 0)) + int(usage.get("completion_tokens", 0))
            self._save()

    def stats_snapshot(self) -> Dict[str, Any]:
        return dict(self._data.get("stats", {}))

    def reset_stats(self) -> None:
        with self._lock:
            fresh = dict(DEFAULT_STATE["stats"])
            fresh["stats_reset_at"] = _now_iso()
            self._data["stats"] = fresh
            self._save()

    def next_ruling_id(self, year: int) -> int:
        """Increment and return the per-year GM ruling counter."""
        with self._lock:
            key = f"gm_ruling_counter_{int(year)}"
            n = int(self._data.get(key, 0)) + 1
            self._data[key] = n
            self._save()
            return n

    def next_war_id(self, year: int) -> int:
        """Increment and return the per-year war counter."""
        with self._lock:
            key = f"war_counter_{int(year)}"
            n = int(self._data.get(key, 0)) + 1
            self._data[key] = n
            self._save()
            return n
