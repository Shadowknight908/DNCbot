"""Persistent storage for War GM mode.

A war is scoped to a single Discord thread. Each war keeps an ordered,
append-only ``log`` of moves and rulings — this verbatim record is the
heart of the war-memory system: every adjudication replays it in full
(within a char budget) rather than relying on lossy vector search. A
maintained ``state_digest`` (forces / territory / casualties) is kept
alongside as an overflow hedge for very long wars.

Schema (wars.json):
{
  "wars": {
    "<thread_id>": {
      "war_id":        "WAR-1900-0001",
      "title":         "The Levant War",
      "thread_id":     "...",
      "guild_id":      "...",
      "status":        "active" | "pending_commit" | "ended" | "cancelled",
      "started_by":    "<user_id>",
      "started_at":    "<iso>",
      "start_year":    1900,
      "ended_at":      "<iso>" | null,
      "sides":         { "Allies": {"tags": ["USA", "GBR"]} },
      "belligerents":  { "USA": {"side": "Allies", "npc": false} },
      "log":           [ {seq, type, ...}, ... ],
      "next_seq":      1,
      "state_digest":  "<markdown>",
      "digest_updated_seq": 0,
      "chronicle_draft":   [ {"title": ..., "text": ...} ]
    }
  }
}
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Statuses that count as "the war is still going" for routing/guard purposes.
LIVE_STATUSES = ("active", "pending_commit")


class WarStore:
    def __init__(self, path: str = "wars.json"):
        self._path = path
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self._path):
            return {"wars": {}}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"wars": {}}
        data.setdefault("wars", {})
        return data

    def _save(self) -> None:
        tmp = f"{self._path}.tmp.{os.getpid()}.{time.time_ns()}"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self._path)
        except Exception:
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            raise

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def get_war(self, thread_id: str) -> Optional[Dict[str, Any]]:
        return self._data["wars"].get(str(thread_id))

    def get_active_war(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Return the war for this thread only if it's still live."""
        war = self._data["wars"].get(str(thread_id))
        if war and war.get("status") in LIVE_STATUSES:
            return war
        return None

    def get_belligerents(self, thread_id: str) -> Dict[str, Dict[str, Any]]:
        war = self.get_war(thread_id)
        return dict(war.get("belligerents", {})) if war else {}

    def get_log(self, thread_id: str) -> List[Dict[str, Any]]:
        war = self.get_war(thread_id)
        return list(war.get("log", [])) if war else []

    def get_digest(self, thread_id: str) -> str:
        war = self.get_war(thread_id)
        return war.get("state_digest", "") if war else ""

    def get_chronicle_draft(self, thread_id: str) -> List[Dict[str, str]]:
        war = self.get_war(thread_id)
        return list(war.get("chronicle_draft", [])) if war else []

    def list_active(self) -> List[str]:
        return [
            tid for tid, w in self._data["wars"].items()
            if w.get("status") in LIVE_STATUSES
        ]

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def create_war(
        self,
        thread_id: str,
        war_id: str,
        title: str,
        guild_id: str,
        started_by: str,
        start_year: int,
    ) -> Dict[str, Any]:
        with self._lock:
            war = {
                "war_id": war_id,
                "title": title or war_id,
                "thread_id": str(thread_id),
                "guild_id": str(guild_id),
                "status": "active",
                "started_by": str(started_by),
                "started_at": _now_iso(),
                "start_year": int(start_year),
                "ended_at": None,
                "sides": {},
                "belligerents": {},
                "log": [],
                "next_seq": 1,
                "state_digest": "",
                "digest_updated_seq": 0,
                "chronicle_draft": [],
            }
            self._data["wars"][str(thread_id)] = war
            self._save()
            return war

    def add_side(
        self,
        thread_id: str,
        side_name: str,
        belligerents: List[Tuple[str, bool]],
    ) -> bool:
        """Register/replace a side and its belligerent tags.

        belligerents: list of (TAG, is_npc). Returns True on success.
        """
        with self._lock:
            war = self._data["wars"].get(str(thread_id))
            if not war:
                return False
            tags = [t.upper() for t, _ in belligerents]
            war.setdefault("sides", {})[side_name] = {"tags": tags}
            bel = war.setdefault("belligerents", {})
            for tag, npc in belligerents:
                bel[tag.upper()] = {"side": side_name, "npc": bool(npc)}
            self._save()
            return True

    def set_npc(self, thread_id: str, tag: str, npc: bool) -> bool:
        with self._lock:
            war = self._data["wars"].get(str(thread_id))
            if not war:
                return False
            bel = war.get("belligerents", {})
            tag = tag.upper()
            if tag not in bel:
                return False
            bel[tag]["npc"] = bool(npc)
            self._save()
            return True

    def append_log(self, thread_id: str, entry: Dict[str, Any]) -> int:
        """Append an entry, assigning it the next sequence number. Returns seq."""
        with self._lock:
            war = self._data["wars"].get(str(thread_id))
            if not war:
                raise KeyError(f"No war for thread {thread_id}")
            seq = int(war.get("next_seq", 1))
            entry = dict(entry)
            entry["seq"] = seq
            entry.setdefault("timestamp", _now_iso())
            war.setdefault("log", []).append(entry)
            war["next_seq"] = seq + 1
            self._save()
            return seq

    def mark_superseded(self, thread_id: str, seq: int) -> bool:
        """Flag a log entry (by seq) as superseded so replay skips it."""
        with self._lock:
            war = self._data["wars"].get(str(thread_id))
            if not war:
                return False
            for e in war.get("log", []):
                if e.get("seq") == seq:
                    e["superseded"] = True
                    self._save()
                    return True
            return False

    def update_ruling_message_ids(
        self, thread_id: str, seq: int, message_ids: List[str]
    ) -> bool:
        with self._lock:
            war = self._data["wars"].get(str(thread_id))
            if not war:
                return False
            for e in war.get("log", []):
                if e.get("seq") == seq:
                    e["discord_message_ids"] = list(message_ids)
                    self._save()
                    return True
            return False

    def set_digest(self, thread_id: str, digest: str, updated_seq: int) -> None:
        with self._lock:
            war = self._data["wars"].get(str(thread_id))
            if not war:
                return
            war["state_digest"] = digest
            war["digest_updated_seq"] = int(updated_seq)
            self._save()

    def set_status(self, thread_id: str, status: str) -> None:
        with self._lock:
            war = self._data["wars"].get(str(thread_id))
            if not war:
                return
            war["status"] = status
            if status in ("ended", "cancelled"):
                war["ended_at"] = _now_iso()
            self._save()

    def set_chronicle_draft(
        self, thread_id: str, entries: List[Dict[str, str]]
    ) -> None:
        with self._lock:
            war = self._data["wars"].get(str(thread_id))
            if not war:
                return
            war["chronicle_draft"] = list(entries)
            self._save()
