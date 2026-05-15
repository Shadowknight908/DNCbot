"""Persistent storage for the espionage system.

Schema (espionage.json):
{
  "spies": {
    "<user_id>": {
      "targets":    ["TAG1", "TAG2"],   // TAGs being actively monitored
      "counterspy": false               // counterintelligence mode active
    }
  }
}
"""
from __future__ import annotations

import json
import os
from typing import List


class EspionageStore:
    def __init__(self, path: str):
        self._path = path
        self._data = self._load()

    def _load(self) -> dict:
        if not os.path.exists(self._path):
            return {"spies": {}}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"spies": {}}

    def _save(self) -> None:
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp, self._path)

    def _entry(self, user_id: str) -> dict:
        """Return (and create if missing) a mutable entry for user_id."""
        if user_id not in self._data["spies"]:
            self._data["spies"][user_id] = {"targets": [], "counterspy": False}
        return self._data["spies"][user_id]

    def _cleanup(self, user_id: str) -> None:
        """Remove entry entirely if it's back to defaults."""
        entry = self._data["spies"].get(user_id)
        if entry and not entry.get("targets") and not entry.get("counterspy"):
            del self._data["spies"][user_id]

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_entry(self, user_id: str) -> dict:
        """Return a snapshot dict: {targets: [...], counterspy: bool}."""
        raw = self._data["spies"].get(user_id, {})
        return {
            "targets": list(raw.get("targets", [])),
            "counterspy": bool(raw.get("counterspy", False)),
        }

    def list_targets(self, user_id: str) -> List[str]:
        return list(self._data["spies"].get(user_id, {}).get("targets", []))

    def is_counterspy(self, user_id: str) -> bool:
        return bool(self._data["spies"].get(user_id, {}).get("counterspy", False))

    def get_spies_for_tag(self, tag: str) -> List[str]:
        """Return user_ids of all players currently watching this TAG."""
        return [
            uid for uid, entry in self._data["spies"].items()
            if tag in entry.get("targets", [])
        ]

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add_target(self, user_id: str, tag: str, max_slots: int) -> str:
        """Try to add a spy target. Returns 'ok', 'duplicate', or 'full'."""
        entry = self._entry(user_id)
        if tag in entry["targets"]:
            return "duplicate"
        if len(entry["targets"]) >= max_slots:
            return "full"
        entry["targets"].append(tag)
        self._save()
        return "ok"

    def remove_target(self, user_id: str, tag: str) -> bool:
        """Remove a spy target. Returns True if it existed."""
        entry = self._entry(user_id)
        if tag not in entry["targets"]:
            return False
        entry["targets"].remove(tag)
        self._cleanup(user_id)
        self._save()
        return True

    def set_counterspy(self, user_id: str, active: bool) -> None:
        entry = self._entry(user_id)
        entry["counterspy"] = active
        self._cleanup(user_id)
        self._save()

    def trim_targets(self, user_id: str, max_slots: int) -> List[str]:
        """Trim targets down to max_slots (keep oldest). Returns removed TAGs."""
        entry = self._entry(user_id)
        if len(entry["targets"]) <= max_slots:
            return []
        removed = entry["targets"][max_slots:]
        entry["targets"] = entry["targets"][:max_slots]
        self._save()
        return removed
