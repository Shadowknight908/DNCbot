"""Opt-out registry.

Stored as a plain text file. One user per line. Format:

    123456789012345678  # username at time of opt-out

Comments and blank lines are ignored on read. Auto-prune removes a
user's entry when they leave the server.
"""
from __future__ import annotations

import os
import threading
from typing import Set


class OptOutStore:
    def __init__(self, path: str = "optouts.txt"):
        self._path = path
        self._lock = threading.Lock()
        if not os.path.exists(self._path):
            with open(self._path, "w", encoding="utf-8") as f:
                f.write("# DNC Lore Bot — opt-out list\n")
                f.write("# One user ID per line; anything after # is a comment.\n")
                f.write("# Users on this list have their messages skipped during ingestion.\n\n")

    def _read_ids(self) -> Set[str]:
        ids: Set[str] = set()
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if line:
                    ids.add(line)
        return ids

    def is_opted_out(self, user_id: str) -> bool:
        return str(user_id) in self._read_ids()

    def add(self, user_id: str, display_name: str) -> bool:
        """Returns True if the user was newly added, False if already present."""
        with self._lock:
            existing = self._read_ids()
            if str(user_id) in existing:
                return False
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(f"{user_id}  # {display_name}\n")
            return True

    def remove(self, user_id: str) -> bool:
        """Returns True if the user was found and removed."""
        with self._lock:
            target = str(user_id)
            kept_lines = []
            removed = False
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.split("#", 1)[0].strip()
                    if stripped == target:
                        removed = True
                        continue
                    kept_lines.append(line)
            if removed:
                with open(self._path, "w", encoding="utf-8") as f:
                    f.writelines(kept_lines)
            return removed

    def count(self) -> int:
        return len(self._read_ids())
