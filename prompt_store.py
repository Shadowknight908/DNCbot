"""Prompt file loader.

Reads system prompts from files on disk so they can be edited without
touching code or YAML. Each prompt is keyed by name (e.g. "memory_extraction",
"lore_query", "chat", "gm") and maps to a file path defined in config.yaml.

Supports hot reload via reload(): re-reads all files without restarting
the bot, so prompts can be iterated on live.
"""
from __future__ import annotations

import logging
import os
from typing import Dict

log = logging.getLogger("dnc.prompts")


class PromptStore:
    def __init__(self, prompt_paths: Dict[str, str]):
        """prompt_paths: name -> filepath. Read at init and on reload()."""
        self._paths = dict(prompt_paths)
        self._cache: Dict[str, str] = {}
        self.reload()

    def reload(self) -> Dict[str, str]:
        """Re-read all configured prompt files. Returns {name: status}.
        Status is "ok" or an error message — useful for displaying
        results of a !DNC reloadprompts command.
        """
        results: Dict[str, str] = {}
        new_cache: Dict[str, str] = {}
        for name, path in self._paths.items():
            if not path:
                results[name] = f"no path configured"
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    new_cache[name] = f.read().strip()
                results[name] = "ok"
                log.info("Loaded prompt %r from %s (%d chars)",
                         name, path, len(new_cache[name]))
            except FileNotFoundError:
                results[name] = f"file not found: {path}"
                log.error("Prompt %r: file not found at %s", name, path)
            except OSError as e:
                results[name] = f"read error: {e}"
                log.error("Prompt %r: read error: %s", name, e)
        # Only swap in the new cache if at least one prompt loaded successfully,
        # so a botched edit doesn't wipe everything.
        if any(v == "ok" for v in results.values()):
            # Preserve old prompts that failed to load this time
            for k, v in self._cache.items():
                if k not in new_cache:
                    new_cache[k] = v
            self._cache = new_cache
        return results

    def get(self, name: str) -> str:
        if name not in self._cache:
            raise KeyError(f"Prompt {name!r} not loaded")
        return self._cache[name]

    def names(self) -> list:
        return list(self._cache.keys())

    def paths(self) -> Dict[str, str]:
        return dict(self._paths)
