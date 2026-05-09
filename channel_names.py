"""Channel name normalization.

Discord channel names can contain emoji, variant selectors (U+FE0E/FE0F),
zero-width joiners (U+200D), and other Unicode characters. Comparing
names with simple `lower()` is unreliable because:

- "☎general☎" and "☎️general☎️" (with variant selectors) look identical
  but compare as unequal.
- Different normalization forms (NFC vs NFD) produce visually identical
  strings that don't match each other.

normalize_channel_name() collapses all of this into a single comparable
form so config entries match Discord's actual channel names regardless
of how they were typed or pasted.
"""
from __future__ import annotations

import logging
import unicodedata
from typing import List, Tuple

log = logging.getLogger("dnc.channels")

# Variation selectors and zero-width characters that are visually invisible
# but break naive equality comparisons.
_INVISIBLE_CHARS = {
    "\uFE0E",  # text variation selector
    "\uFE0F",  # emoji variation selector
    "\u200D",  # zero-width joiner
    "\u200C",  # zero-width non-joiner
    "\u200B",  # zero-width space
    "\uFEFF",  # zero-width no-break space (BOM)
}


def normalize_channel_name(name: str) -> str:
    """Normalize a channel name for comparison.
    NFC-normalize, strip invisible/variant characters, lowercase, strip whitespace.
    """
    if not name:
        return ""
    s = unicodedata.normalize("NFC", name)
    s = "".join(ch for ch in s if ch not in _INVISIBLE_CHARS)
    return s.strip().lower()


def sanitize_channel_list(
    raw: List[str], list_name: str
) -> Tuple[List[str], List[str]]:
    """Clean a config channel list. Returns (clean_names, warnings).

    Drops empty/whitespace-only entries, strips surrounding whitespace,
    and emits a warning for any entry containing internal whitespace
    (which Discord doesn't allow in channel names — almost always means
    two channels got mashed onto one YAML line).
    """
    clean: List[str] = []
    warnings: List[str] = []
    if not raw:
        return clean, warnings

    for i, entry in enumerate(raw):
        if entry is None:
            warnings.append(
                f"{list_name}[{i}]: empty entry (None) — dropped"
            )
            continue
        if not isinstance(entry, str):
            warnings.append(
                f"{list_name}[{i}]: non-string entry {entry!r} — dropped"
            )
            continue
        stripped = entry.strip()
        if not stripped:
            warnings.append(
                f"{list_name}[{i}]: blank/empty entry — dropped"
            )
            continue
        # Internal whitespace is invalid in Discord channel names. This
        # almost always means two channel names ended up on the same
        # YAML line by mistake.
        if any(c.isspace() for c in stripped):
            warnings.append(
                f"{list_name}[{i}]: entry {stripped!r} contains whitespace, "
                f"which is not valid in Discord channel names. "
                f"Did you mean to put each channel on its own line? "
                f"(this entry will not match any channel)"
            )
            # Keep it anyway — it just won't match anything. That's
            # less surprising than silently dropping it.
        clean.append(stripped)
    return clean, warnings


def emit_warnings(warnings: List[str]) -> None:
    for w in warnings:
        log.warning("⚠️  CONFIG: %s", w)
