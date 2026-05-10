"""Extract ISO country tags from Discord display names.

Expected format:  TAG - Full Name
  TAG: 2–4 uppercase ASCII letters (ISO-3166-1 alpha-2/3 or similar)
  Separator: hyphen, en-dash, or em-dash with optional surrounding spaces

Accepted examples:
  "USA - Richard M. Nixon"  → "USA"
  "GBR - Winston Churchill" → "GBR"
  "DE - Konrad Adenauer"    → "DE"
"""
from __future__ import annotations

import re
from typing import Optional

_TAG_RE = re.compile(r'^([A-Z]{2,4})\s*[-–—]\s*')


def parse_tag(nickname: Optional[str]) -> Optional[str]:
    """Return the ISO tag prefix from a nickname, or None if absent."""
    if not nickname:
        return None
    m = _TAG_RE.match(nickname.strip())
    return m.group(1) if m else None
