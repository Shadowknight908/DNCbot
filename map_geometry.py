"""Geometry-aware helpers for the map LLM handler.

Pure-Python helpers exposed as LLM tools so the map handler can reason
about real geography instead of being handed a flat list of province IDs.

Public functions:
    lookup_province(query, max_results)
        Fuzzy-resolve a name/alias to province IDs.
    provinces_in_radius(lat, lon, distance, unit, country_filter)
        Centroid disc query — "everything within 450 mi of (lat, lon)".
    provinces_in_country(country_or_iso2, region_hint)
        List a country's provinces, optionally filtered to a directional
        quadrant ("northern", "southwest", "central", …).
    provinces_bordering(province_id)
        Provinces sharing a land boundary (lazy adjacency, sindex-backed).

All distance math is haversine on WGS84 centroids — fast and accurate
enough for game-scale radii (tens to thousands of km).
"""
from __future__ import annotations

import difflib
import json
import logging
import math
import os
import threading
from typing import Iterable, Optional

import arcgis_provinces

log = logging.getLogger("dnc.map_geometry")

_ALIAS_PATH = os.path.join("map_data", "province_aliases.json")
_EARTH_R_KM = 6371.0088
_MILES_PER_KM = 0.621371

_ALIASES: Optional[dict[str, list[str]]] = None
_NEIGHBOUR_CACHE: dict[str, list[str]] = {}
_NEIGHBOUR_LOCK = threading.Lock()


# ── Alias table ──────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum() or ch.isspace()).strip()


def _load_aliases() -> dict[str, list[str]]:
    """Lazy-load map_data/province_aliases.json.

    Format: {"Common Name": "PROV-ID"} or {"Region Name": ["PID1","PID2",...]}.
    Missing file is fine — returns an empty dict.
    """
    global _ALIASES
    if _ALIASES is not None:
        return _ALIASES
    out: dict[str, list[str]] = {}
    if os.path.exists(_ALIAS_PATH):
        try:
            with open(_ALIAS_PATH, encoding="utf-8") as f:
                raw = json.load(f)
            for k, v in raw.items():
                key = _norm(k)
                if not key:
                    continue
                if isinstance(v, str):
                    out[key] = [v.upper()]
                elif isinstance(v, list):
                    out[key] = [str(x).upper() for x in v]
        except Exception as e:
            log.warning("province_aliases.json failed to load: %s", e)
    _ALIASES = out
    log.info("Loaded %d province aliases", len(out))
    return out


# ── Distance math ────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * _EARTH_R_KM * math.asin(math.sqrt(a))


def _to_km(distance: float, unit: str) -> float:
    u = (unit or "mi").lower()
    if u in ("mi", "mile", "miles"):
        return distance / _MILES_PER_KM
    if u in ("km", "kilometer", "kilometre", "kilometers", "kilometres"):
        return float(distance)
    raise ValueError(f"unknown distance unit: {unit!r}")


def _prov_dict(pid: str, p) -> dict:
    return {
        "province_id": pid,
        "name": p.name,
        "iso_cc": p.iso_cc,
        "admin_type": p.admin_type,
        "centroid": [round(p.centroid[0], 4), round(p.centroid[1], 4)],
    }


# ── Lookup ───────────────────────────────────────────────────────────────────

def _country_meta(iso2: str) -> dict:
    """Build a 'country' meta record so the LLM can pivot to provinces_in_country."""
    provs = arcgis_provinces.get_by_iso2(iso2)
    rec: dict = {
        "kind": "country",
        "iso_cc": iso2.upper(),
        "province_count": len(provs),
    }
    if provs:
        lons = [p.centroid[0] for p in provs]
        lats = [p.centroid[1] for p in provs]
        rec["bbox"] = [round(min(lons), 3), round(min(lats), 3),
                       round(max(lons), 3), round(max(lats), 3)]
        rec["centroid"] = [round(sum(lons) / len(lons), 3),
                           round(sum(lats) / len(lats), 3)]
        rec["sample_provinces"] = [p.id for p in provs[:6]]
    from map_renderer import TAG_TO_ISO2
    rec["game_tags"] = sorted(t for t, i in TAG_TO_ISO2.items() if i == iso2.upper())
    return rec


def lookup_province(query: str, *, max_results: int = 5) -> list[dict]:
    """Resolve free-text to province IDs (or a country meta record).

    Try in order: alias table → exact ID → exact name → substring → difflib.
    Returns up to `max_results` ranked candidates. An alias that points to
    an ISO2 country code yields a single "kind": "country" record with the
    country's bbox, centroid, and game tags, so the model can follow up
    with provinces_in_country.
    """
    q = (query or "").strip()
    if not q:
        return []
    provinces = arcgis_provinces.get_all()
    if not provinces:
        return []
    qnorm = _norm(q)
    qup = q.upper()
    aliases = _load_aliases()
    found: list[tuple[str, str, float]] = []  # (pid_or_iso2, match_kind, score)
    seen: set[str] = set()

    def _push(key: str, kind: str, score: float) -> None:
        if key in seen:
            return
        if kind == "country":
            found.append((key, kind, score))
            seen.add(key)
            return
        if key not in provinces:
            return
        found.append((key, kind, score))
        seen.add(key)

    # Alias table — values may be province IDs or ISO2 country codes.
    if qnorm in aliases:
        for val in aliases[qnorm]:
            if "-" in val:
                _push(val, "alias", 1.0)
            else:
                _push(val.upper(), "country", 1.0)

    if qup in provinces:
        _push(qup, "id", 1.0)

    if len(found) < max_results:
        for pid, p in provinces.items():
            if _norm(p.name) == qnorm:
                _push(pid, "exact_name", 0.95)

    if len(found) < max_results:
        for pid, p in provinces.items():
            if qnorm and qnorm in _norm(p.name):
                _push(pid, "substring", 0.7)

    if len(found) < max_results:
        names = {pid: p.name for pid, p in provinces.items() if pid not in seen}
        close = difflib.get_close_matches(
            q, list(names.values()), n=max_results - len(found), cutoff=0.55
        )
        for name in close:
            for pid, n in names.items():
                if n == name:
                    _push(pid, "fuzzy", 0.5)
                    break

    out: list[dict] = []
    for key, match, score in found[:max_results]:
        if match == "country":
            rec = _country_meta(key)
        else:
            rec = _prov_dict(key, provinces[key])
            rec["kind"] = "province"
        rec["match"] = match
        rec["score"] = score
        out.append(rec)
    return out


# ── Radius query ─────────────────────────────────────────────────────────────

def provinces_in_radius(
    lat: float,
    lon: float,
    distance: float,
    *,
    unit: str = "mi",
    country_filter: Optional[Iterable[str]] = None,
    max_results: int = 500,
) -> list[dict]:
    """Provinces whose centroid lies within `distance` of (lat, lon).

    `country_filter` is a list of ISO2 codes (e.g. ["CN", "MN"]) — passing
    game tags also works, they're auto-mapped via TAG_TO_ISO2 if present.
    """
    radius_km = _to_km(distance, unit)

    iso_filter: Optional[set[str]] = None
    if country_filter:
        from map_renderer import TAG_TO_ISO2
        iso_filter = set()
        for c in country_filter:
            cu = str(c).upper()
            iso_filter.add(TAG_TO_ISO2.get(cu, cu))

    out: list[dict] = []
    for pid, p in arcgis_provinces.get_all().items():
        if iso_filter and p.iso_cc not in iso_filter:
            continue
        clon, clat = p.centroid
        d_km = haversine_km(lat, lon, clat, clon)
        if d_km <= radius_km:
            rec = _prov_dict(pid, p)
            rec["distance_km"] = round(d_km, 1)
            rec["distance_mi"] = round(d_km * _MILES_PER_KM, 1)
            out.append(rec)

    out.sort(key=lambda r: r["distance_km"])
    if len(out) > max_results:
        out = out[:max_results]
    return out


# ── Country + region query ───────────────────────────────────────────────────

_REGION_PREDICATES = {
    "north":      lambda lat, lon, b: lat >  (b[1] + b[3]) / 2,
    "south":      lambda lat, lon, b: lat <= (b[1] + b[3]) / 2,
    "east":       lambda lat, lon, b: lon >  (b[0] + b[2]) / 2,
    "west":       lambda lat, lon, b: lon <= (b[0] + b[2]) / 2,
    "northern":   lambda lat, lon, b: lat >  (b[1] + b[3]) / 2,
    "southern":   lambda lat, lon, b: lat <= (b[1] + b[3]) / 2,
    "eastern":    lambda lat, lon, b: lon >  (b[0] + b[2]) / 2,
    "western":    lambda lat, lon, b: lon <= (b[0] + b[2]) / 2,
    "northeast":  lambda lat, lon, b: lat >  (b[1] + b[3]) / 2 and lon >  (b[0] + b[2]) / 2,
    "northwest":  lambda lat, lon, b: lat >  (b[1] + b[3]) / 2 and lon <= (b[0] + b[2]) / 2,
    "southeast":  lambda lat, lon, b: lat <= (b[1] + b[3]) / 2 and lon >  (b[0] + b[2]) / 2,
    "southwest":  lambda lat, lon, b: lat <= (b[1] + b[3]) / 2 and lon <= (b[0] + b[2]) / 2,
    "central":    lambda lat, lon, b: (
        abs(lat - (b[1] + b[3]) / 2) <= (b[3] - b[1]) / 4
        and abs(lon - (b[0] + b[2]) / 2) <= (b[2] - b[0]) / 4
    ),
}


def _resolve_iso2(code: str) -> str:
    """Game tag or ISO2 → ISO2."""
    from map_renderer import TAG_TO_ISO2
    cu = code.upper()
    return TAG_TO_ISO2.get(cu, cu)


def provinces_in_country(
    country_or_iso2: str,
    *,
    region_hint: Optional[str] = None,
) -> list[dict]:
    """List provinces in a country with an optional directional filter."""
    iso2 = _resolve_iso2(country_or_iso2)
    provs = arcgis_provinces.get_by_iso2(iso2)
    if not provs:
        return []

    predicate = None
    bbox = None
    if region_hint:
        key = _norm(region_hint).replace(" ", "")
        # Longest-key match wins (so "northeast" beats "north")
        for k in sorted(_REGION_PREDICATES, key=len, reverse=True):
            if k in key:
                predicate = _REGION_PREDICATES[k]
                break
        if predicate is not None:
            lons = [p.centroid[0] for p in provs]
            lats = [p.centroid[1] for p in provs]
            bbox = (min(lons), min(lats), max(lons), max(lats))

    out: list[dict] = []
    for p in provs:
        clon, clat = p.centroid
        if predicate and bbox and not predicate(clat, clon, bbox):
            continue
        out.append(_prov_dict(p.id, p))
    return out


# ── Adjacency / bordering ────────────────────────────────────────────────────

def _compute_neighbours(pid: str) -> list[str]:
    """Compute (and cache) the neighbours of `pid` using the rendered GDF."""
    from map_renderer import _province_gdf
    gdf = _province_gdf()
    if gdf is None or gdf.empty:
        return []
    rows = gdf[gdf["province_id"] == pid]
    if rows.empty:
        return []
    target_geom = rows.iloc[0].geometry
    if target_geom is None or target_geom.is_empty:
        return []
    try:
        possible_idx = list(gdf.sindex.intersection(target_geom.bounds))
    except Exception:
        possible_idx = list(range(len(gdf)))

    out: list[tuple[str, float]] = []
    for idx in possible_idx:
        other = gdf.iloc[idx]
        if other["province_id"] == pid:
            continue
        og = other.geometry
        if og is None or og.is_empty:
            continue
        try:
            if og.intersects(target_geom):
                out.append((other["province_id"], float(og.distance(target_geom))))
        except Exception:
            continue
    out.sort(key=lambda x: x[1])
    return [x[0] for x in out]


def provinces_bordering(province_id: str) -> list[dict]:
    """Provinces sharing a boundary with `province_id`. Cached per call."""
    pid = (province_id or "").upper()
    target = arcgis_provinces.get_province(pid)
    if target is None:
        return []
    with _NEIGHBOUR_LOCK:
        cached = _NEIGHBOUR_CACHE.get(pid)
        if cached is None:
            cached = _compute_neighbours(pid)
            _NEIGHBOUR_CACHE[pid] = cached
    out = []
    for nid in cached:
        np = arcgis_provinces.get_province(nid)
        if np:
            out.append(_prov_dict(nid, np))
    return out


def clear_neighbour_cache() -> None:
    """For tests / data reloads — drops the lazy adjacency cache."""
    with _NEIGHBOUR_LOCK:
        _NEIGHBOUR_CACHE.clear()


# ── Convenience: structured tag → ISO2 + game-tag info ───────────────────────

def country_tag_directory() -> dict:
    """Compact directory of {iso2: [game_tags...]} for the LLM prompt.

    Useful so the model can disambiguate "Germany" → ["FRG", "GDR"] etc.
    without us shipping the full province list.
    """
    from map_renderer import TAG_TO_ISO2
    out: dict[str, list[str]] = {}
    for tag, iso2 in TAG_TO_ISO2.items():
        out.setdefault(iso2, []).append(tag)
    for iso2 in out:
        out[iso2].sort()
    return out


def default_tag_for_province(province_id: str) -> Optional[str]:
    """Heuristic: pick the most plausible game tag for a province's location.

    Splits divided countries (Germany, Korea, Vietnam, Yemen, China/Taiwan)
    by centroid. Defaults to the first listed tag otherwise.
    """
    from map_renderer import TAG_TO_ISO2
    prov = arcgis_provinces.get_province(province_id.upper())
    if not prov:
        return None
    iso2 = prov.iso_cc
    tags = [t for t, i in TAG_TO_ISO2.items() if i == iso2]
    if not tags:
        return None
    if len(tags) == 1:
        return tags[0]
    lon, lat = prov.centroid
    # Divided-country rules (1970-era splits).
    if iso2 == "DE":  # Berlin meridian ~13.4°E roughly splits FRG / GDR.
        return "GDR" if lon > 11.5 else "FRG"
    if iso2 == "KP" or iso2 == "KR":
        # ArcGIS may store ROK/DPRK under separate ISO2; if collapsed, split by 38°.
        return "PRK" if lat > 38.0 else "KOR"
    if iso2 == "VN":
        return "DRV" if lat > 17.0 else "RVN"  # 17th parallel.
    if iso2 == "YE":
        return "YAR" if lat > 14.5 else "PDY"
    # Generic fallback: stable alphabetical first.
    return sorted(tags)[0]
