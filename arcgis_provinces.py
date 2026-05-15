"""Real administrative division data from World_Administrative_Divisions.geojson.

Province IDs use the format ISO_CC-ISO_SUB (e.g. "US-AK" for Alaska,
"DE-BB" for Brandenburg). Multiple GeoJSON features sharing the same ISO_CODE
are merged into a single geometry via unary_union.

First load: processes the raw ArcGIS file and writes a merged cache to
map_data/arcgis_provinces_merged.geojson (~2 min, one-time).
Subsequent loads: reads the cache (~1 s).
"""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass

log = logging.getLogger("dnc.arcgis_provinces")

_ARCGIS_PATH = "World_Administrative_Divisions.geojson"
_CACHE_PATH = os.path.join("map_data", "arcgis_provinces_merged.geojson")


@dataclass
class Province:
    id: str          # e.g. "US-AK"
    name: str        # e.g. "Alaska"
    iso_cc: str      # 2-letter country code, e.g. "US"
    iso_sub: str     # subdivision code, e.g. "AK"
    admin_type: str  # e.g. "State"
    geometry: dict   # GeoJSON geometry dict (merged, ready for shapely.shape())
    centroid: tuple  # (lon, lat)


_BY_ID: dict[str, Province] | None = None
_BY_ISO2: dict[str, list[Province]] | None = None


def _build_cache() -> None:
    """Merge multi-part provinces and write map_data/arcgis_provinces_merged.geojson."""
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union

    log.info("Building ArcGIS province cache (one-time, ~1-2 min)…")
    with open(_ARCGIS_PATH, encoding="utf-8") as f:
        fc = json.load(f)

    groups: dict[str, list[dict]] = defaultdict(list)
    for feat in fc["features"]:
        props = feat["properties"]
        iso_code = props.get("ISO_CODE", "").strip()
        if iso_code:
            groups[iso_code].append(feat)

    out_features = []
    for iso_code, feats in groups.items():
        props = feats[0]["properties"]
        iso_cc = props.get("ISO_CC", "").strip()
        iso_sub = props.get("ISO_SUB", "").strip()
        name = props.get("NAME", iso_code).strip()
        admin_type = props.get("ADMINTYPE", "").strip()
        if not iso_cc or not iso_sub:
            continue
        try:
            geoms = [shape(f["geometry"]) for f in feats]
            merged = unary_union(geoms) if len(geoms) > 1 else geoms[0]
            cx, cy = float(merged.centroid.x), float(merged.centroid.y)
            # Simplify to ~0.05° tolerance — invisible at map scale, ~10x smaller file
            simplified = merged.simplify(0.05, preserve_topology=True)
            geom_dict = mapping(simplified)
        except Exception as e:
            log.debug("Skipping %s: %s", iso_code, e)
            continue

        out_features.append({
            "type": "Feature",
            "geometry": geom_dict,
            "properties": {
                "province_id": f"{iso_cc}-{iso_sub}",
                "name": name,
                "iso_cc": iso_cc,
                "iso_sub": iso_sub,
                "admin_type": admin_type,
                "cx": cx,
                "cy": cy,
            },
        })

    os.makedirs(os.path.dirname(os.path.abspath(_CACHE_PATH)), exist_ok=True)
    tmp = _CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": out_features}, f)
    os.replace(tmp, _CACHE_PATH)
    log.info("Province cache written: %d provinces", len(out_features))


def _load() -> tuple[dict[str, Province], dict[str, list[Province]]]:
    if not os.path.exists(_ARCGIS_PATH) and not os.path.exists(_CACHE_PATH):
        log.warning("ArcGIS province data not found (%s)", _ARCGIS_PATH)
        return {}, {}

    if not os.path.exists(_CACHE_PATH):
        _build_cache()

    with open(_CACHE_PATH, encoding="utf-8") as f:
        fc = json.load(f)

    by_id: dict[str, Province] = {}
    by_iso2: dict[str, list[Province]] = defaultdict(list)

    for feat in fc["features"]:
        p = feat["properties"]
        pid = p["province_id"]
        prov = Province(
            id=pid,
            name=p["name"],
            iso_cc=p["iso_cc"],
            iso_sub=p["iso_sub"],
            admin_type=p["admin_type"],
            geometry=feat["geometry"],
            centroid=(p["cx"], p["cy"]),
        )
        by_id[pid] = prov
        by_iso2[p["iso_cc"]].append(prov)

    log.info("Loaded %d provinces from cache", len(by_id))
    return dict(by_id), dict(by_iso2)


def _ensure() -> None:
    global _BY_ID, _BY_ISO2
    if _BY_ID is None:
        _BY_ID, _BY_ISO2 = _load()


def get_all() -> dict[str, Province]:
    _ensure()
    return _BY_ID  # type: ignore[return-value]


def get_by_iso2(iso2: str) -> list[Province]:
    _ensure()
    return (_BY_ISO2 or {}).get(iso2.upper(), [])


def get_province(province_id: str) -> Province | None:
    _ensure()
    return (_BY_ID or {}).get(province_id.upper())
