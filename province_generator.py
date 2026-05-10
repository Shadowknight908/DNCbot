"""Auto-subdivision of country polygons into numbered provinces.

A bounding-box grid is clipped to the actual country boundary using shapely
(available transitively via geopandas). No extra dependencies required.

Standalone test:
    python province_generator.py USSR 15
"""
from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass

from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

log = logging.getLogger("dnc.province_generator")


@dataclass
class Province:
    id: str             # e.g. "USSR-01"
    name: str           # e.g. "USSR Province 1"
    parent_tag: str     # e.g. "USSR"
    geometry: dict      # GeoJSON geometry dict (Polygon / MultiPolygon)
    centroid: tuple     # (lon, lat)


def subdivide_country(
    geojson_feature: dict,
    tag: str,
    n: int,
) -> list[Province]:
    """Subdivide one GeoJSON feature into up to N clipped grid-cell provinces.

    The bounding box is divided into ceil(sqrt(n)) cols × ceil(n/cols) rows.
    Each cell is intersected with the country polygon; empty/tiny cells are
    discarded. Survivors are sorted north-to-south, west-to-east, then
    assigned zero-padded IDs like TAG-01 … TAG-NN.
    """
    tag = tag.upper()
    country_geom = shape(geojson_feature["geometry"])
    minx, miny, maxx, maxy = country_geom.bounds

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    cell_w = (maxx - minx) / cols
    cell_h = (maxy - miny) / rows

    cells: list[tuple[float, float, object]] = []  # (lat, lon, geom)
    for row in range(rows):
        for col in range(cols):
            cell_minx = minx + col * cell_w
            cell_miny = miny + row * cell_h
            cell = box(cell_minx, cell_miny, cell_minx + cell_w, cell_miny + cell_h)
            clipped = country_geom.intersection(cell)
            if clipped.is_empty or clipped.area < 1e-6:
                continue
            c = clipped.centroid
            cells.append((c.y, c.x, clipped))

    # North-to-south, west-to-east ordering for intuitive numbering
    cells.sort(key=lambda c: (-c[0], c[1]))

    pad = max(2, len(str(len(cells))))
    provinces: list[Province] = []
    for i, (lat, lon, geom) in enumerate(cells, start=1):
        pid = f"{tag}-{str(i).zfill(pad)}"
        provinces.append(Province(
            id=pid,
            name=f"{tag} Province {i}",
            parent_tag=tag,
            geometry=mapping(geom),
            centroid=(lon, lat),
        ))
    return provinces


def merge_and_subdivide(
    geojson_features: list[dict],
    tag: str,
    n: int,
) -> list[Province]:
    """Merge multiple GeoJSON features (e.g. island nations) then subdivide."""
    if not geojson_features:
        raise ValueError(f"No GeoJSON features supplied for tag '{tag}'")
    merged = unary_union([shape(f["geometry"]) for f in geojson_features])
    return subdivide_country({"geometry": merged.__geo_interface__}, tag, n)


if __name__ == "__main__":
    import os

    tag = sys.argv[1].upper() if len(sys.argv) > 1 else "USSR"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    geojson_path = os.path.join("map_data", "world_1970.geojson")
    with open(geojson_path, encoding="utf-8") as f:
        fc = json.load(f)

    sys.path.insert(0, ".")
    from map_renderer import TAG_TO_NAME  # noqa: E402

    target_name = TAG_TO_NAME.get(tag, "").strip()
    if not target_name:
        print(f"Tag '{tag}' not found in TAG_TO_NAME")
        sys.exit(1)

    matching = [
        feat for feat in fc["features"]
        if feat["properties"].get("NAME", "").strip() == target_name
    ]
    if not matching:
        print(f"No GeoJSON features found for country name '{target_name}'")
        sys.exit(1)

    provinces = merge_and_subdivide(matching, tag, n)
    print(f"Generated {len(provinces)} provinces for {tag} ({target_name}):")
    for p in provinces:
        print(f"  {p.id}  centroid=({p.centroid[0]:.2f}°E, {p.centroid[1]:.2f}°N)")
