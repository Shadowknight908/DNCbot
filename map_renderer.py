"""Map rendering engine.

Public async API (CPU-bound work runs in a thread executor):
    render_world_map(occupation, province_ownership, province_definitions, units=[]) → PNG bytes
    render_zoom_map(iso_tag, occupation, province_ownership, province_definitions, ...) → PNG bytes

Layer architecture:
    Layer 0 — base choropleth: nation fills (occupied = player color, else gray)
    Layer 1 — province fills: province cells drawn on top for subdivided nations
    Layer 2 — labels: province labels (subdivided nations) or country labels (whole nations)
    Layer 3 — military units (future): NATO-style markers from `units` list

Geometry source: ArcGIS Historic National Boundaries (Year_1970 layer),
cached locally at map_data/world_1970.geojson.
"""
from __future__ import annotations

import asyncio
import logging
import os
import urllib.request
from functools import partial
from io import BytesIO
from typing import Optional

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import shape

from map_store import OccupiedNation

log = logging.getLogger("dnc.map_renderer")

UNOCCUPIED_COLOR = "#cccccc"
_OCEAN_COLOR = "#a8d5e2"
_BORDER_COLOR = "#ffffff"
_PROVINCE_BORDER_COLOR = "#888888"
_HIGHLIGHT_EDGE = "#333333"

_DATA_DIR = "map_data"
_GEOJSON_PATH = os.path.join(_DATA_DIR, "world_1970.geojson")
_ARCGIS_URL = (
    "https://services.arcgis.com/8df8p0NlLFEShl0r/arcgis/rest/services"
    "/Historic_National_Boundaries_NEW/FeatureServer/2/query"
    "?where=1%3D1&outFields=*&f=geojson"
)

# Maps 2–4 letter occupation tags → NAME values in the 1970 GeoJSON.
# Covers the principal 1970-era powers and historical country names.
TAG_TO_NAME: dict[str, str] = {
    # Major powers
    "USA": "United States",
    "USSR": "Soviet Union",
    "GBR": "United Kingdom",
    "FRA": "France",
    "CHN": "China",
    "JPN": "Japan",
    "DEW": "West Germany",
    "FRG": "West Germany",
    "DDR": "East Germany",
    "GDR": "East Germany",
    "ITA": "Italy",
    "CAN": "Canada",
    "AUS": "Australia",
    "NZL": "New Zealand",
    # Europe
    "POL": "Poland",
    "HUN": "Hungary",
    "ROM": "Romania",
    "ROU": "Romania",
    "BUL": "Bulgaria",
    "BGR": "Bulgaria",
    "CZS": "Czechoslovakia",
    "CSK": "Czechoslovakia",
    "YUG": "Yugoslavia",
    "ALB": "Albania",
    "GRC": "Greece",
    "TUR": "Turkey",
    "SWE": "Sweden",
    "NOR": "Norway",
    "DNK": "Denmark",
    "FIN": "Finland",
    "ISL": "Iceland",
    "IRL": "Ireland",
    "NLD": "Netherlands",
    "BEL": "Belgium",
    "LUX": "Luxembourg",
    "CHE": "Switzerland",
    "AUT": "Austria",
    "ESP": "Spain",
    "PRT": "Portugal",
    "MCO": "Monaco",
    "AND": "Andorra",
    "SMR": "San Marino",
    "VAT": "Vatican City",
    "MLT": "Malta",
    "CYP": "Cyprus",
    # Middle East / North Africa
    "ISR": "Israel",
    "JOR": "Jordan",
    "SYR": "Syria",
    "LBN": "Lebanon",
    "IRQ": "Iraq",
    "IRN": "Iran",
    "SAU": "Saudi Arabia",
    "KWT": "Kuwait",
    "BHR": "Bahrain",
    "QAT": "Qatar",
    "TRS": "Trucial States",
    "UAE": "Trucial States",
    "YAR": "North Yemen",
    "NYE": "North Yemen",
    "SYE": "Southern Yemen",
    "PDY": "Southern Yemen",
    "EGY": "United Arab Republic",
    "UAR": "United Arab Republic",
    "LBA": "Libya",
    "TUN": "Tunisia",
    "ALG": "Algeria",
    "DZA": "Algeria",
    "MAR": "Morocco",
    "OMA": "Oman",
    "OMN": "Oman",
    # Asia
    "IND": "India",
    "PAK": "Pakistan",
    "BGD": "Pakistan",   # Bangladesh was East Pakistan in 1970
    "LKA": "Ceylon",
    "CEY": "Ceylon",
    "NEP": "Nepal",
    "BTN": "Bhutan",
    "MDV": "Maldives",
    "MMR": "Burma",
    "BUR": "Burma",
    "THA": "Thailand",
    "KHM": "Cambodia",
    "LAO": "Laos",
    "VNS": "South Vietnam",
    "RVN": "South Vietnam",
    "SVN": "South Vietnam",
    "VND": "North Vietnam",
    "DRV": "North Vietnam",
    "NVN": "North Vietnam",
    "MYS": "Malaysia",
    "SGP": "Singapore",
    "IDN": "Indonesia",
    "PHL": "Philippines",
    "TWN": "Taiwan",
    "PRK": "North Korea",
    "KPN": "North Korea",
    "NKO": "North Korea",
    "KOR": "South Korea",
    "SKO": "South Korea",
    "MNG": "Mongolian Republic",
    "MPR": "Mongolian Republic",
    "AFG": "Afghanistan",
    "HKG": "Hong Kong",
    "MAC": "Macau",
    "BRN": "Brunei ",   # trailing space matches the GeoJSON value
    "TMP": "Timor",
    "TLS": "Timor",
    # Africa
    "ETH": "Ethiopia",
    "SOM": "Somali Republic",
    "KEN": "Kenya",
    "TZA": "Tanzania",
    "UGA": "Uganda",
    "RWA": "Rwanda",
    "BDI": "Burundi",
    "MOZ": "Mozambique",
    "ZMB": "Zambia",
    "RHO": "Rhodesia",
    "ZWE": "Rhodesia",
    "SWZ": "Swaziland",
    "LSO": "Lesotho",
    "BWA": "Botswana",
    "ZAF": "South Africa",
    "SWA": "South-West Africa",
    "NAM": "South-West Africa",
    "AGO": "Angola",
    "COD": "Congo DRC",
    "ZAR": "Congo DRC",
    "COG": "Congo",
    "GAB": "Gabon",
    "CAF": "Central African Republic",
    "CMR": "Cameroon",
    "NGA": "Nigeria",
    "GHA": "Ghana",
    "CIV": "Ivory Coast",
    "LBR": "Liberia",
    "SLE": "Sierra Leone",
    "GIN": "Guinea",
    "GNB": "Portuguese Guinea",
    "PGU": "Portuguese Guinea",
    "SEN": "Senegal",
    "GMB": "Gambia",
    "MLI": "Mali",
    "NER": "Niger",
    "BFA": "Upper Volta",
    "UPV": "Upper Volta",
    "TGO": "Togo",
    "BEN": "Dahomey",
    "DAH": "Dahomey",
    "GNQ": "Equatorial Guinea",
    "SDN": "Sudan",
    "TCD": "Chad",
    "MDG": "Madagascar",
    "MUS": "Mauritius",
    "COM": "Comoros",
    "CPV": "Cabo Verde",
    "STP": "Sao Tome and Principe",
    "DJI": "Djibouti",
    "ERI": "Ethiopia",   # Eritrea was part of Ethiopia in 1970
    "MRT": "Mauritania",
    # Americas
    "MEX": "Mexico",
    "GTM": "Guatemala",
    "BLZ": "British Honduras",
    "BHO": "British Honduras",
    "HND": "Honduras",
    "SLV": "El Salvador",
    "NIC": "Nicaragua",
    "CRI": "Costa Rica",
    "PAN": "Panama",
    "CUB": "Cuba",
    "HTI": "Haiti",
    "DOM": "Dominican Republic",
    "JAM": "Jamaica",
    "TTO": "Trinidad and Tobago",
    "BRB": "Barbados",
    "GUY": "Guyana",
    "SUR": "Suriname",
    "VEN": "Venezuela",
    "COL": "Colombia",
    "ECU": "Ecuador",
    "PER": "Peru",
    "BOL": "Bolivia",
    "BRA": "Brazil",
    "PRY": "Paraguay",
    "URY": "Uruguay",
    "ARG": "Argentina",
    "CHL": "Chile",
    # Oceania
    "PNG": "Papua New Guinea",
    "SLB": "Solomon Islands",
    "FJI": "Fiji",
    "WSM": "Samoa",
    "TON": "Tonga",
    "KIR": "Kiribati",
    "NRU": "Nauru",
    "TUV": "Tuvalu",
    "VUT": "Vanuatu",
    "NCL": "New Caledonia",
}

# Reverse lookup: NAME → tag (first-registered wins; used for zoom by name)
_NAME_TO_TAG: dict[str, str] = {}
for _t, _n in TAG_TO_NAME.items():
    _NAME_TO_TAG.setdefault(_n.strip(), _t)

_WORLD_GDF: Optional[gpd.GeoDataFrame] = None


def _load_world_gdf() -> gpd.GeoDataFrame:
    """Load 1970 boundary GeoJSON, downloading it if not cached."""
    if not os.path.exists(_GEOJSON_PATH):
        log.info("Downloading 1970 world boundary data (one-time)…")
        os.makedirs(_DATA_DIR, exist_ok=True)
        tmp = _GEOJSON_PATH + ".tmp"
        try:
            urllib.request.urlretrieve(_ARCGIS_URL, tmp)
            os.replace(tmp, _GEOJSON_PATH)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    gdf = gpd.read_file(_GEOJSON_PATH)
    # Normalise: strip trailing spaces from NAME
    gdf["NAME"] = gdf["NAME"].str.strip()
    return gdf


def _world() -> gpd.GeoDataFrame:
    global _WORLD_GDF
    if _WORLD_GDF is None:
        _WORLD_GDF = _load_world_gdf()
    return _WORLD_GDF


def _name_to_color(
    name: str,
    occupation: dict[str, OccupiedNation],
) -> str:
    """Return player color if a tag maps to this country name, else gray."""
    tag = _NAME_TO_TAG.get(name)
    if tag and tag in occupation:
        return occupation[tag].color
    if name.upper() in occupation:
        return occupation[name.upper()].color
    return UNOCCUPIED_COLOR


def _fill_colors(
    world: gpd.GeoDataFrame,
    occupation: dict[str, OccupiedNation],
) -> list[str]:
    return [_name_to_color(name, occupation) for name in world["NAME"]]


def _add_province_layer(
    ax,
    province_definitions: dict,
    province_ownership: dict,
    subdivided_tags: set[str],
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    fontsize: int = 4,
) -> None:
    """Draw province cells (Layer 1) and their labels (Layer 2a) for subdivided nations."""
    if not province_definitions:
        return

    for pid, prov in province_definitions.items():
        tag = prov.parent_tag
        if tag not in subdivided_tags:
            continue

        ownership = province_ownership.get(pid)
        fill = ownership.color if ownership else UNOCCUPIED_COLOR

        try:
            geom = shape(prov.geometry)
        except Exception:
            continue

        # Clip check: skip if centroid is outside view bounds
        cx, cy = prov.centroid
        if xlim and not (xlim[0] <= cx <= xlim[1]):
            continue
        if ylim and not (ylim[0] <= cy <= ylim[1]):
            continue

        prov_gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        prov_gdf.plot(
            ax=ax,
            color=fill,
            edgecolor=_PROVINCE_BORDER_COLOR,
            linewidth=0.3,
            alpha=0.85,
        )

        # Province label: ID on first line, player name on second if owned
        label = pid
        if ownership:
            label += f"\n{ownership.display_name[:12]}"
        ax.annotate(
            label,
            xy=(cx, cy),
            ha="center", va="center",
            fontsize=fontsize, color="#111111",
            bbox=dict(boxstyle="round,pad=0.08", fc="white", alpha=0.4, ec="none"),
        )


def _add_labels(
    ax,
    world: gpd.GeoDataFrame,
    occupation: dict[str, OccupiedNation],
    subdivided_tags: Optional[set[str]] = None,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    fontsize: int = 5,
) -> None:
    """Draw country-level labels (Layer 2b). Skips nations that are subdivided into provinces."""
    skip = subdivided_tags or set()
    for tag, nation in occupation.items():
        if tag in skip:
            continue
        name = TAG_TO_NAME.get(tag)
        if name is None:
            continue
        subset = world[world["NAME"] == name.strip()]
        if subset.empty:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            centroid = subset.geometry.centroid.iloc[0]
        if xlim and not (xlim[0] <= centroid.x <= xlim[1]):
            continue
        if ylim and not (ylim[0] <= centroid.y <= ylim[1]):
            continue
        ax.annotate(
            f"{tag}\n{nation.display_name[:14]}",
            xy=(centroid.x, centroid.y),
            ha="center", va="center",
            fontsize=fontsize, color="#111111", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.45, ec="none"),
        )


def _render_world_sync(
    occupation: dict[str, OccupiedNation],
    province_ownership: dict,
    province_definitions: dict,
    subdivided_tags: set[str],
    units: list,
) -> bytes:
    world = _world()
    colors = _fill_colors(world, occupation)

    fig, ax = plt.subplots(1, 1, figsize=(22, 11), facecolor=_OCEAN_COLOR)
    ax.set_facecolor(_OCEAN_COLOR)
    world.plot(ax=ax, color=colors, edgecolor=_BORDER_COLOR, linewidth=0.4)

    _add_province_layer(ax, province_definitions, province_ownership, subdivided_tags, fontsize=4)
    _add_labels(ax, world, occupation, subdivided_tags=subdivided_tags, fontsize=5)

    ax.set_axis_off()
    ax.set_title("DNC World Map — 1970", pad=8, fontsize=16, fontweight="bold")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=_OCEAN_COLOR)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _render_zoom_sync(
    iso_tag: str,
    occupation: dict[str, OccupiedNation],
    province_ownership: dict,
    province_definitions: dict,
    subdivided_tags: set[str],
    units: list,
    buffer_deg: float,
) -> bytes:
    world = _world()
    tag = iso_tag.upper()
    name = TAG_TO_NAME.get(tag)
    if name is None:
        raise ValueError(f"Tag '{iso_tag}' has no 1970 map entry; update TAG_TO_NAME")
    name = name.strip()
    target = world[world["NAME"] == name]
    if target.empty:
        raise ValueError(f"Country '{name}' not found in 1970 map dataset")

    bounds = target.total_bounds  # (minx, miny, maxx, maxy)
    xlim = (bounds[0] - buffer_deg, bounds[2] + buffer_deg)
    ylim = (bounds[1] - buffer_deg, bounds[3] + buffer_deg)
    colors = _fill_colors(world, occupation)

    fig, ax = plt.subplots(1, 1, figsize=(14, 9), facecolor=_OCEAN_COLOR)
    ax.set_facecolor(_OCEAN_COLOR)
    world.plot(ax=ax, color=colors, edgecolor=_BORDER_COLOR, linewidth=0.6)

    target_color = occupation[tag].color if tag in occupation else UNOCCUPIED_COLOR
    target.plot(ax=ax, color=target_color, edgecolor=_HIGHLIGHT_EDGE, linewidth=2.5)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    _add_province_layer(
        ax, province_definitions, province_ownership, subdivided_tags,
        xlim=xlim, ylim=ylim, fontsize=6,
    )
    _add_labels(
        ax, world, occupation,
        subdivided_tags=subdivided_tags,
        xlim=xlim, ylim=ylim, fontsize=7,
    )
    ax.set_axis_off()

    title = tag
    if tag in occupation:
        title += f"  —  {occupation[tag].display_name}"
    ax.set_title(title, pad=8, fontsize=16, fontweight="bold")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=_OCEAN_COLOR)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


async def render_world_map(
    occupation: dict[str, OccupiedNation],
    province_ownership: Optional[dict] = None,
    province_definitions: Optional[dict] = None,
    subdivided_tags: Optional[set[str]] = None,
    units: Optional[list] = None,
) -> bytes:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        partial(
            _render_world_sync,
            occupation,
            province_ownership or {},
            province_definitions or {},
            subdivided_tags or set(),
            units or [],
        ),
    )


async def render_zoom_map(
    iso_tag: str,
    occupation: dict[str, OccupiedNation],
    province_ownership: Optional[dict] = None,
    province_definitions: Optional[dict] = None,
    subdivided_tags: Optional[set[str]] = None,
    units: Optional[list] = None,
    buffer_deg: float = 5.0,
) -> bytes:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        partial(
            _render_zoom_sync,
            iso_tag,
            occupation,
            province_ownership or {},
            province_definitions or {},
            subdivided_tags or set(),
            units or [],
            buffer_deg,
        ),
    )
