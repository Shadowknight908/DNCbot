"""Map rendering engine.

Public async API (CPU-bound work runs in a thread executor):
    render_world_map(occupation, province_ownership) → PNG bytes
    render_zoom_map(iso_tag, occupation, province_ownership, ...) → PNG bytes

Layer architecture:
    Layer 0 — base choropleth: nation fills (occupied = player color, else gray)
    Layer 1 — province fills: real ArcGIS admin divisions, always visible
    Layer 2 — nation borders: drawn over provinces for clean country outlines
    Layer 3 — labels: province labels (zoom) or country labels (world)
    Layer 4 — military units (future): NATO-style markers from `units` list

Nation boundary source: ArcGIS Historic National Boundaries (Year_1970),
cached locally at map_data/world_1970.geojson.

Province source: World_Administrative_Divisions.geojson (ArcGIS admin divisions).
Province IDs are ISO_CC-ISO_SUB (e.g. "US-AK", "DE-BB").
Game tags map to ISO2 codes via TAG_TO_ISO2.
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
from matplotlib.patches import Patch
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape

import arcgis_provinces
from faction_store import FactionMembership
from map_store import OccupiedNation

log = logging.getLogger("dnc.map_renderer")

UNOCCUPIED_COLOR = "#cccccc"
_OCEAN_COLOR = "#2a2a2a"
_BORDER_COLOR = "#ffffff"
_PROVINCE_BORDER_COLOR = "#888888"
_HIGHLIGHT_EDGE = "#333333"
_FACTION_HATCHES = ("////", "\\\\\\\\", "xxxx", "....", "++++", "oooo")

_EQUAL_EARTH_CRS = "+proj=eqearth +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

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
    "RUS": "Soviet Union",
    "USSR": "Soviet Union",
    "GBR": "United Kingdom",
    "FRA": "France",
    "CHN": "China",
    "JPN": "Japan",
    "FRG": "West Germany",   # primary — wins _NAME_TO_TAG reverse lookup
    "DEW": "West Germany",
    "GDR": "East Germany",   # primary — wins _NAME_TO_TAG reverse lookup
    "DDR": "East Germany",
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

# Maps game tags → ISO 3166-1 alpha-2 codes for ArcGIS province lookups.
# Historical states map to their closest modern equivalent.
TAG_TO_ISO2: dict[str, str] = {
    # Major powers
    "USA": "US", "USSR": "RU", "GBR": "GB", "FRA": "FR", "CHN": "CN",
    "JPN": "JP", "FRG": "DE", "DEW": "DE", "GDR": "DE", "DDR": "DE",
    "ITA": "IT", "CAN": "CA", "AUS": "AU", "NZL": "NZ",
    # Europe
    "POL": "PL", "HUN": "HU", "ROM": "RO", "ROU": "RO",
    "BUL": "BG", "BGR": "BG", "CZS": "CZ", "CSK": "CZ",
    "YUG": "RS", "ALB": "AL", "GRC": "GR", "TUR": "TR",
    "SWE": "SE", "NOR": "NO", "DNK": "DK", "FIN": "FI",
    "ISL": "IS", "IRL": "IE", "NLD": "NL", "BEL": "BE",
    "LUX": "LU", "CHE": "CH", "AUT": "AT", "ESP": "ES", "PRT": "PT",
    "MCO": "MC", "AND": "AD", "SMR": "SM", "VAT": "VA", "MLT": "MT",
    "CYP": "CY",
    # Middle East / North Africa
    "ISR": "IL", "JOR": "JO", "SYR": "SY", "LBN": "LB",
    "IRQ": "IQ", "IRN": "IR", "SAU": "SA", "KWT": "KW",
    "BHR": "BH", "QAT": "QA", "TRS": "AE", "UAE": "AE",
    "YAR": "YE", "NYE": "YE", "SYE": "YE", "PDY": "YE",
    "EGY": "EG", "UAR": "EG", "LBA": "LY", "TUN": "TN",
    "ALG": "DZ", "DZA": "DZ", "MAR": "MA", "OMA": "OM", "OMN": "OM",
    # Asia
    "IND": "IN", "PAK": "PK", "BGD": "BD", "LKA": "LK", "CEY": "LK",
    "NEP": "NP", "BTN": "BT", "MDV": "MV", "MMR": "MM", "BUR": "MM",
    "THA": "TH", "KHM": "KH", "LAO": "LA",
    "VNS": "VN", "RVN": "VN", "SVN": "VN", "VND": "VN", "DRV": "VN", "NVN": "VN",
    "MYS": "MY", "SGP": "SG", "IDN": "ID", "PHL": "PH",
    "TWN": "TW", "PRK": "KP", "KPN": "KP", "NKO": "KP",
    "KOR": "KR", "SKO": "KR", "MNG": "MN", "MPR": "MN",
    "AFG": "AF", "HKG": "HK",
    # Africa
    "ETH": "ET", "ERI": "ER", "SOM": "SO", "KEN": "KE",
    "TZA": "TZ", "UGA": "UG", "RWA": "RW", "BDI": "BI",
    "MOZ": "MZ", "ZMB": "ZM", "RHO": "ZW", "ZWE": "ZW",
    "SWZ": "SZ", "LSO": "LS", "BWA": "BW", "ZAF": "ZA",
    "SWA": "NA", "NAM": "NA", "AGO": "AO",
    "COD": "CD", "ZAR": "CD", "COG": "CG", "GAB": "GA",
    "CAF": "CF", "CMR": "CM", "NGA": "NG", "GHA": "GH",
    "CIV": "CI", "LBR": "LR", "SLE": "SL", "GIN": "GN",
    "GNB": "GW", "SEN": "SN", "GMB": "GM", "MLI": "ML",
    "NER": "NE", "BFA": "BF", "UPV": "BF", "TGO": "TG",
    "BEN": "BJ", "DAH": "BJ", "GNQ": "GQ", "SDN": "SD",
    "TCD": "TD", "MDG": "MG", "MUS": "MU", "COM": "KM",
    "CPV": "CV", "STP": "ST", "DJI": "DJ", "MRT": "MR",
    # Americas
    "MEX": "MX", "GTM": "GT", "BLZ": "BZ", "BHO": "BZ",
    "HND": "HN", "SLV": "SV", "NIC": "NI", "CRI": "CR",
    "PAN": "PA", "CUB": "CU", "HTI": "HT", "DOM": "DO",
    "JAM": "JM", "TTO": "TT", "BRB": "BB", "GUY": "GY",
    "SUR": "SR", "VEN": "VE", "COL": "CO", "ECU": "EC",
    "PER": "PE", "BOL": "BO", "BRA": "BR", "PRY": "PY",
    "URY": "UY", "ARG": "AR", "CHL": "CL",
    # Oceania
    "PNG": "PG", "SLB": "SB", "FJI": "FJ", "WSM": "WS",
    "TON": "TO", "VUT": "VU",
}

# Reverse: ISO2 → list of game tags that map to it (for color lookup during render)
_ISO2_TO_TAGS: dict[str, list[str]] = {}
for _tag, _iso2 in TAG_TO_ISO2.items():
    _ISO2_TO_TAGS.setdefault(_iso2, []).append(_tag)

_WORLD_GDF: Optional[gpd.GeoDataFrame] = None
_PROVINCE_GDF: Optional[gpd.GeoDataFrame] = None


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


def _province_gdf() -> gpd.GeoDataFrame:
    """Build (once) a GeoDataFrame of all ArcGIS provinces with province_id and iso_cc columns."""
    global _PROVINCE_GDF
    if _PROVINCE_GDF is not None:
        return _PROVINCE_GDF

    provinces = arcgis_provinces.get_all()
    rows = []
    for pid, prov in provinces.items():
        try:
            geom = shape(prov.geometry)
        except Exception:
            continue
        rows.append({
            "province_id": pid,
            "iso_cc": prov.iso_cc,
            "name": prov.name,
            "cx": prov.centroid[0],
            "cy": prov.centroid[1],
            "geometry": geom,
        })
    _PROVINCE_GDF = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    log.info("Built province GeoDataFrame with %d features", len(_PROVINCE_GDF))
    return _PROVINCE_GDF


def _tags_for_country_name(name: str) -> list[str]:
    """Return all game tags whose 1970 map name matches a GeoJSON NAME."""
    stripped = name.strip()
    tags = [tag for tag, mapped in TAG_TO_NAME.items() if mapped.strip() == stripped]
    if stripped.upper() not in tags:
        tags.append(stripped.upper())
    return tags


def _name_to_color(
    name: str,
    occupation: dict[str, OccupiedNation],
) -> str:
    """Return player color if a tag maps to this country name, else gray."""
    for tag in _tags_for_country_name(name):
        if tag in occupation:
            return occupation[tag].color
    return UNOCCUPIED_COLOR


def _name_to_factions(
    name: str,
    memberships: dict[str, list[FactionMembership]],
) -> list[FactionMembership]:
    """Return faction memberships for every map tag matching this country name."""
    found: list[FactionMembership] = []
    seen: set[tuple[str, str]] = set()
    for tag in _tags_for_country_name(name):
        for membership in memberships.get(tag, []):
            key = (membership.name, membership.color)
            if key in seen:
                continue
            seen.add(key)
            found.append(membership)
    return found


def _fill_colors(
    world: gpd.GeoDataFrame,
    occupation: dict[str, OccupiedNation],
) -> list[str]:
    return [_name_to_color(name, occupation) for name in world["NAME"]]


def _faction_fill_colors(
    world: gpd.GeoDataFrame,
    memberships: dict[str, list[FactionMembership]],
) -> list[str]:
    colors = []
    for name in world["NAME"]:
        factions = _name_to_factions(name, memberships)
        colors.append(factions[0].color if factions else UNOCCUPIED_COLOR)
    return colors


def _add_province_layer(
    ax,
    province_ownership: dict,
    occupation: dict,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    fontsize: int = 4,
    show_labels: bool = False,
    crs: Optional[str] = None,
    geo_xlim: Optional[tuple[float, float]] = None,
    geo_ylim: Optional[tuple[float, float]] = None,
) -> None:
    """Draw real ArcGIS province divisions (always visible).

    Color priority per province:
      1. Explicit owner color (from province_ownership)
      2. Parent country occupation color (if the owning game tag is in occupation)
      3. UNOCCUPIED_COLOR

    On the world map labels are suppressed (show_labels=False); on zoom they show.
    A single GeoDataFrame.plot() call handles all provinces for performance.
    When crs is set the GDF is reprojected (world map); otherwise geographic
    coordinates are used with xlim/ylim spatial filtering (zoom map).
    geo_xlim/geo_ylim filter provinces by geographic centroid BEFORE reprojection
    (used on zoom maps to avoid reprojecting the full global province set).
    """
    gdf = _province_gdf()
    if gdf.empty:
        return

    if crs:
        # Geographic pre-filter before reprojection for performance
        if geo_xlim or geo_ylim:
            mask = pd.Series(True, index=gdf.index)
            if geo_xlim:
                mask &= gdf["cx"].between(geo_xlim[0], geo_xlim[1])
            if geo_ylim:
                mask &= gdf["cy"].between(geo_ylim[0], geo_ylim[1])
            gdf = gdf[mask]
        if gdf.empty:
            return
        gdf = gdf.to_crs(crs)
    elif xlim or ylim:
        # Spatial filter for zoom maps (geographic coords)
        mask = pd.Series(True, index=gdf.index)
        if xlim:
            mask &= gdf["cx"].between(xlim[0], xlim[1])
        if ylim:
            mask &= gdf["cy"].between(ylim[0], ylim[1])
        gdf = gdf[mask]
        if gdf.empty:
            return

    # Build per-row color array
    colors = []
    for _, row in gdf.iterrows():
        pid = row["province_id"]
        iso_cc = row["iso_cc"]
        own = province_ownership.get(pid)
        if own:
            colors.append(own.color)
        else:
            parent_color = UNOCCUPIED_COLOR
            for tag in _ISO2_TO_TAGS.get(iso_cc, []):
                if tag in occupation:
                    parent_color = occupation[tag].color
                    break
            colors.append(parent_color)

    gdf.plot(
        ax=ax,
        color=colors,
        edgecolor=_PROVINCE_BORDER_COLOR,
        linewidth=0.15,
        alpha=0.9,
    )

    if show_labels:
        for _, row in gdf.iterrows():
            pid = row["province_id"]
            own = province_ownership.get(pid)
            label = row["name"][:14]
            if own:
                label += f"\n{own.display_name[:12]}"
            if crs:
                centroid = row.geometry.centroid
                cx, cy = centroid.x, centroid.y
            else:
                cx, cy = row["cx"], row["cy"]
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
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    fontsize: int = 5,
) -> None:
    """Draw country-level labels for occupied nations."""
    for tag, nation in occupation.items():
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
) -> bytes:
    world = _world()
    world_proj = world.to_crs(_EQUAL_EARTH_CRS)
    # Antarctica appears visually detached in Equal Earth projection — exclude it
    world_proj = world_proj[world_proj["NAME"] != "Antarctica"]
    colors = _fill_colors(world_proj, occupation)

    fig, ax = plt.subplots(1, 1, figsize=(16, 9), facecolor=_OCEAN_COLOR)
    ax.set_facecolor(_OCEAN_COLOR)

    # Layer 0: base nation fills (Equal Earth projection)
    world_proj.plot(ax=ax, color=colors, edgecolor=_BORDER_COLOR, linewidth=0.4)

    # Layer 1: province subdivisions (always visible, no labels at world scale)
    _add_province_layer(ax, province_ownership, occupation, show_labels=False, crs=_EQUAL_EARTH_CRS)

    # Layer 2: nation borders redrawn on top for clean outlines
    world_proj.plot(ax=ax, color="none", edgecolor=_BORDER_COLOR, linewidth=0.5)

    # Layer 3: country labels (centroids computed in projected space)
    _add_labels(ax, world_proj, occupation, fontsize=5)

    ax.set_axis_off()
    ax.set_title("DNC World Map — 1970", pad=8, fontsize=16, fontweight="bold", color="white")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor=_OCEAN_COLOR)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _render_faction_world_sync(
    memberships: dict[str, list[FactionMembership]],
) -> bytes:
    world = _world()
    world_proj = world.to_crs(_EQUAL_EARTH_CRS)
    world_proj = world_proj[world_proj["NAME"] != "Antarctica"]
    colors = _faction_fill_colors(world_proj, memberships)

    fig, ax = plt.subplots(1, 1, figsize=(16, 9), facecolor=_OCEAN_COLOR)
    ax.set_facecolor(_OCEAN_COLOR)

    world_proj.plot(ax=ax, color=colors, edgecolor=_BORDER_COLOR, linewidth=0.4)

    for row_idx, row in world_proj.iterrows():
        factions = _name_to_factions(row["NAME"], memberships)
        if len(factions) < 2:
            continue
        feature = world_proj.loc[[row_idx]]
        for hatch_idx, faction in enumerate(factions[1:]):
            feature.plot(
                ax=ax,
                facecolor="none",
                edgecolor=faction.color,
                linewidth=0.15,
                hatch=_FACTION_HATCHES[hatch_idx % len(_FACTION_HATCHES)],
            )

    world_proj.plot(ax=ax, color="none", edgecolor=_BORDER_COLOR, linewidth=0.5)

    legend_items: dict[str, str] = {}
    for faction_list in memberships.values():
        for faction in faction_list:
            legend_items.setdefault(faction.name, faction.color)
    if legend_items:
        handles = [
            Patch(facecolor=color, edgecolor=_BORDER_COLOR, label=name)
            for name, color in legend_items.items()
        ]
        ncol = 1 if len(handles) <= 12 else 2
        ax.legend(
            handles=handles,
            title="Factions",
            loc="lower left",
            fontsize=8,
            title_fontsize=9,
            framealpha=0.9,
            ncol=ncol,
        )

    ax.set_axis_off()
    ax.set_title("DNC Faction Map — 1970", pad=8, fontsize=16, fontweight="bold", color="white")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor=_OCEAN_COLOR)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _geographic_center_lon(target: gpd.GeoDataFrame) -> float:
    """Area-weighted projection-centre longitude, with antimeridian adjustment.

    Handles two failure modes of simple bounding-box midpoints:
    - Countries split at ±180° (Russia, Fiji, NZ): bbox midpoint ≈ 0°.
    - Countries with many tiny remote island parts (USA Aleutians): bbox
      midpoint is pulled far from the main body by the extreme minx/maxx.

    Finds the area-weighted median longitude as an anchor, shifts any parts
    that straddle the antimeridian, then returns the area-weighted mean.
    """
    parts = target.explode(index_parts=False)
    pairs = [(float(p.centroid.x), float(p.area)) for p in parts.geometry]

    pairs_sorted = sorted(pairs, key=lambda t: t[0])
    total_area = sum(a for _, a in pairs_sorted) or 1.0

    # Area-weighted median — robust anchor for antimeridian detection
    cum = 0.0
    anchor = pairs_sorted[0][0]
    for lon, area in pairs_sorted:
        cum += area
        if cum >= total_area / 2:
            anchor = lon
            break

    # Shift parts that are >180° from the anchor (antimeridian wrap)
    adj_pairs = [
        (lon + 360 if lon < anchor - 180 else lon - 360 if lon > anchor + 180 else lon, area)
        for lon, area in pairs
    ]

    lon_0 = sum(lon * area for lon, area in adj_pairs) / total_area
    return ((lon_0 + 180) % 360) - 180


def _render_zoom_sync(
    iso_tag: str,
    occupation: dict[str, OccupiedNation],
    province_ownership: dict,
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

    # Geographic bounds in EPSG:4326
    geo_bounds = target.total_bounds  # (minx, miny, maxx, maxy)

    lat_0 = round((geo_bounds[1] + geo_bounds[3]) / 2, 4)
    lat_0 = max(-89.0, min(89.0, lat_0))  # avoid aeqd singularity at poles
    lon_0 = round(_geographic_center_lon(target), 4)

    # Local azimuthal equidistant projection centered on the country.
    # Correctly wraps antimeridian-spanning geometries (Russia, etc.).
    local_crs = (
        f"+proj=aeqd +lat_0={lat_0} +lon_0={lon_0} "
        "+datum=WGS84 +units=m +no_defs"
    )

    world_proj = world.to_crs(local_crs)
    target_proj = world_proj[world_proj["NAME"] == name]
    colors = _fill_colors(world_proj, occupation)

    # Bounds in projected meters; convert buffer from degrees (≈111 km/°)
    bounds = target_proj.total_bounds
    buf_m = buffer_deg * 111_000
    xlim = (bounds[0] - buf_m, bounds[2] + buf_m)
    ylim = (bounds[1] - buf_m, bounds[3] + buf_m)

    # Geographic pre-filter for province layer — skip lon pre-filter for
    # antimeridian-spanning countries (their cx values may scatter across ±180°)
    geo_pad = buffer_deg + 5
    geo_xlim: Optional[tuple[float, float]] = (
        max(-180.0, geo_bounds[0] - geo_pad),
        min(180.0, geo_bounds[2] + geo_pad),
    )
    geo_ylim: Optional[tuple[float, float]] = (
        max(-90.0, geo_bounds[1] - geo_pad),
        min(90.0, geo_bounds[3] + geo_pad),
    )
    if geo_bounds[2] - geo_bounds[0] > 150:
        geo_xlim = None  # antimeridian-crossing country — lat filter is enough

    fig, ax = plt.subplots(1, 1, figsize=(16, 10), facecolor=_OCEAN_COLOR)
    ax.set_facecolor(_OCEAN_COLOR)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Layer 0: base nation fills (local projection)
    world_proj.plot(ax=ax, color=colors, edgecolor=_BORDER_COLOR, linewidth=0.6)

    # Layer 1: province subdivisions with labels at zoom scale
    _add_province_layer(
        ax, province_ownership, occupation,
        fontsize=6, show_labels=True, crs=local_crs,
        geo_xlim=geo_xlim, geo_ylim=geo_ylim,
    )

    # Layer 2: highlighted target country border redrawn on top
    target_proj.plot(ax=ax, color="none", edgecolor=_HIGHLIGHT_EDGE, linewidth=2.5)

    # Layer 3: country labels (centroids in projected coords)
    _add_labels(ax, world_proj, occupation, xlim=xlim, ylim=ylim, fontsize=7)

    ax.set_axis_off()

    title = tag
    if tag in occupation:
        title += f"  —  {occupation[tag].display_name}"
    ax.set_title(title, pad=8, fontsize=16, fontweight="bold", color="white")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor=_OCEAN_COLOR)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


async def render_world_map(
    occupation: dict[str, OccupiedNation],
    province_ownership: Optional[dict] = None,
    # Legacy kwargs accepted and ignored for backward compatibility
    **_,
) -> bytes:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        partial(
            _render_world_sync,
            occupation,
            province_ownership or {},
        ),
    )


async def render_faction_map(
    memberships: dict[str, list[FactionMembership]],
    # Legacy kwargs accepted and ignored for forward compatibility
    **_,
) -> bytes:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        partial(
            _render_faction_world_sync,
            memberships,
        ),
    )


async def render_zoom_map(
    iso_tag: str,
    occupation: dict[str, OccupiedNation],
    province_ownership: Optional[dict] = None,
    buffer_deg: float = 5.0,
    # Legacy kwargs accepted and ignored for backward compatibility
    **_,
) -> bytes:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        partial(
            _render_zoom_sync,
            iso_tag,
            occupation,
            province_ownership or {},
            buffer_deg,
        ),
    )
