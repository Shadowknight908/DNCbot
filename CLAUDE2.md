# Geopolitical Discord War Game — Project Architecture

## Project Overview

A mostly-automated, real-time geopolitical war game engine built on top of an existing Discord bot with an LLM ingestion pipeline. Players claim countries via Discord nickname tags, post military actions in natural language, and an automated system parses those posts, updates province ownership, renders NATO unit symbols, and generates NPC nation responses — all reflected on a live-updating world map.

---

## Core Design Principles

- The **map is the source of truth** for all territorial ownership
- The **GM is a human override layer**, not a micromanager — the system handles bookkeeping
- **NPC nations respond intelligently** but cannot call allies (hard constraint for simplicity)
- **Province-based territory** — discrete regions rather than freehand borders
- **Contested state** — if the LLM is uncertain about ownership, province is marked contested rather than forcing a binary decision
- **Prototype first on modern map**, swap for historical 1971 map once system is validated

---

## Tech Stack (Recommended)

| Layer | Tool |
|---|---|
| Discord Bot | discord.py or discord.js |
| LLM Pipeline | Existing ingestion pipeline (already in place) |
| Map Rendering | D3.js or Leaflet |
| NATO Symbology | milsymbol.js (MIL-STD-2525 standard) |
| Province Data (prototype) | Natural Earth admin-1 boundaries (GeoJSON) |
| Province Data (final) | Custom QGIS-authored historical GeoJSON |
| Province Database | Simple key-value or relational DB (province → owner) |
| Backend | Lightweight server to receive bot events and trigger map re-renders |

---

## System Components

### 1. Country Tag Parser
- Reads Discord guild member nicknames
- Extracts three-letter country tag (e.g. `[USA]`, `GER`, `URS`)
- Maps tag to a country in the province ownership database
- Triggers map update when a user is assigned or leaves

### 2. Province Ownership Database
- Schema: `province_id → { name, owner_tag, status }`
- Status values: `owned | contested | unclaimed`
- Updated by:
  - GM/mod bot commands (e.g. `/occupy [province] [tag]`)
  - LLM post parser (automated)
  - Season reset (admin redraws baseline)

### 3. LLM Post Parser
- Monitors designated Discord channels for military action posts
- Extracts from natural language:
  - **Unit identity** (e.g. "3rd Guards Tank Army")
  - **Unit type** (armor, infantry, artillery, naval, air)
  - **Nationality** → maps to country tag
  - **Location** (resolved against province/gazetteer index)
  - **Action** (advancing, holding, retreating, attacking)
- If parsing is ambiguous → mark province as `contested`, flag for GM review
- Confidence threshold system to avoid bad automated updates

### 4. Named Location Gazetteer
- Index of every province name, major city, geographic feature, and historical military reference point
- Used by LLM parser to resolve location strings to province IDs or coordinates
- Critical for things like "crosses the Fulda Gap" → resolves to a specific province/coordinate
- Built alongside the province map

### 5. NATO Symbology Renderer
- Uses **milsymbol.js** to generate MIL-STD-2525 compliant unit icons
- LLM classifies unit type and affiliation → generates SIDC code → renders symbol
- Symbols placed at province centroid or specified coordinate
- Renders on top of map layer, updates in real time with posts

### 6. Map Renderer
- Renders province GeoJSON as a colored world map
- Province colors reflect ownership (by country color) or contested state (neutral/middle color)
- Overlays NATO unit symbols
- Re-renders and posts updated map image to a designated Discord channel on any update
- Hosted as a web dashboard, updated via backend events

### 7. NPC Nation Response System
- Triggered when a province belonging to a non-player-controlled nation is attacked
- Context fed to LLM:
  - Defender's current military strength (from player-maintained records)
  - Geographic defensibility of contested province
  - Attacking force as described in the triggering post
  - Prior post history / attrition state
- Output:
  - Narrative flavor text posted to Discord (e.g. *"Romanian forces establish a defensive line along the Carpathians..."*)
  - Province status update (contested, fallback, etc.)
- **Hard constraints:**
  - NPC nations cannot call allies
  - GM can override any NPC response

### 8. Military Strength Ledger
- Nations are required to post their military strength
- LLM reads and maintains an updated record based on posts
- Fed as context into NPC response system
- No manual GM input required for routine updates

---

## Data Flow

```
Discord Post
     │
     ▼
LLM Ingestion Pipeline
     │
     ├─► Unit Type + Nationality → NATO Symbol → Map Overlay
     │
     ├─► Location → Province ID (via Gazetteer)
     │
     └─► Action + Confidence
              │
              ├─► High confidence → Update Province Ownership DB
              │
              ├─► Low confidence → Mark Contested, Flag GM
              │
              └─► NPC nation? → NPC Response LLM → Flavor Post + Province Update
                       │
                       ▼
                  Map Re-renders → Posted to Discord
```

---

## Map Data Strategy

### Prototype Phase
- Use **Natural Earth admin-1 boundaries** (modern, free, GeoJSON-ready)
- Download from naturalearthdata.com
- No editing required — load and go
- Validates entire pipeline before historical work begins

### Final Phase
- Use **QGIS** to author custom historical province layer
- Start from Natural Earth or GADM data
- Merge/dissolve provinces where game granularity doesn't require detail
- Redraw borders where 1971 accuracy demands it:
  - East Germany / West Germany (DDR / BRD)
  - North Vietnam / South Vietnam
  - North Yemen / South Yemen
  - USSR as unified entity
  - Rhodesia (not Zimbabwe)
  - Bangladesh (independence declared 1971, war ongoing)
  - etc.
- Export as GeoJSON → drop-in replacement for prototype data
- Everything else in the system stays identical

### Province Granularity Principle
- Major powers / likely conflict zones → finer subdivision
- Minor / peripheral regions → coarser subdivision
- Granularity should support meaningful front-line movement in wars adjudicated by GM
- Roughly inspired by historical military district / army group level

---

## Bot Commands (Draft)

| Command | Action |
|---|---|
| `/occupy [province] [tag]` | Assign province to a country |
| `/contest [province]` | Mark province as contested |
| `/cede [province] [from] [to]` | Transfer province between players |
| `/mappost` | Force a map render and post to channel |
| `/season_reset` | Load new baseline GeoJSON for new season |

---

## Build Order

1. **Province map + GeoJSON** — modern Natural Earth data, load as-is for prototype
2. **Basic map renderer** — D3.js/Leaflet, color provinces by owner tag
3. **Discord bot + province DB** — nickname parser, manual `/occupy` commands, map posts
4. **LLM post parser** — unit extraction, location resolution, province updates
5. **NATO symbology layer** — milsymbol.js, unit rendering on map
6. **NPC response system** — defensive response generation, flavor text posting
7. **Historical map swap** — replace GeoJSON with QGIS-authored 1971 layer

---

## Open Questions / Future Decisions

- Exact confidence threshold for LLM province auto-update vs flagging GM
- How GM override is surfaced (bot command, web dashboard, Discord reaction?)
- Whether unit symbols persist on map or clear each round
- How seasons are structured and what resets between them
- Web dashboard vs pure Discord-channel map posts
