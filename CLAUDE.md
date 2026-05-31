# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DNC Lore Bot** is a Discord roleplay/nation-sim bot (v0.8) that automatically archives roleplay messages into a local vector store and answers lore queries in character. It is powered by OpenRouter (or any OpenAI-compatible API) for chat, vision, and embeddings, and uses ChromaDB for persistent memory. Beyond lore archiving it provides GM adjudication, a thread-based **war system**, an **espionage** layer, and a full **spatial mapping** system (world/zoom/faction maps with real province ownership).

A separate `EMBEDDING_API_KEY` can route embeddings through a different provider; `TAVILY_API_KEY` enables model-driven web search in chat modes.

### Core message pipeline

Non-command messages in scanned channels are buffered (consecutive same-author posts grouped) and passed to the **Archivist** LLM, which extracts a lore summary + tags and stores it as an embedding in ChromaDB. Command messages (`!DNC …`) are routed to the relevant handler; a bare `!DNC <text>` falls back to a semantic **Chronicler** lore query.

### Feature summary (v0.8)

- Multi-turn conversations via reply-chain reconstruction (chat / chatuc / query / gm modes)
- Role-based permissions: `admin`, `gm`, `chatuc`, `war` roles
- GM adjudication (`!DNC gm <link>`) with author-history + semantic retrieval, query expansion, revise-by-reply, 🎲 reroll
- **War GM mode** (`!DNC war …`): thread-scoped wars with verbatim move memory, AI-controlled belligerents, and end-of-war chronicles
- Vision/OCR ingestion of images; PDF/DOCX text extraction + chunking
- Espionage: spies intercept secret-channel posts probabilistically
- Spatial mapping: world map, country zoom, faction map, real ArcGIS provinces, LLM-driven ownership changes
- Void/unvoid with tombstone retention; year rollover; stats; export; opt-out

## File Structure

**Core**
- `bot.py` (~4600 lines): `LoreBot` (Discord client; routing, all handlers, subsystem lifecycle) and `LoreCog` (command registration). Entry point via `main()`.

**Inference & memory**
- `inference_client.py`: async wrapper over any OpenAI-compatible API — chat (SSE streaming, thinking tokens, tool calling), vision, embeddings. Returns `(text, usage)`.
- `memory_store.py`: ChromaDB wrapper (`lore` collection) — add/search, metadata filters, author/source lookups, void/unvoid with tombstones.
- `prompt_store.py`: hot-reloadable prompt loader keyed by config paths.
- `state_store.py`: atomic JSON runtime state (year, stats, per-year ruling/war ID counters).

**Subsystems**
- `war_store.py`: JSON store for thread-scoped wars (verbatim move/ruling log + maintained state digest + chronicle draft).
- `espionage_store.py`: JSON store for spy targets / counterspy state.
- `optout_store.py`: plain-text opt-out registry (thread-safe).
- `chat_blacklist.py`: plain-text blacklist for chat commands.
- `file_logging.py`: daily-rotating ingestion/command logs, append-only void + map-change logs.
- `tavily_client.py`: async Tavily web-search tool exposed to chat modes.

**Spatial mapping**
- `map_store.py`: runtime tag→player occupation (rebuilt from nicknames).
- `map_colors.py`: persistent tag→hex color assignments.
- `map_renderer.py`: matplotlib/geopandas world/zoom/faction renderers; `TAG_TO_ISO2` mappings.
- `map_geometry.py`: geometry helpers (radius, bordering, in-country) for the map LLM.
- `map_llm.py`: tool-driven LLM frontend that proposes province-ownership changes.
- `province_store.py` / `province_generator.py` / `arcgis_provinces.py`: real administrative-division province data + ownership state.
- `map_cache.py`: in-memory + on-disk cache for the rendered world map.
- `map_scheduler.py` / `year_scheduler.py`: periodic nickname scans / year-rollover task.

**Utilities**
- `channel_names.py`: Unicode-normalize channel names (emoji, variant selectors) for reliable matching.
- `nickname_parser.py`: extract ISO nation TAG from `TAG - Name` nicknames.
- `reply_chain.py`: walk reply references, detect conversation mode, build OpenAI-format multi-turn messages.

**Config & data**
- `config.yaml`: all tuning knobs. `.env`: secrets. `requirements.txt`: deps.
- `memory_store/` (ChromaDB), `state.json`, `wars.json`, `espionage.json`, `optouts.txt`, `chat_blacklist.txt`, `map_colors.txt`, `province_ownership.json`, `factions`, `map_data/`, `logs/`, `exports/`, `prompts/`.

## Setup & Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill DISCORD_TOKEN, OPENROUTER_API_KEY (+ optional EMBEDDING_API_KEY, TAVILY_API_KEY)
python bot.py
```

Startup: load `config.yaml` → init Chroma/state/wars/espionage/optouts/prompts/map subsystems → purge expired tombstones → connect to Discord → (optional) prewarm map cache + start scheduled tasks.

### Development / testing

No automated test suite. To verify: edit code/prompts, restart (`python bot.py`), exercise in Discord, watch console (color-coded inference logs) and `logs/`. Prompts hot-reload via `!DNC reloadprompts` (admin) without restart.

## Key Concepts & Patterns

### Message routing (`on_message`)
1. **Reply-to-bot**: reply to a bot message not starting with the prefix → war-ruling revision (if the parent is a cached war ruling and replier is a war GM), else `_handle_reply_to_bot` continues the conversation by detected mode.
2. **Commands**: prefix + registered command → dispatched via `commands.Bot`.
3. **Fallback query**: prefix + unknown command → treated as a lore query.
4. **Ingestion**: otherwise, if in a scanned channel (and not an active war thread, not opted-out, not obviously OOC, meets min length) → buffer for grouped ingestion. Espionage interception runs independently for secret channels.

### Conversation modes (from reply-chain root)
- `chat`: general conversation with memory context (anyone can continue)
- `chatuc`: admin/`chatuc`-role "unhinged" personality
- `gm`: GM adjudication; only GM roles may revise
- `query`: lore retrieval + in-character answer

### Permission checks
- `_is_admin_user`: Manage Server perm OR an `admin` role
- `_is_gm_user`: a `gm` role (falls back to admin if `gm.roles` empty)
- `_is_chatuc_user`: a `chatuc` role (falls back to admin)
- `_is_war_gm`: a `war` role (falls back to GM/admin)
- Admin commands generally require admin channel **and** admin user.

### Memory ingestion
Group consecutive same-author messages (within `chain_delay_seconds`) → optional vision OCR + PDF/DOCX extraction → Archivist LLM → drop if `NO_LORE` (when `filter_non_lore`) → parse summary + tags (entities/action_type/claims) → embed → store with rich metadata. Documents are additionally chunked and embedded as `entry_type: "document_chunk"`.

### GM mode
Fetch target action (may span consecutive posts) → optional LLM query expansion (entities/prerequisites + inferred `action_type`) → retrieve author priors (recency + semantic) and wider archive context (with optional action_type boost) → GM Ruling LLM → post (in a dedicated output channel or inline) → cache ruling into `lore` as `entry_type: "ruling"`. GM replies revise; the action author 🎲-reacts to reroll. Per-year IDs `GM-<year>-NNNN`.

### War GM mode
Thread-scoped wars adjudicated with maximal memory fidelity. Lifecycle:
1. `!DNC war start <title>` (GM) inside a thread → registers the thread as a war (`WAR-<year>-NNNN`).
2. `!DNC war side <Side> <TAG|@player>…` (GM) → registers belligerents. Nations with no human occupant (per `map_store` occupation) are **auto-flagged AI-controlled**; `!DNC war npc <TAG> on|off` overrides.
3. `!DNC war move <text|message-link>` (a registered belligerent or GM) → the move is appended to the war log and **adjudicated immediately**. The ruling context is the **entire verbatim war log** (within `war.context.log_char_budget`, with older entries covered by the digest) + the maintained **state digest** + the acting nation's lore pulled from the main archive + the move. Five-tier outcomes, posted in-thread.
4. **AI belligerents**: when a player move targets an AI nation, the bot generates that nation's in-character move (grounded in its archive lore), posts it (`🤖 TAG responds:`), and adjudicates it through the same engine. NPC moves never cascade into further NPC moves. `!DNC war npcact <TAG> [guidance]` triggers one manually.
5. After each ruling the **state digest** (forces/territory/casualties per side) is regenerated incrementally by a cheap model.
6. Revise a ruling by **replying** to it (GM), or **🎲-react** to reroll (submitter or GM). The old ruling is superseded in the log and its messages deleted.
7. `!DNC war end` (GM) → drafts one or more chronicle entries in-thread (split on `=== ENTRY: … ===`) and sets status `pending_commit`. `!DNC war commit` (GM) embeds and writes each entry to the main `lore` DB as `entry_type: "war_chronicle"` and closes the war. `!DNC war cancel` abandons it (log retained, nothing archived).

**Memory design**: a war is bounded, so the war log is replayed *verbatim* rather than retrieved — the highest-fidelity option. The digest is an overflow hedge for very long wars. War-thread chatter is isolated from the main archive while the war is live; only the committed chronicle enters `lore`. The reroll/revise cache is in-memory (lost on restart), like the GM 🎲 cache; the war log, digest, and commands persist in `wars.json`.

### Espionage
For each post in a secret channel, every spy watching that nation's TAG rolls against `base_chance` (+ per-TAG modifier, × `counterspy_multiplier` if the target runs counterspy). On success the spy is DM'd the intercepted content. Spies manage targets via `spy`/`unspy`/`counterspy`/`spylist`.

### Spatial mapping
Occupation is rebuilt from member nicknames (`TAG - Name`) and kept in sync via `on_member_update`. `map`/`mapzoom`/`mapfaction` render PNGs (cached). Provinces come from real ArcGIS administrative divisions; ownership is assigned manually (`mapset`/`maprelease`/`mapmerge`) or via the tool-driven LLM (`mapparse` on a post, `mapchange` from instructions), which proposes changes for ✅/❌ confirmation.

### Void / unvoid
Voided memories are written to `memory_store/voided_memories.jsonl` (tombstones with group id, reason, who, when), deleted from Chroma, and recoverable for `void_retention_days`. Expired tombstones are purged on startup. `unvoid <id>` restores a group.

### Reply chain reconstruction
Walk `message.reference` backward to the root invocation (bounded), keep up to `max_chain_depth` recent messages, truncate each to `max_history_message_chars`, and build OpenAI-format messages (user/assistant by author), prefixing user turns with the display name.

## Configuration Reference (`config.yaml`)

| Section | Key | Purpose |
|---|---|---|
| bot | command_prefix | Trigger phrase (`!DNC`) |
| bot | min_message_length / min_image_message_length | Ingest thresholds |
| bot | chain_delay_seconds | Group window for multi-part posts (also GM action grouping) |
| bot/conversation | max_chain_depth / max_history_message_chars | Reply-chain limits |
| bot/year_rollover | enabled / interval_days / announcement_* | Year rollover task |
| discord/channels | scan / ignored / admin | Ingestion whitelist / blacklist / admin command channels |
| discord/roles | admin / chatuc | Role names |
| gm | roles / channels / output_channel / retrieval.* | GM mode permissions, location, retrieval tuning |
| **war** | **roles / npc_auto_respond / npc_max_responses_per_move / context.{log_char_budget,archive_top_k,archive_item_chars}** | **War mode permissions, NPC behavior, ruling context budgets** |
| models/provider | base_url / embedding_base_url / site_* | API endpoints |
| models/defaults | model / temperature | Fallback LLM settings |
| models | vision_model / embedding_model | Specialized models |
| models/modes | chat, chatuc, query, gm, **war, war_npc, war_digest, war_chronicle**, archivist | Per-mode model/temperature/thinking_budget |
| memory | db_path / top_k / filter_non_lore / void_retention_days / store_full_message | Vector store behavior |
| tavily | enabled_modes / max_results / search_depth | Web search per mode |
| spatial_mapping | enabled / output_channel / map_llm_* / province_*_path / color_palette | Map system |
| espionage | enabled / command_channels / secret_channels / base_chance / counterspy_multiplier / base_modifiers | Espionage system |
| prompts | * | File paths for each system prompt |

`thinking_budget` accepts an int (max reasoning tokens), a string effort (`"low"`/`"medium"`/`"high"`/`"none"`), or `0` (off).

## Command Reference

**Public** (anyone, any channel):
- `!DNC <question>` — query the archive
- `!DNC chat <message>` — chat with memory context
- `!DNC gm <message-link> [question]` — GM adjudication (GM/admin)
- `!DNC war …` — war GM mode (see below; run inside the war's thread)
- `!DNC year` / `!DNC whoami` / `!DNC help`
- `!DNC optout` / `!DNC optin`
- `!DNC map` / `!DNC mapzoom <TAG>` / `!DNC mapfaction` / `!DNC maplist [TAG]`
- `!DNC spy <TAG>` / `!DNC unspy <TAG>` / `!DNC counterspy` / `!DNC spylist` (espionage command channels)

**War mode** (inside the war's thread):
- `!DNC war start <title>` (GM), `!DNC war side <Side> <TAG|@player>…` (GM), `!DNC war npc <TAG> [on|off]` (GM)
- `!DNC war move <text|message-link>` (belligerent/GM), `!DNC war npcact <TAG> [guidance]` (GM)
- `!DNC war status`, `!DNC war end` (GM), `!DNC war commit` (GM), `!DNC war cancel` (GM)

**Admin** (admin channels + admin users):
- `!DNC chatuc <message>` — unhinged chat
- `!DNC ingest [#channel] <N | date-range | link/ID> [year <N>]` — backfill / single ingest
- `!DNC void <@user|link|ID>` / `!DNC unvoid <ID>`
- `!DNC yearset <year>` / `!DNC yearroll` / `!DNC purge year <year>`
- `!DNC export` / `!DNC stats [reset]` / `!DNC channels` / `!DNC reloadprompts`
- `!DNC chatban <@user|ID>` / `!DNC chatunban <@user|ID>`
- `!DNC mapset <PROVINCE_ID> @player` / `!DNC maprelease <PROVINCE_ID>` / `!DNC mapmerge <TAG>` / `!DNC mapdivide [TAG]`
- `!DNC mapparse <message-link>` / `!DNC mapchange <instructions>`

## Common Development Tasks

### Add a new command
1. Add a handler `async def _cmd_x(self, message, arg=…)` on `LoreBot`.
2. Register it on `LoreCog` with `@commands.command(name="x")` + permission checks.
3. Log admin commands via `self.bot.flog.log_command(...)`.
4. Surface it in `_help_text()`.

### Modify a system prompt
Edit `prompts/<name>.txt`, then `!DNC reloadprompts` in an admin channel. `reload()` only swaps the cache if at least one prompt loads, so a botched edit won't wipe everything.

### Add a new LLM mode
Add a block under `models.modes.<mode>` in `config.yaml`; read it via `self._mode_settings("<mode>")` and pass to `llm.chat`/`llm.chat_messages`. Track tokens with `self.state.add_usage("<kind>", usage)` (add stat keys in `state_store.DEFAULT_STATE` and the `stats` display if you want them surfaced).

### Tune memory / retrieval
`memory.top_k` (query breadth), `memory.void_retention_days`, `conversation.*`, `gm.retrieval.*`, `war.context.*`.

## State & Data Persistence

- **state.json**: year, stats, per-year `gm_ruling_counter_*` / `war_counter_*`. Atomic unique-temp-then-rename.
- **wars.json**: per-thread war records (sides, belligerents, ordered log, digest, chronicle draft).
- **espionage.json / optouts.txt / chat_blacklist.txt / map_colors.txt / province_ownership.json**: feature state.
- **memory_store/**: ChromaDB — back this up if lore matters. `voided_memories.jsonl` holds tombstones.
- **logs/**: daily ingestion/command logs, append-only void + map-change logs.

To reset lore: delete `memory_store/` and restart.

## Known Patterns & Gotchas

1. **`Dict` annotations**: several instance-attribute annotations use `Dict` though only `List/Optional/Tuple` are imported. PEP 526 does not evaluate local/attribute annotations, so this is inert — but don't use `Dict` in an evaluated position (function signature, class-body) without importing it.
2. **War reroll/revise cache is in-memory**: rulings issued before a restart can't be rerolled/revised (war log + digest still persist). Same limitation as the GM 🎲 cache.
3. **Discord rate limits**: backfill sleeps 0.05s/message.
4. **Message links**: `MSG_LINK_RE` matches `…/channels/{guild}/{channel}/{message}`.
5. **Unicode channel names**: always compare via `normalize_channel_name()`; config entries must match the normalized form.
6. **Threads**: war commands run inside `discord.Thread`s. Active war threads are explicitly excluded from ingestion regardless of `scan` config.
7. **Void tombstones**: expired tombstones are purged on startup, not in real time.
8. **Embedding limits**: large backfills/chronicles may hit provider token/batch limits; inputs are trimmed before embedding.
9. **Multi-provider keys**: `OPENROUTER_API_KEY` for chat/vision; `EMBEDDING_API_KEY` optional for embeddings; `TAVILY_API_KEY` optional for web search.
