# DNC Lore Bot

A Discord roleplay bot (v0.8) that automatically archives roleplay messages into a local vector store and answers lore queries in-character. Features a full spatial mapping system, GM adjudication mode, multi-turn conversation support, and an optional espionage system.

Powered by [OpenRouter](https://openrouter.ai) for chat, vision, and embeddings (configurable per mode). ChromaDB handles persistent memory storage locally.

## Architecture

### Message Pipeline

```
on_message ─┬─► reply to bot message? ──► reconstruct reply chain ──► continue conversation
            │
            ├─► starts with "!DNC "? ──► registered command? ──► dispatch command
            │                                                    │
            │                         └─► no match ────────────┤
            │                                                    ▼
            │                                              embed query ──► top-k search
            │                                                    │
            │                                                    ▼
            │                                          Chronicler LLM ──► reply
            │
            └─► long enough? in scanned channel? not opted out?
                    │
                    ▼
              (vision OCR if image attached)
                    │
                    ▼
              Archivist LLM ──► embed ──► ChromaDB (./memory_store/)
```

### Conversation Modes

| Mode | Trigger | Who can use | Description |
|---|---|---|---|
| `query` | `!DNC <question>` | Anyone | Semantic search + in-character lore answer |
| `chat` | `!DNC chat <msg>` | Anyone | General conversation with memory context |
| `chatuc` | `!DNC chatuc <msg>` | Admin / chatuc role | "Unhinged" chat with alternate personality |
| `gm` | `!DNC gm <link>` | GM roles / admin | GM adjudication of a roleplay action |

After any bot response, anyone with the appropriate permissions can continue the conversation by **replying directly** to the bot's message (no prefix needed), up to `max_chain_depth` messages of context.

### Subsystems

| File | Role |
|---|---|
| `bot.py` | Main entry point; `LoreBot` (Discord client) + `LoreCog` (command dispatcher) |
| `inference_client.py` | Async OpenRouter wrapper (chat, vision, embeddings, tool use) |
| `memory_store.py` | ChromaDB wrapper — store, retrieve, void/unvoid |
| `state_store.py` | Atomic JSON state (current year, stats) |
| `prompt_store.py` | Hot-reloadable system prompt loader |
| `optout_store.py` | Plain-text opt-out registry |
| `file_logging.py` | Daily-rotating ingestion + command logs, append-only void log |
| `reply_chain.py` | Walk message reply chains; detect conversation mode |
| `year_scheduler.py` | Year rollover scheduler |
| `channel_names.py` | Unicode normalization for reliable channel matching |
| `chat_blacklist.py` | Per-user chat ban list |
| `tavily_client.py` | Optional Tavily web-search tool (chat / mapparse modes) |
| `espionage_store.py` | Spy / counterspy state persistence |
| `faction_store.py` | Human-editable faction membership file loader |
| `map_store.py` | Province ownership persistence |
| `map_renderer.py` | GeoJSON → PNG world/zoom map renderer |
| `map_llm.py` | Tool-driven LLM frontend for `mapparse` / `mapchange` |
| `map_geometry.py` | Spatial query tools (radius, bordering, country lookup) |
| `map_scheduler.py` | Background auto-map render task |
| `map_cache.py` | Disk-backed PNG cache to avoid redundant renders |
| `province_store.py` | Province definition storage |
| `map_colors.py` | Per-nation color assignment |
| `nickname_parser.py` | Country TAG resolver |

## Setup

### 1. Create a Discord Bot

- Go to <https://discord.com/developers/applications> and create an application.
- Under **Bot**, copy the token.
- Enable **Message Content Intent** under *Privileged Gateway Intents*.
- Invite to your server with `bot` scope and at minimum `Read Messages`, `Send Messages`, and `Read Message History` permissions.

### 2. Get an OpenRouter API Key

Sign up at <https://openrouter.ai> and create an API key.

An optional separate key for embeddings (`EMBEDDING_API_KEY`) can be set if you use a different provider for that.

### 3. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configure

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```
DISCORD_TOKEN=your_discord_token
OPENROUTER_API_KEY=your_openrouter_api_key
EMBEDDING_API_KEY=          # optional: separate key for embeddings
TAVILY_API_KEY=             # optional: enables web search in chat/map modes
```

Then edit `config.yaml` to set your channels, roles, and preferred models.

### 5. Run

```bash
python bot.py
```

## Configuration Reference

All tuning knobs live in `config.yaml`.

### Bot Settings

| Key | Default | Purpose |
|---|---|---|
| `bot.command_prefix` | `!DNC` | Trigger phrase |
| `bot.min_message_length` | `100` | Min chars for text-only ingestion |
| `bot.min_image_message_length` | `10` | Min chars for posts with an image |
| `bot.chain_delay_seconds` | `30` | Window to group multi-part posts from same author |
| `bot.conversation.max_chain_depth` | `5` | Max prior messages in reply chain context |
| `bot.conversation.max_history_message_chars` | `2000` | Per-message truncation limit for history |

### Discord / Channels / Roles

| Key | Purpose |
|---|---|
| `discord.channels.scan` | Channels whitelisted for ingestion (empty = all non-ignored) |
| `discord.channels.ignored` | Channels never scanned |
| `discord.channels.admin` | Channels where admin commands work |
| `discord.roles.admin` | Role names for admin access (also grants Manage Server perm) |
| `discord.roles.chatuc` | Roles allowed to use `!DNC chatuc` |
| `gm.roles` | Roles allowed to use `!DNC gm` and revise rulings |
| `gm.channels` | Channels where `!DNC gm` can be invoked |
| `gm.output_channel` | Channel where GM rulings are posted |

### Models

| Key | Default | Purpose |
|---|---|---|
| `models.provider.base_url` | OpenRouter | LLM provider endpoint |
| `models.provider.embedding_base_url` | (same) | Separate embeddings endpoint |
| `models.defaults.model` | `deepseek/deepseek-v4-flash` | Fallback model for all modes |
| `models.vision_model` | `google/gemini-2.5-flash` | Vision/OCR model |
| `models.embedding_model` | `Qwen/Qwen3-Embedding-8B` | Embedding model |
| `models.modes.*` | per-mode | Per-mode model, temperature, thinking_budget overrides |

### Memory

| Key | Default | Purpose |
|---|---|---|
| `memory.db_path` | `./memory_store` | ChromaDB directory |
| `memory.top_k` | `20` | Search results per query |
| `memory.filter_non_lore` | `true` | Drop Archivist summaries tagged `NO_LORE` |
| `memory.void_retention_days` | `30` | Days voided memories stay recoverable |
| `memory.store_full_message` | `true` | Store original message text in metadata |

### Spatial Mapping

| Key | Default | Purpose |
|---|---|---|
| `spatial_mapping.enabled` | `true` | Enable the map system |
| `spatial_mapping.output_channel` | `maps` | Auto-render destination channel |
| `spatial_mapping.update_interval_hours` | `24` | Auto-render frequency |
| `spatial_mapping.zoom_buffer_deg` | `5.0` | Degree padding for `!DNC mapzoom` |
| `spatial_mapping.map_llm_model` | `moonshotai/kimi-k2.6` | Model for `mapparse`/`mapchange` |
| `spatial_mapping.map_llm_max_tool_depth` | `6` | Max sequential tool rounds for map LLM |

### Espionage (Optional)

| Key | Default | Purpose |
|---|---|---|
| `espionage.enabled` | `false` | Enable the spy system |
| `espionage.base_chance` | `30` | Base interception probability (%) |
| `espionage.counterspy_multiplier` | `0.25` | Multiplier when target has counterspy active |

## Command Reference

### Public Commands (anyone, any channel)

| Command | Description |
|---|---|
| `!DNC <question>` | Query the lore archive; bot searches memory and replies in-character |
| `!DNC chat <message>` | General conversation with memory context; continue by replying to the bot |
| `!DNC gm <discord-message-link>` | GM adjudication — fetches the post, reviews author history, issues a ruling *(requires gm role or admin, in a GM channel)* |
| `!DNC year` | Show the current in-game year |
| `!DNC optout` | Remove your messages from future ingestion |
| `!DNC optin` | Re-enable ingestion after a previous opt-out |
| `!DNC whoami` | Show your current permission level |
| `!DNC help` | Show command help (output varies by permissions) |

### Map Commands — Public

| Command | Description |
|---|---|
| `!DNC map` | Render and post the full world map coloured by player ownership |
| `!DNC mapfaction` | Render the full world map coloured by faction membership |
| `!DNC mapzoom <TAG>` | Zoom into a specific country (e.g. `!DNC mapzoom USSR`) |
| `!DNC maplist` | List all countries subdivided into provinces |
| `!DNC maplist <TAG>` | List all provinces within a country and their owners |

### Map Commands — Admin Only

| Command | Description |
|---|---|
| `!DNC mapdivide <TAG> <N>` | Subdivide a country into N auto-generated provinces (2–100) |
| `!DNC mapmerge <TAG>` | Remove all provinces for a country, reverting to whole-country ownership |
| `!DNC mapset <PROVINCE_ID> @player` | Manually assign a province to a player |
| `!DNC maprelease <PROVINCE_ID>` | Release a province to uncontrolled |
| `!DNC mapparse <discord-message-link>` | Feed a roleplay post to the LLM; proposes ownership changes for admin ✅/❌ confirmation |

### Admin Commands — General

*(Requires admin role + admin channel, unless noted)*

| Command | Description |
|---|---|
| `!DNC chatuc <message>` | "Unhinged" chat mode with alternate personality (any channel) |
| `!DNC chatban @user` | Ban a user from `!DNC chat` / `chatuc` / reply-chain conversation |
| `!DNC chatunban @user` | Lift a chat ban |
| `!DNC ingest [#channel] <N>` | Backfill N most recent messages into the lore archive |
| `!DNC ingest [#channel] <YYYY-MM-DD> <YYYY-MM-DD>` | Backfill all messages in a date range |
| `!DNC ingest <discord-message-link>` | Ingest a single specific message |
| `!DNC void @user` | Void all archived memories attributed to a user |
| `!DNC void <discord-message-link>` | Void the memory derived from a specific message |
| `!DNC void <memory-ID>` | Void a specific memory by its ChromaDB ID |
| `!DNC unvoid <void-group-ID>` | Restore a voided memory group (within retention window) |
| `!DNC yearset <year>` | Set the current in-game year silently |
| `!DNC yearroll` | Increment the in-game year by 1 and post an announcement |
| `!DNC purge year <year>` | Void all memories tagged with a specific in-game year |
| `!DNC export` | Export all memories to `exports/` as JSON and Markdown |
| `!DNC stats` | Show token usage and message statistics |
| `!DNC stats reset` | Reset all statistics counters |
| `!DNC channels` | Diagnostic: compare config-listed channels against actual Discord channels |
| `!DNC reloadprompts` | Hot-reload all prompt files from disk without restarting |

### Reply-Chain Continuation

After any bot response to `!DNC chat`, `!DNC chatuc`, `!DNC gm`, or a query, reply directly to the bot's message (without a prefix) to continue the conversation. The bot walks up to `max_chain_depth` (default 5) messages of prior context.

Permission rules for continuations match the original command:
- `chat`: anyone can reply
- `chatuc`: admin / chatuc role only
- `gm`: gm roles or admin can revise a ruling

## Data & Files

```
bot.py                      Main entry point
config.yaml                 All tuning knobs
.env                        Secrets (DISCORD_TOKEN, OPENROUTER_API_KEY, etc.)
requirements.txt            Python dependencies
prompts/                    System prompts (hot-reloadable via !DNC reloadprompts)
memory_store/               ChromaDB persistent directory — back this up!
state.json                  Runtime state (year, stats counters)
optouts.txt                 Opt-out list (one user ID per line)
chat_blacklist.txt          Chat ban list
espionage.json              Spy / counterspy state
province_ownership.json     Current province owner assignments
map_data/                   Province definitions + cached map renders
factions                    Human-editable faction membership file
exports/                    Output from !DNC export
logs/                       Daily-rotating ingestion + command logs, void log
```

**To reset lore from scratch:** delete `memory_store/` and restart.

## Map System Quick-Start

```
!DNC map                         # see the starting map
!DNC mapdivide USSR 12           # split USSR into 12 provinces
!DNC mapset USSR-01 @Khrushchev  # assign province 1 to a player
!DNC maplist USSR                # review all provinces and owners
!DNC mapparse <rp-post-link>     # let the LLM auto-detect changes
!DNC mapzoom USSR                # zoom in to see the result
!DNC mapmerge USSR               # undo all provinces if needed
```

## GM Mode

1. Post a roleplay action in Discord.
2. GM runs `!DNC gm <link-to-that-post>` in a configured GM channel.
3. Bot fetches the post, retrieves the author's recent history and semantic context, then calls the GM Ruling LLM.
4. The ruling is posted to the configured GM output channel and cached as a `ruling` memory entry.
5. A GM can reply directly to the ruling to revise it.

## Monitoring

| Log | Contents |
|---|---|
| `logs/ingestion-YYYY-MM-DD.log` | Archived vs filtered messages |
| `logs/commands-YYYY-MM-DD.log` | Admin command audit trail |
| `logs/voids.log` | All void / unvoid events |
| `!DNC stats` | Token usage + message counts |
