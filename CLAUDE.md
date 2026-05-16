# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DNC Lore Bot** is a Discord roleplay bot (v0.6) that automatically archives roleplay messages into a local vector store and answers lore queries. It's powered by OpenRouter's API for chat, vision, and embeddings, and uses ChromaDB for persistent memory storage. A separate `DEEPINFRA_API_KEY` can optionally be set to route embeddings through DeepInfra instead.

### Core Architecture

The bot operates on a two-path message pipeline:

1. **Ingestion Path**: Non-command messages in scanned channels are passed to the Archivist LLM, which extracts lore summaries and stores them as embeddings in ChromaDB
2. **Query Path**: Command messages (`!DNC <query>`) trigger semantic search on accumulated memories, followed by the Chronicler LLM providing in-character responses

Key features (v0.6):
- Multi-turn conversation support with reply chain reconstruction (up to 5 messages of context)
- Role-based permissions (`admin_roles`, `gm_roles`) for access control
- GM adjudication mode (`!DNC gm <message-link>`) with author history + semantic context
- Vision/OCR support for image ingestion
- Year rollover scheduling with manual safeguards
- Void/unvoid system with 30-day tombstone retention
- Comprehensive statistics and export capabilities

## File Structure

**Core Bot Logic**
- `bot.py` (1514 lines): Main entry point; contains `LoreBot` class (Discord client) and `LoreCog` (command dispatcher)
  - Message routing (ingestion vs commands)
  - Reply chain detection and mode routing (chat/chatuc/gm/query)
  - All command implementations
  - Subsystem initialization and lifecycle

**Subsystems**
- `inference_client.py`: Async HTTP wrapper around any OpenAI-compatible inference API (chat, vision, embeddings); supports OpenRouter, HuggingFace TGI, vLLM, etc.
- `memory_store.py`: ChromaDB wrapper with vector storage, retrieval, void/unvoid, metadata filtering
- `state_store.py`: Persistent JSON state (current year, stats, timestamps) with atomic writes via temp-then-rename
- `prompt_store.py`: Hot-reloadable prompt file loader from config paths
- `optout_store.py`: Plain-text opt-out registry with threading safety
- `file_logging.py`: Daily-rotating ingestion + command logs, append-only void log

**Utilities**
- `channel_names.py`: Unicode normalization for reliable channel matching (handles emoji, variant selectors)
- `reply_chain.py`: Walk reply messages backward to reconstruct conversation history and detect mode
- `year_scheduler.py`: Year rollover task with console-based safeguard prompts

**Configuration**
- `config.yaml`: All tuning knobs (channels, models, prompts, memory settings)
- `.env`: Secrets (DISCORD_TOKEN, DEEPINFRA_API_KEY)
- `requirements.txt`: Dependencies (discord.py, chromadb, httpx, PyYAML, python-dotenv)

**Data**
- `memory_store/`: ChromaDB persistent directory
- `state.json`: Runtime state (year, stats)
- `optouts.txt`: Opt-out list
- `logs/`: Daily-rotating ingestion/command logs + void log
- `prompts/`: System prompts for memory extraction, lore queries, chat, GM rulings, vision

## Setup & Running

### Installation
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
# Fill in DISCORD_TOKEN and DEEPINFRA_API_KEY in .env
```

### Running the Bot
```bash
python bot.py
```

The bot will:
1. Load config.yaml
2. Initialize ChromaDB, state, optouts, and prompts
3. Connect to Discord
4. Start listening for messages and commands

### Development/Testing

There are no built-in tests. To verify changes:
1. Update code and/or prompts in `config.yaml`
2. Restart the bot: `python bot.py`
3. Test in Discord (commands, ingestion, replies)
4. Check logs in `logs/` for errors or ingestion details

For prompt iteration: edit `prompts/*.txt`, then run `!DNC reloadprompts` in an admin channel—changes take effect immediately without restarting.

## Key Concepts & Patterns

### Message Routing (bot.py, on_message)
Messages are routed in this order:
1. **Reply-to-bot path**: If message.reference points to a bot message AND doesn't start with prefix, call `_handle_reply_to_bot()` for conversation continuation
2. **Command path**: If message starts with prefix and is a registered command, dispatch via discord.py commands.Bot
3. **Fallback query**: If starts with prefix but no registered command, treat as a lore query
4. **Ingestion path**: If none above and message is in scanned channel, ingest it

### Conversation Modes
Detected from the root message's text (`!DNC chat`, `!DNC chatuc`, `!DNC gm`, or anything else = query):
- `chat`: General conversation with memory context, anyone can continue
- `chatuc`: Admin-only "unhinged" chat with different personality prompt
- `gm`: GM adjudication mode; only gm_roles can revise via reply
- `query`: Lore retrieval + in-character response based on search results

### Permission Checks
- `_is_admin_user()`: Checks Manage Server perm OR admin_roles (case-insensitive)
- `_is_gm_user()`: Checks gm_roles; falls back to admin if gm_roles empty
- Admin commands only work in admin_channels AND by admin users

### Memory Ingestion
1. Extract image OCR (if attached) via vision model
2. Call Archivist LLM with message + channel + author context
3. Filter if tagged `NO_LORE` (if `filter_non_lore: true`)
4. Embed summary + metadata, store in ChromaDB
5. Log to ingestion.log with year + channel + author

### GM Mode (Lore Adjudication)
1. Fetch the target message from a Discord link
2. Retrieve author's prior posts (before action timestamp, newest-first)
3. Embed action text, search for wider semantic context
4. Build comprehensive user content block with action + history
5. Call GM Ruling LLM
6. Cache ruling back into memory with `entry_type: "ruling"`
7. If GM replies to ruling, re-fetch and revise via same process

### Void/Unvoid System
- Voided memories written to `memory_store/voided_memories.jsonl` (per-memory tombstone with void_group_id, void_reason, voided_at, voided_by)
- Tombstones kept for `void_retention_days` (config, default 30)
- Purge expired tombstones on startup
- `unvoid <id>` recovers all voided memories in a group

### Reply Chain Reconstruction
- Walk backward via message.reference until no parent or bot message not found
- Collect messages as (oldest, ..., newest) up to max_chain_depth
- Detect conversation mode from root message text
- Truncate prior messages to max_history_message_chars each
- Build OpenAI-format messages for multi-turn API call

## Configuration Reference

Key settings in `config.yaml`:

| Section | Key | Default | Purpose |
|---------|-----|---------|---------|
| bot | command_prefix | `!DNC` | Trigger phrase |
| bot | min_message_length | 0 | Chars required to ingest (10 if image) |
| bot/conversation | max_chain_depth | 5 | Prior messages to include in context |
| bot/year_rollover | enabled | false | Auto year rollover scheduler |
| discord/channels | scan | list | Whitelist; empty = scan all non-ignored |
| discord/channels | ignored | list | Never scan for ingestion |
| discord/channels | admin | list | Only place for admin commands |
| discord/roles | admin | list | Role names for admin |
| gm | roles | list | Role names for GM mode |
| gm | channels | list | Where !DNC gm can be invoked |
| models/provider | base_url | ... | OpenRouter/API base URL |
| models/defaults | model | ... | Fallback LLM for all modes |
| models/modes | * | per-mode | LLM settings overrides |
| memory | db_path | `./memory_store` | ChromaDB location |
| memory | top_k | 20 | Search results per query |
| prompts | * | file paths | System prompts for each mode |

## Command Reference

**Public Commands** (anyone, any channel):
- `!DNC <question>` — Query the lore archive
- `!DNC chat <message>` — Chat with the bot (with memory context)
- `!DNC year` — Show current in-game year
- `!DNC optout` / `!DNC optin` — Manage personal opt-out
- `!DNC whoami` — Show your role/permission status
- `!DNC help` — Show help text (generates dynamically based on permissions)
- `!DNC gm <message-link>` — GM mode: adjudicate an action (gm_roles or admin only)

**Admin Commands** (admin_channels + admin users):
- `!DNC chatuc <message>` — Unhinged chat (admin-only personality)
- `!DNC ingest [#channel] <N|date-range>` — Backfill N recent or date-range messages
- `!DNC ingest <message-link|ID>` — Ingest a single message
- `!DNC void <@user|message-link|ID>` — Void memory(ies)
- `!DNC unvoid <ID>` — Restore voided memory group
- `!DNC yearset <year>` — Set current year without announcement
- `!DNC yearroll` — Increment year + announce
- `!DNC purge year <year>` — Void all memories from a year
- `!DNC export` — Export all memories as JSON + Markdown to `exports/`
- `!DNC stats [reset]` — Show/reset token counts and message stats
- `!DNC channels` — Diagnostic: show config vs actual Discord channels
- `!DNC reloadprompts` — Hot-reload prompt files from disk

## Common Development Tasks

### Adding a New Command
1. Write the command handler: `async def _cmd_mycommand(self, message, arg=...)`
2. Add it to `LoreCog`: `@commands.command(name="mycommand")` + permission checks if needed
3. Log if admin: `self.bot.flog.log_command(...)`
4. Update help text in `_help_text()` (currently missing implementation; add if needed)

### Modifying System Prompts
1. Edit the file in `prompts/` (e.g., `prompts/lore_query.txt`)
2. Run `!DNC reloadprompts` in an admin channel
3. Test by running the corresponding command

### Changing Channel Matching Logic
Channel names support emoji and Unicode. Normalization happens in `channel_names.py` via NFC-normalize + invisible-char stripping + lowercase. Config entries must match the normalized form.

### Adjusting Memory Retention / Search
- `memory.top_k`: Change how many archived memories to retrieve per query
- `memory.void_retention_days`: Change how long voided memories stay recoverable
- `conversation.max_chain_depth`: Change how many prior messages are included in multi-turn
- `conversation.max_history_message_chars`: Truncate longer conversation history

### Monitoring Bot Health
- Check `logs/ingestion-YYYY-MM-DD.log` for archived vs filtered messages
- Check `logs/commands-YYYY-MM-DD.log` for admin command audit
- Check `logs/voids.log` for void/unvoid events
- Run `!DNC stats` in admin channel for token usage + message counts

## State & Data Persistence

- **state.json**: Current year, stats counters, timestamps. Atomic writes via unique temp files.
- **optouts.txt**: Plain text, one user ID per line with optional comment.
- **memory_store/**: ChromaDB directory; back this up if lore matters.
- **voided_memories.jsonl**: Append-only tombstone log; entries auto-purged after void_retention_days.
- **logs/**: Daily-rotating files; 90-day backups for ingestion + commands, unlimited append for voids.

To reset lore from scratch: delete `memory_store/` directory and restart.

## Known Patterns & Gotchas

1. **_help_text() undefined**: The `_help_text()` method referenced in bot.py is not currently implemented. If you need dynamic help, implement it as a property that builds text based on user permissions.

2. **Discord Rate Limits**: Backfill (`ingest`) sleeps 0.05s between messages to avoid rate limits. Adjust if needed.

3. **Message Link Parsing**: MSG_LINK_RE matches `https://discord.com/channels/{guild}/{channel}/{message}`. Ensure links are formatted correctly in commands.

4. **Unicode in Channel Names**: Always use `normalize_channel_name()` when comparing channel names from config to Discord objects.

5. **Vision Model Context**: Image context is only extracted if at least one image is attached. Reduces min_message_length to 10 if images present.

6. **Void Tombstones**: Expired tombstones are purged on startup, not in real-time. Multiple voids of the same message create duplicate tombstones with the same void_group_id.

7. **Embedding Limits**: The embedding API (OpenRouter or DeepInfra) may have token/batch limits; monitor for errors during large backfills.

8. **Prompt Reloading**: `reload()` only swaps cache if at least one prompt loads successfully, so a botched edit doesn't wipe all prompts.

