# DNC Lore Bot

A Discord roleplay bot that quietly listens to long-form messages, has an LLM
turn them into lore-archive entries, embeds them, and stores them locally.
Members can query the accumulated lore with `!DNC <question>` and the bot
answers in-character based on what it has seen.

Powered by [DeepInfra](https://deepinfra.com) for both chat and embeddings
(Qwen by default — easily swappable).

## Architecture

```
on_message ─┬─► starts with "!DNC "?  ──► embed query ──► top-k search
            │                                              │
            │                                              ▼
            │                                  Chronicler LLM ──► reply
            │
            └─► long enough? in scanned channel?
                    │
                    ▼
              Archivist LLM ──► embed ──► ChromaDB (./memory_store/)
```

## Setup

1. **Create a Discord bot** at <https://discord.com/developers/applications>
   - Add a bot user, copy the token.
   - Under *Privileged Gateway Intents*, enable **Message Content Intent**.
   - Invite to your server with `bot` scope and at minimum the
     `Read Messages`, `Send Messages`, and `Read Message History` permissions.

2. **Get a DeepInfra API key** at <https://deepinfra.com/dash/api_keys>.

3. **Install + configure**:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env       # then fill in DISCORD_TOKEN and DEEPINFRA_API_KEY
   ```

4. **Run**:
   ```bash
   python bot.py
   ```

## Configuration

Everything tunable lives in `config.yaml`:

| Key | What it does |
|---|---|
| `discord.command_prefix` | Trigger phrase for queries (default `!DNC`) |
| `discord.scan_channels` | Whitelist — if empty, scans all non-ignored channels |
| `discord.ignored_channels` | Channels never scanned for ingestion |
| `discord.min_message_length` | Skip messages shorter than this (default 200 chars) |
| `deepinfra.chat_model` | Any model DeepInfra hosts (e.g. `Qwen/Qwen2.5-72B-Instruct`) |
| `deepinfra.embedding_model` | Any embedding model (e.g. `Qwen/Qwen3-Embedding-8B`) |
| `memory.top_k` | How many memories to retrieve per query |
| `memory.filter_non_lore` | If true, drop summaries the LLM tags `NO_LORE` |
| `prompts.memory_extraction` | System prompt for condensing messages → archive entries |
| `prompts.lore_query` | System prompt for answering queries in-character |

Verify the exact embedding model ID on
<https://deepinfra.com/models?type=embeddings> — DeepInfra occasionally
renames or rotates models.

## Usage

In any non-ignored channel, just chat normally. Anything ≥200 chars gets
read by the Archivist and (if it contains lore) added to the on-device store.

To ask about the lore:

```
!DNC what happened in 1971?
!DNC who is Marcus Vell and what faction does he serve?
!DNC summarize the war of the bronze gates
```

The bot replies in the same channel.

## Files

```
bot.py                Main entry point + Discord event handlers
deepinfra_client.py   Async wrapper for chat + embedding API calls
memory_store.py       ChromaDB-backed local vector store
config.yaml           All tuning knobs (channels, models, prompts)
.env.example          Secrets template
requirements.txt      Python dependencies
```

The vector store persists in `./memory_store/` — back this up if the lore
matters to you. To wipe and start fresh, delete the directory.

## Easy extensions

- **Backfill** existing channel history on startup (loop over
  `channel.history(limit=N)` and call `_ingest_memory` on each).
- **Admin commands**: `!DNC-stats`, `!DNC-clear`, `!DNC-recent 10`.
- **Per-channel personalities**: split memory into multiple ChromaDB
  collections keyed by channel.
- **Source links**: include `metadata.source_message_id` in replies so users
  can jump back to the original message via `https://discord.com/channels/...`.
# DNCbot
