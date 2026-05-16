# GEMINI.md

This file provides project-wide instructions and architectural guidance for the DNC Lore Bot.

## Project Overview

**DNC Lore Bot** is a Discord roleplay bot (v0.6) that archives roleplay messages into a local vector store (ChromaDB) and provides lore queries and GM adjudication via LLMs (OpenRouter/DeepInfra).

## Core Architecture

- **Ingestion Path**: Processes messages in scanned channels, extracts lore via Archivist LLM, and stores embeddings in ChromaDB.
- **Query Path**: `!DNC <query>` triggers semantic search and Chronicler LLM response.
- **GM Mode**: `!DNC gm <link>` for adjudication, including author history and semantic context.
- **Multi-turn Conversations**: Supports reply chain reconstruction (up to 5 messages).
- **Vision Support**: OCR/image ingestion via vision model.
- **State Management**: Persistent JSON state for years, stats, etc.

## Key Subsystems

- `bot.py`: Main Discord client and command dispatcher.
- `inference_client.py`: `InferenceClient` for Chat/Vision/Embeddings via any OpenAI-compatible API (OpenRouter, DeepInfra, etc.).
- `memory_store.py`: ChromaDB wrapper for vector operations.
- `state_store.py`: Atomic JSON state management.
- `prompt_store.py`: Hot-reloadable prompts.
- `reply_chain.py`: Conversation history reconstruction.

## Configuration Guidelines

- `config.yaml` is the primary configuration file.
- Sensitive keys (tokens, API keys) must be in `.env`.
- Prompts are stored in `prompts/` and can be reloaded via `!DNC reloadprompts`.

## Development Standards

- **Permissions**: Admin users have `Manage Server` or roles in `admin_roles`. GMs have roles in `gm_roles`.
- **Channel Matching**: Use `normalize_channel_name()` from `channel_names.py` for reliable matching.
- **Logging**: Use `file_logging.py` for structured daily-rotating logs.
- **Testing**: Manual verification in Discord; check `logs/` for errors.
