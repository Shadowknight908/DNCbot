# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python Discord roleplay/lore bot. `bot.py` is the main entry point and command dispatcher. Core subsystems live as top-level modules: `memory_store.py` for ChromaDB persistence, `state_store.py` for JSON runtime state, `prompt_store.py` for prompt loading, `reply_chain.py` for conversation context, and map-related logic in `map_*.py`, `province_*.py`, and `arcgis_provinces.py`. Prompt assets are in `prompts/`; map and GIS assets are in `map_data/` plus `World_Administrative_Divisions.geojson`. Runtime data such as `memory_store/`, `logs/`, `state.json`, `optouts.txt`, and generated ownership files should be treated as local state unless a change intentionally updates fixtures or seed data.

## Build, Test, and Development Commands

Create and activate a virtual environment before development:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the bot locally with:

```bash
python bot.py
```

Configuration is loaded from `config.yaml` and secrets from `.env`; start from `.env.example`. Prompt-only changes can usually be tested by editing `prompts/*.txt` and running `!DNC reloadprompts` in an admin channel.

## Coding Style & Naming Conventions

Use standard Python style with 4-space indentation, descriptive snake_case names, and async functions for Discord/API operations. Keep new code consistent with the existing module-per-subsystem layout instead of expanding `bot.py` unless the behavior is command routing or Discord lifecycle glue. Prefer structured config in `config.yaml` over hard-coded IDs, model names, or channel lists.

## Testing Guidelines

There is no committed automated test suite at present. For changes, run the bot and verify the affected Discord command or ingestion path manually. Check `logs/ingestion-YYYY-MM-DD.log`, `logs/commands-YYYY-MM-DD.log`, and startup output for errors. If adding tests, place them under `tests/`, name files `test_<module>.py`, and prefer focused unit tests for stores, parsers, schedulers, and map helpers before integration tests requiring Discord.

## Commit & Pull Request Guidelines

Recent history uses short imperative messages, sometimes with Conventional Commit prefixes such as `fix:`. Prefer concise subjects like `fix: preserve map label visibility` or `add GM ruling prompt reload`. Pull requests should describe the behavior change, list manual verification steps, note config or migration impacts, and include screenshots or rendered map examples when UI/map output changes.

## Security & Configuration Tips

Do not commit `.env`, API keys, Discord tokens, or private server data. Be careful with large generated artifacts and persistent stores; back up `memory_store/` before destructive memory, purge, or schema changes.
