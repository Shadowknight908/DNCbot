"""File-based logging for the DNC bot.

Three loggers, each writing to its own daily-rolling file in ./logs/:
- ingestion-YYYY-MM-DD.log — archived + filtered (NO_LORE) only
- commands-YYYY-MM-DD.log  — every admin command invocation
- voids.log                — append-only, human-readable, includes
                             void/unvoid/expired-purge events (no rotation)
"""
from __future__ import annotations

import logging
import os
from logging.handlers import TimedRotatingFileHandler


def _make_rotating_logger(name: str, filename_stem: str, log_dir: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    os.makedirs(log_dir, exist_ok=True)
    handler = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, f"{filename_stem}.log"),
        when="midnight",
        backupCount=90,
        encoding="utf-8",
        utc=True,
    )
    handler.suffix = "%Y-%m-%d"
    handler.setFormatter(logging.Formatter("%(asctime)sZ  %(message)s"))
    logger.addHandler(handler)
    return logger


def _make_append_logger(name: str, filename: str, log_dir: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    os.makedirs(log_dir, exist_ok=True)
    handler = logging.FileHandler(
        filename=os.path.join(log_dir, filename),
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)sZ  %(message)s"))
    logger.addHandler(handler)
    return logger


class FileLoggers:
    def __init__(self, log_dir: str = "logs"):
        self.ingestion = _make_rotating_logger("dnc.ingestion", "ingestion", log_dir)
        self.commands = _make_rotating_logger("dnc.commands", "commands", log_dir)
        self.voids = _make_append_logger("dnc.voids", "voids.log", log_dir)
        self.map_changes = _make_append_logger("dnc.map_changes", "map_changes.log", log_dir)

    # Convenience helpers — keeps message format consistent everywhere.
    def log_archived(self, msg_id: str, author: str, channel: str, chars: int, year: int):
        self.ingestion.info(
            f"ARCHIVED  msg={msg_id}  author={author}  channel={channel}  chars={chars}  year={year}"
        )

    def log_non_lore(self, msg_id: str, author: str, channel: str, chars: int):
        self.ingestion.info(
            f"FILTERED  msg={msg_id}  author={author}  channel={channel}  chars={chars}  reason=non_lore"
        )

    def log_command(self, command: str, user: str, channel: str, args: str):
        self.commands.info(
            f"CMD={command}  user={user}  channel={channel}  args={args!r}"
        )

    def log_void_msg(self, msg_id: str, author: str, by: str, group_id: str):
        self.voids.info(
            f"VOIDED-MSG     msg={msg_id}  author={author}  by={by}  group={group_id}"
        )

    def log_void_user(self, user_id: str, user_name: str, count: int, by: str, group_id: str):
        self.voids.info(
            f"VOIDED-USER    user={user_name} ({user_id})  memories={count}  by={by}  group={group_id}"
        )

    def log_unvoid(self, descriptor: str, count: int, by: str, group_id: str):
        self.voids.info(
            f"UNVOIDED       {descriptor}  memories={count}  by={by}  group={group_id}"
        )

    def log_purge_expired(self, count: int):
        self.voids.info(
            f"PURGED-EXPIRED memories={count}  (retention window passed)"
        )

    def log_map_change(
        self, command: str, by: str, action: str, pid: str,
        *, player_tag: str = "", from_tag: str = "", to_tag: str = "",
        reason: str = "",
    ):
        parts = [
            f"CMD={command}", f"by={by}", f"action={action}", f"province={pid}",
        ]
        if player_tag:
            parts.append(f"tag={player_tag}")
        if from_tag:
            parts.append(f"from={from_tag}")
        if to_tag:
            parts.append(f"to={to_tag}")
        if reason:
            parts.append(f"reason={reason!r}")
        self.map_changes.info("  ".join(parts))
