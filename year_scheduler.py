"""Year rollover scheduler + console-based sanity check.

Auto-rollover is a simple periodic task that compares (now - last_rollover_at)
to the configured interval. When overdue, it prompts the operator on the host
console rather than firing automatically — see prompt_for_overdue_rollover().
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Awaitable, Callable, List, Optional

import discord

from state_store import StateStore

log = logging.getLogger("dnc.year")


def _parse_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


async def fetch_year_check_preview(
    client: discord.Client, channel_name: str, limit: int = 5
) -> List[str]:
    """Best-effort fetch of the last N messages from the configured year-check
    channel. Returns formatted strings ready for console display, or [] on error.
    """
    if not channel_name:
        return []
    target = None
    for guild in client.guilds:
        for ch in guild.text_channels:
            if ch.name.lower() == channel_name.lower():
                target = ch
                break
        if target:
            break
    if target is None:
        return [f"(channel '{channel_name}' not found in any guild)"]
    try:
        msgs = [m async for m in target.history(limit=limit)]
    except discord.Forbidden:
        return [f"(no permission to read #{channel_name})"]
    except Exception as e:
        return [f"(error reading #{channel_name}: {e})"]

    out = []
    for m in msgs:
        ts = m.created_at.strftime("%Y-%m-%d %H:%M")
        author = m.author.display_name
        body = (m.content or "").replace("\n", " ")
        if len(body) > 120:
            body = body[:117] + "..."
        out.append(f"  [{ts} by {author}]  {body}")
    return out


async def prompt_for_overdue_rollover(
    state: StateStore,
    client: discord.Client,
    year_check_channel: str,
    on_announce: Callable[[int], Awaitable[None]],
) -> None:
    """Console prompt fired when auto-rollover detects an overdue rollover.
    Blocks the scheduler loop until the operator answers.

    Choices:
      [Enter]   keep current state, reset timer, no announcement
      <number>  set year to that value, reset timer, no announcement
      r<number> set year AND post announcement to configured channels
      q         leave year/timer alone — operator will deal with it manually
    """
    state_year = state.current_year
    last = state.last_rollover_at or "never"

    print()
    print("=" * 70)
    print("  ⚠️  YEAR ROLLOVER CHECK")
    print(f"  The auto-rollover scheduler is overdue.")
    print(f"  State file says current year is: {state_year}")
    print(f"  Last rollover timestamp:         {last}")
    print()

    if year_check_channel:
        print(f"  Last 5 messages in #{year_check_channel}:")
        preview = await fetch_year_check_preview(client, year_check_channel, 5)
        if preview:
            for line in preview:
                print(line)
        else:
            print(f"  (no recent messages found)")
        print()

    print("  Options:")
    print("    [Enter]    keep year as is, reset timer, no announcement")
    print("    <number>   set year to that value, reset timer, no announcement")
    print("    r<number>  set year AND post the rollover announcement")
    print("    q          leave timer alone, decide manually later")
    print("=" * 70)

    loop = asyncio.get_running_loop()
    choice = await loop.run_in_executor(None, lambda: input("  Choice: ").strip())

    if not choice:
        state.reset_rollover_timer()
        print(f"  → Kept year at {state.current_year}, reset rollover timer.")
        return

    if choice.lower() == "q":
        print("  → Left timer alone. The scheduler will prompt again next cycle.")
        return

    announce = False
    year_str = choice
    if choice.lower().startswith("r"):
        announce = True
        year_str = choice[1:].strip()

    try:
        new_year = int(year_str)
    except ValueError:
        print(f"  → Couldn't parse '{choice}' as a year. Leaving timer alone.")
        return

    state.set_year(new_year, mark_rollover=True)
    if announce:
        print(f"  → Set year to {new_year} and posting announcement…")
        try:
            await on_announce(new_year)
            print("  → Announcement posted.")
        except Exception as e:
            print(f"  → Announcement failed: {e}")
    else:
        print(f"  → Set year to {new_year}, no announcement.")


def is_overdue(state: StateStore, interval_days: int) -> bool:
    last = _parse_iso(state.last_rollover_at)
    if last is None:
        # Never rolled — set baseline now without flagging overdue.
        state.reset_rollover_timer()
        return False
    return datetime.now(timezone.utc) - last >= timedelta(days=interval_days)
