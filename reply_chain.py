"""Reply chain reconstruction for multi-turn conversations.

When a user replies to a bot message, walk back up the reply chain to
reconstruct the full conversation. Detect what mode the conversation
started in (chat / chatuc / query / gm) by finding the original
!DNC <subcommand> invocation at the root.

The mode is encoded into bot responses via a tiny invisible marker in
metadata so we don't have to parse text. But Discord doesn't give us
custom metadata on messages — so instead we use stateless detection:
walk back to the root and look at the first invocation message's text.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import discord

log = logging.getLogger("dnc.replies")


# Detection of conversation mode from the chain root message text.
# Order matters: longer prefixes first so "chatuc" doesn't match "chat".
MODE_PREFIXES = [
    ("chatuc", "chatuc"),
    ("chat", "chat"),
    ("gm", "gm"),
    # Any other root → treated as a lore query
]


def detect_mode_from_root(root_text: str, command_prefix: str) -> str:
    """Given the original invocation text (e.g. '!DNC chat hello'), return
    the conversation mode: 'chat', 'chatuc', 'gm', or 'query'.
    """
    if not root_text.startswith(command_prefix):
        return "query"
    rest = root_text[len(command_prefix):].strip()
    if not rest:
        return "query"
    first_word = rest.split(maxsplit=1)[0].lower()
    for prefix, mode in MODE_PREFIXES:
        if first_word == prefix:
            return mode
    return "query"


async def walk_reply_chain(
    message: discord.Message,
    bot_user_id: int,
    max_depth: int = 5,
) -> Tuple[List[discord.Message], Optional[discord.Message]]:
    """Walk backward from `message` through reply.references.

    Returns (chain, root) where:
      chain is a list of messages [oldest, ..., newest=message] up to max_depth
            messages of context BEFORE the current `message` (not counting it)
      root is the original triggering message (the one with !DNC <subcommand>)
            or None if not found

    The chain stops at max_depth messages OR when a reference can't be fetched.
    """
    chain_back: List[discord.Message] = []
    root: Optional[discord.Message] = None
    current = message

    # We allow the walk to go further than max_depth to find the root,
    # but only keep the most recent max_depth messages in `chain_back`.
    walked = 0
    max_total_walk = 30  # sanity cap to prevent runaway walks
    while walked < max_total_walk:
        ref = current.reference
        if not ref or not ref.message_id:
            break
        try:
            parent = await current.channel.fetch_message(ref.message_id)
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            break
        chain_back.append(parent)
        # Keep the trail bounded
        if len(chain_back) > max_total_walk:
            break
        current = parent
        walked += 1

    # Find the root: walk back through chain_back looking for the first
    # message authored by a non-bot user (the original asker) that was
    # NOT a bot reply. The root is the message that started the conversation.
    # Heuristic: it's the oldest non-bot message in the chain.
    if chain_back:
        # chain_back is [parent1, parent2, ...] where parent1 is the
        # immediate parent of `message`. The root is the FURTHEST back one
        # that was authored by a real user.
        for m in reversed(chain_back):
            if m.author.id != bot_user_id:
                root = m
                break  # found the oldest non-bot message; stop
        # The above gives us the oldest non-bot message in the walk,
        # which should be the original invocation.

    # Trim chain_back to max_depth most recent (i.e., closest to `message`)
    chain_back = chain_back[:max_depth]
    # Reverse so it's oldest → newest
    chain_back.reverse()

    return chain_back, root


def build_messages_from_chain(
    chain_back: List[discord.Message],
    current_message: discord.Message,
    bot_user_id: int,
    system_prompt: str,
    max_message_chars: int = 2000,
) -> List[dict]:
    """Convert a reply chain into an OpenAI-format messages list.

    Each chain message becomes either {"role": "user"} or
    {"role": "assistant"} depending on author. User messages get a
    "[username]: " prefix so multi-user threads still attribute correctly.
    """
    messages: List[dict] = [{"role": "system", "content": system_prompt}]

    def fmt_user_content(msg: discord.Message) -> str:
        # Strip the bot's command prefix from the text so the LLM doesn't
        # see "!DNC chat hello" as the user's words — just "hello".
        content = (msg.content or "").strip()
        # Strip "!DNC <subcommand>" prefix if present
        if content.lower().startswith("!dnc"):
            parts = content.split(maxsplit=2)
            if len(parts) >= 2 and parts[1].lower() in ("chat", "chatuc", "gm"):
                content = parts[2] if len(parts) > 2 else ""
            elif len(parts) >= 2:
                content = parts[1] if len(parts) > 1 else ""
                # Re-glue rest if more parts existed
                if len(parts) > 2:
                    content = " ".join(parts[1:])
        # Truncate
        if len(content) > max_message_chars:
            content = content[:max_message_chars] + "…"
        # Show both character name (nick) and real Discord account name when
        # they differ, so the LLM knows who it's talking to outside the RP.
        char_name = msg.author.display_name
        account_name = msg.author.name
        author_label = (f"{char_name} (@{account_name})"
                        if char_name != account_name else char_name)
        return f"[{author_label}]: {content}"

    def fmt_assistant_content(msg: discord.Message) -> str:
        content = (msg.content or "").strip()
        if len(content) > max_message_chars:
            content = content[:max_message_chars] + "…"
        return content

    for msg in chain_back:
        if msg.author.id == bot_user_id:
            messages.append({"role": "assistant", "content": fmt_assistant_content(msg)})
        else:
            messages.append({"role": "user", "content": fmt_user_content(msg)})

    # Current message
    messages.append({"role": "user", "content": fmt_user_content(current_message)})
    return messages
