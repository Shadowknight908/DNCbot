"""Async wrapper around OpenRouter (chat/vision) and DeepInfra (embeddings).

OpenRouter provides access to frontier models (Gemini 2.5, Claude 4, etc.) via an
OpenAI-compatible API.  Embeddings are served by DeepInfra because OpenRouter does
not offer embedding models.

chat() and embed() return (result, usage_dict) so callers can track tokens.
chat_messages() takes a pre-built list of messages for multi-turn conversations.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import asyncio
import json
import re

import httpx

# Matches <think>…</think> or <thinking>…</thinking> (Gemini 2.5, DeepSeek-R1, etc.)
_THINK_RE = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.DOTALL | re.IGNORECASE)

# ANSI — dim yellow for thinking output so it's visually distinct from bot console logs
_C_THINK = "\033[2;33m"
_C_RESET = "\033[0m"
_THINK_W = 66


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        chat_model: str = "",
        embedding_model: str = "",
        vision_model: str = "",
        embedding_base_url: str = "",
        embedding_api_key: str = "",
        site_url: str = "",
        site_name: str = "DNC Lore Bot",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is missing")
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.vision_model = vision_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._base_url = base_url.rstrip("/")
        self._embed_base_url = (embedding_base_url or base_url).rstrip("/")

        chat_headers: Dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if site_url:
            chat_headers["HTTP-Referer"] = site_url
        if site_name:
            chat_headers["X-Title"] = site_name

        self._chat_client = httpx.AsyncClient(
            timeout=httpx.Timeout(180.0, connect=10.0),
            headers=chat_headers,
        )
        self._embed_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            headers={
                "Authorization": f"Bearer {embedding_api_key or api_key}",
                "Content-Type": "application/json",
            },
        )

    async def chat(
        self, system_prompt: str, user_message: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        thinking_budget: Optional[Union[int, str]] = None,
    ) -> Tuple[str, Dict[str, int]]:
        """Single-turn convenience wrapper."""
        return await self.chat_messages(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            model=model, temperature=temperature,
            top_p=top_p, top_k=top_k, max_tokens=max_tokens,
            thinking_budget=thinking_budget,
        )

    async def chat_messages(
        self, messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        thinking_budget: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_executor: Optional[Callable] = None,
        _tool_depth: int = 0,
    ) -> Tuple[str, Dict[str, int]]:
        """Multi-turn chat using SSE streaming.

        reasoning_content (thinking) tokens are streamed to console in real-time.
        Response content is buffered and returned in full to the caller.
        Inline <think>…</think> tags in the content are stripped and echoed to
        console after the stream ends (covers models that embed tags in content).

        If tools and tool_executor are provided, the model may call tools. Results
        are fed back automatically (up to 3 rounds) before the final reply.
        """
        payload: Dict[str, Any] = {
            "model": model or self.chat_model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        effective_max = max_tokens if max_tokens is not None else self.max_tokens
        if effective_max is not None:
            payload["max_tokens"] = effective_max
        if top_p is not None:
            payload["top_p"] = top_p
        if top_k is not None:
            payload["top_k"] = top_k
        if thinking_budget is not None:
            if isinstance(thinking_budget, str):
                payload["reasoning"] = {"effort": thinking_budget}
            elif thinking_budget == 0:
                payload["reasoning"] = {"effort": "none"}
            else:
                payload["reasoning"] = {"max_tokens": thinking_budget}
                # OpenRouter requires top-level max_tokens > reasoning max_tokens
                if "max_tokens" not in payload:
                    payload["max_tokens"] = thinking_budget + 4096
        if tools and _tool_depth < 3:
            payload["tools"] = tools

        content_parts: List[str] = []
        usage_data: Dict[str, Any] = {}
        think_open = False  # tracks whether we've printed the [THINK] header
        # tool_calls_acc: index -> {id, name, arguments_chunks}
        tool_calls_acc: Dict[int, Dict[str, Any]] = {}

        async with self._chat_client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:].strip()
                if raw == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                # Usage arrives in the final chunk when stream_options is set
                if chunk.get("usage"):
                    usage_data = chunk["usage"]

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}

                # reasoning: stream each token to console as it arrives (OpenRouter field name)
                r = delta.get("reasoning") or ""
                if r:
                    if not think_open:
                        print(f"\n{_C_THINK}┌─ THINKING {'─' * _THINK_W}", flush=True)
                        think_open = True
                    print(r, end="", flush=True)

                # response content: buffer silently
                c = delta.get("content") or ""
                if c:
                    content_parts.append(c)

                # tool call deltas: accumulate by index across chunks
                for tc_delta in delta.get("tool_calls") or []:
                    idx = tc_delta.get("index", 0)
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": "", "name": "", "args": ""}
                    if tc_delta.get("id"):
                        tool_calls_acc[idx]["id"] = tc_delta["id"]
                    fn = tc_delta.get("function") or {}
                    if fn.get("name"):
                        tool_calls_acc[idx]["name"] = fn["name"]
                    if fn.get("arguments"):
                        tool_calls_acc[idx]["args"] += fn["arguments"]

        if think_open:
            print(f"\n└{'─' * (_THINK_W + 11)}{_C_RESET}", flush=True)

        # Post-process buffered content for inline <think> tags (some models embed
        # thinking inside the content field rather than using reasoning_content)
        full_text = "".join(content_parts).strip()
        inline_thoughts: List[str] = []
        def _capture(m: re.Match) -> str:
            inline_thoughts.append(m.group(1).strip())
            return ""
        clean_text = _THINK_RE.sub(_capture, full_text).strip()
        if inline_thoughts:
            print(f"\n{_C_THINK}┌─ THINKING (inline tags) {'─' * (_THINK_W - 14)}", flush=True)
            print("---".join(inline_thoughts), flush=True)
            print(f"└{'─' * (_THINK_W + 11)}{_C_RESET}", flush=True)

        first_usage = self._extract_usage({"usage": usage_data})

        # If the model made tool calls, execute them and loop
        if tool_calls_acc and tool_executor is not None and _tool_depth < 3:
            calls = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
            tool_call_objs = [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["args"]},
                }
                for c in calls
            ]
            # Append the assistant turn that triggered the tool calls
            next_messages = list(messages) + [
                {"role": "assistant", "content": clean_text or None, "tool_calls": tool_call_objs}
            ]
            # Execute all tool calls concurrently
            async def _run_one(c: Dict[str, Any]) -> str:
                try:
                    args = json.loads(c["args"]) if c["args"] else {}
                except json.JSONDecodeError:
                    args = {}
                return await tool_executor(c["name"], args)

            results = await asyncio.gather(*[_run_one(c) for c in calls])

            for tc_obj, result in zip(tool_call_objs, results):
                next_messages.append({
                    "role": "tool",
                    "tool_call_id": tc_obj["id"],
                    "content": result,
                })

            continuation, cont_usage = await self.chat_messages(
                next_messages,
                model=model, temperature=temperature,
                top_p=top_p, top_k=top_k, max_tokens=max_tokens,
                thinking_budget=thinking_budget,
                tools=tools, tool_executor=tool_executor,
                _tool_depth=_tool_depth + 1,
            )
            merged = {
                "prompt_tokens":     first_usage["prompt_tokens"] + cont_usage["prompt_tokens"],
                "completion_tokens": first_usage["completion_tokens"] + cont_usage["completion_tokens"],
                "total_tokens":      first_usage["total_tokens"] + cont_usage["total_tokens"],
            }
            return continuation, merged

        return clean_text, first_usage

    async def vision_extract(
        self, image_urls: List[str], prompt: str
    ) -> Tuple[str, Dict[str, int]]:
        """Extract text/description from one or more images in a single vision call."""
        if not self.vision_model:
            raise ValueError("vision_model not configured")

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})

        return await self.chat_messages(
            [{"role": "user", "content": content}], model=self.vision_model
        )

    async def embed(self, text: str) -> Tuple[List[float], Dict[str, int]]:
        resp = await self._embed_client.post(
            f"{self._embed_base_url}/embeddings",
            json={
                "model": self.embedding_model,
                "input": text,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        vector = data["data"][0]["embedding"]
        usage = self._extract_usage(data)
        return vector, usage

    @staticmethod
    def _extract_usage(data: Dict[str, Any]) -> Dict[str, int]:
        u = data.get("usage", {}) or {}
        return {
            "prompt_tokens": int(u.get("prompt_tokens", 0)),
            "completion_tokens": int(u.get("completion_tokens", 0)),
            "total_tokens": int(u.get("total_tokens", 0)),
        }

    async def close(self) -> None:
        await self._chat_client.aclose()
        await self._embed_client.aclose()
