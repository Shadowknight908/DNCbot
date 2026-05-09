"""Async wrapper around the Tavily Search API for model-driven web search."""
from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx

_TAVILY_URL = "https://api.tavily.com/search"

_C_SEND = "\033[96m"
_C_RECV = "\033[36m"
_C_RST  = "\033[0m"

_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information. Use this when you need "
            "up-to-date facts, recent events, or details not in your training data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                }
            },
            "required": ["query"],
        },
    },
}


class TavilyClient:
    def __init__(self, api_key: str, max_results: int = 5, search_depth: str = "basic"):
        self.max_results = max_results
        self.search_depth = search_depth
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )

    def tool_definition(self) -> Dict[str, Any]:
        return _TOOL_DEFINITION

    async def search(self, query: str) -> str:
        print(f"{_C_SEND}[TAVILY  →] {query!r}{_C_RST}", flush=True)
        try:
            resp = await self._client.post(
                _TAVILY_URL,
                json={
                    "query": query,
                    "max_results": self.max_results,
                    "search_depth": self.search_depth,
                },
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
        except Exception as exc:
            print(f"{_C_RECV}[TAVILY  ←] error: {exc}{_C_RST}", flush=True)
            return f"Search failed: {exc}"

        print(f"{_C_RECV}[TAVILY  ←] {len(results)} result(s){_C_RST}", flush=True)
        if not results:
            return "No results found."

        lines = []
        for r in results:
            title   = r.get("title", "")
            url     = r.get("url", "")
            content = r.get("content", "")
            lines.append(f"{title}\n{url}\n{content}")
        return "\n\n".join(lines)

    async def execute(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        if tool_name == "web_search":
            query = tool_args.get("query", "").strip()
            if not query:
                return "Error: no query provided."
            return await self.search(query)
        return f"Unknown tool: {tool_name!r}"

    async def close(self) -> None:
        await self._client.aclose()
