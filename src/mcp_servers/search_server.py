#!/usr/bin/env python3
"""Standalone MCP server providing Google search results via the Serper API."""

from __future__ import annotations

import json
import os
from typing import Any

import anyio
import click
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.shared._httpx_utils import create_mcp_http_client


ContentList = list[types.ContentBlock]
JSONDict = dict[str, Any]


def _json_content(payload: Any) -> ContentList:
    """Serialize payload into a text content block."""

    return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]


class SerperClient:
    """Async client for the Serper Google Search API."""

    BASE_URL = "https://google.serper.dev/search"

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise RuntimeError("SERPER_API_KEY environment variable is not set")
        self._api_key = api_key

    async def search(
        self,
        query: str,
        location: str,
        gl: str,
        hl: str,
        num_results: int,
    ) -> JSONDict:
        payload = {
            "q": query,
            "location": location,
            "gl": gl,
            "hl": hl,
            "num": max(1, min(num_results, 10)),
        }
        headers = {"X-API-KEY": self._api_key, "Content-Type": "application/json"}

        async with create_mcp_http_client() as client:
            response = await client.post(
                self.BASE_URL,
                json=payload,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()


def _build_serper_tool() -> types.Tool:
    return types.Tool(
        name="serper_google_search",
        title="Serper Google Search",
        description=(
            "Run a Google Search using the Serper API. Returns structured JSON "
            'containing AI overviews ("answerBox", if available), organic results, answer boxes, '
            "top stories, discussions, videos, shopping data, and related entities."
            " Each section mirrors Serper's response format and can be filtered using "
            "the 'sections' argument."
        ),
        inputSchema={
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to send to Google via Serper.",
                },
                "location": {
                    "type": "string",
                    "description": "Geographic location for the search.",
                    "default": "United States",
                },
                "gl": {
                    "type": "string",
                    "description": "Country code (gl parameter) for Google search.",
                    "default": "us",
                },
                "hl": {
                    "type": "string",
                    "description": "UI language (hl parameter) for Google search.",
                    "default": "en",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return from each section (1-10).",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                },
                "sections": {
                    "type": "array",
                    "description": (
                        "Optional list of response sections to include (e.g. 'answerBox', 'organic', "
                        "'topStories'). When omitted the full response is returned."
                    ),
                    "items": {"type": "string"},
                },
            },
        },
    )


def _filter_sections(
    data: JSONDict, sections: list[str] | None, num_results: int
) -> JSONDict:
    """Return only the selected sections, truncating list-based entries."""

    if not sections:
        return data

    filtered: JSONDict = {}
    for section in sections:
        value = data.get(section)
        if isinstance(value, list):
            filtered[section] = value[:num_results]
        else:
            filtered[section] = value
    return filtered


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    show_default=True,
)
@click.option(
    "--host", default="127.0.0.1", show_default=True, help="Host for SSE transport"
)
@click.option("--port", default=8000, show_default=True, help="Port for SSE transport")
def main(transport: str, host: str, port: int) -> int:
    api_key = os.getenv("SERPER_API_KEY") or ""
    client = SerperClient(api_key)
    tool = _build_serper_tool()
    server = Server("mcp-serper")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [tool]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> ContentList:
        if name != "serper_google_search":
            raise ValueError(f"Unknown tool: {name}")
        args = arguments or {}
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("'query' must be a non-empty string")
        location = str(args.get("location") or "United States")
        gl = str(args.get("gl") or "us")
        hl = str(args.get("hl") or "en")
        num_results = int(args.get("num_results") or 5)
        sections = args.get("sections")
        if sections is not None:
            if not isinstance(sections, list) or not all(
                isinstance(item, str) for item in sections
            ):
                raise ValueError("'sections' must be a list of strings when provided")

        data = await client.search(query, location, gl, hl, num_results)
        filtered = _filter_sections(data, sections, num_results)
        return _json_content(filtered)

    if transport == "sse":
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import Response
        from starlette.routing import Mount, Route

        sse = SseServerTransport("/messages/")

        async def handle_sse(request: Request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:  # type: ignore[attr-defined]
                await server.run(
                    streams[0], streams[1], server.create_initialization_options()
                )
            return Response()

        starlette_app = Starlette(
            debug=False,
            routes=[
                Route("/messages/", handle_sse, methods=["GET"]),
                Mount("/", app=sse.app),
            ],
        )

        import uvicorn

        uvicorn.run(starlette_app, host=host, port=port)
        return 0
    else:
        from mcp.server.stdio import stdio_server

        async def run_stdio() -> None:
            async with stdio_server() as streams:
                await server.run(
                    streams[0], streams[1], server.create_initialization_options()
                )

        anyio.run(run_stdio)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
