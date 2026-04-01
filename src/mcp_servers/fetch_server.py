#!/usr/bin/env python3
"""MCP server that fetches web pages and converts them to Markdown."""

from __future__ import annotations

import re
from typing import Any

import anyio
import click
import markdownify
import mcp.types as types
from dotenv import load_dotenv
from httpx import HTTPError
from mcp.server.lowlevel import Server
from mcp.shared._httpx_utils import create_mcp_http_client

load_dotenv()

ContentList = list[types.ContentBlock]


_DROP_CONTENT_TAGS = (
    "script",
    "style",
    "noscript",
    "template",
    "canvas",
    "math",
    "iframe",
    "object",
    "embed",
    "applet",
    "audio",
    "video",
)

_DROP_EMPTY_TAGS = (
    "meta",
    "link",
    "base",
    "source",
    "track",
    "param",
    "area",
    "col",
    "input",
)


def _sanitize_html(html: str) -> str:
    """Remove non-textual tags plus their payloads while preserving links and images."""
    if not html:
        return html

    cleaned = html
    for tag in _DROP_CONTENT_TAGS:
        pattern = re.compile(
            rf"<{tag}(?:\s[^>]*)?>.*?</{tag}\s*>", flags=re.IGNORECASE | re.DOTALL
        )
        # Re-run until no matches remain to clear nested occurrences.
        while True:
            cleaned, count = pattern.subn("", cleaned)
            if count == 0:
                break

    for tag in _DROP_EMPTY_TAGS:
        cleaned = re.sub(
            rf"<{tag}(?:\s[^>]*)?/?>", "", cleaned, flags=re.IGNORECASE
        )

    # Drop HTML comments to avoid stray scripts in comment blocks.
    cleaned = re.sub(r"<!--.*?-->", "", cleaned, flags=re.DOTALL)
    return cleaned


class FetchTools:
    """Utilities for retrieving web pages and exposing Markdown or raw HTML."""

    DEFAULT_TIMEOUT = 30

    async def fetch(self, url: str, raw_html: bool) -> str:
        async with create_mcp_http_client() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    timeout=self.DEFAULT_TIMEOUT,
                    headers={
                        "User-Agent": "AARFetch/1.0 (+https://github.com/TheAmazingAgentRace)",
                    },
                )
                response.raise_for_status()
            except HTTPError as exc:  # pragma: no cover - network failure
                raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc

        html = response.text
        html = _sanitize_html(html)
        if raw_html:
            return html
        markdown = markdownify.markdownify(
            html,
            heading_style=markdownify.ATX,
            strip=["script", "style", "noscript"],
        ).strip()
        return markdown or "(No readable content found.)"


def _text_content(text: str) -> ContentList:
    return [types.TextContent(type="text", text=text)]


def _build_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="fetch_webpage",
            title="Fetch Webpage",
            description=(
                "Fetch the contents of a URL and return a Markdown representation of the page."
                " Set 'raw_html' to true to receive the original HTML instead."
            ),
            inputSchema={
                "type": "object",
                "required": ["url"],
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Absolute URL to retrieve.",
                    },
                    "raw_html": {
                        "type": "boolean",
                        "description": "Return raw HTML instead of Markdown.",
                        "default": False,
                    },
                },
            },
        )
    ]


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
    fetch_tools = FetchTools()
    tools = _build_tools()
    server = Server("mcp-fetch")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> ContentList:
        if name != "fetch_webpage":
            raise ValueError(f"Unknown tool: {name}")

        args = arguments or {}
        url = args.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ValueError("'url' must be a non-empty string")
        url = url.strip()

        raw_flag = args.get("raw_html", False)
        if raw_flag is not None and not isinstance(raw_flag, bool):
            raise ValueError("'raw_html' must be a boolean when provided")

        result = await fetch_tools.fetch(url, bool(raw_flag))
        return _text_content(result)

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
        from mcp.server.stdio import stdio_server  # type: ignore

        async def run_stdio() -> None:
            async with stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream, write_stream, server.create_initialization_options()
                )

        anyio.run(run_stdio)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
