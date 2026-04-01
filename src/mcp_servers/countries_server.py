#!/usr/bin/env python3
"""Standalone MCP server exposing REST Countries API as tools.

Uses the free, no-auth REST Countries API (https://restcountries.com/v3.1/).
"""

from __future__ import annotations

import json
from typing import Any

import anyio
import click
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.shared._httpx_utils import create_mcp_http_client

ContentList = list[types.ContentBlock]


def _json_content(payload: Any) -> ContentList:
    return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]


def _require_argument(arguments: dict[str, Any], name: str) -> Any:
    if name not in arguments or arguments[name] is None:
        raise ValueError(f"Missing required argument '{name}'")
    return arguments[name]


class RestCountriesAPI:
    """Async client for the REST Countries v3.1 API."""

    BASE_URL = "https://restcountries.com/v3.1"

    async def get_by_name(self, name: str) -> dict[str, Any]:
        """Fetch country data by name. Returns the first match."""
        async with create_mcp_http_client() as client:
            response = await client.get(
                f"{self.BASE_URL}/name/{name}",
                params={"fields": "name,population,area,capital,currencies,languages,timezones,region,subregion,latlng"},
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()

        if not isinstance(data, list) or not data:
            raise RuntimeError(f"No country found for '{name}'")
        return data[0]

    async def population(self, country: str) -> ContentList:
        data = await self.get_by_name(country)
        payload = {
            "country": data.get("name", {}).get("common", country),
            "population": data.get("population"),
        }
        return _json_content(payload)

    async def area(self, country: str) -> ContentList:
        data = await self.get_by_name(country)
        payload = {
            "country": data.get("name", {}).get("common", country),
            "area_km2": data.get("area"),
        }
        return _json_content(payload)


COUNTRIES_TOOLS: list[types.Tool] = [
    types.Tool(
        name="countries_population",
        title="Country Population",
        description=(
            "Look up the population of a country by name using the REST Countries API. "
            "Returns JSON with `country` (common name) and `population` (integer)."
        ),
        inputSchema={
            "type": "object",
            "required": ["country"],
            "properties": {
                "country": {
                    "type": "string",
                    "description": "The country name to look up (e.g. 'Nepal', 'France')",
                }
            },
        },
    ),
    types.Tool(
        name="countries_area",
        title="Country Area",
        description=(
            "Look up the area (in km²) of a country by name using the REST Countries API. "
            "Returns JSON with `country` (common name) and `area_km2` (number)."
        ),
        inputSchema={
            "type": "object",
            "required": ["country"],
            "properties": {
                "country": {
                    "type": "string",
                    "description": "The country name to look up (e.g. 'Nepal', 'France')",
                }
            },
        },
    ),
]


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    show_default=True,
)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True)
def main(transport: str, host: str, port: int) -> int:
    api = RestCountriesAPI()
    server = Server("mcp-countries")

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return COUNTRIES_TOOLS

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict[str, Any] | None
    ) -> ContentList:
        args = arguments or {}
        if name == "countries_population":
            country = str(_require_argument(args, "country"))
            return await api.population(country)
        if name == "countries_area":
            country = str(_require_argument(args, "country"))
            return await api.area(country)
        raise ValueError(f"Unknown tool: {name}")

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
            ) as streams:
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
