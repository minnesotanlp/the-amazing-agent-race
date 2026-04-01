#!/usr/bin/env python3
"""Standalone MCP server exposing Open-Meteo weather tools."""

from __future__ import annotations

import json
from typing import Any, Iterable

import anyio
import click
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.shared._httpx_utils import create_mcp_http_client

JSONDict = dict[str, Any]
ContentList = list[types.ContentBlock]


def _json_content(payload: Any) -> ContentList:
    """Serialize payload as pretty JSON MCP text content."""

    return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]


def _require_argument(arguments: dict[str, Any], name: str) -> Any:
    if name not in arguments or arguments[name] is None:
        raise ValueError(f"Missing required argument '{name}'")
    return arguments[name]


def _extract_by_paths(data: JSONDict, selectors: Iterable[str]) -> dict[str, Any]:
    """Return values mapped by dotted selector strings, falling back to None."""

    result: dict[str, Any] = {}
    for selector in selectors:
        result[selector] = _resolve_selector(data, selector)
    return result


def _resolve_selector(data: Any, selector: str) -> Any:
    """Resolve dot + bracket notation (e.g. daily.values[0]) into a JSON value."""

    current = data
    for segment in selector.split("."):
        if segment == "":
            return None
        for token in _iter_segment_tokens(segment):
            if current is None:
                return None
            try:
                if isinstance(token, int):
                    current = current[token]
                else:
                    current = current[token]
            except (KeyError, IndexError, TypeError):
                return None
    return current


def _iter_segment_tokens(segment: str) -> Iterable[int | str]:
    """Yield dictionary keys and list indices for a single selector segment."""

    buffer: list[str] = []
    i = 0
    while i < len(segment):
        char = segment[i]
        if char == "[":
            if buffer:
                yield "".join(buffer)
                buffer.clear()
            closing = segment.find("]", i)
            if closing == -1:
                return
            index_str = segment[i + 1 : closing]
            try:
                yield int(index_str)
            except ValueError:
                return
            i = closing + 1
        else:
            buffer.append(char)
            i += 1
    if buffer:
        yield "".join(buffer)


class OpenMeteoAPI:
    """Thin async client around the Open-Meteo endpoints used by this server."""

    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    async def get_historical(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        selectors: list[str] | None,
    ) -> ContentList:
        """Return Open-Meteo archive data as MCP text content.

        The payload mirrors the `/v1/archive` response with fields such as
        `latitude`, `longitude`, `generationtime_ms`, `timezone`, `daily_units`,
        and `daily`. Within `daily`, the arrays `time`,
        `apparent_temperature_mean`, `apparent_temperature_min`,
        `apparent_temperature_max`, `rain_sum`, `precipitation_sum`,
        `snowfall_sum`, `precipitation_hours`, `sunrise`, `sunset`,
        `daylight_duration`, and `sunshine_duration` are included. When
        selectors are supplied, the JSON instead maps each selector to its
        resolved value (or null).
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(
                [
                    "apparent_temperature_mean",
                    "rain_sum",
                    "precipitation_sum",
                    "snowfall_sum",
                    "precipitation_hours",
                    "sunrise",
                    "sunset",
                    "daylight_duration",
                    "sunshine_duration",
                    "apparent_temperature_min",
                    "apparent_temperature_max",
                ]
            ),
            "timezone": "auto",
        }

        data = await self._fetch_json(self.ARCHIVE_URL, params, "historical weather")
        payload = _extract_by_paths(data, selectors) if selectors else data
        return _json_content(payload)

    async def get_forecast(
        self,
        latitude: float,
        longitude: float,
        selectors: list[str] | None,
    ) -> ContentList:
        """Return Open-Meteo forecast data as MCP text content.

        The JSON matches the `/v1/forecast` response and includes top-level
        metadata plus `current_units`, `current` (e.g. `temperature_2m`,
        `relative_humidity_2m`, `weather_code`), and `daily_units`/`daily`
        arrays (`time`, `temperature_2m_max`, `temperature_2m_min`, `rain_sum`,
        `sunrise`, `sunset`, `weather_code`). Hourly data is omitted unless the
        API returns it by default. Providing selectors yields a mapping from
        each selector to the extracted value.
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": ",".join(
                [
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "rain_sum",
                    "sunrise",
                    "sunset",
                    "weather_code",
                ]
            ),
            "current": ",".join(
                ["temperature_2m", "relative_humidity_2m", "weather_code"]
            ),
            "timezone": "auto",
        }

        data = await self._fetch_json(self.FORECAST_URL, params, "weather forecast")
        payload = _extract_by_paths(data, selectors) if selectors else data
        return _json_content(payload)

    async def _fetch_json(
        self, url: str, params: dict[str, Any], context: str
    ) -> JSONDict:
        async with create_mcp_http_client() as client:
            try:
                response = await client.get(url, params=params, timeout=30)
                response.raise_for_status()
            except Exception as exc:  # pragma: no cover - network failure
                raise RuntimeError(f"Failed to fetch {context}: {exc}") from exc
        return response.json()


def _build_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="weather_historical",
            title="Historical Weather",
            description=(
                "Fetch historical daily aggregates from Open-Meteo. Returns JSON "
                "with standard archive metadata plus daily arrays (time, apparent"
                " temperatures, precipitation totals, daylight metrics). When "
                "`select` is provided the response maps selectors to values."
            ),
            inputSchema={
                "type": "object",
                "required": ["latitude", "longitude", "start_date", "end_date"],
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "Latitude in decimal degrees",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Longitude in decimal degrees",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date (inclusive) in YYYY-MM-DD",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date (inclusive) in YYYY-MM-DD",
                    },
                    "select": {
                        "type": "array",
                        "description": "Optional dotted paths into the JSON response",
                        "items": {"type": "string"},
                    },
                },
            },
        ),
        types.Tool(
            name="weather_forecast",
            title="Weather Forecast",
            description=(
                "Fetch the current conditions and daily forecast from Open-Meteo. "
                "Returns JSON containing current metrics (temperature, humidity, "
                "weather code) and daily arrays (time, max/min temperature, rain, "
                "sunrise/sunset, weather code), or selector results if `select` is set."
            ),
            inputSchema={
                "type": "object",
                "required": ["latitude", "longitude"],
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "Latitude in decimal degrees",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Longitude in decimal degrees",
                    },
                    "select": {
                        "type": "array",
                        "description": "Optional dotted paths into the JSON response",
                        "items": {"type": "string"},
                    },
                },
            },
        ),
    ]


def _normalize_select(select: Any) -> list[str] | None:
    if select is None:
        return None
    if not isinstance(select, list):
        raise ValueError("'select' must be a list of strings when provided")
    normalized: list[str] = []
    for item in select:
        if not isinstance(item, str):
            raise ValueError("All select entries must be strings")
        normalized.append(item)
    return normalized


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
    api = OpenMeteoAPI()
    tools = _build_tools()
    server = Server("mcp-weather")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> ContentList:
        args = arguments or {}

        if name == "weather_historical":
            latitude = float(_require_argument(args, "latitude"))
            longitude = float(_require_argument(args, "longitude"))
            start_date = str(_require_argument(args, "start_date"))
            end_date = str(_require_argument(args, "end_date"))
            selectors = _normalize_select(args.get("select"))
            return await api.get_historical(
                latitude, longitude, start_date, end_date, selectors
            )

        if name == "weather_forecast":
            latitude = float(_require_argument(args, "latitude"))
            longitude = float(_require_argument(args, "longitude"))
            selectors = _normalize_select(args.get("select"))
            return await api.get_forecast(latitude, longitude, selectors)

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
