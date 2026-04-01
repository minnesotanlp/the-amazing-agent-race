#!/usr/bin/env python3
"""Standalone MCP server exposing Binance public API crypto tools."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import anyio
import click
import mcp.types as types
from mcp.shared._httpx_utils import create_mcp_http_client

ContentList = list[types.ContentBlock]


def _json_content(payload: Any) -> ContentList:
    """Serialize payload as pretty JSON MCP text content."""
    return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]


def _require_argument(arguments: dict[str, Any], name: str) -> Any:
    if name not in arguments or arguments[name] is None:
        raise ValueError(f"Missing required argument '{name}'")
    return arguments[name]


class BinanceAPI:
    """Thin async client around the Binance public klines endpoint."""

    # Use binance.us for US-accessible public API; falls back to binance.com
    KLINES_URLS = [
        "https://api.binance.us/api/v3/klines",
        "https://api.binance.com/api/v3/klines",
    ]

    async def _fetch_daily_kline(self, symbol: str, date_str: str) -> list | None:
        """Fetch the daily kline (candlestick) for a symbol on a given date.

        Returns the raw kline array:
        [openTime, open, high, low, close, volume, closeTime,
         quoteAssetVolume, numberOfTrades, takerBuyBaseVol, takerBuyQuoteVol, ignore]
        """
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_ms = int(dt.timestamp() * 1000)
        # End is start of next day
        end_ms = start_ms + 86_400_000

        params = {
            "symbol": symbol.upper(),
            "interval": "1d",
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1,
        }

        last_exc: Exception | None = None
        data = None
        async with create_mcp_http_client() as client:
            for url in self.KLINES_URLS:
                try:
                    response = await client.get(url, params=params, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    break
                except Exception as exc:
                    last_exc = exc
                    continue

        if data is None:
            raise RuntimeError(
                f"Failed to fetch Binance kline for {symbol} on {date_str}: {last_exc}"
            )
        if not data:
            return None
        return data[0]

    async def historical_price(self, symbol: str, date_str: str) -> ContentList:
        """Return the historical closing price of a crypto pair."""
        kline = await self._fetch_daily_kline(symbol, date_str)
        if kline is None:
            raise RuntimeError(f"No kline data for {symbol} on {date_str}")
        payload = {
            "symbol": symbol.upper(),
            "date": date_str,
            "close_price": float(kline[4]),
        }
        return _json_content(payload)

    async def volume(self, symbol: str, date_str: str) -> ContentList:
        """Return the 24h trading volume of a crypto pair."""
        kline = await self._fetch_daily_kline(symbol, date_str)
        if kline is None:
            raise RuntimeError(f"No kline data for {symbol} on {date_str}")
        payload = {
            "symbol": symbol.upper(),
            "date": date_str,
            "volume": float(kline[5]),
            "quote_volume": float(kline[7]),
        }
        return _json_content(payload)


CRYPTO_TOOLS = [
    types.Tool(
        name="crypto_historical_price",
        title="Historical Crypto Price",
        description=(
            "Fetch the historical closing price of a cryptocurrency trading "
            "pair on a given date using the Binance API. Returns JSON with "
            "`symbol`, `date`, and `close_price`. No API key required."
        ),
        inputSchema={
            "type": "object",
            "required": ["symbol", "date"],
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": (
                        "Trading pair symbol (e.g. BTCUSDT, ETHUSDT, BNBUSDT)"
                    ),
                },
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format (after 2017-08-17)",
                },
            },
        },
    ),
    types.Tool(
        name="crypto_volume",
        title="Historical Crypto Volume",
        description=(
            "Fetch the 24h trading volume of a cryptocurrency pair on a "
            "given date. Returns JSON with `symbol`, `date`, `volume` "
            "(base asset), and `quote_volume` (quote asset)."
        ),
        inputSchema={
            "type": "object",
            "required": ["symbol", "date"],
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": (
                        "Trading pair symbol (e.g. BTCUSDT, ETHUSDT, BNBUSDT)"
                    ),
                },
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format (after 2017-08-17)",
                },
            },
        },
    ),
]


def _build_tools() -> list[types.Tool]:
    return CRYPTO_TOOLS


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
    api = BinanceAPI()
    tools = _build_tools()
    from mcp.server.lowlevel import Server

    server = Server("mcp-crypto")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> ContentList:
        args = arguments or {}

        if name == "crypto_historical_price":
            symbol = str(_require_argument(args, "symbol")).upper()
            date_str = str(_require_argument(args, "date"))
            return await api.historical_price(symbol, date_str)

        if name == "crypto_volume":
            symbol = str(_require_argument(args, "symbol")).upper()
            date_str = str(_require_argument(args, "date"))
            return await api.volume(symbol, date_str)

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
