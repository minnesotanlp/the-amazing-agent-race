#!/usr/bin/env python3
"""Standalone MCP server exposing Yahoo Finance stock tools via yfinance."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

import anyio
import click
import mcp.types as types

ContentList = list[types.ContentBlock]


def _json_content(payload: Any) -> ContentList:
    """Serialize payload as pretty JSON MCP text content."""
    return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]


def _require_argument(arguments: dict[str, Any], name: str) -> Any:
    if name not in arguments or arguments[name] is None:
        raise ValueError(f"Missing required argument '{name}'")
    return arguments[name]


class StockAPI:
    """Thin async wrapper around yfinance for historical stock data."""

    async def _fetch_day_data(self, ticker: str, date_str: str) -> dict[str, Any] | None:
        """Fetch OHLCV data for a ticker on a given date.

        Uses a 5-day window to handle weekends/holidays — returns the closest
        trading day on or before the requested date.
        """
        import yfinance as yf

        def _sync_fetch() -> dict[str, Any] | None:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            # Fetch a 5-day window to handle weekends/holidays
            start = dt - timedelta(days=5)
            end = dt + timedelta(days=1)
            t = yf.Ticker(ticker)
            hist = t.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
            if hist.empty:
                return None
            # Find the closest trading day on or before the requested date
            hist.index = hist.index.tz_localize(None)
            valid = hist[hist.index <= dt]
            if valid.empty:
                valid = hist
            row = valid.iloc[-1]
            actual_date = valid.index[-1].strftime("%Y-%m-%d")
            return {
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
                "actual_date": actual_date,
            }

        return await anyio.to_thread.run_sync(_sync_fetch)

    async def historical_price(self, ticker: str, date_str: str) -> ContentList:
        """Return the historical closing price of a stock."""
        data = await self._fetch_day_data(ticker, date_str)
        if data is None:
            raise RuntimeError(f"No data found for {ticker} around {date_str}")
        payload = {
            "ticker": ticker,
            "date": data["actual_date"],
            "close_price": round(data["close"], 2),
            "currency": "USD",
        }
        return _json_content(payload)

    async def volume(self, ticker: str, date_str: str) -> ContentList:
        """Return the historical trading volume of a stock."""
        data = await self._fetch_day_data(ticker, date_str)
        if data is None:
            raise RuntimeError(f"No data found for {ticker} around {date_str}")
        payload = {
            "ticker": ticker,
            "date": data["actual_date"],
            "volume": data["volume"],
        }
        return _json_content(payload)


STOCK_TOOLS = [
    types.Tool(
        name="stock_historical_price",
        title="Historical Stock Price",
        description=(
            "Fetch the historical closing price of a stock on a given date. "
            "Returns JSON with `ticker`, `date`, `close_price`, and `currency`. "
            "If the date falls on a weekend or holiday, returns the nearest "
            "prior trading day."
        ),
        inputSchema={
            "type": "object",
            "required": ["ticker", "date"],
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g. AAPL, MSFT, GOOGL)",
                },
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format",
                },
            },
        },
    ),
    types.Tool(
        name="stock_volume",
        title="Historical Stock Volume",
        description=(
            "Fetch the historical trading volume of a stock on a given date. "
            "Returns JSON with `ticker`, `date`, and `volume` (integer). "
            "If the date falls on a weekend or holiday, returns the nearest "
            "prior trading day."
        ),
        inputSchema={
            "type": "object",
            "required": ["ticker", "date"],
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g. AAPL, MSFT, GOOGL)",
                },
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format",
                },
            },
        },
    ),
]


def _build_tools() -> list[types.Tool]:
    return STOCK_TOOLS


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
    api = StockAPI()
    tools = _build_tools()
    from mcp.server.lowlevel import Server

    server = Server("mcp-stock")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> ContentList:
        args = arguments or {}

        if name == "stock_historical_price":
            ticker = str(_require_argument(args, "ticker")).upper()
            date_str = str(_require_argument(args, "date"))
            return await api.historical_price(ticker, date_str)

        if name == "stock_volume":
            ticker = str(_require_argument(args, "ticker")).upper()
            date_str = str(_require_argument(args, "date"))
            return await api.volume(ticker, date_str)

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
