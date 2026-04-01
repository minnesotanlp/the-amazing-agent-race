"""Tool registry: discovers and wraps MCP tools for programmatic execution.

Extracted from the old benchmark/explorer.py — contains only ToolRegistry
and its supporting dataclasses.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Awaitable, Callable, Mapping

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolAvailability:
    """Represents whether a tool can be executed in the current environment."""

    is_available: bool
    reason: str | None = None


@dataclass(slots=True)
class ToolExecutionResult:
    """Captures the result of executing an MCP tool."""

    tool_name: str
    arguments: dict[str, Any]
    output_text: str
    raw_output: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "output_text": self.output_text,
            "raw_output": self._serialize_raw(),
        }

    def _serialize_raw(self) -> Any:
        if isinstance(self.raw_output, list):
            serialized: list[Any] = []
            for block in self.raw_output:
                if hasattr(block, "dict"):
                    serialized.append(block.dict())
                elif hasattr(block, "model_dump"):
                    serialized.append(block.model_dump())
                elif hasattr(block, "__dict__"):
                    serialized.append(dict(block.__dict__))
                else:
                    serialized.append(str(block))
            return serialized
        if hasattr(self.raw_output, "dict"):
            return self.raw_output.dict()
        if hasattr(self.raw_output, "model_dump"):
            return self.raw_output.model_dump()
        return self.raw_output


@dataclass(slots=True)
class ToolSpec:
    """Metadata and executor for an MCP tool."""

    name: str
    title: str
    description: str
    input_schema: Mapping[str, Any]
    executor: Callable[[dict[str, Any]], Awaitable[ToolExecutionResult]]
    availability: ToolAvailability
    origin_module: str

    async def execute(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        if not self.availability.is_available:
            reason = self.availability.reason or "Tool marked as unavailable"
            raise RuntimeError(f"Tool '{self.name}' is unavailable: {reason}")
        return await self.executor(arguments)

    def describe(self) -> dict[str, Any]:
        availability = (
            "available"
            if self.availability.is_available
            else f"unavailable: {self.availability.reason}"
        )
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "input_schema": self.input_schema,
            "availability": availability,
            "origin": self.origin_module,
        }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _coerce_output_text(payload: Any) -> str:
    """Convert tool outputs into a printable string."""

    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        fragments: list[str] = []
        for block in payload:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                fragments.append(text)
                continue
            dict_fn = getattr(block, "dict", None)
            dump_fn = getattr(block, "model_dump", None)
            if callable(dict_fn):
                fragments.append(json.dumps(dict_fn(), indent=2, default=str))
            elif callable(dump_fn):
                fragments.append(json.dumps(dump_fn(), indent=2, default=str))
            else:
                fragments.append(repr(block))
        return "\n\n".join(fragments)
    return json.dumps(payload, indent=2, default=str)


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Discovers and executes MCP tools exposed under mcp.*."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._load_builtin_tools()

    def available_tools(self) -> dict[str, ToolSpec]:
        return dict(self._tools)

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Unknown tool '{name}'")
        return self._tools[name]

    def describe_for_prompt(self) -> str:
        payload = [spec.describe() for spec in self._tools.values()]
        return json.dumps(payload, indent=2, sort_keys=True)

    def _register(self, spec: ToolSpec) -> None:
        logger.debug("Registering MCP tool %s", spec.name)
        self._tools[spec.name] = spec

    def _load_builtin_tools(self) -> None:
        self._register_fetch_tools()
        self._register_weather_tools()
        self._register_code_tools()
        self._register_search_tools()
        self._register_maps_tools()
        self._register_countries_tools()
        self._register_stock_tools()
        self._register_crypto_tools()

    def _register_fetch_tools(self) -> None:
        module_name = "mcp_servers.fetch_server"
        try:
            module = import_module(module_name)
        except ModuleNotFoundError:
            logger.warning("Fetch server module missing; skipping fetch tools")
            return

        builder = getattr(module, "_build_tools", None)
        fetch_class = getattr(module, "FetchTools", None)
        if builder is None or fetch_class is None:
            logger.warning("Fetch server missing required attributes; skipping")
            return

        tools = builder()
        fetcher = fetch_class()

        async def execute(arguments: dict[str, Any]) -> ToolExecutionResult:
            url = arguments.get("url")
            if not isinstance(url, str) or not url.strip():
                raise ValueError("'url' must be a non-empty string")
            raw_html = arguments.get("raw_html", False)
            if raw_html not in (True, False):
                raise ValueError("'raw_html' must be a boolean if provided")
            result = await fetcher.fetch(url.strip(), bool(raw_html))
            output_text = _coerce_output_text(result)
            return ToolExecutionResult(
                tool_name="fetch_webpage",
                arguments={"url": url.strip(), "raw_html": bool(raw_html)},
                output_text=output_text,
                raw_output=result,
            )

        for tool in tools:
            if tool.name != "fetch_webpage":
                continue
            spec = ToolSpec(
                name=tool.name,
                title=tool.title,
                description=tool.description,
                input_schema=tool.inputSchema,
                executor=execute,
                availability=ToolAvailability(is_available=True),
                origin_module=module_name,
            )
            self._register(spec)

    def _register_weather_tools(self) -> None:
        module_name = "mcp_servers.weather_server"
        try:
            module = import_module(module_name)
        except ModuleNotFoundError:
            logger.info("Weather server module missing; skipping")
            return

        builder = getattr(module, "_build_tools", None)
        api_cls = getattr(module, "OpenMeteoAPI", None)
        normalize_select = getattr(module, "_normalize_select", None)
        require_argument = getattr(module, "_require_argument", None)
        if None in (builder, api_cls, normalize_select, require_argument):
            logger.warning("Weather server missing helpers; skipping")
            return

        api = api_cls()
        tools = builder()

        async def execute_historical(arguments: dict[str, Any]) -> ToolExecutionResult:
            latitude = float(require_argument(arguments, "latitude"))
            longitude = float(require_argument(arguments, "longitude"))
            start_date = str(require_argument(arguments, "start_date"))
            end_date = str(require_argument(arguments, "end_date"))
            selectors = normalize_select(arguments.get("select"))
            raw = await api.get_historical(
                latitude, longitude, start_date, end_date, selectors
            )
            output_text = _coerce_output_text(raw)
            return ToolExecutionResult(
                tool_name="weather_historical",
                arguments={
                    "latitude": latitude,
                    "longitude": longitude,
                    "start_date": start_date,
                    "end_date": end_date,
                    "select": selectors,
                },
                output_text=output_text,
                raw_output=raw,
            )

        async def execute_forecast(arguments: dict[str, Any]) -> ToolExecutionResult:
            latitude = float(require_argument(arguments, "latitude"))
            longitude = float(require_argument(arguments, "longitude"))
            selectors = normalize_select(arguments.get("select"))
            raw = await api.get_forecast(latitude, longitude, selectors)
            output_text = _coerce_output_text(raw)
            return ToolExecutionResult(
                tool_name="weather_forecast",
                arguments={
                    "latitude": latitude,
                    "longitude": longitude,
                    "select": selectors,
                },
                output_text=output_text,
                raw_output=raw,
            )

        for tool in tools:
            if tool.name == "weather_historical":
                spec = ToolSpec(
                    name=tool.name,
                    title=tool.title,
                    description=tool.description,
                    input_schema=tool.inputSchema,
                    executor=execute_historical,
                    availability=ToolAvailability(is_available=True),
                    origin_module=module_name,
                )
                self._register(spec)
            elif tool.name == "weather_forecast":
                spec = ToolSpec(
                    name=tool.name,
                    title=tool.title,
                    description=tool.description,
                    input_schema=tool.inputSchema,
                    executor=execute_forecast,
                    availability=ToolAvailability(is_available=True),
                    origin_module=module_name,
                )
                self._register(spec)

    def _register_code_tools(self) -> None:
        module_name = "mcp_servers.code_server"
        try:
            module = import_module(module_name)
        except ModuleNotFoundError:
            logger.info("Code server module missing; skipping")
            return

        builder = getattr(module, "_build_tools", None)
        tools_cls = getattr(module, "PythonCodeTools", None)
        executor_cls = getattr(module, "PythonExecutor", None)
        if None in (builder, tools_cls, executor_cls):
            logger.warning("Code server missing required attributes; skipping")
            return

        tools = builder()
        try:
            code_tools = tools_cls()
            executor = executor_cls()
        except Exception as exc:
            reason = str(exc)
            logger.warning("Code tools unavailable: %s", reason)

            async def unavailable_executor(_arguments: dict[str, Any]) -> ToolExecutionResult:
                raise RuntimeError("Code tools unavailable: " + reason)

            for tool in tools:
                spec = ToolSpec(
                    name=tool.name,
                    title=tool.title,
                    description=tool.description,
                    input_schema=tool.inputSchema,
                    executor=unavailable_executor,
                    availability=ToolAvailability(is_available=False, reason=reason),
                    origin_module=module_name,
                )
                self._register(spec)
            return

        async def execute_generate(arguments: dict[str, Any]) -> ToolExecutionResult:
            description = str(arguments.get("description") or "").strip()
            if not description:
                raise ValueError("'description' must be a non-empty string")
            model = arguments.get("model")
            if model is not None and not isinstance(model, str):
                raise ValueError("'model' must be a string when provided")
            extra_instructions = arguments.get("extra_instructions")
            if extra_instructions is not None and not isinstance(extra_instructions, str):
                raise ValueError("'extra_instructions' must be a string when provided")

            code = await code_tools.generate_python(
                description,
                model if isinstance(model, str) else None,
                extra_instructions,
            )
            return ToolExecutionResult(
                tool_name="python_generate_code",
                arguments={
                    "description": description,
                    "model": model,
                    "extra_instructions": extra_instructions,
                },
                output_text=code,
                raw_output=code,
            )

        async def execute_run(arguments: dict[str, Any]) -> ToolExecutionResult:
            code = arguments.get("code")
            if not isinstance(code, str) or not code.strip():
                raise ValueError("'code' must be a non-empty string")
            timeout = arguments.get("timeout_seconds")
            timeout_value = float(timeout) if timeout is not None else None
            result = await executor.run(code, timeout_value)
            output_text = json.dumps(result, indent=2)
            return ToolExecutionResult(
                tool_name="python_execute_code",
                arguments={"code": code, "timeout_seconds": timeout},
                output_text=output_text,
                raw_output=result,
            )

        executors_map = {
            "python_generate_code": execute_generate,
            "python_execute_code": execute_run,
        }

        for tool in tools:
            executor_fn = executors_map.get(tool.name)
            if executor_fn is None:
                logger.debug("Skipping code tool %s; no executor bound", tool.name)
                continue
            spec = ToolSpec(
                name=tool.name,
                title=tool.title,
                description=tool.description,
                input_schema=tool.inputSchema,
                executor=executor_fn,
                availability=ToolAvailability(is_available=True),
                origin_module=module_name,
            )
            self._register(spec)

    def _register_search_tools(self) -> None:
        module_name = "mcp_servers.search_server"
        try:
            module = import_module(module_name)
        except ModuleNotFoundError:
            logger.info("Search server module missing; skipping")
            return

        build_tool = getattr(module, "_build_serper_tool", None)
        client_cls = getattr(module, "SerperClient", None)
        filter_sections = getattr(module, "_filter_sections", None)
        if None in (build_tool, client_cls, filter_sections):
            logger.warning("Search server missing helpers; skipping")
            return

        tool = build_tool()
        api_key = os.getenv("SERPER_API_KEY") or ""
        try:
            client = client_cls(api_key)
        except Exception as exc:
            reason = str(exc)
            logger.warning("Serper client unavailable: %s", reason)

            async def unavailable_executor(_arguments: dict[str, Any]) -> ToolExecutionResult:
                raise RuntimeError("Search tool unavailable: " + reason)

            spec = ToolSpec(
                name=tool.name,
                title=tool.title,
                description=tool.description,
                input_schema=tool.inputSchema,
                executor=unavailable_executor,
                availability=ToolAvailability(is_available=False, reason=reason),
                origin_module=module_name,
            )
            self._register(spec)
            return

        async def execute(arguments: dict[str, Any]) -> ToolExecutionResult:
            query = str(arguments.get("query") or "")
            if not query.strip():
                raise ValueError("'query' must be a non-empty string")
            location = str(arguments.get("location") or "United States")
            gl = str(arguments.get("gl") or "us")
            hl = str(arguments.get("hl") or "en")
            num_results = int(arguments.get("num_results") or 5)
            sections = arguments.get("sections")
            if sections is not None:
                if not isinstance(sections, list) or not all(
                    isinstance(item, str) for item in sections
                ):
                    raise ValueError("'sections' must be a list of strings when provided")

            data = await client.search(query, location, gl, hl, num_results)
            filtered = filter_sections(data, sections, num_results)
            output_text = json.dumps(filtered, indent=2)
            return ToolExecutionResult(
                tool_name=tool.name,
                arguments={
                    "query": query,
                    "location": location,
                    "gl": gl,
                    "hl": hl,
                    "num_results": num_results,
                    "sections": sections,
                },
                output_text=output_text,
                raw_output=filtered,
            )

        spec = ToolSpec(
            name=tool.name,
            title=tool.title,
            description=tool.description,
            input_schema=tool.inputSchema,
            executor=execute,
            availability=ToolAvailability(is_available=True),
            origin_module=module_name,
        )
        self._register(spec)

    def _register_maps_tools(self) -> None:
        module_name = "mcp_servers.google_maps_server"
        try:
            module = import_module(module_name)
        except ModuleNotFoundError:
            logger.info("Google Maps server module missing; skipping")
            return

        maps_tools = getattr(module, "MAPS_TOOLS", None)
        api_cls = getattr(module, "GoogleMapsAPI", None)
        require_argument = getattr(module, "_require_argument", None)
        parse_coord = getattr(module, "_parse_coordinate_argument", None)
        parse_coords = getattr(module, "_parse_coordinates_argument", None)
        Coordinate = getattr(module, "Coordinate", None)
        if None in (maps_tools, api_cls, require_argument, parse_coord, parse_coords, Coordinate):
            logger.warning("Google Maps server missing helpers; skipping")
            return

        api_key = os.getenv("GOOGLE_API_KEY") or ""
        try:
            maps_api = api_cls(api_key)
        except Exception as exc:
            reason = str(exc)
            logger.warning("Google Maps API unavailable: %s", reason)

            async def unavailable_executor(_arguments: dict[str, Any]) -> ToolExecutionResult:
                raise RuntimeError("Maps tool unavailable: " + reason)

            for tool in maps_tools:
                spec = ToolSpec(
                    name=tool.name,
                    title=tool.title,
                    description=tool.description,
                    input_schema=tool.inputSchema,
                    executor=unavailable_executor,
                    availability=ToolAvailability(is_available=False, reason=reason),
                    origin_module=module_name,
                )
                self._register(spec)
            return

        async def wrap_geocode(arguments: dict[str, Any]) -> ToolExecutionResult:
            address = str(require_argument(arguments, "address"))
            content = await maps_api.geocode(address)
            output_text = _coerce_output_text(content)
            return ToolExecutionResult(
                tool_name="maps_geocode",
                arguments={"address": address},
                output_text=output_text,
                raw_output=content,
            )

        async def wrap_reverse_geocode(arguments: dict[str, Any]) -> ToolExecutionResult:
            latitude = float(require_argument(arguments, "latitude"))
            longitude = float(require_argument(arguments, "longitude"))
            content = await maps_api.reverse_geocode(Coordinate(latitude, longitude))
            output_text = _coerce_output_text(content)
            return ToolExecutionResult(
                tool_name="maps_reverse_geocode",
                arguments={"latitude": latitude, "longitude": longitude},
                output_text=output_text,
                raw_output=content,
            )

        async def wrap_search_places(arguments: dict[str, Any]) -> ToolExecutionResult:
            query = str(require_argument(arguments, "query"))
            location_arg = arguments.get("location")
            radius = arguments.get("radius")
            coordinate = None
            if location_arg is not None:
                coordinate = parse_coord(location_arg)
            content = await maps_api.search_places(
                query,
                coordinate,
                int(radius) if radius is not None else None,
            )
            output_text = _coerce_output_text(content)
            return ToolExecutionResult(
                tool_name="maps_search_places",
                arguments={
                    "query": query,
                    "location": location_arg,
                    "radius": radius,
                },
                output_text=output_text,
                raw_output=content,
            )

        async def wrap_place_details(arguments: dict[str, Any]) -> ToolExecutionResult:
            place_id = str(require_argument(arguments, "place_id"))
            content = await maps_api.place_details(place_id)
            output_text = _coerce_output_text(content)
            return ToolExecutionResult(
                tool_name="maps_place_details",
                arguments={"place_id": place_id},
                output_text=output_text,
                raw_output=content,
            )

        async def wrap_distance_matrix(arguments: dict[str, Any]) -> ToolExecutionResult:
            origins = require_argument(arguments, "origins")
            destinations = require_argument(arguments, "destinations")
            if not isinstance(origins, list) or not origins:
                raise ValueError("'origins' must be a non-empty list")
            if not isinstance(destinations, list) or not destinations:
                raise ValueError("'destinations' must be a non-empty list")
            mode = arguments.get("mode")
            content = await maps_api.distance_matrix(
                [str(origin) for origin in origins],
                [str(dest) for dest in destinations],
                str(mode) if mode else None,
            )
            output_text = _coerce_output_text(content)
            return ToolExecutionResult(
                tool_name="maps_distance_matrix",
                arguments={
                    "origins": origins,
                    "destinations": destinations,
                    "mode": mode,
                },
                output_text=output_text,
                raw_output=content,
            )

        async def wrap_elevation(arguments: dict[str, Any]) -> ToolExecutionResult:
            locations_payload = require_argument(arguments, "locations")
            locations = parse_coords(locations_payload)
            content = await maps_api.elevation(locations)
            output_text = _coerce_output_text(content)
            return ToolExecutionResult(
                tool_name="maps_elevation",
                arguments={"locations": locations_payload},
                output_text=output_text,
                raw_output=content,
            )

        async def wrap_directions(arguments: dict[str, Any]) -> ToolExecutionResult:
            origin = str(require_argument(arguments, "origin"))
            destination = str(require_argument(arguments, "destination"))
            mode = arguments.get("mode")
            content = await maps_api.directions(
                origin,
                destination,
                str(mode) if mode else None,
            )
            output_text = _coerce_output_text(content)
            return ToolExecutionResult(
                tool_name="maps_directions",
                arguments={
                    "origin": origin,
                    "destination": destination,
                    "mode": mode,
                },
                output_text=output_text,
                raw_output=content,
            )

        executors_map = {
            "maps_geocode": wrap_geocode,
            "maps_reverse_geocode": wrap_reverse_geocode,
            "maps_search_places": wrap_search_places,
            "maps_place_details": wrap_place_details,
            "maps_distance_matrix": wrap_distance_matrix,
            "maps_elevation": wrap_elevation,
            "maps_directions": wrap_directions,
        }

        for tool in maps_tools:
            executor_fn = executors_map.get(tool.name)
            if executor_fn is None:
                logger.debug("Skipping Google Maps tool %s; no executor", tool.name)
                continue
            spec = ToolSpec(
                name=tool.name,
                title=tool.title,
                description=tool.description,
                input_schema=tool.inputSchema,
                executor=executor_fn,
                availability=ToolAvailability(is_available=True),
                origin_module=module_name,
            )
            self._register(spec)

    def _register_countries_tools(self) -> None:
        module_name = "mcp_servers.countries_server"
        try:
            module = import_module(module_name)
        except ModuleNotFoundError:
            logger.info("Countries server module missing; skipping")
            return

        tools_list = getattr(module, "COUNTRIES_TOOLS", None)
        api_cls = getattr(module, "RestCountriesAPI", None)
        require_argument = getattr(module, "_require_argument", None)
        if None in (tools_list, api_cls, require_argument):
            logger.warning("Countries server missing helpers; skipping")
            return

        api = api_cls()

        async def execute_population(arguments: dict[str, Any]) -> ToolExecutionResult:
            country = str(require_argument(arguments, "country"))
            raw = await api.population(country)
            output_text = _coerce_output_text(raw)
            return ToolExecutionResult(
                tool_name="countries_population",
                arguments={"country": country},
                output_text=output_text,
                raw_output=raw,
            )

        async def execute_area(arguments: dict[str, Any]) -> ToolExecutionResult:
            country = str(require_argument(arguments, "country"))
            raw = await api.area(country)
            output_text = _coerce_output_text(raw)
            return ToolExecutionResult(
                tool_name="countries_area",
                arguments={"country": country},
                output_text=output_text,
                raw_output=raw,
            )

        executors_map = {
            "countries_population": execute_population,
            "countries_area": execute_area,
        }

        for tool in tools_list:
            executor_fn = executors_map.get(tool.name)
            if executor_fn is None:
                continue
            spec = ToolSpec(
                name=tool.name,
                title=tool.title,
                description=tool.description,
                input_schema=tool.inputSchema,
                executor=executor_fn,
                availability=ToolAvailability(is_available=True),
                origin_module=module_name,
            )
            self._register(spec)

    def _register_stock_tools(self) -> None:
        module_name = "mcp_servers.stock_server"
        try:
            module = import_module(module_name)
        except ModuleNotFoundError:
            logger.info("Stock server module missing; skipping")
            return

        tools_list = getattr(module, "STOCK_TOOLS", None)
        api_cls = getattr(module, "StockAPI", None)
        require_argument = getattr(module, "_require_argument", None)
        if None in (tools_list, api_cls, require_argument):
            logger.warning("Stock server missing helpers; skipping")
            return

        api = api_cls()

        async def execute_historical_price(arguments: dict[str, Any]) -> ToolExecutionResult:
            ticker = str(require_argument(arguments, "ticker")).upper()
            date_str = str(require_argument(arguments, "date"))
            raw = await api.historical_price(ticker, date_str)
            output_text = _coerce_output_text(raw)
            return ToolExecutionResult(
                tool_name="stock_historical_price",
                arguments={"ticker": ticker, "date": date_str},
                output_text=output_text,
                raw_output=raw,
            )

        async def execute_stock_volume(arguments: dict[str, Any]) -> ToolExecutionResult:
            ticker = str(require_argument(arguments, "ticker")).upper()
            date_str = str(require_argument(arguments, "date"))
            raw = await api.volume(ticker, date_str)
            output_text = _coerce_output_text(raw)
            return ToolExecutionResult(
                tool_name="stock_volume",
                arguments={"ticker": ticker, "date": date_str},
                output_text=output_text,
                raw_output=raw,
            )

        executors_map = {
            "stock_historical_price": execute_historical_price,
            "stock_volume": execute_stock_volume,
        }

        for tool in tools_list:
            executor_fn = executors_map.get(tool.name)
            if executor_fn is None:
                continue
            spec = ToolSpec(
                name=tool.name,
                title=tool.title,
                description=tool.description,
                input_schema=tool.inputSchema,
                executor=executor_fn,
                availability=ToolAvailability(is_available=True),
                origin_module=module_name,
            )
            self._register(spec)

    def _register_crypto_tools(self) -> None:
        module_name = "mcp_servers.crypto_server"
        try:
            module = import_module(module_name)
        except ModuleNotFoundError:
            logger.info("Crypto server module missing; skipping")
            return

        tools_list = getattr(module, "CRYPTO_TOOLS", None)
        api_cls = getattr(module, "BinanceAPI", None)
        require_argument = getattr(module, "_require_argument", None)
        if None in (tools_list, api_cls, require_argument):
            logger.warning("Crypto server missing helpers; skipping")
            return

        api = api_cls()

        async def execute_crypto_historical_price(arguments: dict[str, Any]) -> ToolExecutionResult:
            symbol = str(require_argument(arguments, "symbol")).upper()
            date_str = str(require_argument(arguments, "date"))
            raw = await api.historical_price(symbol, date_str)
            output_text = _coerce_output_text(raw)
            return ToolExecutionResult(
                tool_name="crypto_historical_price",
                arguments={"symbol": symbol, "date": date_str},
                output_text=output_text,
                raw_output=raw,
            )

        async def execute_crypto_volume(arguments: dict[str, Any]) -> ToolExecutionResult:
            symbol = str(require_argument(arguments, "symbol")).upper()
            date_str = str(require_argument(arguments, "date"))
            raw = await api.volume(symbol, date_str)
            output_text = _coerce_output_text(raw)
            return ToolExecutionResult(
                tool_name="crypto_volume",
                arguments={"symbol": symbol, "date": date_str},
                output_text=output_text,
                raw_output=raw,
            )

        executors_map = {
            "crypto_historical_price": execute_crypto_historical_price,
            "crypto_volume": execute_crypto_volume,
        }

        for tool in tools_list:
            executor_fn = executors_map.get(tool.name)
            if executor_fn is None:
                continue
            spec = ToolSpec(
                name=tool.name,
                title=tool.title,
                description=tool.description,
                input_schema=tool.inputSchema,
                executor=executor_fn,
                availability=ToolAvailability(is_available=True),
                origin_module=module_name,
            )
            self._register(spec)
