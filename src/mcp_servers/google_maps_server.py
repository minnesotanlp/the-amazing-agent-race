#!/usr/bin/env python3
"""Standalone MCP server exposing Google Maps APIs as tools."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import anyio
import click
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.shared._httpx_utils import create_mcp_http_client
from dotenv import load_dotenv

load_dotenv()

JSONDict = dict[str, Any]
ContentList = list[types.ContentBlock]


def _json_content(payload: Any) -> ContentList:
    """Serialize payload as pretty JSON MCP text content."""

    text = json.dumps(payload, indent=2)
    return [types.TextContent(type="text", text=text)]


def _require_argument(arguments: dict[str, Any], name: str) -> Any:
    if name not in arguments or arguments[name] is None:
        raise ValueError(f"Missing required argument '{name}'")
    return arguments[name]


@dataclass(slots=True)
class Coordinate:
    latitude: float
    longitude: float

    @classmethod
    def from_mapping(cls, value: Any) -> "Coordinate":
        if not isinstance(value, dict):
            raise ValueError("Expected an object with latitude and longitude")
        try:
            lat = float(value["latitude"])
            lng = float(value["longitude"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid latitude/longitude pair") from error
        return cls(latitude=lat, longitude=lng)


class GoogleMapsAPI:
    """Thin async client around Google Maps REST endpoints used by the server."""

    BASE_URL = "https://maps.googleapis.com/maps/api"

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable is not set")
        self._api_key = api_key

    async def _get(self, path: str, params: dict[str, Any]) -> JSONDict:
        query = {key: value for key, value in params.items() if value is not None}
        query["key"] = self._api_key
        async with create_mcp_http_client() as client:
            response = await client.get(
                f"{self.BASE_URL}/{path}", params=query, timeout=30
            )
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _ensure_ok(data: JSONDict, context: str) -> None:
        status = data.get("status")
        if status != "OK":
            message = data.get("error_message") or status or "Unknown error"
            raise RuntimeError(f"{context} failed: {message}")

    async def geocode(self, address: str) -> ContentList:
        data = await self._get("geocode/json", {"address": address})
        self._ensure_ok(data, "Geocoding")
        results = data.get("results") or []
        if not results:
            raise RuntimeError("Geocoding returned no results")
        first = results[0]
        payload = {
            "location": first["geometry"]["location"],
            "formatted_address": first["formatted_address"],
            "place_id": first["place_id"],
        }
        return _json_content(payload)

    async def reverse_geocode(self, coordinate: Coordinate) -> ContentList:
        params = {"latlng": f"{coordinate.latitude},{coordinate.longitude}"}
        data = await self._get("geocode/json", params)
        self._ensure_ok(data, "Reverse geocoding")
        results = data.get("results") or []
        if not results:
            raise RuntimeError("Reverse geocoding returned no results")
        first = results[0]
        payload = {
            "formatted_address": first["formatted_address"],
            "place_id": first["place_id"],
            "address_components": first.get("address_components", []),
        }
        return _json_content(payload)

    async def search_places(
        self,
        query: str,
        location: Optional[Coordinate],
        radius: Optional[int],
    ) -> ContentList:
        params: dict[str, Any] = {"query": query}
        if location:
            params["location"] = f"{location.latitude},{location.longitude}"
        if radius:
            params["radius"] = int(radius)
        data = await self._get("place/textsearch/json", params)
        self._ensure_ok(data, "Place search")
        places = [
            {
                "name": place.get("name"),
                "formatted_address": place.get("formatted_address"),
                "location": place.get("geometry", {}).get("location"),
                "place_id": place.get("place_id"),
                "rating": place.get("rating"),
                "user_ratings_total": place.get("user_ratings_total"),
                "types": place.get("types", []),
            }
            for place in data.get("results", [])
        ]
        return _json_content({"places": places})

    async def place_details(self, place_id: str) -> ContentList:
        data = await self._get("place/details/json", {"place_id": place_id})
        self._ensure_ok(data, "Place details request")
        result = data.get("result") or {}
        payload = {
            "name": result.get("name"),
            "formatted_address": result.get("formatted_address"),
            "location": result.get("geometry", {}).get("location"),
            "formatted_phone_number": result.get("formatted_phone_number"),
            "website": result.get("website"),
            "rating": result.get("rating"),
            "user_ratings_total": result.get("user_ratings_total"),
            "reviews": result.get("reviews"),
            "opening_hours": result.get("opening_hours"),
        }
        return _json_content(payload)

    async def distance_matrix(
        self,
        origins: Iterable[str],
        destinations: Iterable[str],
        mode: Optional[str],
    ) -> ContentList:
        params: dict[str, Any] = {
            "origins": "|".join(origins),
            "destinations": "|".join(destinations),
        }
        params["mode"] = mode or "driving"
        data = await self._get("distancematrix/json", params)
        self._ensure_ok(data, "Distance matrix request")
        rows = [
            {
                "elements": [
                    {
                        "status": element.get("status"),
                        "duration": element.get("duration"),
                        "distance": element.get("distance"),
                    }
                    for element in row.get("elements", [])
                ]
            }
            for row in data.get("rows", [])
        ]
        payload = {
            "origin_addresses": data.get("origin_addresses", []),
            "destination_addresses": data.get("destination_addresses", []),
            "results": rows,
        }
        return _json_content(payload)

    async def elevation(self, locations: list[Coordinate]) -> ContentList:
        location_str = "|".join(f"{loc.latitude},{loc.longitude}" for loc in locations)
        data = await self._get("elevation/json", {"locations": location_str})
        self._ensure_ok(data, "Elevation request")
        payload = {
            "results": [
                {
                    "elevation": result.get("elevation"),
                    "location": result.get("location"),
                    "resolution": result.get("resolution"),
                }
                for result in data.get("results", [])
            ]
        }
        return _json_content(payload)

    async def directions(
        self,
        origin: str,
        destination: str,
        mode: Optional[str],
    ) -> ContentList:
        params: dict[str, Any] = {
            "origin": origin,
            "destination": destination,
            "mode": mode or "driving",
        }
        data = await self._get("directions/json", params)
        self._ensure_ok(data, "Directions request")
        routes_payload = []
        for route in data.get("routes", []):
            legs = route.get("legs") or []
            if not legs:
                continue
            primary_leg = legs[0]
            routes_payload.append(
                {
                    "summary": route.get("summary"),
                    "distance": primary_leg.get("distance"),
                    "duration": primary_leg.get("duration"),
                    "steps": [
                        {
                            "instructions": step.get("html_instructions"),
                            "distance": step.get("distance"),
                            "duration": step.get("duration"),
                            "travel_mode": step.get("travel_mode"),
                        }
                        for step in primary_leg.get("steps", [])
                    ],
                }
            )
        return _json_content({"routes": routes_payload})


MAPS_TOOLS: list[types.Tool] = [
    types.Tool(
        name="maps_geocode",
        title="Geocode Address",
        description=(
            "Convert an address into geographic coordinates. Returns JSON with "
            "`location` (lat/lng), `formatted_address`, and the Google `place_id`."
        ),
        inputSchema={
            "type": "object",
            "required": ["address"],
            "properties": {
                "address": {
                    "type": "string",
                    "description": "The address to geocode",
                }
            },
        },
    ),
    types.Tool(
        name="maps_reverse_geocode",
        title="Reverse Geocode",
        description=(
            "Convert geographic coordinates into a formatted address. The payload includes "
            "`formatted_address`, `place_id`, and the raw `address_components` array."
        ),
        inputSchema={
            "type": "object",
            "required": ["latitude", "longitude"],
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Latitude coordinate",
                },
                "longitude": {
                    "type": "number",
                    "description": "Longitude coordinate",
                },
            },
        },
    ),
    types.Tool(
        name="maps_search_places",
        title="Places Search",
        description=(
            "Search for places using the Google Places Text Search API. Returns JSON with a `places` "
            "array; each entry provides `name`, `formatted_address`, `location`, `place_id`, rating "
            "metadata, and Google place `types`."
        ),
        inputSchema={
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text query describing the place you want to find",
                },
                "location": {
                    "type": "object",
                    "properties": {
                        "latitude": {"type": "number"},
                        "longitude": {"type": "number"},
                    },
                    "description": "Optional center point to bias results",
                },
                "radius": {
                    "type": "number",
                    "description": "Optional search radius in meters (max 50000)",
                },
            },
        },
    ),
    types.Tool(
        name="maps_place_details",
        title="Place Details",
        description=(
            "Fetch detailed information about a Google Maps place by place_id. Useful for retrieving "
            "contact info, website, ratings, opening hours, and canonical address/geometry."
        ),
        inputSchema={
            "type": "object",
            "required": ["place_id"],
            "properties": {
                "place_id": {
                    "type": "string",
                    "description": "The Google Maps place ID to look up",
                }
            },
        },
    ),
    types.Tool(
        name="maps_distance_matrix",
        title="Distance Matrix",
        description=(
            "Calculate travel distances and durations between origin and destination sets. "
            "The response mirrors the Distance Matrix API with `origin_addresses`, `destination_addresses`, "
            "and `results` containing per-pair `distance`, `duration`, and `status` values."
        ),
        inputSchema={
            "type": "object",
            "required": ["origins", "destinations"],
            "properties": {
                "origins": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of origin addresses or coordinates",
                },
                "destinations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of destination addresses or coordinates",
                },
                "mode": {
                    "type": "string",
                    "enum": ["driving", "walking", "bicycling", "transit"],
                    "description": "Optional travel mode",
                },
            },
        },
    ),
    types.Tool(
        name="maps_elevation",
        title="Elevation Lookup",
        description=(
            "Get elevation data for one or more coordinate pairs. Results include an array of objects "
            "with `elevation` (meters), `location`, and `resolution` (meters of vertical accuracy)."
        ),
        inputSchema={
            "type": "object",
            "required": ["locations"],
            "properties": {
                "locations": {
                    "type": "array",
                    "description": "Array of coordinate objects",
                    "items": {
                        "type": "object",
                        "required": ["latitude", "longitude"],
                        "properties": {
                            "latitude": {"type": "number"},
                            "longitude": {"type": "number"},
                        },
                    },
                }
            },
        },
    ),
    types.Tool(
        name="maps_directions",
        title="Directions",
        description=(
            "Get step-by-step directions between two locations. Response JSON exposes a `routes` array "
            "with summary distance/duration plus detailed `steps` (HTML instructions, distance, duration, mode)."
        ),
        inputSchema={
            "type": "object",
            "required": ["origin", "destination"],
            "properties": {
                "origin": {
                    "type": "string",
                    "description": "Starting point address or coordinates",
                },
                "destination": {
                    "type": "string",
                    "description": "Ending point address or coordinates",
                },
                "mode": {
                    "type": "string",
                    "enum": ["driving", "walking", "bicycling", "transit"],
                    "description": "Optional travel mode",
                },
            },
        },
    ),
]


def _parse_coordinate_argument(argument: Any) -> Coordinate:
    return Coordinate.from_mapping(argument)


def _parse_coordinates_argument(argument: Any) -> list[Coordinate]:
    if not isinstance(argument, list) or not argument:
        raise ValueError("Expected at least one location with latitude and longitude")
    return [Coordinate.from_mapping(item) for item in argument]


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
    api_key = os.getenv("GOOGLE_API_KEY")
    maps_api = GoogleMapsAPI(api_key or "")
    server = Server("mcp-google-maps")

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return MAPS_TOOLS

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict[str, Any] | None
    ) -> ContentList:
        args = arguments or {}
        if name == "maps_geocode":
            address = str(_require_argument(args, "address"))
            return await maps_api.geocode(address)
        if name == "maps_reverse_geocode":
            latitude = float(_require_argument(args, "latitude"))
            longitude = float(_require_argument(args, "longitude"))
            return await maps_api.reverse_geocode(Coordinate(latitude, longitude))
        if name == "maps_search_places":
            query = str(_require_argument(args, "query"))
            location = args.get("location")
            radius = args.get("radius")
            coord = (
                _parse_coordinate_argument(location) if location is not None else None
            )
            return await maps_api.search_places(
                query, coord, int(radius) if radius is not None else None
            )
        if name == "maps_place_details":
            place_id = str(_require_argument(args, "place_id"))
            return await maps_api.place_details(place_id)
        if name == "maps_distance_matrix":
            origins = args.get("origins")
            destinations = args.get("destinations")
            if not isinstance(origins, list) or not origins:
                raise ValueError("'origins' must be a non-empty list of strings")
            if not isinstance(destinations, list) or not destinations:
                raise ValueError("'destinations' must be a non-empty list of strings")
            mode = args.get("mode")
            return await maps_api.distance_matrix(
                [str(o) for o in origins],
                [str(d) for d in destinations],
                str(mode) if mode else None,
            )
        if name == "maps_elevation":
            locations = _parse_coordinates_argument(
                _require_argument(args, "locations")
            )
            return await maps_api.elevation(locations)
        if name == "maps_directions":
            origin = str(_require_argument(args, "origin"))
            destination = str(_require_argument(args, "destination"))
            mode = args.get("mode")
            return await maps_api.directions(
                origin, destination, str(mode) if mode else None
            )
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
