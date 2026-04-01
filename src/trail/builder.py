"""Trail construction with LLM-planned coherent themes.

Three-phase build:
  Phase 1 — _plan_route(): LLM plans a thematic route through crawled pages.
  Phase 2 — _build_stops_from_plan(): Convert planned route into Stop objects.
  Phase 3 — _build_bridges(): Connect consecutive stops.
"""

from __future__ import annotations

import json
import logging
import random
import re
import uuid
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

from openai import OpenAI

from trail.extractor import ExtractableFact, FactExtractor
from trail.models import (
    Bridge,
    Stop,
    Trail,
    TrailDifficulty,
    TrailDifficultyConfig,
)
from trail.wiki_graph import WikiGraph

if TYPE_CHECKING:
    from mcp_servers.registry import ToolRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool stop templates
# ---------------------------------------------------------------------------

TOOL_TEMPLATES = [
    {
        "name": "geocode_elevation",
        "requires": ["location_name"],
        "produces": "number",
        "description": "Look up the elevation at {location}",
        "chain_builder": "_build_geocode_elevation_chain",
    },
    {
        "name": "geocode_weather_historical",
        "requires": ["location_name", "date"],
        "produces": "number",
        "description": "Find the historical max temperature at {location} on {date}",
        "chain_builder": "_build_geocode_weather_chain",
    },
    {
        "name": "geocode_distance",
        "requires": ["location_name_a", "location_name_b"],
        "produces": "number",
        "description": "Calculate the driving distance between {location_a} and {location_b}",
        "chain_builder": "_build_geocode_distance_chain",
    },
    {
        "name": "date_computation",
        "requires": ["date"],
        "produces": "number",
        "description": "Calculate the number of days between {date} and {reference_date}",
        "chain_builder": "_build_date_computation_chain",
    },
    {
        "name": "geocode_directions_duration",
        "requires": ["location_name_a", "location_name_b"],
        "produces": "number",
        "description": "Get the driving duration in minutes between {location_a} and {location_b}",
        "chain_builder": "_build_geocode_directions_duration_chain",
    },
    {
        "name": "geocode_weather_precipitation",
        "requires": ["location_name", "date"],
        "produces": "number",
        "description": "Find the total precipitation at {location} on {date}",
        "chain_builder": "_build_geocode_weather_precipitation_chain",
    },
    {
        "name": "math_conversion",
        "requires": ["numeric_value"],
        "produces": "number",
        "description": "Convert {value} using a unit conversion and extract a digit",
        "chain_builder": "_build_math_conversion_chain",
    },
    {
        "name": "nearby_poi_count",
        "requires": ["location_name"],
        "produces": "number",
        "description": "Count the number of {poi_type} near {location}",
        "chain_builder": "_build_nearby_poi_count_chain",
    },
    {
        "name": "place_rating",
        "requires": ["location_name"],
        "produces": "number",
        "description": "Look up the Google Maps rating of {location}",
        "chain_builder": "_build_place_rating_chain",
    },
    {
        "name": "country_population",
        "requires": ["country_name"],
        "produces": "number",
        "description": "Look up the population of {country}",
        "chain_builder": "_build_country_population_chain",
    },
    {
        "name": "country_area",
        "requires": ["country_name"],
        "produces": "number",
        "description": "Look up the area (km²) of {country}",
        "chain_builder": "_build_country_area_chain",
    },
    {
        "name": "historical_snowfall",
        "requires": ["location_name", "date"],
        "produces": "number",
        "description": "Find the snowfall at {location} on {date}",
        "chain_builder": "_build_historical_snowfall_chain",
    },
    {
        "name": "historical_sunshine",
        "requires": ["location_name", "date"],
        "produces": "number",
        "description": "Find the sunshine duration at {location} on {date}",
        "chain_builder": "_build_historical_sunshine_chain",
    },
    {
        "name": "stock_price",
        "requires": ["ticker_symbol", "date"],
        "produces": "number",
        "description": "Look up the historical closing price of {ticker} on {date}",
        "chain_builder": "_build_stock_price_chain",
    },
    {
        "name": "stock_volume",
        "requires": ["ticker_symbol", "date"],
        "produces": "number",
        "description": "Look up the trading volume of {ticker} on {date}",
        "chain_builder": "_build_stock_volume_chain",
    },
    {
        "name": "crypto_price",
        "requires": ["crypto_symbol", "date"],
        "produces": "number",
        "description": "Look up the historical closing price of {symbol} on {date}",
        "chain_builder": "_build_crypto_price_chain",
    },
    {
        "name": "crypto_volume",
        "requires": ["crypto_symbol", "date"],
        "produces": "number",
        "description": "Look up the trading volume of {symbol} on {date}",
        "chain_builder": "_build_crypto_volume_chain",
    },
]

COMPUTE_EXPRESSIONS = [
    ("digital_root", "def digital_root(n):\n    n = abs(int(n))\n    return n if n == 0 else ((n - 1) % 9) + 1\nresult = digital_root({sum_expr})"),
    ("mod10", "result = ({sum_expr}) % 10"),
    ("abs_diff_mod10", "result = abs({a} - {b}) % 10"),
    ("product_mod10", "result = ({a} * {b}) % 10"),
]


# ---------------------------------------------------------------------------
# Tool chain builders
# ---------------------------------------------------------------------------


def _build_geocode_elevation_chain(location: str) -> list[dict[str, Any]]:
    return [
        {
            "tool_name": "maps_geocode",
            "arguments": {"address": location},
            "expected_output": None,
            "output_key": "coords",
        },
        {
            "tool_name": "maps_elevation",
            "arguments": {"__from_previous_as_locations": "coords"},
            "expected_output": None,
            "output_key": "elevation",
        },
    ]


def _build_geocode_weather_chain(
    location: str, event_date: str
) -> list[dict[str, Any]]:
    return [
        {
            "tool_name": "maps_geocode",
            "arguments": {"address": location},
            "expected_output": None,
            "output_key": "coords",
        },
        {
            "tool_name": "weather_historical",
            "arguments": {"__from_previous": "coords", "date": event_date},
            "expected_output": None,
            "output_key": "temperature",
        },
    ]


def _build_geocode_distance_chain(
    location_a: str, location_b: str
) -> list[dict[str, Any]]:
    return [
        {
            "tool_name": "maps_geocode",
            "arguments": {"address": location_a},
            "expected_output": None,
            "output_key": "coords_a",
        },
        {
            "tool_name": "maps_geocode",
            "arguments": {"address": location_b},
            "expected_output": None,
            "output_key": "coords_b",
        },
        {
            "tool_name": "maps_distance_matrix",
            "arguments": {
                "__from_previous_as_origins_destinations": ["coords_a", "coords_b"],
            },
            "expected_output": None,
            "output_key": "distance_km",
        },
    ]


def _build_date_computation_chain(
    event_date: str, reference_date: str
) -> list[dict[str, Any]]:
    code = (
        f"from datetime import date\n"
        f"d1 = date.fromisoformat('{event_date}')\n"
        f"d2 = date.fromisoformat('{reference_date}')\n"
        f"result = abs((d2 - d1).days)\n"
        f"print(result)"
    )
    return [
        {
            "tool_name": "python_execute_code",
            "arguments": {"code": code},
            "expected_output": None,
            "output_key": "days",
        },
    ]


def _build_geocode_directions_duration_chain(
    location_a: str, location_b: str
) -> list[dict[str, Any]]:
    """Build a maps_directions chain that extracts driving duration in minutes."""
    return [
        {
            "tool_name": "maps_directions",
            "arguments": {"origin": location_a, "destination": location_b, "mode": "driving"},
            "expected_output": None,
            "output_key": "duration",
        },
    ]


def _build_geocode_weather_precipitation_chain(
    location: str, event_date: str
) -> list[dict[str, Any]]:
    """Build a maps_geocode -> weather_historical chain extracting precipitation_sum."""
    return [
        {
            "tool_name": "maps_geocode",
            "arguments": {"address": location},
            "expected_output": None,
            "output_key": "coords",
        },
        {
            "tool_name": "weather_historical",
            "arguments": {
                "__from_previous": "coords",
                "start_date": event_date,
                "end_date": event_date,
                "select": ["daily.precipitation_sum[0]"],
            },
            "expected_output": None,
            "output_key": "precipitation",
        },
    ]


def _build_math_conversion_chain(
    value: float, from_unit: str, to_unit: str, factor: float
) -> list[dict[str, Any]]:
    """Build a python_execute_code chain for unit conversion."""
    code = (
        f"# Convert {from_unit} to {to_unit}\n"
        f"value = {value}\n"
        f"converted = value * {factor}\n"
        f"result = int(converted)\n"
        f"print(result)"
    )
    return [
        {
            "tool_name": "python_execute_code",
            "arguments": {"code": code},
            "expected_output": None,
            "output_key": "converted",
        },
    ]


# POI types for nearby_poi_count
_POI_TYPES = [
    "museums", "restaurants", "parks", "hotels", "hospitals",
    "universities", "libraries", "stadiums", "temples", "churches",
]


def _build_nearby_poi_count_chain(
    location: str, poi_type: str
) -> list[dict[str, Any]]:
    """Search for POIs near a location and count them."""
    return [
        {
            "tool_name": "maps_search_places",
            "arguments": {"query": f"{poi_type} near {location}"},
            "expected_output": None,
            "output_key": "poi_count",
        },
    ]


def _build_place_rating_chain(location: str) -> list[dict[str, Any]]:
    """Search for a place and extract its Google rating."""
    return [
        {
            "tool_name": "maps_search_places",
            "arguments": {"query": location},
            "expected_output": None,
            "output_key": "place_rating",
        },
    ]


def _build_country_population_chain(country: str) -> list[dict[str, Any]]:
    """Look up country population via REST Countries API."""
    return [
        {
            "tool_name": "countries_population",
            "arguments": {"country": country},
            "expected_output": None,
            "output_key": "population",
        },
    ]


def _build_country_area_chain(country: str) -> list[dict[str, Any]]:
    """Look up country area via REST Countries API."""
    return [
        {
            "tool_name": "countries_area",
            "arguments": {"country": country},
            "expected_output": None,
            "output_key": "area_km2",
        },
    ]


def _build_historical_snowfall_chain(
    location: str, event_date: str
) -> list[dict[str, Any]]:
    """Geocode → weather_historical with snowfall_sum selector."""
    return [
        {
            "tool_name": "maps_geocode",
            "arguments": {"address": location},
            "expected_output": None,
            "output_key": "coords",
        },
        {
            "tool_name": "weather_historical",
            "arguments": {
                "__from_previous": "coords",
                "start_date": event_date,
                "end_date": event_date,
                "select": ["daily.snowfall_sum[0]"],
            },
            "expected_output": None,
            "output_key": "snowfall",
        },
    ]


def _build_historical_sunshine_chain(
    location: str, event_date: str
) -> list[dict[str, Any]]:
    """Geocode → weather_historical with sunshine_duration selector."""
    return [
        {
            "tool_name": "maps_geocode",
            "arguments": {"address": location},
            "expected_output": None,
            "output_key": "coords",
        },
        {
            "tool_name": "weather_historical",
            "arguments": {
                "__from_previous": "coords",
                "start_date": event_date,
                "end_date": event_date,
                "select": ["daily.sunshine_duration[0]"],
            },
            "expected_output": None,
            "output_key": "sunshine",
        },
    ]


def _build_stock_price_chain(ticker: str, event_date: str) -> list[dict[str, Any]]:
    """Look up historical stock closing price."""
    return [
        {
            "tool_name": "stock_historical_price",
            "arguments": {"ticker": ticker, "date": event_date},
            "expected_output": None,
            "output_key": "stock_price",
        },
    ]


def _build_stock_volume_chain(ticker: str, event_date: str) -> list[dict[str, Any]]:
    """Look up historical stock trading volume."""
    return [
        {
            "tool_name": "stock_volume",
            "arguments": {"ticker": ticker, "date": event_date},
            "expected_output": None,
            "output_key": "stock_volume",
        },
    ]


def _build_crypto_price_chain(symbol: str, event_date: str) -> list[dict[str, Any]]:
    """Look up historical crypto closing price."""
    return [
        {
            "tool_name": "crypto_historical_price",
            "arguments": {"symbol": symbol, "date": event_date},
            "expected_output": None,
            "output_key": "crypto_price",
        },
    ]


def _build_crypto_volume_chain(symbol: str, event_date: str) -> list[dict[str, Any]]:
    """Look up historical crypto trading volume."""
    return [
        {
            "tool_name": "crypto_volume",
            "arguments": {"symbol": symbol, "date": event_date},
            "expected_output": None,
            "output_key": "crypto_volume",
        },
    ]


# Unit conversion options: (from_unit, to_unit, factor, description_template)
_UNIT_CONVERSIONS = [
    ("meters", "feet", 3.28084, "Convert {value} meters to feet"),
    ("km", "miles", 0.621371, "Convert {value} km to miles"),
    ("meters", "yards", 1.09361, "Convert {value} meters to yards"),
    ("celsius", "fahrenheit_offset", 1.8, "Convert {value}°C to a Fahrenheit-scale offset"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_markdown(text: str) -> str:
    """Strip markdown artifacts from extracted values.

    Handles: [text](url) links, Wikipedia citation brackets, bullet prefixes,
    pipe chars from tables.
    """
    # Strip Wikipedia citation brackets: [[1]](#cite_note-1), [[2]](#cite_note-...)
    text = re.sub(r"\[\[\d+\]\]\(#cite_note[^)]*\)", "", text)
    # Also handle plain citation brackets without links: [1], [2], [note 1]
    text = re.sub(r"\[(\d+|note \d+)\]", "", text)
    # Strip markdown links: [text](url "title") → text
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # Strip leading bullet/list markers
    text = re.sub(r"^\s*[\*\-•]\s+", "", text)
    # Strip stray pipes from table fragments
    text = text.strip().strip("|").strip()
    return text


def _is_geocodable(name: str) -> bool:
    """Heuristic check: is this string likely a geocodable place name?

    Rejects person names, abstract concepts, organization names, etc.
    """
    if not name or len(name) < 2:
        return False
    lower = name.lower()
    # Reject common non-place patterns
    non_place_keywords = [
        "national", "team", "union", "league", "championship", "cup",
        "sultanate", "dynasty", "empire", "kingdom",  # historical polities
        "olympique", "athletic", "football", "rugby", "cricket",  # sports orgs
        "university", "institute", "academy", "college",  # institutions
        "wikipedia", "disambiguation",
    ]
    # Allow if it looks like a place with these in it (e.g., "United Kingdom")
    place_keywords = [
        "city", "town", "village", "county", "district", "province",
        "state", "island", "mount", "lake", "river", "bridge", "park",
        "station", "airport", "harbor", "harbour", "port",
    ]
    if any(kw in lower for kw in place_keywords):
        return True
    if any(kw in lower for kw in non_place_keywords):
        return False
    # Reject if it looks like a person name (2-3 capitalized words, no comma)
    words = name.split()
    if 2 <= len(words) <= 4 and "," not in name:
        # If all words are capitalized and none are place-like, likely a person
        if all(w[0].isupper() for w in words if w[0].isalpha()):
            # Check if any word is a common place word
            if not any(kw in lower for kw in ["city", "bay", "mount", "new", "san", "los", "el", "la", "fort", "st.", "saint"]):
                # Could be a person — check for common first/last name length
                if all(len(w) < 15 for w in words):
                    return False
    return True


def _get_page_location(page: Any) -> str | None:
    """Extract a geocodable location name from a page's infobox or title.

    Tries multiple infobox fields in priority order, cleans markdown,
    and validates that the result looks like a real place.
    """
    # Priority-ordered infobox keys for location extraction
    location_keys = [
        "location", "city", "capital", "headquarters", "seat",
        "place", "venue", "ground", "stadium",
        "birth_place", "birthplace", "birth place",
        "death_place", "deathplace", "death place",
        "home_town", "hometown", "home town",
        "native_place", "residence",
        "coordinates", "coord",
    ]
    location_keys_normalized = {k.replace(" ", "_") for k in location_keys}
    for f in page.infobox:
        key_lower = f.key.lower().replace(" ", "_")
        # Use startswith matching to handle keys like "coordinates: [...]"
        key_base = key_lower.split(":")[0].strip("_").strip()
        if key_base in location_keys_normalized or key_lower in location_keys_normalized:
            raw = _clean_markdown(f.value)
            # Take first comma-separated part (e.g. "London, England" → "London")
            candidate = raw.split(",")[0].strip()
            if candidate and _is_geocodable(candidate):
                return candidate

    # Fallback: use title only if the page has coordinates (likely a place)
    _coord_prefixes = ("coordinates", "coord", "coords", "latd", "latitude")
    has_coords = any(
        any(f.key.lower().startswith(p) for p in _coord_prefixes)
        for f in page.infobox
    )
    title = page.title
    if has_coords and len(title) < 50:
        cleaned = _clean_markdown(title)
        if _is_geocodable(cleaned):
            return cleaned

    # Last resort: title if it passes geocodability check
    if len(title) < 40 and "(" not in title and _is_geocodable(title):
        return title

    return None


def _extract_location_hint(fact: ExtractableFact, page: Any) -> str | None:
    if fact.value_type != "text":
        return None
    value = _clean_markdown(str(fact.value).strip())
    if len(value) < 60 and not any(c.isdigit() for c in value) and _is_geocodable(value):
        return value
    return None


def _get_page_country(page: Any) -> str | None:
    """Extract a country name from a page's infobox."""
    for f in page.infobox:
        clean_key = _clean_markdown(f.key).lower().strip()
        if clean_key in ("country", "nation", "sovereign state"):
            raw = _clean_markdown(f.value)
            return raw.split(",")[0].strip()
    return None


def _get_page_ticker(page: Any) -> str | None:
    """Extract a stock ticker symbol from a page's infobox.

    Looks for 'traded_as', 'stock_symbol', 'ticker_symbol' keys and
    parses patterns like 'NYSE: AAPL', 'NASDAQ: MSFT', or 'Nasdaq: TSLA'.
    Also handles infobox keys that are markdown links like '[Traded as](...)'.
    """
    ticker_keys = {"traded_as", "ticker_symbol", "stock_symbol", "trading_symbol", "traded as"}
    for f in page.infobox:
        # Clean the key: strip markdown links like [Traded as](/wiki/...)
        clean_key = _clean_markdown(f.key).lower().strip()
        normalized_key = clean_key.replace(" ", "_")
        if normalized_key in ticker_keys or clean_key in ticker_keys:
            raw = _clean_markdown(f.value)
            # Parse "NYSE: AAPL" or "Nasdaq: TSLA" patterns (case-insensitive exchange)
            match = re.search(
                r"(?:NYSE|NASDAQ|TSE|LSE|HKEX|SGX|SZSE|SSE|KRX|TSX)[:\s]+([A-Z]{1,5})",
                raw.upper(),
            )
            if match:
                return match.group(1)
            # Try just an all-caps ticker (1-5 letters, not common words)
            match = re.search(r"\b([A-Z]{1,5})\b", raw.upper())
            if match and match.group(1) not in {
                "THE", "AND", "FOR", "INC", "LTD", "PLC", "CO", "USD", "EUR",
                "GBP", "JPY", "CNY", "HKD", "CAD",
            }:
                return match.group(1)
    return None


# Well-known crypto name → Binance trading pair
_CRYPTO_SYMBOL_MAP = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "binance coin": "BNBUSDT",
    "bnb": "BNBUSDT",
    "solana": "SOLUSDT",
    "cardano": "ADAUSDT",
    "dogecoin": "DOGEUSDT",
    "ripple": "XRPUSDT",
    "xrp": "XRPUSDT",
    "polkadot": "DOTUSDT",
    "litecoin": "LTCUSDT",
    "avalanche": "AVAXUSDT",
    "chainlink": "LINKUSDT",
    "polygon": "MATICUSDT",
    "tron": "TRXUSDT",
    "shiba inu": "SHIBUSDT",
    "uniswap": "UNIUSDT",
    "stellar": "XLMUSDT",
    "monero": "XMRUSDT",
    "toncoin": "TONUSDT",
}


def _get_page_crypto_symbol(page: Any) -> str | None:
    """Extract a Binance trading pair symbol from a page's title or infobox."""
    title_lower = page.title.lower()
    for name, symbol in _CRYPTO_SYMBOL_MAP.items():
        if name in title_lower:
            return symbol
    # Check infobox for ticker/symbol/code fields
    for f in page.infobox:
        clean_key = _clean_markdown(f.key).lower().strip()
        if clean_key in ("ticker_symbol", "symbol", "code", "ticker symbol"):
            raw = _clean_markdown(f.value).upper().strip()
            if 2 <= len(raw) <= 6 and raw.isalpha():
                candidate = raw + "USDT"
                # Only return well-known pairs
                if candidate in {v for v in _CRYPTO_SYMBOL_MAP.values()}:
                    return candidate
    return None


def _adjust_to_trading_day(date_str: str) -> str:
    """Shift a date to the nearest prior weekday (Mon-Fri) for stock markets.

    If the date falls on Saturday, moves to Friday.
    If Sunday, moves to Friday.
    Does not account for market holidays, but avoids the most common issue.
    """
    dt = date.fromisoformat(date_str)
    weekday = dt.weekday()  # 0=Mon ... 6=Sun
    if weekday == 5:  # Saturday → Friday
        dt = dt - timedelta(days=1)
    elif weekday == 6:  # Sunday → Friday
        dt = dt - timedelta(days=2)
    return dt.isoformat()


def _try_parse_date(value: str) -> str | None:
    """Try to extract a valid ISO date from a string.

    Only returns dates that ``date.fromisoformat`` can handle so that
    downstream ``_build_date_computation_chain`` never produces broken code.
    """
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)

    # Reject obviously non-parseable patterns early
    if re.search(r"\b(BC|BCE|AD|CE|century|centuries|era)\b", cleaned, re.IGNORECASE):
        return None
    if re.search(r"\d{4}\s*[–—-]\s*\d{4}", cleaned):  # date ranges like 1478–1519
        return None
    if re.search(r"\d{4}s\b", cleaned):  # decades like 1960s
        return None

    # Full ISO date
    iso_match = re.search(r"(\d{4}-\d{2}-\d{2})", cleaned)
    if iso_match:
        candidate = iso_match.group(1)
        try:
            date.fromisoformat(candidate)
            return candidate
        except ValueError:
            pass

    # "Month Day, Year" or "Day Month Year" patterns
    month_day_year = re.search(
        r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})",
        cleaned,
    ) or re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})",
        cleaned,
    )
    if month_day_year:
        groups = month_day_year.groups()
        try:
            from datetime import datetime as _dt
            for fmt in ("%d %B %Y", "%B %d %Y", "%B %d, %Y"):
                try:
                    parsed = _dt.strptime(" ".join(groups), fmt)
                    return parsed.date().isoformat()
                except ValueError:
                    continue
        except Exception:
            pass

    # Bare 4-digit year in modern range (1800-2029) → use Jan 1
    year_match = re.search(r"\b(1[89]\d{2}|20[0-2]\d)\b", cleaned)
    if year_match:
        candidate = f"{year_match.group(0)}-01-01"
        try:
            date.fromisoformat(candidate)
            return candidate
        except ValueError:
            pass

    return None



# ---------------------------------------------------------------------------
# Phase 1: LLM route planning
# ---------------------------------------------------------------------------

_PLAN_SYSTEM_PROMPT = """\
You are a scavenger-hunt puzzle designer. Given a set of Wikipedia pages that have \
been crawled, plan a coherent thematic trail through them.

The trail should:
- Start from the seed page
- Visit {num_stops} pages total (including the seed)
- NEVER repeat the same page_url — every stop must use a DIFFERENT page
- Have a clear narrative theme connecting the pages
- Include {num_tool_stops} tool-based stops. VARY the tool types — avoid using \
only geocode_elevation and geocode_distance. Prefer a mix from: \
geocode_elevation, geocode_distance, geocode_directions_duration, \
country_population, country_area, nearby_poi_count, place_rating, \
historical_snowfall, historical_sunshine, geocode_weather_historical, \
geocode_weather_precipitation, math_conversion, date_computation, \
stock_price, stock_volume, crypto_price, crypto_volume
- Tool selection guide (match tool to page content):
  * stock_price / stock_volume — use on pages about publicly traded companies \
(those with "traded_as" or stock ticker in infobox, e.g. Apple Inc., Tesla)
  * crypto_price / crypto_volume — use on pages about major cryptocurrencies \
(Bitcoin, Ethereum, Solana, Dogecoin, Litecoin, Cardano, etc.)
  * geocode_elevation / geocode_distance / geocode_directions_duration / \
nearby_poi_count / place_rating — use on pages about places/landmarks/cities
  * country_population / country_area — use on pages mentioning countries
  * geocode_weather_* / historical_snowfall / historical_sunshine — use on \
pages about places that also mention a date
  * date_computation — use on pages mentioning specific dates
  * math_conversion — use when a numeric value was extracted in a prior stop
- Prefer pages with infoboxes (marked has_infobox=true) for extraction stops

Output a JSON object:
{{
  "theme": "A short description of the trail's narrative theme",
  "stops": [
    {{
      "type": "page",
      "page_url": "https://en.wikipedia.org/wiki/...",
      "extraction_hint": "what fact to extract, e.g. 'elevation' or 'population'",
      "narrative_reason": "why this page fits the theme"
    }},
    {{
      "type": "tool",
      "tool_type": "geocode_elevation|geocode_weather_historical|geocode_distance|geocode_directions_duration|geocode_weather_precipitation|historical_snowfall|historical_sunshine|nearby_poi_count|place_rating|country_population|country_area|math_conversion|date_computation|stock_price|stock_volume|crypto_price|crypto_volume",
      "page_url": "https://en.wikipedia.org/wiki/...",
      "narrative_reason": "why this tool stop fits"
    }}
  ]
}}

The last stop will be auto-generated as a compute stop (not in your list).
Output ONLY valid JSON, no markdown fencing."""

_PLAN_USER_TEMPLATE = """\
Seed page: {seed_title} ({seed_url})

Available pages:
{page_summaries}

Plan a trail with {num_stops} stops ({num_page_stops} page stops + \
{num_tool_stops} tool stops). The final compute stop is added automatically."""


def _format_page_summaries(
    graph: WikiGraph,
    *,
    max_pages: int = 200,
    avoid_pages: set[str] | None = None,
) -> str:
    pages = graph.pages()
    # Filter out pages that have been used in prior samples
    if avoid_pages:
        pages = [p for p in pages if p.url not in avoid_pages]
    # Prioritise pages with infoboxes (more extractable facts), then by link count
    pages.sort(key=lambda p: (bool(p.infobox), len(p.outgoing_links)), reverse=True)
    pages = pages[:max_pages]
    lines = []
    for page in pages:
        has_infobox = "true" if page.infobox else "false"
        paragraph = page.first_paragraph[:200] if page.first_paragraph else ""
        # Add ticker/crypto annotations to help the planner pick stock/crypto tools
        extras = []
        ticker = _get_page_ticker(page)
        if ticker:
            extras.append(f"stock_ticker={ticker}")
        crypto = _get_page_crypto_symbol(page)
        if crypto:
            extras.append(f"crypto_symbol={crypto}")
        extra_str = " | " + " | ".join(extras) if extras else ""
        lines.append(
            f"- {page.title} | {page.url} | has_infobox={has_infobox}{extra_str}\n"
            f"  {paragraph}"
        )
    return "\n".join(lines)


async def _plan_route(
    seed_page: Any,
    graph: WikiGraph,
    difficulty: TrailDifficultyConfig,
    llm_client: OpenAI,
    model: str,
    rng: random.Random,
    *,
    max_retries: int = 3,
    avoid_pages: set[str] | None = None,
) -> dict[str, Any] | None:
    """Use LLM to plan a thematic route through crawled pages."""
    depth = rng.randint(*difficulty.depth_range)
    num_tool_stops = rng.randint(*difficulty.tool_stops_range)

    # Over-plan tool stops: request extra to absorb geocoding/directions failures
    # during pre-validation. The surplus is trimmed after validation if not needed.
    tool_overplan = max(1, num_tool_stops // 3)  # ~33% extra
    planned_tool_stops = num_tool_stops + tool_overplan

    # depth includes the final compute stop, so page+tool stops = depth - 1
    num_content_stops = depth - 1 + tool_overplan
    num_page_stops = num_content_stops - planned_tool_stops

    if num_page_stops < 1:
        num_page_stops = 1
        planned_tool_stops = num_content_stops - 1

    page_summaries = _format_page_summaries(graph, avoid_pages=avoid_pages)

    system_prompt = _PLAN_SYSTEM_PROMPT.format(
        num_stops=num_content_stops,
        num_tool_stops=planned_tool_stops,
    )
    user_prompt = _PLAN_USER_TEMPLATE.format(
        seed_title=seed_page.title,
        seed_url=seed_page.url,
        page_summaries=page_summaries,
        num_stops=num_content_stops,
        num_page_stops=num_page_stops,
        num_tool_stops=planned_tool_stops,
    )

    for attempt in range(max_retries):
        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = (response.choices[0].message.content or "").strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                lines = raw.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                raw = "\n".join(lines)

            plan = json.loads(raw)
            if not isinstance(plan, dict) or "stops" not in plan:
                logger.warning("Plan attempt %d: invalid structure", attempt + 1)
                continue

            # Validate all page_urls exist in graph and are unique
            valid = True
            seen_urls: set[str] = set()
            for stop in plan["stops"]:
                if stop.get("type") == "page" or stop.get("page_url"):
                    page_url = stop.get("page_url", "")
                    if page_url and graph.get_page(page_url) is None:
                        logger.warning(
                            "Plan attempt %d: page %s not in graph",
                            attempt + 1, page_url,
                        )
                        valid = False
                        break
                    if page_url in seen_urls:
                        logger.warning(
                            "Plan attempt %d: duplicate page %s",
                            attempt + 1, page_url,
                        )
                        valid = False
                        break
                    if page_url:
                        seen_urls.add(page_url)

            if valid:
                plan["_depth"] = depth
                plan["_num_tool_stops"] = num_tool_stops
                plan["_planned_tool_stops"] = planned_tool_stops
                plan["_tool_overplan"] = tool_overplan
                return plan

        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Plan attempt %d failed: %s", attempt + 1, e)

    return None


# ---------------------------------------------------------------------------
# Phase 2: Build stops from plan
# ---------------------------------------------------------------------------


async def _build_stops_from_plan(
    plan: dict[str, Any],
    graph: WikiGraph,
    extractor: FactExtractor,
    rng: random.Random,
) -> tuple[list[Stop], dict[int, float], list[tuple[int, str]], dict[int, tuple[Any, str]]]:
    """Convert a planned route into Stop objects.

    Returns (stops, numeric_values, location_names, all_values).
    all_values maps stop index -> (value, value_type) for every page stop.
    """
    stops: list[Stop] = []
    numeric_values: dict[int, float] = {}
    location_names: list[tuple[int, str]] = []
    date_values: list[tuple[int, str]] = []
    country_names: list[tuple[int, str]] = []
    all_values: dict[int, tuple[Any, str]] = {}

    for idx, planned_stop in enumerate(plan["stops"]):
        stop_type = planned_stop.get("type", "page")

        if stop_type == "page":
            page_url = planned_stop.get("page_url", "")
            page = graph.get_page(page_url)
            if page is None:
                logger.warning("Page %s not found in graph, skipping", page_url)
                return [], {}, [], {}

            # Extract facts and match to extraction_hint
            extraction_hint = planned_stop.get("extraction_hint", "")
            facts = await extractor.extract_facts(page, include_prose=True)

            # Find best matching fact via keyword overlap
            best_fact = _match_fact_to_hint(facts, extraction_hint)
            if best_fact is None and facts:
                best_fact = facts[0]

            if best_fact is None:
                logger.warning("No extractable facts for %s", page.title)
                return [], {}, [], {}

            # Clean markdown from text values
            if isinstance(best_fact.value, str):
                cleaned = _clean_markdown(best_fact.value)
                if cleaned != best_fact.value:
                    best_fact = ExtractableFact(
                        page_url=best_fact.page_url,
                        description=best_fact.description,
                        section=best_fact.section,
                        value=cleaned,
                        value_type=best_fact.value_type,
                        confidence=best_fact.confidence,
                        outgoing_links=best_fact.outgoing_links,
                        source_tier=best_fact.source_tier,
                    )

            # Reject empty values — try fallback facts
            if isinstance(best_fact.value, str) and not best_fact.value.strip():
                logger.warning(
                    "Empty value for fact '%s' on %s, trying fallback",
                    best_fact.description, page.title,
                )
                fallback = None
                for f in facts:
                    if f is best_fact:
                        continue
                    fval = _clean_markdown(str(f.value)) if isinstance(f.value, str) else f.value
                    if fval and str(fval).strip():
                        fallback = ExtractableFact(
                            page_url=f.page_url,
                            description=f.description,
                            section=f.section,
                            value=fval if isinstance(fval, str) else f.value,
                            value_type=f.value_type,
                            confidence=f.confidence,
                            outgoing_links=f.outgoing_links,
                            source_tier=f.source_tier,
                        )
                        break
                if fallback is None:
                    logger.warning("No non-empty facts for %s", page.title)
                    return [], {}, [], {}
                best_fact = fallback

            # Track all extracted values
            all_values[idx] = (best_fact.value, best_fact.value_type)

            # Record numeric values
            if best_fact.value_type == "number" and isinstance(best_fact.value, (int, float)):
                numeric_values[idx] = float(best_fact.value)

            # Record location names — from text facts or page title
            if best_fact.value_type == "text":
                loc = _extract_location_hint(best_fact, page)
                if loc:
                    location_names.append((idx, loc))
            page_loc = _get_page_location(page)
            if page_loc and not any(name == page_loc for _, name in location_names):
                location_names.append((idx, page_loc))

            # Record dates (only if parseable as ISO)
            if best_fact.value_type == "date":
                parsed_date = _try_parse_date(str(best_fact.value))
                if parsed_date:
                    date_values.append((idx, parsed_date))

            # Record country names from infobox
            country = _get_page_country(page)
            if country and not any(c == country for _, c in country_names):
                country_names.append((idx, country))

            stop = Stop(
                index=idx,
                stop_type="page",
                page_url=page.url,
                extraction_target=best_fact.description,
                extraction_section=best_fact.section,
                extracted_value=best_fact.value,
                extracted_value_type=best_fact.value_type,
                bridge=Bridge(bridge_type="link_follow"),  # Placeholder, filled in phase 3
                reasoning=planned_stop.get("narrative_reason", ""),
            )
            stops.append(stop)

        elif stop_type == "tool":
            tool_type = planned_stop.get("tool_type", "geocode_elevation")
            page_url = planned_stop.get("page_url", "")
            page = graph.get_page(page_url) if page_url else None

            tool_stop = _build_tool_stop_from_plan(
                idx, tool_type, page, rng,
                numeric_values, location_names, date_values, country_names,
            )
            if tool_stop is None:
                # Fallback: try other tool types
                fallback_types = [
                    "geocode_elevation",
                    "nearby_poi_count",
                    "place_rating",
                    "country_population",
                    "country_area",
                    "geocode_weather_historical",
                    "historical_snowfall",
                    "historical_sunshine",
                    "geocode_distance",
                    "geocode_directions_duration",
                    "geocode_weather_precipitation",
                    "math_conversion",
                    "date_computation",
                ]
                for fallback in fallback_types:
                    if fallback == tool_type:
                        continue
                    tool_stop = _build_tool_stop_from_plan(
                        idx, fallback, page, rng,
                        numeric_values, location_names, date_values, country_names,
                    )
                    if tool_stop is not None:
                        logger.info(
                            "Fell back from %s to %s at index %d",
                            tool_type, fallback, idx,
                        )
                        break
            if tool_stop is None:
                logger.warning("Failed to build any tool stop at index %d, skipping", idx)
                continue

            stops.append(tool_stop)

    # Re-index stops (some tool stops may have been skipped)
    idx_remap: dict[int, int] = {}
    for new_idx, s in enumerate(stops):
        idx_remap[s.index] = new_idx
        s.index = new_idx
    numeric_values = {idx_remap[k]: v for k, v in numeric_values.items() if k in idx_remap}
    all_values = {idx_remap[k]: v for k, v in all_values.items() if k in idx_remap}
    location_names = [(idx_remap[k], v) for k, v in location_names if k in idx_remap]

    return stops, numeric_values, location_names, all_values


def _match_fact_to_hint(
    facts: list[ExtractableFact], hint: str
) -> ExtractableFact | None:
    """Find the fact whose description best matches the extraction hint.

    Ties are broken in favor of numeric facts (more useful for compute stops).
    """
    if not hint or not facts:
        return None

    hint_words = set(hint.lower().split())
    best_score = -1
    best_fact = None

    for fact in facts:
        desc_words = set(fact.description.lower().split())
        overlap = len(hint_words & desc_words)
        # Slight bonus for numeric facts — prefer them on ties
        numeric_bonus = 0.5 if fact.value_type == "number" else 0
        score = overlap + numeric_bonus
        if score > best_score:
            best_score = score
            best_fact = fact

    return best_fact


def _build_tool_stop_from_plan(
    step_idx: int,
    tool_type: str,
    page: Any | None,
    rng: random.Random,
    numeric_values: dict[int, float],
    location_names: list[tuple[int, str]],
    date_values: list[tuple[int, str]],
    country_names: list[tuple[int, str]] | None = None,
) -> Stop | None:
    """Build a tool stop of the specified type."""
    raw_location = _get_page_location(page) if page else None
    current_location = _clean_markdown(raw_location) if raw_location else None
    # Filter location_names to only geocodable ones
    geocodable_locations = [
        (idx, _clean_markdown(name))
        for idx, name in location_names
        if _is_geocodable(_clean_markdown(name))
    ]

    if tool_type == "geocode_elevation":
        if not current_location:
            return None
        chain = _build_geocode_elevation_chain(current_location)
        description = f"Look up the elevation at {current_location}"
        return Stop(
            index=step_idx,
            stop_type="tool",
            page_url=page.url if page else None,
            extraction_target=description,
            extraction_section=None,
            extracted_value=None,
            extracted_value_type="number",
            bridge=Bridge(bridge_type="tool_call", tool_chain=chain),
            reasoning=f"Tool stop: geocode+elevation for {current_location}",
        )

    if tool_type == "geocode_weather_historical":
        if not current_location:
            return None
        available_dates = list(date_values)
        if page:
            for f in page.infobox:
                if any(kw in f.key.lower() for kw in ["date", "founded", "opened", "established"]):
                    d = _try_parse_date(f.value)
                    if d:
                        available_dates.append((step_idx, d))
            year_match = re.search(r'\b(1[89]\d{2}|20[0-2]\d)\b', page.first_paragraph)
            if year_match:
                available_dates.append((step_idx, f"{year_match.group(0)}-07-01"))
        if not available_dates:
            return None
        _, event_date = rng.choice(available_dates)
        chain = _build_geocode_weather_chain(current_location, event_date)
        description = (
            f"Find the historical maximum temperature at {current_location} "
            f"on {event_date}"
        )
        return Stop(
            index=step_idx,
            stop_type="tool",
            page_url=page.url if page else None,
            extraction_target=description,
            extraction_section=None,
            extracted_value=None,
            extracted_value_type="number",
            bridge=Bridge(bridge_type="tool_call", tool_chain=chain),
            reasoning=f"Tool stop: geocode+weather_historical for {current_location} on {event_date}",
        )

    if tool_type == "geocode_distance":
        if not current_location or not geocodable_locations:
            return None
        _, other_location = rng.choice(geocodable_locations)
        if other_location == current_location:
            return None
        chain = _build_geocode_distance_chain(current_location, other_location)
        description = (
            f"Calculate the driving distance between {current_location} "
            f"and {other_location}"
        )
        return Stop(
            index=step_idx,
            stop_type="tool",
            page_url=page.url if page else None,
            extraction_target=description,
            extraction_section=None,
            extracted_value=None,
            extracted_value_type="number",
            bridge=Bridge(bridge_type="tool_call", tool_chain=chain),
            reasoning=f"Tool stop: distance between {current_location} and {other_location}",
        )

    if tool_type == "date_computation":
        available_dates = list(date_values)
        if page:
            for f in page.infobox:
                if any(kw in f.key.lower() for kw in ["date", "founded", "opened"]):
                    d = _try_parse_date(f.value)
                    if d:
                        available_dates.append((step_idx, d))
        if not available_dates:
            return None
        _, event_date = rng.choice(available_dates)
        reference_date = date.today().isoformat()
        chain = _build_date_computation_chain(event_date, reference_date)
        description = (
            f"Calculate the number of days between {event_date} "
            f"and {reference_date}"
        )
        return Stop(
            index=step_idx,
            stop_type="tool",
            page_url=page.url if page else None,
            extraction_target=description,
            extraction_section=None,
            extracted_value=None,
            extracted_value_type="number",
            bridge=Bridge(bridge_type="tool_call", tool_chain=chain),
            reasoning=f"Tool stop: date computation {event_date} to {reference_date} (pinned)",
        )

    if tool_type == "geocode_directions_duration":
        if not current_location or not geocodable_locations:
            return None
        _, other_location = rng.choice(geocodable_locations)
        if other_location == current_location:
            return None
        chain = _build_geocode_directions_duration_chain(current_location, other_location)
        description = (
            f"Get the driving duration in minutes from {current_location} "
            f"to {other_location}"
        )
        return Stop(
            index=step_idx,
            stop_type="tool",
            page_url=page.url if page else None,
            extraction_target=description,
            extraction_section=None,
            extracted_value=None,
            extracted_value_type="number",
            bridge=Bridge(bridge_type="tool_call", tool_chain=chain),
            reasoning=f"Tool stop: directions duration {current_location} to {other_location}",
        )

    if tool_type == "geocode_weather_precipitation":
        if not current_location:
            return None
        available_dates = list(date_values)
        if page:
            for f in page.infobox:
                if any(kw in f.key.lower() for kw in ["date", "founded", "opened", "established"]):
                    d = _try_parse_date(f.value)
                    if d:
                        available_dates.append((step_idx, d))
            year_match = re.search(r'\b(1[89]\d{2}|20[0-2]\d)\b', page.first_paragraph)
            if year_match:
                available_dates.append((step_idx, f"{year_match.group(0)}-07-01"))
        if not available_dates:
            return None
        _, event_date = rng.choice(available_dates)
        chain = _build_geocode_weather_precipitation_chain(current_location, event_date)
        description = (
            f"Find the total precipitation at {current_location} "
            f"on {event_date}"
        )
        return Stop(
            index=step_idx,
            stop_type="tool",
            page_url=page.url if page else None,
            extraction_target=description,
            extraction_section=None,
            extracted_value=None,
            extracted_value_type="number",
            bridge=Bridge(bridge_type="tool_call", tool_chain=chain),
            reasoning=f"Tool stop: precipitation at {current_location} on {event_date}",
        )

    if tool_type == "math_conversion":
        if not numeric_values:
            return None
        # Pick a numeric value from a previous stop
        source_idx = rng.choice(list(numeric_values.keys()))
        source_value = numeric_values[source_idx]
        conversion = rng.choice(_UNIT_CONVERSIONS)
        from_unit, to_unit, factor, desc_template = conversion
        chain = _build_math_conversion_chain(source_value, from_unit, to_unit, factor)
        description = desc_template.format(value=int(source_value))
        return Stop(
            index=step_idx,
            stop_type="tool",
            page_url=page.url if page else None,
            extraction_target=description,
            extraction_section=None,
            extracted_value=None,
            extracted_value_type="number",
            bridge=Bridge(bridge_type="tool_call", tool_chain=chain),
            reasoning=f"Tool stop: convert {int(source_value)} {from_unit} to {to_unit}",
        )

    if tool_type == "nearby_poi_count":
        if not current_location:
            return None
        poi_type = rng.choice(_POI_TYPES)
        chain = _build_nearby_poi_count_chain(current_location, poi_type)
        description = f"Count the number of {poi_type} near {current_location}"
        return Stop(
            index=step_idx,
            stop_type="tool",
            page_url=page.url if page else None,
            extraction_target=description,
            extraction_section=None,
            extracted_value=None,
            extracted_value_type="number",
            bridge=Bridge(bridge_type="tool_call", tool_chain=chain),
            reasoning=f"Tool stop: count {poi_type} near {current_location}",
        )

    if tool_type == "place_rating":
        if not current_location:
            return None
        chain = _build_place_rating_chain(current_location)
        description = f"Look up the Google Maps rating of {current_location}"
        return Stop(
            index=step_idx,
            stop_type="tool",
            page_url=page.url if page else None,
            extraction_target=description,
            extraction_section=None,
            extracted_value=None,
            extracted_value_type="number",
            bridge=Bridge(bridge_type="tool_call", tool_chain=chain),
            reasoning=f"Tool stop: Google rating of {current_location}",
        )

    if tool_type == "country_population":
        country_list = list(country_names or [])
        # Also try extracting country from current page
        if page:
            c = _get_page_country(page)
            if c and not any(name == c for _, name in country_list):
                country_list.append((step_idx, c))
        if not country_list:
            return None
        _, country = rng.choice(country_list)
        chain = _build_country_population_chain(country)
        description = f"Look up the population of {country}"
        return Stop(
            index=step_idx,
            stop_type="tool",
            page_url=page.url if page else None,
            extraction_target=description,
            extraction_section=None,
            extracted_value=None,
            extracted_value_type="number",
            bridge=Bridge(bridge_type="tool_call", tool_chain=chain),
            reasoning=f"Tool stop: population of {country}",
        )

    if tool_type == "country_area":
        country_list = list(country_names or [])
        if page:
            c = _get_page_country(page)
            if c and not any(name == c for _, name in country_list):
                country_list.append((step_idx, c))
        if not country_list:
            return None
        _, country = rng.choice(country_list)
        chain = _build_country_area_chain(country)
        description = f"Look up the area in km² of {country}"
        return Stop(
            index=step_idx,
            stop_type="tool",
            page_url=page.url if page else None,
            extraction_target=description,
            extraction_section=None,
            extracted_value=None,
            extracted_value_type="number",
            bridge=Bridge(bridge_type="tool_call", tool_chain=chain),
            reasoning=f"Tool stop: area of {country}",
        )

    if tool_type == "historical_snowfall":
        if not current_location:
            return None
        available_dates = list(date_values)
        if page:
            for f in page.infobox:
                if any(kw in f.key.lower() for kw in ["date", "founded", "opened", "established"]):
                    d = _try_parse_date(f.value)
                    if d:
                        available_dates.append((step_idx, d))
            year_match = re.search(r'\b(1[89]\d{2}|20[0-2]\d)\b', page.first_paragraph)
            if year_match:
                available_dates.append((step_idx, f"{year_match.group(0)}-01-15"))
        if not available_dates:
            return None
        _, event_date = rng.choice(available_dates)
        chain = _build_historical_snowfall_chain(current_location, event_date)
        description = f"Find the snowfall at {current_location} on {event_date}"
        return Stop(
            index=step_idx,
            stop_type="tool",
            page_url=page.url if page else None,
            extraction_target=description,
            extraction_section=None,
            extracted_value=None,
            extracted_value_type="number",
            bridge=Bridge(bridge_type="tool_call", tool_chain=chain),
            reasoning=f"Tool stop: snowfall at {current_location} on {event_date}",
        )

    if tool_type == "historical_sunshine":
        if not current_location:
            return None
        available_dates = list(date_values)
        if page:
            for f in page.infobox:
                if any(kw in f.key.lower() for kw in ["date", "founded", "opened", "established"]):
                    d = _try_parse_date(f.value)
                    if d:
                        available_dates.append((step_idx, d))
            year_match = re.search(r'\b(1[89]\d{2}|20[0-2]\d)\b', page.first_paragraph)
            if year_match:
                available_dates.append((step_idx, f"{year_match.group(0)}-06-21"))
        if not available_dates:
            return None
        _, event_date = rng.choice(available_dates)
        chain = _build_historical_sunshine_chain(current_location, event_date)
        description = f"Find the sunshine duration (seconds) at {current_location} on {event_date}"
        return Stop(
            index=step_idx,
            stop_type="tool",
            page_url=page.url if page else None,
            extraction_target=description,
            extraction_section=None,
            extracted_value=None,
            extracted_value_type="number",
            bridge=Bridge(bridge_type="tool_call", tool_chain=chain),
            reasoning=f"Tool stop: sunshine duration at {current_location} on {event_date}",
        )

    if tool_type == "stock_price":
        if not page:
            return None
        ticker = _get_page_ticker(page)
        if not ticker:
            return None
        available_dates = list(date_values)
        if page:
            for f in page.infobox:
                if any(kw in f.key.lower() for kw in ["date", "founded", "ipo", "listed", "opened"]):
                    d = _try_parse_date(f.value)
                    if d:
                        available_dates.append((step_idx, d))
            year_match = re.search(r'\b(20[0-2]\d)\b', page.first_paragraph)
            if year_match:
                available_dates.append((step_idx, f"{year_match.group(0)}-06-15"))
        if not available_dates:
            available_dates.append((step_idx, "2024-06-15"))
        _, event_date = rng.choice(available_dates)
        event_date = _adjust_to_trading_day(event_date)
        chain = _build_stock_price_chain(ticker, event_date)
        description = f"Look up the closing price of {ticker} on {event_date}"
        return Stop(
            index=step_idx,
            stop_type="tool",
            page_url=page.url if page else None,
            extraction_target=description,
            extraction_section=None,
            extracted_value=None,
            extracted_value_type="number",
            bridge=Bridge(bridge_type="tool_call", tool_chain=chain),
            reasoning=f"Tool stop: stock price {ticker} on {event_date}",
        )

    if tool_type == "stock_volume":
        if not page:
            return None
        ticker = _get_page_ticker(page)
        if not ticker:
            return None
        available_dates = list(date_values)
        if page:
            for f in page.infobox:
                if any(kw in f.key.lower() for kw in ["date", "founded", "ipo", "listed", "opened"]):
                    d = _try_parse_date(f.value)
                    if d:
                        available_dates.append((step_idx, d))
            year_match = re.search(r'\b(20[0-2]\d)\b', page.first_paragraph)
            if year_match:
                available_dates.append((step_idx, f"{year_match.group(0)}-06-15"))
        if not available_dates:
            available_dates.append((step_idx, "2024-06-15"))
        _, event_date = rng.choice(available_dates)
        event_date = _adjust_to_trading_day(event_date)
        chain = _build_stock_volume_chain(ticker, event_date)
        description = f"Look up the trading volume of {ticker} on {event_date}"
        return Stop(
            index=step_idx,
            stop_type="tool",
            page_url=page.url if page else None,
            extraction_target=description,
            extraction_section=None,
            extracted_value=None,
            extracted_value_type="number",
            bridge=Bridge(bridge_type="tool_call", tool_chain=chain),
            reasoning=f"Tool stop: stock volume {ticker} on {event_date}",
        )

    if tool_type == "crypto_price":
        if not page:
            return None
        symbol = _get_page_crypto_symbol(page)
        if not symbol:
            return None
        available_dates = list(date_values)
        if page:
            for f in page.infobox:
                if any(kw in f.key.lower() for kw in ["date", "released", "launched"]):
                    d = _try_parse_date(f.value)
                    if d:
                        available_dates.append((step_idx, d))
            year_match = re.search(r'\b(20[12]\d)\b', page.first_paragraph)
            if year_match:
                available_dates.append((step_idx, f"{year_match.group(0)}-06-15"))
        if not available_dates:
            available_dates.append((step_idx, "2024-06-15"))
        _, event_date = rng.choice(available_dates)
        chain = _build_crypto_price_chain(symbol, event_date)
        description = f"Look up the closing price of {symbol} on {event_date}"
        return Stop(
            index=step_idx,
            stop_type="tool",
            page_url=page.url if page else None,
            extraction_target=description,
            extraction_section=None,
            extracted_value=None,
            extracted_value_type="number",
            bridge=Bridge(bridge_type="tool_call", tool_chain=chain),
            reasoning=f"Tool stop: crypto price {symbol} on {event_date}",
        )

    if tool_type == "crypto_volume":
        if not page:
            return None
        symbol = _get_page_crypto_symbol(page)
        if not symbol:
            return None
        available_dates = list(date_values)
        if page:
            for f in page.infobox:
                if any(kw in f.key.lower() for kw in ["date", "released", "launched"]):
                    d = _try_parse_date(f.value)
                    if d:
                        available_dates.append((step_idx, d))
            year_match = re.search(r'\b(20[12]\d)\b', page.first_paragraph)
            if year_match:
                available_dates.append((step_idx, f"{year_match.group(0)}-06-15"))
        if not available_dates:
            available_dates.append((step_idx, "2024-06-15"))
        _, event_date = rng.choice(available_dates)
        chain = _build_crypto_volume_chain(symbol, event_date)
        description = f"Look up the trading volume of {symbol} on {event_date}"
        return Stop(
            index=step_idx,
            stop_type="tool",
            page_url=page.url if page else None,
            extraction_target=description,
            extraction_section=None,
            extracted_value=None,
            extracted_value_type="number",
            bridge=Bridge(bridge_type="tool_call", tool_chain=chain),
            reasoning=f"Tool stop: crypto volume {symbol} on {event_date}",
        )

    return None


# ---------------------------------------------------------------------------
# Phase 2.7: Inject reason (analytical transform) stops
# ---------------------------------------------------------------------------

REASON_TRANSFORMS: dict[str, dict[str, Any]] = {
    # --- Number theory (requires integer input >= 2) ---
    "next_prime": {
        "input_type": "number",
        "riddle_hint": "the next prime number after that figure",
        "code_template": (
            "def next_prime(n):\n"
            "    n = abs(int(n))\n"
            "    if n < 2: return 2\n"
            "    candidate = n + 1\n"
            "    while True:\n"
            "        if all(candidate % i != 0 for i in range(2, int(candidate**0.5)+1)):\n"
            "            return candidate\n"
            "        candidate += 1\n"
            "result = next_prime({value})\nprint(result)"
        ),
    },
    "num_divisors": {
        "input_type": "number",
        "riddle_hint": "the number of divisors of that figure",
        "code_template": (
            "n = abs(int({value}))\n"
            "count = sum(1 for i in range(1, n+1) if n % i == 0) if n > 0 else 0\n"
            "result = count\nprint(result)"
        ),
    },
    "sum_prime_factors": {
        "input_type": "number",
        "riddle_hint": "the sum of its distinct prime factors",
        "code_template": (
            "def prime_factors(n):\n"
            "    n = abs(int(n))\n"
            "    factors = []\n"
            "    d = 2\n"
            "    while d * d <= n:\n"
            "        while n % d == 0:\n"
            "            factors.append(d)\n"
            "            n //= d\n"
            "        d += 1\n"
            "    if n > 1: factors.append(n)\n"
            "    return factors\n"
            "result = sum(set(prime_factors({value})))\nprint(result)"
        ),
    },
    "largest_prime_factor": {
        "input_type": "number",
        "riddle_hint": "the largest prime factor of that figure",
        "code_template": (
            "def largest_prime_factor(n):\n"
            "    n = abs(int(n))\n"
            "    if n <= 1: return n\n"
            "    d = 2\n"
            "    while d * d <= n:\n"
            "        while n % d == 0: n //= d\n"
            "        d += 1\n"
            "    return n\n"
            "result = largest_prime_factor({value})\nprint(result)"
        ),
    },
    # --- Representation (requires integer input) ---
    "roman_numeral_letter_count": {
        "input_type": "number",
        "riddle_hint": "the number of characters when written as a Roman numeral",
        "code_template": (
            "def to_roman(n):\n"
            "    n = abs(int(n))\n"
            "    vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),\n"
            "            (90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]\n"
            "    r = ''\n"
            "    for v, s in vals:\n"
            "        while n >= v:\n"
            "            r += s; n -= v\n"
            "    return r\n"
            "result = len(to_roman({value}))\nprint(result)"
        ),
    },
    "binary_1bit_count": {
        "input_type": "number",
        "riddle_hint": "the number of 1-bits in its binary representation",
        "code_template": "result = bin(abs(int({value}))).count('1')\nprint(result)",
    },
    "digit_reversal": {
        "input_type": "number",
        "riddle_hint": "the number you get by reversing its digits",
        "code_template": "result = int(str(abs(int({value})))[::-1])\nprint(result)",
    },
    "digital_root": {
        "input_type": "number",
        "riddle_hint": "the digital root (iterated digit sum until single digit)",
        "code_template": (
            "n = abs(int({value}))\n"
            "result = n if n == 0 else ((n - 1) % 9) + 1\nprint(result)"
        ),
    },
    "digit_sum": {
        "input_type": "number",
        "riddle_hint": "the sum of its digits",
        "code_template": "result = sum(int(d) for d in str(abs(int({value}))))\nprint(result)",
    },
    # --- Calendar (requires year in Gregorian range) ---
    "day_of_week": {
        "input_type": "number",
        "riddle_hint": "the day of the week for January 1st of that year (1=Mon, 7=Sun)",
        "code_template": (
            "from datetime import date\n"
            "result = date(abs(int({value})), 1, 1).isoweekday()\nprint(result)"
        ),
    },
    "leap_years_to_2000": {
        "input_type": "number",
        "riddle_hint": "the count of leap years between that year and 2000",
        "code_template": (
            "import calendar\n"
            "y = abs(int({value}))\n"
            "lo, hi = min(y, 2000), max(y, 2000)\n"
            "result = sum(1 for yr in range(lo, hi+1) if calendar.isleap(yr))\nprint(result)"
        ),
    },
    # --- String (requires non-empty text input) ---
    "scrabble_score": {
        "input_type": "text",
        "riddle_hint": "the Scrabble score of that word",
        "code_template": (
            "scores = {{'A':1,'B':3,'C':3,'D':2,'E':1,'F':4,'G':2,'H':4,'I':1,'J':8,\n"
            "          'K':5,'L':1,'M':3,'N':1,'O':1,'P':3,'Q':10,'R':1,'S':1,'T':1,\n"
            "          'U':1,'V':4,'W':4,'X':8,'Y':4,'Z':10}}\n"
            "result = sum(scores.get(c.upper(), 0) for c in \"{value}\")\nprint(result)"
        ),
    },
    "vowel_count": {
        "input_type": "text",
        "riddle_hint": "the number of vowels in that name",
        "code_template": "result = sum(1 for c in \"{value}\".lower() if c in 'aeiou')\nprint(result)",
    },
    "alpha_position_first_letter": {
        "input_type": "text",
        "riddle_hint": "the alphabetical position of its first letter (A=1, B=2, ...)",
        "code_template": (
            "first = \"{value}\"[0].upper()\n"
            "result = ord(first) - ord('A') + 1 if first.isalpha() else 0\nprint(result)"
        ),
    },
}


# Per-transform minimum value thresholds.  Avoids degenerate outputs
# (e.g. digit_sum(3)==3) while still allowing useful transforms for small
# numbers (e.g. next_prime(2)==3, binary_1bit_count(3)==2).
_TRANSFORM_MIN_VALUE: dict[str, int] = {
    "next_prime": 2,
    "binary_1bit_count": 2,
    "roman_numeral_letter_count": 1,
    "num_divisors": 4,
    "sum_prime_factors": 4,
    "largest_prime_factor": 4,
    "digit_sum": 10,
    "digital_root": 10,
    "digit_reversal": 10,
    "day_of_week": 1583,
    "leap_years_to_2000": 1583,
}


def _select_applicable_transforms(
    value: Any, value_type: str,
) -> list[str]:
    """Return transform names applicable to the given value/type.

    Accepts ``value_type`` of ``"number"`` or ``"date"`` (year integers
    are treated as numeric inputs for transforms).
    """
    applicable = []

    # Date-typed year values can be treated as numbers for transforms
    numeric_eligible = value_type in ("number",)
    if value_type == "date":
        try:
            int(float(value))
            numeric_eligible = True  # bare year integer like 1933
        except (ValueError, TypeError):
            pass

    for name, meta in REASON_TRANSFORMS.items():
        if meta["input_type"] == "number" and numeric_eligible:
            try:
                n = abs(int(float(value)))
            except (ValueError, TypeError):
                continue
            min_val = _TRANSFORM_MIN_VALUE.get(name, 4)
            if n < min_val:
                continue
            if name in ("day_of_week", "leap_years_to_2000"):
                if not (1583 <= n <= 2100):
                    continue
            if name == "roman_numeral_letter_count" and n > 3999:
                continue  # Roman numerals go up to 3999
            applicable.append(name)
        elif meta["input_type"] == "text" and value_type in ("text",):
            s = str(value).strip()
            if len(s) < 2:
                continue
            if name == "alpha_position_first_letter" and not s[0].isalpha():
                continue
            applicable.append(name)
    return applicable


def _build_reason_code(transform_name: str, source_value: Any) -> str:
    """Build executable Python code for a reason transform."""
    meta = REASON_TRANSFORMS[transform_name]
    template = meta["code_template"]
    if meta["input_type"] == "text":
        safe_value = str(source_value).replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
        return template.replace("{value}", safe_value)
    else:
        return template.replace("{value}", str(int(float(source_value))))


async def _inject_reason_stops(
    stops: list[Stop],
    numeric_values: dict[int, float],
    all_values: dict[int, tuple[Any, str]],
    difficulty: TrailDifficultyConfig,
    tool_registry: "ToolRegistry",
    rng: random.Random,
) -> tuple[list[Stop], dict[int, float], dict[int, tuple[Any, str]]]:
    """Insert analytical transform stops after selected page/tool stops.

    Returns updated (stops, numeric_values, all_values).
    """
    min_r, max_r = difficulty.reason_stops_range
    if max_r <= 0:
        return stops, numeric_values, all_values

    non_compute = len(stops)  # compute stop not yet appended
    target_count = max(min_r, min(max_r, non_compute // 3))

    # Collect candidates: (list_position, applicable_transforms)
    candidates: list[tuple[int, list[str]]] = []
    for i, stop in enumerate(stops):
        if stop.stop_type not in ("page", "tool"):
            continue
        vtype = stop.extracted_value_type or ""
        transforms = _select_applicable_transforms(stop.extracted_value, vtype)
        if transforms:
            candidates.append((i, transforms))

    if not candidates:
        logger.info("No stops eligible for reason transforms")
        return stops, numeric_values, all_values

    rng.shuffle(candidates)
    selected = candidates[:target_count]
    # Sort by position (descending) so insertions don't shift earlier indices
    selected.sort(key=lambda x: x[0], reverse=True)

    # Get the python executor
    spec = tool_registry.available_tools().get("python_execute_code")
    if spec is None or spec.executor is None:
        logger.warning("python_execute_code not available; skipping reason stops")
        return stops, numeric_values, all_values

    # Build reason stops (inserted after their source)
    insertions: list[tuple[int, Stop]] = []  # (insert_after_position, stop)
    for list_pos, transforms in selected:
        source_stop = stops[list_pos]
        rng.shuffle(transforms)

        created = False
        for transform_name in transforms:
            code = _build_reason_code(transform_name, source_stop.extracted_value)

            # Dry-run the transform
            try:
                result = await spec.executor({"code": code, "timeout_seconds": 10})
                output = result.output_text if hasattr(result, "output_text") else str(result)
                stripped = output.strip()
                # Parse output (may be JSON with stdout)
                try:
                    data = json.loads(stripped)
                    if isinstance(data, dict) and "stdout" in data:
                        stripped = data["stdout"].strip()
                except (json.JSONDecodeError, TypeError):
                    pass
                transformed_value = int(float(stripped))
            except Exception as exc:
                logger.debug(
                    "Reason transform %s failed for stop %d: %s",
                    transform_name, list_pos, exc,
                )
                continue

            # Reject binary/trivial outputs (0 or 1)
            if transformed_value in (0, 1):
                continue

            meta = REASON_TRANSFORMS[transform_name]
            reason_stop = Stop(
                index=-1,  # will be assigned during re-indexing
                stop_type="reason",
                page_url=source_stop.page_url,
                extraction_target=f"Reason: {meta['riddle_hint']}",
                extraction_section=None,
                extracted_value=transformed_value,
                extracted_value_type="number",
                bridge=Bridge(bridge_type="compute"),
                reasoning=f"Analytical transform ({transform_name}) on stop {list_pos} value",
                reason_transform=transform_name,
                reason_source_stop=source_stop.index,
                reason_code=code,
            )
            insertions.append((list_pos, reason_stop))
            created = True
            break

        if not created:
            logger.debug("No valid transform for stop %d", list_pos)

    if not insertions:
        logger.info("No reason stops could be created")
        return stops, numeric_values, all_values

    # Insert reason stops (already sorted descending, so insert from end)
    new_stops = list(stops)
    for insert_after, reason_stop in insertions:
        new_stops.insert(insert_after + 1, reason_stop)

    # Re-index all stops and rebuild value dicts
    # Track which original indices got reason stops
    source_indices_with_reason: set[int] = set()
    new_numeric: dict[int, float] = {}
    new_all: dict[int, tuple[Any, str]] = {}

    for new_idx, stop in enumerate(new_stops):
        old_idx = stop.index
        stop.index = new_idx

        if stop.stop_type == "reason":
            # Update reason_source_stop to the new index of its source
            # (source is always at new_idx - 1 since we inserted right after)
            stop.reason_source_stop = new_idx - 1
            source_indices_with_reason.add(new_idx - 1)
            # Add transformed value to dicts
            new_numeric[new_idx] = float(stop.extracted_value)
            new_all[new_idx] = (stop.extracted_value, "number")
        elif stop.stop_type in ("page", "tool"):
            val = stop.extracted_value
            vtype = stop.extracted_value_type or ""
            new_all[new_idx] = (val, vtype)
            num = _value_to_number(val, vtype)
            if num is not None:
                new_numeric[new_idx] = float(num)

    # Remove source stops that have reason transforms from numeric dict
    # (so compute step uses the transformed value, not the original)
    for src_idx in source_indices_with_reason:
        new_numeric.pop(src_idx, None)

    logger.info(
        "Injected %d reason stops (target was %d)",
        len(insertions), target_count,
    )
    return new_stops, new_numeric, new_all


# ---------------------------------------------------------------------------
# Phase 3: Build bridges between stops
# ---------------------------------------------------------------------------


def _build_bridges(stops: list[Stop], graph: WikiGraph) -> None:
    """Connect consecutive stops with appropriate bridges (in-place)."""
    for i in range(len(stops) - 1):
        current = stops[i]
        next_stop = stops[i + 1]

        if current.stop_type == "tool":
            # Tool stops already have their bridge set (tool_call)
            continue

        if current.stop_type == "reason":
            # Reason stops have no navigation; bridge is already set
            continue

        if next_stop.page_url and current.page_url:
            # Check if there's a direct link from current page to next page
            current_page = graph.get_page(current.page_url)
            if current_page:
                has_direct_link = any(
                    link.target_url == next_stop.page_url
                    for link in current_page.outgoing_links
                )
                if has_direct_link:
                    current.bridge = Bridge(
                        bridge_type="link_follow",
                        target_url=next_stop.page_url,
                    )
                    continue

        # Fallback: search_query bridge
        if next_stop.page_url:
            next_page = graph.get_page(next_stop.page_url)
            query = next_page.title if next_page else next_stop.page_url
            current.bridge = Bridge(
                bridge_type="search_query",
                search_query=f"{query} wikipedia",
                expected_result_url=next_stop.page_url,
            )
        else:
            current.bridge = Bridge(bridge_type="link_follow")


# ---------------------------------------------------------------------------
# Compute stop
# ---------------------------------------------------------------------------


def _value_to_number(value: Any, value_type: str) -> int | None:
    """Convert an extracted value to an integer for compute expressions.

    Returns None for non-numeric values — character-count operations on text
    are deliberately excluded because the exact string varies across Wikipedia
    fetches, making len()-based computations fragile and ambiguous.
    """
    if value_type == "number" and isinstance(value, (int, float)):
        return int(value)
    # Try parsing numeric strings like "1899" or "3,456"
    try:
        text = str(value).strip().replace(",", "")
        return int(float(text))
    except (ValueError, OverflowError):
        return None


def _value_to_code_literal(value: Any, value_type: str) -> tuple[str, str] | None:
    """Return (code_snippet, expression_label) that evaluates to an integer.

    Returns None for non-numeric values — len() operations on text strings
    are excluded to avoid ambiguity from varying Wikipedia phrasing.
    """
    if value_type == "number" and isinstance(value, (int, float)):
        return str(int(value)), str(int(value))
    # Try parsing numeric strings
    try:
        text = str(value).strip().replace(",", "")
        num = int(float(text))
        return str(num), str(num)
    except (ValueError, OverflowError):
        return None


def _build_compute_stop(
    step_idx: int,
    numeric_values: dict[int, float],
    all_values: dict[int, tuple[Any, str]],
    rng: random.Random,
) -> Stop | None:
    """Build a final compute stop that produces a single-digit passcode.

    Only uses values that can be cleanly converted to integers — text/date
    values that would require len() are excluded to avoid fragile
    character-count operations that produce ambiguous results.
    """
    if not all_values and not numeric_values:
        logger.warning("No extracted values available for compute stop")
        return None

    # Build a unified map: index -> (value, value_type)
    unified: dict[int, tuple[Any, str]] = dict(all_values)

    if not unified:
        # Fall back to numeric_values only
        for idx, v in numeric_values.items():
            unified[idx] = (v, "number")

    # Filter to only values that can be converted to integers
    # (excludes text values that would require fragile len() operations)
    numeric_unified: dict[int, tuple[Any, str]] = {}
    for idx, (value, vtype) in unified.items():
        if _value_to_number(value, vtype) is not None:
            numeric_unified[idx] = (value, vtype)

    if len(numeric_unified) < 2:
        logger.warning(
            "Not enough numeric values for compute stop (have %d, need 2)",
            len(numeric_unified),
        )
        return None

    indices = sorted(numeric_unified.keys())
    num_to_use = min(len(indices), rng.randint(2, 4))
    selected_indices = rng.sample(indices, num_to_use)
    selected = {i: numeric_unified[i] for i in selected_indices}

    # Convert all values to code snippets and compute passcode
    code_parts: list[str] = []  # code expressions
    label_parts: list[str] = []  # human-readable labels
    int_vals: list[int] = []  # actual integer values for pre-computing passcode
    used_indices: list[int] = []  # track which stop indices we actually use

    for idx in sorted(selected.keys()):
        value, vtype = selected[idx]
        result = _value_to_code_literal(value, vtype)
        int_val = _value_to_number(value, vtype)
        if result is None or int_val is None:
            continue  # skip non-numeric values
        code_lit, label_lit = result
        code_parts.append(code_lit)
        label_parts.append(label_lit)
        int_vals.append(int_val)
        used_indices.append(idx)

    if len(int_vals) < 2:
        logger.warning("Not enough convertible values for compute stop")
        return None

    if len(int_vals) >= 2:
        expr_type = rng.choice(["digital_root", "mod10", "abs_diff_mod10"])

        if expr_type == "digital_root":
            sum_code = " + ".join(code_parts)
            sum_label = " + ".join(label_parts)
            code = (
                "def digital_root(n):\n"
                "    n = abs(int(n))\n"
                "    return n if n == 0 else ((n - 1) % 9) + 1\n"
                f"result = digital_root({sum_code})\n"
                "print(result)"
            )
            expression = f"digital_root({sum_label})"
            total = abs(sum(int_vals))
            passcode = total if total == 0 else ((total - 1) % 9) + 1

        elif expr_type == "mod10":
            sum_code = " + ".join(code_parts)
            sum_label = " + ".join(label_parts)
            code = f"result = ({sum_code}) % 10\nprint(result)"
            expression = f"({sum_label}) % 10"
            passcode = sum(int_vals) % 10

        else:  # abs_diff_mod10
            code = f"result = abs({code_parts[0]} - {code_parts[1]}) % 10\nprint(result)"
            expression = f"abs({label_parts[0]} - {label_parts[1]}) % 10"
            passcode = abs(int_vals[0] - int_vals[1]) % 10
    else:
        code = f"result = abs({code_parts[0]}) % 10\nprint(result)"
        expression = f"abs({label_parts[0]}) % 10"
        passcode = abs(int_vals[0]) % 10

    return Stop(
        index=step_idx,
        stop_type="compute",
        page_url=None,
        extraction_target=f"Compute: {expression}",
        extraction_section=None,
        extracted_value=passcode,
        extracted_value_type="number",
        bridge=Bridge(
            bridge_type="compute",
            expression=expression,
            expression_code=code,
            referenced_stops=used_indices,
        ),
        reasoning=f"Compute stop: {expression} = {passcode}",
    )


# ---------------------------------------------------------------------------
# TrailBuilder
# ---------------------------------------------------------------------------


class TrailBuilder:
    """Constructs trails using LLM-planned coherent themes."""

    def __init__(
        self,
        wiki_graph: WikiGraph,
        fact_extractor: FactExtractor,
        llm_client: OpenAI,
        model: str = "gpt-4o-mini",
        tool_registry: ToolRegistry | None = None,
        rng: random.Random | None = None,
    ):
        self._graph = wiki_graph
        self._extractor = fact_extractor
        self._llm = llm_client
        self._model = model
        self._tool_registry = tool_registry
        self._rng = rng or random.Random()

    async def _prevalidate_tool_stops(
        self, stops: list[Stop],
    ) -> tuple[list[Stop], dict[int, int]]:
        """Dry-run tool chains; drop stops whose chains fail.

        Returns (validated_stops, old_to_new_index_map).
        """
        from trail.golden import _execute_tool_chain

        registry = self._tool_registry
        assert registry is not None

        validated: list[Stop] = []
        for stop in stops:
            if stop.stop_type != "tool" or not stop.bridge or not stop.bridge.tool_chain:
                validated.append(stop)
                continue

            try:
                value, records = await _execute_tool_chain(
                    stop.bridge.tool_chain, registry,
                )
                if value is None:
                    errors = [r.get("error") for r in records if r.get("error")]
                    logger.info(
                        "Dropping tool stop %d: chain returned no value (%s)",
                        stop.index, "; ".join(errors) if errors else "unknown",
                    )
                    continue
                # Store the validated value so golden execution can confirm it later
                stop.extracted_value = value
                validated.append(stop)
            except Exception as exc:
                logger.info("Dropping tool stop %d: chain error: %s", stop.index, exc)
                continue

        # Build old-to-new index mapping and re-index
        old_to_new: dict[int, int] = {}
        for new_idx, stop in enumerate(validated):
            old_to_new[stop.index] = new_idx
            stop.index = new_idx

        return validated, old_to_new

    async def build_trail(
        self,
        seed_url: str,
        difficulty: TrailDifficultyConfig,
        avoid_pages: set[str] | None = None,
    ) -> Trail | None:
        """Build a single trail starting from seed_url using LLM planning.

        Args:
            avoid_pages: Page URLs used in prior samples — the planner will
                         try to avoid these to improve cross-sample diversity.

        Returns None if construction fails.
        """
        seed_page = self._graph.get_page(seed_url)
        if not seed_page:
            logger.warning("Seed page not in graph: %s", seed_url)
            return None

        # Phase 1: LLM plans a thematic route
        plan = await _plan_route(
            seed_page, self._graph, difficulty,
            self._llm, self._model, self._rng,
            avoid_pages=avoid_pages,
        )
        if plan is None:
            logger.warning("Failed to plan route from %s", seed_url)
            return None

        # Phase 2: Build stops from plan
        stops, numeric_values, location_names, all_values = await _build_stops_from_plan(
            plan, self._graph, self._extractor, self._rng,
        )
        if not stops:
            logger.warning("Failed to build stops from plan")
            return None

        # Phase 2.5: Pre-validate tool stops by dry-running their chains
        if self._tool_registry:
            stops, idx_map = await self._prevalidate_tool_stops(stops)
            if not stops:
                logger.warning("All stops removed after geocode pre-validation")
                return None
            # Remap value dicts to new indices
            numeric_values = {idx_map[k]: v for k, v in numeric_values.items() if k in idx_map}
            all_values = {idx_map[k]: v for k, v in all_values.items() if k in idx_map}

        # Check minimum depth after pre-validation (stops + compute stop = len(stops) + 1)
        min_depth = difficulty.depth_range[0]
        if len(stops) + 1 < min_depth:
            logger.warning(
                "Trail too short after pre-validation: %d stops (need %d), retrying",
                len(stops) + 1, min_depth,
            )
            return None

        # Trim excess stops from over-planning (keep at most depth-1 content stops)
        target_depth = plan.get("_depth", min_depth)
        max_content_stops = target_depth - 1  # -1 for compute stop
        if len(stops) > max_content_stops:
            excess = len(stops) - max_content_stops
            # Remove excess tool stops from the end (preserve page stops)
            to_remove: set[int] = set()
            for s in reversed(stops):
                if len(to_remove) >= excess:
                    break
                if s.stop_type == "tool":
                    to_remove.add(s.index)
            # If not enough tool stops to remove, trim from end
            if len(to_remove) < excess:
                for s in reversed(stops):
                    if len(to_remove) >= excess:
                        break
                    to_remove.add(s.index)
            stops = [s for s in stops if s.index not in to_remove]
            # Re-index stops and rebuild value dicts
            new_numeric = {}
            new_all = {}
            for new_idx, s in enumerate(stops):
                old_idx = s.index
                s.index = new_idx
                if old_idx in numeric_values:
                    new_numeric[new_idx] = numeric_values[old_idx]
                if old_idx in all_values:
                    new_all[new_idx] = all_values[old_idx]
            numeric_values = new_numeric
            all_values = new_all

        # Phase 2.7: Inject analytical reasoning stops
        if self._tool_registry and difficulty.reason_stops_range[1] > 0:
            stops, numeric_values, all_values = await _inject_reason_stops(
                stops, numeric_values, all_values,
                difficulty, self._tool_registry, self._rng,
            )

        # Phase 3: Build bridges between stops
        _build_bridges(stops, self._graph)

        # Add final compute stop
        compute_stop = _build_compute_stop(len(stops), numeric_values, all_values, self._rng)
        if compute_stop is None:
            logger.warning("Failed to build compute stop")
            return None
        stops.append(compute_stop)

        # Assemble trail
        trail_id = f"trail_{uuid.uuid4().hex[:8]}"
        trail = Trail(
            trail_id=trail_id,
            seed_url=seed_url,
            seed_title=seed_page.title,
            stops=stops,
            metadata={
                "theme": plan.get("theme", ""),
                "difficulty": difficulty.level,
            },
        )

        # Compute passcode
        if trail.stops and trail.stops[-1].stop_type == "compute":
            passcode_val = trail.stops[-1].extracted_value
            if isinstance(passcode_val, (int, float)):
                trail.passcode = int(passcode_val) % 10

        # Compute difficulty metadata
        actual_tool_count = sum(1 for s in trail.stops if s.stop_type == "tool")
        actual_reason_count = sum(1 for s in trail.stops if s.stop_type == "reason")
        max_extraction = "infobox"
        for s in trail.stops:
            if s.stop_type == "page" and s.extraction_section:
                if s.extraction_section not in ("infobox",):
                    if "cross" in s.extraction_section.lower() or s.extraction_section == "cross_section":
                        max_extraction = "cross_section"
                    elif max_extraction != "cross_section":
                        max_extraction = "prose"

        has_search_bridge = any(
            s.bridge and s.bridge.bridge_type == "search_query"
            for s in trail.stops
        )
        trail.difficulty = TrailDifficulty(
            level=difficulty.level,
            depth=len(trail.stops),
            tool_stop_count=actual_tool_count,
            reason_stop_count=actual_reason_count,
            extraction_difficulty=max_extraction,
            bridge_obscurity="search" if has_search_bridge else "direct_link",
        )

        return trail
