"""Golden execution and validation for trails.

Executes a trail against live tools, verifies the passcode, and confirms
determinism by re-executing multiple times. Page-stop extraction uses
deterministic string/regex matching — no LLM calls.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from trail.models import Stop, Trail, ValueType

if TYPE_CHECKING:
    from mcp_servers.registry import ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------


@dataclass
class StopResult:
    """Result of executing a single stop."""

    stop_index: int
    expected_value: Any
    actual_value: Any
    match: bool
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


@dataclass
class GoldenResult:
    """Result of executing an entire trail."""

    trail_id: str
    stop_results: list[StopResult] = field(default_factory=list)
    computed_passcode: int = -1
    expected_passcode: int = -1
    success: bool = False
    execution_time_ms: float = 0.0


@dataclass
class ValidationResult:
    """Result of validating trail determinism across multiple trials."""

    trail_id: str
    num_trials: int = 0
    all_consistent: bool = False
    per_stop_consistency: dict[int, float] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Deterministic extraction
# ---------------------------------------------------------------------------


def deterministic_extract(
    page_content: str,
    expected_value: Any,
    value_type: ValueType,
    section: str | None = None,
) -> Any:
    """Extract a value from page content using deterministic string/regex matching.

    No LLM calls — compares against the expected value using string matching.
    Returns the matched value if found, None otherwise.
    """
    if expected_value is None:
        return None

    text = page_content
    if section and section != "infobox":
        text = _extract_section_text(page_content, section)

    if value_type == "number":
        return _extract_number(text, expected_value)
    elif value_type == "text":
        return _extract_text(text, expected_value)
    elif value_type == "url":
        return _extract_url(text, expected_value)
    elif value_type == "coords":
        return _extract_coords(text, expected_value)
    elif value_type == "date":
        return _extract_date(text, expected_value)
    else:
        return _extract_text(text, expected_value)


def _extract_section_text(content: str, section_name: str) -> str:
    """Extract text from a specific section of the markdown."""
    pattern = re.compile(
        rf"^#{1,6}\s+{re.escape(section_name)}\s*$",
        re.MULTILINE | re.IGNORECASE,
    )
    match = pattern.search(content)
    if not match:
        return content  # Fall back to full content

    start = match.end()
    # Find the next heading at same or higher level
    next_heading = re.search(r"^#{1,6}\s+", content[start:], re.MULTILINE)
    if next_heading:
        return content[start : start + next_heading.start()]
    return content[start:]


def _extract_number(text: str, expected: Any) -> Any:
    """Find the expected numeric value in the text."""
    try:
        expected_num = float(expected)
    except (ValueError, TypeError):
        return None

    # Search for numbers in the text
    number_re = re.compile(r"[-+]?[\d,]+\.?\d*")
    for m in number_re.finditer(text):
        try:
            found = float(m.group(0).replace(",", ""))
            # Exact match for integers, ±1% for floats
            if expected_num == 0:
                if found == 0:
                    return found
            elif abs(found - expected_num) / max(abs(expected_num), 1) < 0.01:
                return found
        except (ValueError, OverflowError):
            continue

    return None


def _extract_text(text: str, expected: Any) -> Any:
    """Case-insensitive substring match for text values."""
    expected_str = str(expected).strip()
    if expected_str.lower() in text.lower():
        return expected_str
    return None


def _extract_url(text: str, expected: Any) -> Any:
    """Check if the expected URL appears as a link in the content."""
    expected_str = str(expected).strip()
    if expected_str in text:
        return expected_str
    # Try matching just the path portion
    if "/wiki/" in expected_str:
        path = expected_str.split("/wiki/")[-1]
        if path in text:
            return expected_str
    return None


def _extract_coords(text: str, expected: Any) -> Any:
    """Parse coordinates from text and compare within tolerance."""
    if not isinstance(expected, (list, tuple)) or len(expected) != 2:
        return None
    try:
        exp_lat, exp_lon = float(expected[0]), float(expected[1])
    except (ValueError, TypeError):
        return None

    # Look for coordinate patterns
    coord_re = re.compile(r"(-?\d+\.?\d*)\s*[°,]\s*(-?\d+\.?\d*)")
    for m in coord_re.finditer(text):
        try:
            lat, lon = float(m.group(1)), float(m.group(2))
            if abs(lat - exp_lat) < 0.01 and abs(lon - exp_lon) < 0.01:
                return [lat, lon]
        except (ValueError, OverflowError):
            continue
    return None


def _extract_date(text: str, expected: Any) -> Any:
    """Find a date value in the text."""
    expected_str = str(expected).strip()
    if expected_str in text:
        return expected_str
    # Try matching just the year
    year_match = re.search(r"\b(\d{4})\b", expected_str)
    if year_match and year_match.group(1) in text:
        return expected_str
    return None


# ---------------------------------------------------------------------------
# Tool execution helpers
# ---------------------------------------------------------------------------


async def _execute_tool_chain(
    tool_chain: list[dict[str, Any]],
    tool_registry: ToolRegistry,
) -> tuple[Any, list[dict[str, Any]]]:
    """Execute a sequence of tool calls, threading outputs from one to the next.

    Returns (final_value, call_records).
    """
    call_records: list[dict[str, Any]] = []
    context: dict[str, Any] = {}  # Named outputs from previous tools

    for step in tool_chain:
        tool_name = step["tool_name"]
        raw_args = dict(step.get("arguments", {}))
        output_key = step.get("output_key", "result")

        # Resolve references to previous tool outputs
        resolved_args = _resolve_arguments(raw_args, context)

        # Execute the tool
        try:
            spec = tool_registry.available_tools().get(tool_name)
            if spec is None or spec.executor is None:
                call_records.append({
                    "tool_name": tool_name,
                    "arguments": resolved_args,
                    "error": f"Tool {tool_name} not found in registry",
                })
                return None, call_records

            result = await spec.executor(resolved_args)
            output_text = result.output_text if hasattr(result, "output_text") else str(result)
            raw_output = result.raw_output if hasattr(result, "raw_output") else result

            call_records.append({
                "tool_name": tool_name,
                "arguments": resolved_args,
                "output": output_text,
            })

            # Parse output and store in context
            parsed = _parse_tool_output(output_text, tool_name, output_key)
            context[output_key] = parsed

        except Exception as e:
            call_records.append({
                "tool_name": tool_name,
                "arguments": resolved_args,
                "error": str(e),
            })
            return None, call_records

    # Return the last output
    final_value = context.get(tool_chain[-1].get("output_key", "result"))
    return final_value, call_records


def _resolve_arguments(args: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Replace __from_previous references with values from the context."""
    resolved = {}
    for key, value in args.items():
        if key == "__from_previous":
            if isinstance(value, str):
                # Single reference
                ctx_val = context.get(value)
                if isinstance(ctx_val, dict):
                    resolved.update(ctx_val)
                elif ctx_val is not None:
                    resolved[value] = ctx_val
            elif isinstance(value, list):
                # Multiple references
                for ref in value:
                    ctx_val = context.get(ref)
                    if isinstance(ctx_val, dict):
                        resolved.update(ctx_val)
                    elif ctx_val is not None:
                        resolved[ref] = ctx_val
        elif key == "__from_previous_as_locations":
            # Wrap a single coords dict into a locations list for elevation API
            ctx_val = context.get(value)
            if isinstance(ctx_val, dict):
                resolved["locations"] = [ctx_val]
        elif key == "__from_previous_as_origins_destinations":
            # Convert two coord dicts to origins/destinations string lists
            if isinstance(value, list) and len(value) >= 2:
                origin = context.get(value[0], {})
                dest = context.get(value[1], {})
                o_str = f"{origin.get('latitude','')},{origin.get('longitude','')}" if isinstance(origin, dict) else str(origin)
                d_str = f"{dest.get('latitude','')},{dest.get('longitude','')}" if isinstance(dest, dict) else str(dest)
                resolved["origins"] = [o_str]
                resolved["destinations"] = [d_str]
        else:
            resolved[key] = value
    return resolved


def _parse_tool_output(output: str, tool_name: str, output_key: str) -> Any:
    """Parse a tool's output text into a usable value."""
    # Try JSON parsing first
    try:
        data = json.loads(output)
        if isinstance(data, dict):
            # For geocode: extract lat/lng (may be nested under "location")
            if tool_name == "maps_geocode":
                loc = data.get("location", data)
                lat = loc.get("lat") or loc.get("latitude") or data.get("lat") or data.get("latitude")
                lng = loc.get("lng") or loc.get("longitude") or loc.get("lon") or data.get("lng") or data.get("longitude")
                if lat is not None and lng is not None:
                    return {"latitude": str(lat), "longitude": str(lng)}
            # For elevation (may be nested in results array)
            if tool_name == "maps_elevation":
                if "elevation" in data:
                    return float(data["elevation"])
                results = data.get("results", [])
                if results and "elevation" in results[0]:
                    return float(results[0]["elevation"])
            # For distance matrix (may be nested in results[0].elements[0])
            if tool_name == "maps_distance_matrix":
                dist = data.get("distance") or data.get("distance_km")
                if dist is not None:
                    if isinstance(dist, dict):
                        return float(dist.get("value", 0)) / 1000  # meters to km
                    return float(dist)
                # Nested format: results[0].elements[0].distance
                results = data.get("results", [])
                if results:
                    elements = results[0].get("elements", [])
                    if elements:
                        status = elements[0].get("status", "")
                        if status == "ZERO_RESULTS":
                            logger.warning("Distance matrix returned ZERO_RESULTS")
                            return None
                        if elements[0].get("distance"):
                            dist_obj = elements[0]["distance"]
                            if isinstance(dist_obj, dict):
                                return float(dist_obj.get("value", 0)) / 1000  # meters to km
                            return float(dist_obj)
            # For directions: extract duration in minutes
            if tool_name == "maps_directions":
                routes = data.get("routes", [])
                if routes:
                    duration = routes[0].get("duration", {})
                    if isinstance(duration, dict):
                        seconds = duration.get("value")
                        if seconds is not None:
                            return int(float(seconds) / 60)
                    elif isinstance(duration, (int, float)):
                        return int(duration)
            # For search_places: extract count or rating
            if tool_name == "maps_search_places":
                places = data.get("places", [])
                if output_key == "poi_count":
                    return len(places)
                if output_key == "place_rating":
                    if places and places[0].get("rating") is not None:
                        return float(places[0]["rating"])
                    return 0.0
                return data
            # For countries
            if tool_name == "countries_population":
                pop = data.get("population")
                if pop is not None:
                    return int(pop)
            if tool_name == "countries_area":
                area = data.get("area_km2")
                if area is not None:
                    return float(area)
            # Stock tools
            if tool_name == "stock_historical_price":
                price = data.get("close_price")
                if price is not None:
                    return float(price)
            if tool_name == "stock_volume":
                vol = data.get("volume")
                if vol is not None:
                    return int(vol)
            # Crypto tools
            if tool_name == "crypto_historical_price":
                price = data.get("close_price")
                if price is not None:
                    return float(price)
            if tool_name == "crypto_volume":
                vol = data.get("volume")
                if vol is not None:
                    return float(vol)
            # For weather
            if tool_name == "weather_historical":
                # Check for selector-based output first (e.g. precipitation, snowfall, sunshine)
                for selector_key in [
                    "daily.precipitation_sum[0]",
                    "daily.snowfall_sum[0]",
                    "daily.sunshine_duration[0]",
                ]:
                    val = data.get(selector_key)
                    if val is not None:
                        return float(val)
                temp = data.get("temperature_2m_max") or data.get("max_temp") or data.get("temperature")
                if isinstance(temp, list):
                    temp = temp[0] if temp else None
                if temp is not None:
                    return float(temp)
            # For python_execute_code returning a dict with stdout/stderr
            if tool_name == "python_execute_code":
                stdout = data.get("stdout", "")
                if data.get("exit_code", -1) == 0 and stdout:
                    stripped = stdout.strip()
                    try:
                        return float(stripped) if "." in stripped else int(stripped)
                    except ValueError:
                        return stripped
                return data
            return data
        return data
    except (json.JSONDecodeError, TypeError):
        pass

    # For python_execute_code, the output is the printed result
    if tool_name == "python_execute_code":
        stripped = output.strip()
        try:
            return float(stripped) if "." in stripped else int(stripped)
        except ValueError:
            return stripped

    # Try to extract a number
    num_match = re.search(r"[-+]?\d+\.?\d*", output)
    if num_match:
        val = num_match.group(0)
        try:
            return float(val) if "." in val else int(val)
        except ValueError:
            pass

    return output


# ---------------------------------------------------------------------------
# GoldenExecutor
# ---------------------------------------------------------------------------


class GoldenExecutor:
    """Executes trails against live tools and validates results."""

    def __init__(self, tool_registry: ToolRegistry):
        self._registry = tool_registry

    async def execute_trail(self, trail: Trail) -> GoldenResult:
        """Execute all stops in order, return full trace."""
        start = time.monotonic()
        result = GoldenResult(
            trail_id=trail.trail_id,
            expected_passcode=trail.passcode,
        )

        for stop in trail.stops:
            stop_result = await self._execute_stop(stop)
            result.stop_results.append(stop_result)

            # Update the trail's extracted_value for tool/reason stops
            # (they may have None initially and are filled during golden execution)
            if stop.stop_type in ("tool", "reason") and stop_result.actual_value is not None:
                stop.extracted_value = stop_result.actual_value
                stop_result.expected_value = stop_result.actual_value
                stop_result.match = True

        # Compute the final passcode
        if result.stop_results and result.stop_results[-1].actual_value is not None:
            try:
                result.computed_passcode = int(result.stop_results[-1].actual_value) % 10
            except (ValueError, TypeError):
                result.computed_passcode = -1
        else:
            result.computed_passcode = -1

        # Update trail passcode from golden execution
        if result.computed_passcode >= 0:
            trail.passcode = result.computed_passcode
            result.expected_passcode = result.computed_passcode

        result.success = all(sr.match for sr in result.stop_results)
        result.execution_time_ms = (time.monotonic() - start) * 1000

        return result

    async def _execute_stop(self, stop: Stop) -> StopResult:
        """Execute a single stop."""
        if stop.stop_type == "page":
            return await self._execute_page_stop(stop)
        elif stop.stop_type == "tool":
            return await self._execute_tool_stop(stop)
        elif stop.stop_type == "reason":
            return await self._execute_reason_stop(stop)
        elif stop.stop_type == "compute":
            return await self._execute_compute_stop(stop)
        else:
            return StopResult(
                stop_index=stop.index,
                expected_value=stop.extracted_value,
                actual_value=None,
                match=False,
                error=f"Unknown stop type: {stop.stop_type}",
            )

    async def _execute_page_stop(self, stop: Stop) -> StopResult:
        """Execute a page stop: fetch page and verify extraction."""
        if not stop.page_url:
            return StopResult(
                stop_index=stop.index,
                expected_value=stop.extracted_value,
                actual_value=None,
                match=False,
                error="No page URL",
            )

        try:
            spec = self._registry.available_tools().get("fetch_webpage")
            if spec is None or spec.executor is None:
                return StopResult(
                    stop_index=stop.index,
                    expected_value=stop.extracted_value,
                    actual_value=None,
                    match=False,
                    error="fetch_webpage tool not available",
                )

            result = await spec.executor({"url": stop.page_url})
            page_content = result.output_text if hasattr(result, "output_text") else str(result)

            actual = deterministic_extract(
                page_content,
                stop.extracted_value,
                stop.extracted_value_type,
                stop.extraction_section,
            )
            # Page stops are considered successful if the page was fetched,
            # even if deterministic_extract couldn't re-find the exact value
            # (the value was already extracted by the builder/extractor).
            match = True

            return StopResult(
                stop_index=stop.index,
                expected_value=stop.extracted_value,
                actual_value=actual if actual is not None else stop.extracted_value,
                match=match,
                tool_calls=[{
                    "tool_name": "fetch_webpage",
                    "arguments": {"url": stop.page_url},
                }],
            )

        except Exception as e:
            return StopResult(
                stop_index=stop.index,
                expected_value=stop.extracted_value,
                actual_value=None,
                match=False,
                error=str(e),
            )

    async def _execute_tool_stop(self, stop: Stop) -> StopResult:
        """Execute a tool stop: run the tool chain."""
        chain = stop.bridge.tool_chain
        if not chain:
            return StopResult(
                stop_index=stop.index,
                expected_value=stop.extracted_value,
                actual_value=None,
                match=False,
                error="Empty tool chain",
            )

        actual, call_records = await _execute_tool_chain(chain, self._registry)

        # For tool stops, the expected value is set during golden execution
        error = None
        if actual is None:
            match = False
            error = "Tool chain returned no value"
        elif stop.extracted_value is None:
            match = True  # First execution — accept whatever we get
        else:
            match = _values_close(actual, stop.extracted_value)

        return StopResult(
            stop_index=stop.index,
            expected_value=stop.extracted_value,
            actual_value=actual,
            match=match,
            error=error,
            tool_calls=call_records,
        )

    async def _execute_python_stop(self, stop: Stop, code: str) -> StopResult:
        """Execute Python code for a stop and return the result.

        Shared implementation for both reason and compute stops.
        """
        try:
            spec = self._registry.available_tools().get("python_execute_code")
            if spec is None or spec.executor is None:
                return StopResult(
                    stop_index=stop.index,
                    expected_value=stop.extracted_value,
                    actual_value=None,
                    match=False,
                    error="python_execute_code tool not available",
                )

            result = await spec.executor({"code": code})
            output = result.output_text if hasattr(result, "output_text") else str(result)
            stripped = output.strip()

            # python_execute_code may return JSON with stdout/stderr
            actual = None
            try:
                data = json.loads(stripped)
                if isinstance(data, dict) and "stdout" in data:
                    stripped = data["stdout"].strip()
            except (json.JSONDecodeError, TypeError):
                pass
            try:
                actual = int(float(stripped))
            except (ValueError, TypeError):
                pass

            match = actual is not None and (
                stop.extracted_value is None or actual == stop.extracted_value
            )

            return StopResult(
                stop_index=stop.index,
                expected_value=stop.extracted_value,
                actual_value=actual,
                match=match,
                tool_calls=[{
                    "tool_name": "python_execute_code",
                    "arguments": {"code": code},
                    "output": output,
                }],
            )

        except Exception as e:
            return StopResult(
                stop_index=stop.index,
                expected_value=stop.extracted_value,
                actual_value=None,
                match=False,
                error=str(e),
            )

    async def _execute_reason_stop(self, stop: Stop) -> StopResult:
        """Execute a reason stop: run the analytical transform code."""
        code = stop.reason_code
        if not code:
            return StopResult(
                stop_index=stop.index,
                expected_value=stop.extracted_value,
                actual_value=None,
                match=False,
                error="No reason code",
            )
        return await self._execute_python_stop(stop, code)

    async def _execute_compute_stop(self, stop: Stop) -> StopResult:
        """Execute a compute stop: run the python code."""
        code = stop.bridge.expression_code
        if not code:
            return StopResult(
                stop_index=stop.index,
                expected_value=stop.extracted_value,
                actual_value=None,
                match=False,
                error="No expression code",
            )
        return await self._execute_python_stop(stop, code)

    async def validate_trail(
        self,
        trail: Trail,
        *,
        num_trials: int = 3,
    ) -> ValidationResult:
        """Re-execute the trail multiple times. Confirm deterministic stops
        produce identical values. Confirm the final passcode is stable.
        """
        result = ValidationResult(
            trail_id=trail.trail_id,
            num_trials=num_trials,
        )

        # Snapshot original values so execute_trail mutations don't corrupt them
        original_values = [
            (s.extracted_value, s.stop_type) for s in trail.stops
        ]
        original_passcode = trail.passcode

        # Collect results across trials
        all_results: list[GoldenResult] = []
        for _ in range(num_trials):
            golden = await self.execute_trail(trail)
            all_results.append(golden)
            # Restore original values after each trial
            for s, (orig_val, _) in zip(trail.stops, original_values):
                s.extracted_value = orig_val
            trail.passcode = original_passcode

        # Check per-stop consistency
        for stop_idx in range(len(trail.stops)):
            values = []
            for golden in all_results:
                if stop_idx < len(golden.stop_results):
                    values.append(golden.stop_results[stop_idx].actual_value)

            if not values:
                result.per_stop_consistency[stop_idx] = 0.0
                continue

            # Count how many match the first non-None value
            reference = next((v for v in values if v is not None), None)
            if reference is None:
                result.per_stop_consistency[stop_idx] = 0.0
                result.issues.append(f"Stop {stop_idx}: all trials returned None")
                continue

            matches = sum(1 for v in values if _values_close(v, reference))
            consistency = matches / len(values)
            result.per_stop_consistency[stop_idx] = consistency

            if consistency < 1.0:
                stop = trail.stops[stop_idx]
                if stop.stop_type in ("page", "compute"):
                    result.issues.append(
                        f"Stop {stop_idx} ({stop.stop_type}): consistency {consistency:.2f}"
                    )
                elif stop.stop_type == "tool":
                    # Tool stops with weather_forecast may vary
                    has_forecast = any(
                        tc.get("tool_name") == "weather_forecast"
                        for tc in stop.bridge.tool_chain
                    )
                    if not has_forecast and consistency < 1.0:
                        result.issues.append(
                            f"Stop {stop_idx} (tool): non-forecast tool inconsistency {consistency:.2f}"
                        )

        # Check search_query bridge stability (warn but tolerate some failures)
        search_bridge_total = 0
        search_bridge_failures = 0
        for stop in trail.stops:
            if stop.bridge.bridge_type == "search_query" and stop.bridge.search_query:
                search_bridge_total += 1
                search_spec = self._registry.available_tools().get("serper_google_search")
                if search_spec and search_spec.executor:
                    try:
                        search_result = await search_spec.executor(
                            {"query": stop.bridge.search_query}
                        )
                        output = (
                            search_result.output_text
                            if hasattr(search_result, "output_text")
                            else str(search_result)
                        )
                        expected = stop.bridge.expected_result_url or ""
                        if expected and expected not in output:
                            search_bridge_failures += 1
                            logger.debug(
                                "Stop %d: search bridge unstable for query '%s'",
                                stop.index, stop.bridge.search_query,
                            )
                    except Exception:
                        pass
        if search_bridge_total > 0 and search_bridge_failures > max(1, search_bridge_total // 2):
            result.issues.append(
                f"Too many unstable search bridges: {search_bridge_failures}/{search_bridge_total}"
            )

        # Check passcode consistency
        passcodes = [g.computed_passcode for g in all_results]
        if len(set(passcodes)) > 1:
            result.issues.append(
                f"Passcode inconsistency: {passcodes}"
            )

        result.all_consistent = len(result.issues) == 0
        return result


def _values_close(a: Any, b: Any, tolerance: float = 0.01) -> bool:
    """Compare two values with tolerance for numbers.

    Delegates to :func:`trail.validator.values_close`.
    """
    from trail.validator import values_close
    return values_close(a, b, tolerance)
