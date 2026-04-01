"""Trail augmentation framework component.

Provides :class:`TrailAugmenter` which adds tool stops and/or reason stops to
existing trails, rebuilds the compute stop, and re-verbalizes the riddle.

Refactored from ``scripts/augment_stops.py`` into a reusable, file-I/O-free
class suitable for programmatic use.
"""

from __future__ import annotations

import copy
import json
import logging
import random
import re
from typing import Any
from urllib.parse import unquote

from mcp_servers.registry import ToolRegistry
from trail.builder import (
    REASON_TRANSFORMS,
    _build_compute_stop,
    _build_country_area_chain,
    _build_country_population_chain,
    _build_geocode_elevation_chain,
    _build_geocode_weather_chain,
    _build_nearby_poi_count_chain,
    _build_place_rating_chain,
    _build_reason_code,
    _is_geocodable,
    _POI_TYPES,
    _select_applicable_transforms,
    _value_to_code_literal,
    _value_to_number,
)
from trail.golden import _execute_tool_chain
from trail.models import (
    Bridge,
    DIFFICULTY_CONFIGS,
    Stop,
    Trail,
    TrailDifficultyConfig,
)
from trail.verbalizer import TrailVerbalizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level location helpers
# ---------------------------------------------------------------------------

_KNOWN_COUNTRIES = {
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Argentina",
    "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain",
    "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin",
    "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil",
    "Brunei", "Bulgaria", "Burkina Faso", "Burundi", "Cambodia", "Cameroon",
    "Canada", "Chad", "Chile", "China", "Colombia", "Congo", "Costa Rica",
    "Croatia", "Cuba", "Cyprus", "Czech Republic", "Denmark", "Djibouti",
    "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Estonia",
    "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia",
    "Germany", "Ghana", "Greece", "Guatemala", "Guinea", "Guyana", "Haiti",
    "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq",
    "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan",
    "Kenya", "Kuwait", "Laos", "Latvia", "Lebanon", "Libya", "Lithuania",
    "Luxembourg", "Madagascar", "Malaysia", "Mali", "Malta", "Mexico",
    "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique",
    "Myanmar", "Namibia", "Nepal", "Netherlands", "New Zealand", "Nicaragua",
    "Niger", "Nigeria", "North Korea", "North Macedonia", "Norway", "Oman",
    "Pakistan", "Panama", "Paraguay", "Peru", "Philippines", "Poland",
    "Portugal", "Qatar", "Romania", "Russia", "Rwanda", "Saudi Arabia",
    "Senegal", "Serbia", "Singapore", "Slovakia", "Slovenia", "Somalia",
    "South Africa", "South Korea", "Spain", "Sri Lanka", "Sudan", "Suriname",
    "Sweden", "Switzerland", "Syria", "Taiwan", "Tanzania", "Thailand",
    "Togo", "Trinidad and Tobago", "Tunisia", "Turkey", "Uganda", "Ukraine",
    "United Arab Emirates", "United Kingdom", "United States", "Uruguay",
    "Uzbekistan", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe",
}


def _location_from_page_url(page_url: str) -> str | None:
    """Derive a location name from a Wikipedia page URL."""
    if not page_url or "/wiki/" not in page_url:
        return None
    title = unquote(page_url.split("/wiki/")[-1]).replace("_", " ")
    # Strip common suffixes / disambiguation
    title = re.sub(r"\s*\(.*?\)\s*$", "", title)
    if not title or len(title) < 2:
        return None
    return title


def _location_from_stop(stop: Stop) -> str | None:
    """Extract a plausible location name from a page stop.

    Tries extraction_target first, then falls back to the page URL title.
    """
    if stop.extracted_value_type == "text" and stop.extracted_value:
        val = str(stop.extracted_value).strip()
        if _is_geocodable(val) and len(val) < 60:
            return val
    return _location_from_page_url(stop.page_url or "")


def _country_from_stop(stop: Stop) -> str | None:
    """Try to extract a country name from a stop."""
    title = _location_from_page_url(stop.page_url or "")
    if title and title in _KNOWN_COUNTRIES:
        return title

    if stop.extracted_value_type == "text" and stop.extracted_value:
        val = str(stop.extracted_value).strip()
        if val in _KNOWN_COUNTRIES:
            return val

    return None


# ---------------------------------------------------------------------------
# TrailAugmenter
# ---------------------------------------------------------------------------


class TrailAugmenter:
    """Augments existing trails by adding tool/reason stops, rebuilding
    compute, and re-verbalizing."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        verbalizer: TrailVerbalizer | None = None,
        rng: random.Random | None = None,
    ):
        self._tool_registry = tool_registry
        self._verbalizer = verbalizer
        self._rng = rng or random.Random()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def augment(
        self,
        trail: Trail,
        *,
        add_tool_stops: int = 0,
        add_reason_stops: int = 0,
        skip_reverbalize: bool = False,
    ) -> Trail | None:
        """Add exactly N tool/reason stops.  Returns augmented trail or
        ``None`` on failure.

        Works on a deep copy -- never mutates the input trail.  Handles all
        side effects: re-indexing, compute rebuild, re-verbalization, and
        integrity check.

        Parameters
        ----------
        trail:
            The source trail.  Not mutated.
        add_tool_stops:
            Number of tool stops to add.
        add_reason_stops:
            Number of reason stops to add.
        skip_reverbalize:
            If *True*, skip riddle re-generation even when stops were added.
            The caller accepts responsibility for an inconsistent riddle.

        Returns
        -------
        Trail | None
            The augmented trail, or ``None`` if re-verbalization fails or
            integrity checks fail.
        """
        if add_tool_stops <= 0 and add_reason_stops <= 0:
            return trail  # nothing to do

        trail = copy.deepcopy(trail)

        total_added = 0

        # --- tool stops ---------------------------------------------------
        if add_tool_stops > 0:
            tool_insertions = await self._build_tool_stops(trail, add_tool_stops)
            if tool_insertions:
                has_compute = trail.stops and trail.stops[-1].stop_type == "compute"
                content_stops = [s for s in trail.stops if s.stop_type != "compute"]
                # Sort descending so inserts don't shift indices
                tool_insertions.sort(key=lambda x: x[0], reverse=True)
                for insert_after, stop in tool_insertions:
                    content_stops.insert(insert_after + 1, stop)
                trail.stops = self._reindex_and_rebuild(
                    content_stops, self._rng, has_compute,
                )
                total_added += len(tool_insertions)

        # --- reason stops -------------------------------------------------
        if add_reason_stops > 0:
            reason_insertions = await self._build_reason_stops(trail, add_reason_stops)
            if reason_insertions:
                has_compute = trail.stops and trail.stops[-1].stop_type == "compute"
                content_stops = [s for s in trail.stops if s.stop_type != "compute"]
                reason_insertions.sort(key=lambda x: x[0], reverse=True)
                for insert_after, stop in reason_insertions:
                    content_stops.insert(insert_after + 1, stop)
                trail.stops = self._reindex_and_rebuild(
                    content_stops, self._rng, has_compute,
                )
                total_added += len(reason_insertions)

        if total_added == 0:
            logger.warning(
                "Trail %s: requested augmentation but could not add any stops",
                trail.trail_id,
            )
            return None

        # --- metadata & passcode ------------------------------------------
        self._update_trail_metadata(trail)

        # --- re-verbalization ---------------------------------------------
        if not skip_reverbalize:
            if self._verbalizer is None:
                logger.warning(
                    "Trail %s: stops were added but no verbalizer provided; "
                    "returning None (pass skip_reverbalize=True to opt out)",
                    trail.trail_id,
                )
                return None

            new_riddle = await self._reverbalize(trail)
            if new_riddle is None:
                logger.error(
                    "Trail %s: re-verbalization failed after augmentation",
                    trail.trail_id,
                )
                return None
            trail.riddle = new_riddle

        # --- integrity check ----------------------------------------------
        from trail.validator import validate_trail as _validate_trail
        validation = _validate_trail(trail)
        if not validation.is_valid:
            logger.error(
                "Trail %s: integrity check failed after augmentation: %s",
                trail.trail_id,
                "; ".join(str(i) for i in validation.critical_issues),
            )
            return None

        logger.info(
            "Trail %s augmented: +%d stop(s), passcode=%d",
            trail.trail_id, total_added, trail.passcode,
        )
        return trail

    async def augment_to_config(
        self,
        trail: Trail,
        config: TrailDifficultyConfig | None = None,
    ) -> Trail | None:
        """Add stops to meet difficulty config minimums.

        If *config* is ``None``, uses ``trail.difficulty.level`` to look up
        :data:`DIFFICULTY_CONFIGS`.  Returns the original trail unchanged if
        it already meets minimums.  Returns ``None`` on failure.
        """
        needed = self.compute_needed(trail, config)
        tool_needed = needed["tool_stops_needed"]
        reason_needed = needed["reason_stops_needed"]

        if tool_needed <= 0 and reason_needed <= 0:
            logger.info(
                "Trail %s already meets config minimums", trail.trail_id,
            )
            return trail

        return await self.augment(
            trail,
            add_tool_stops=max(tool_needed, 0),
            add_reason_stops=max(reason_needed, 0),
        )

    def compute_needed(
        self,
        trail: Trail,
        config: TrailDifficultyConfig | None = None,
    ) -> dict[str, int]:
        """Pure computation: how many stops are needed to meet config
        minimums.

        Returns
        -------
        dict
            ``{"tool_stops_needed": N, "reason_stops_needed": M}``
        """
        if config is None:
            level = trail.difficulty.level if trail.difficulty else "easy"
            config = DIFFICULTY_CONFIGS.get(level)
            if config is None:
                logger.warning("Unknown difficulty level: %s", level)
                return {"tool_stops_needed": 0, "reason_stops_needed": 0}

        current_tool = sum(1 for s in trail.stops if s.stop_type == "tool")
        current_reason = sum(1 for s in trail.stops if s.stop_type == "reason")

        return {
            "tool_stops_needed": max(0, config.tool_stops_range[0] - current_tool),
            "reason_stops_needed": max(0, config.reason_stops_range[0] - current_reason),
        }

    # ------------------------------------------------------------------
    # Internal: building stops
    # ------------------------------------------------------------------

    async def _build_tool_stops(
        self, trail: Trail, count: int,
    ) -> list[tuple[int, Stop]]:
        """Build up to *count* new tool stops from trail's page stops.

        Returns a list of ``(insert_after_position, tool_stop)`` tuples.
        """
        page_stops = [
            (i, s)
            for i, s in enumerate(trail.stops)
            if s.stop_type == "page" and s.page_url
        ]
        self._rng.shuffle(page_stops)

        insertions: list[tuple[int, Stop]] = []
        for list_pos, source_stop in page_stops:
            if len(insertions) >= count:
                break
            tool_stop = await self._try_build_single_tool_stop(source_stop)
            if tool_stop is not None:
                insertions.append((list_pos, tool_stop))

        if not insertions:
            logger.warning(
                "Could not build any tool stops for trail %s", trail.trail_id,
            )

        return insertions

    async def _try_build_single_tool_stop(
        self, source_stop: Stop,
    ) -> Stop | None:
        """Try multiple chain types for one source stop.  Rejects zero
        values.  Returns the first successful stop or ``None``.
        """
        location = _location_from_stop(source_stop)
        country = _country_from_stop(source_stop)

        candidates: list[tuple[str, list[dict[str, Any]], str]] = []

        if location and _is_geocodable(location):
            candidates.append((
                "elevation",
                _build_geocode_elevation_chain(location),
                f"Look up the elevation of {location}",
            ))
            candidates.append((
                "place_rating",
                _build_place_rating_chain(location),
                f"Look up the Google rating of {location}",
            ))
            poi_type = self._rng.choice(_POI_TYPES)
            candidates.append((
                "nearby_poi",
                _build_nearby_poi_count_chain(location, poi_type),
                f"Count {poi_type} near {location}",
            ))
            # Historical weather — use a stable past date
            weather_date = "2024-01-15"
            candidates.append((
                "weather",
                _build_geocode_weather_chain(location, weather_date),
                f"Look up the historical temperature in {location} on {weather_date}",
            ))

        if country:
            candidates.append((
                "population",
                _build_country_population_chain(country),
                f"Look up the population of {country}",
            ))
            candidates.append((
                "area",
                _build_country_area_chain(country),
                f"Look up the area of {country} in km\u00b2",
            ))

        if not candidates:
            return None

        self._rng.shuffle(candidates)

        for desc, chain, extraction_target in candidates:
            try:
                value, records = await _execute_tool_chain(
                    chain, self._tool_registry,
                )
                if value is None:
                    errors = [r.get("error") for r in records if r.get("error")]
                    logger.debug(
                        "Chain %s failed for %s: %s",
                        desc,
                        source_stop.page_url,
                        "; ".join(errors) if errors else "no value",
                    )
                    continue

                # Reject zero values
                try:
                    numeric_val = float(value)
                    if numeric_val == 0:
                        logger.debug(
                            "Chain %s returned zero value, skipping", desc,
                        )
                        continue
                except (ValueError, TypeError):
                    pass

                # Determine value type
                value_type = "number"
                try:
                    float(value)
                except (ValueError, TypeError):
                    value_type = "text"

                tool_stop = Stop(
                    index=-1,  # assigned during re-indexing
                    stop_type="tool",
                    page_url=source_stop.page_url,
                    extraction_target=extraction_target,
                    extraction_section=None,
                    extracted_value=value,
                    extracted_value_type=value_type,
                    bridge=Bridge(
                        bridge_type="tool_call",
                        tool_chain=chain,
                    ),
                    reasoning=(
                        f"Tool stop ({desc}) derived from "
                        f"{source_stop.page_url}"
                    ),
                )
                logger.info(
                    "Built tool stop (%s) for %s -> value=%s",
                    desc, source_stop.page_url, value,
                )
                return tool_stop

            except Exception as exc:
                logger.debug("Chain %s error: %s", desc, exc)
                continue

        return None

    async def _build_reason_stops(
        self, trail: Trail, count: int,
    ) -> list[tuple[int, Stop]]:
        """Build up to *count* new reason stops from eligible stops.

        Returns a list of ``(insert_after_position, reason_stop)`` tuples.
        """
        spec = self._tool_registry.available_tools().get("python_execute_code")
        if spec is None or spec.executor is None:
            logger.warning(
                "python_execute_code not available; skipping reason stops",
            )
            return []

        # Determine which stops already have a reason stop targeting them
        existing_reason_sources: set[int] = set()
        for s in trail.stops:
            if s.stop_type == "reason" and s.reason_source_stop is not None:
                existing_reason_sources.add(s.reason_source_stop)

        candidates: list[tuple[int, list[str]]] = []
        for i, stop in enumerate(trail.stops):
            if stop.stop_type not in ("page", "tool"):
                continue
            if stop.index in existing_reason_sources:
                continue
            vtype = stop.extracted_value_type or ""
            transforms = _select_applicable_transforms(
                stop.extracted_value, vtype,
            )
            if transforms:
                candidates.append((i, transforms))

        if not candidates:
            logger.warning(
                "No eligible stops for reason transforms in trail %s",
                trail.trail_id,
            )
            return []

        self._rng.shuffle(candidates)
        selected = candidates[:count]
        # Sort descending by position for safe insertion
        selected.sort(key=lambda x: x[0], reverse=True)

        insertions: list[tuple[int, Stop]] = []

        for list_pos, transforms in selected:
            source_stop = trail.stops[list_pos]
            self._rng.shuffle(transforms)

            created = False
            for transform_name in transforms:
                code = _build_reason_code(
                    transform_name, source_stop.extracted_value,
                )

                # Dry-run the transform
                try:
                    result = await spec.executor(
                        {"code": code, "timeout_seconds": 10},
                    )
                    output = (
                        result.output_text
                        if hasattr(result, "output_text")
                        else str(result)
                    )
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

                # Reject trivial outputs
                if transformed_value in (0, 1):
                    continue

                meta = REASON_TRANSFORMS[transform_name]
                reason_stop = Stop(
                    index=-1,  # assigned during re-indexing
                    stop_type="reason",
                    page_url=source_stop.page_url,
                    extraction_target=f"Reason: {meta['riddle_hint']}",
                    extraction_section=None,
                    extracted_value=transformed_value,
                    extracted_value_type="number",
                    bridge=Bridge(bridge_type="compute"),
                    reasoning=(
                        f"Analytical transform ({transform_name}) on "
                        f"stop {list_pos} value"
                    ),
                    reason_transform=transform_name,
                    reason_source_stop=source_stop.index,
                    reason_code=code,
                )
                insertions.append((list_pos, reason_stop))
                created = True
                logger.info(
                    "Built reason stop (%s) for stop %d value=%s -> %d",
                    transform_name,
                    list_pos,
                    source_stop.extracted_value,
                    transformed_value,
                )
                break

            if not created:
                logger.debug(
                    "No valid transform for stop at position %d", list_pos,
                )

        if not insertions:
            logger.warning(
                "Could not create any reason stops for trail %s",
                trail.trail_id,
            )

        return insertions

    # ------------------------------------------------------------------
    # Internal: re-indexing, metadata, integrity, re-verbalization
    # ------------------------------------------------------------------

    @staticmethod
    def _reindex_and_rebuild(
        content_stops: list[Stop],
        rng: random.Random,
        add_compute: bool = True,
    ) -> list[Stop]:
        """Re-index stops, fix ``reason_source_stop`` references, and
        rebuild the compute stop.

        Parameters
        ----------
        content_stops:
            All stops *except* compute (which will be rebuilt).
        rng:
            Random instance for compute stop expression selection.
        add_compute:
            Whether to append a new compute stop.

        Returns
        -------
        list[Stop]
            The final list of stops including the rebuilt compute stop.
        """
        # Build old-index -> new-index mapping
        old_to_new: dict[int, int] = {}

        for new_idx, stop in enumerate(content_stops):
            if stop.index >= 0 and stop.stop_type != "reason":
                old_to_new[stop.index] = new_idx
            stop.index = new_idx

        # Fix reason_source_stop references
        for stop in content_stops:
            if stop.stop_type == "reason":
                if stop.reason_source_stop is not None:
                    if stop.reason_source_stop in old_to_new:
                        stop.reason_source_stop = old_to_new[
                            stop.reason_source_stop
                        ]
                    else:
                        # Fallback: point to the stop immediately before
                        stop.reason_source_stop = max(0, stop.index - 1)

        if not add_compute:
            return content_stops

        # Build value dicts for compute stop
        numeric_values: dict[int, float] = {}
        all_values: dict[int, tuple[Any, str]] = {}

        # Track which stops have reason transforms applied to them
        reason_source_indices: set[int] = set()
        for stop in content_stops:
            if stop.stop_type == "reason" and stop.reason_source_stop is not None:
                reason_source_indices.add(stop.reason_source_stop)

        for stop in content_stops:
            if stop.stop_type in ("page", "tool", "reason"):
                val = stop.extracted_value
                vtype = stop.extracted_value_type or ""
                all_values[stop.index] = (val, vtype)
                num = _value_to_number(val, vtype)
                if num is not None:
                    numeric_values[stop.index] = float(num)

        # Remove source stops whose values are superseded by reason transforms
        for src_idx in reason_source_indices:
            numeric_values.pop(src_idx, None)

        compute_stop = _build_compute_stop(
            len(content_stops), numeric_values, all_values, rng,
        )
        if compute_stop is None:
            logger.warning(
                "Failed to rebuild compute stop; keeping stops without compute",
            )
            return content_stops

        content_stops.append(compute_stop)
        return content_stops

    @staticmethod
    def _update_trail_metadata(trail: Trail) -> None:
        """Update trail difficulty counts, depth, and passcode."""
        actual_tool_count = sum(
            1 for s in trail.stops if s.stop_type == "tool"
        )
        actual_reason_count = sum(
            1 for s in trail.stops if s.stop_type == "reason"
        )

        if trail.difficulty:
            trail.difficulty.depth = len(trail.stops)
            trail.difficulty.tool_stop_count = actual_tool_count
            trail.difficulty.reason_stop_count = actual_reason_count

        # Update passcode from the compute stop
        if trail.stops and trail.stops[-1].stop_type == "compute":
            passcode_val = trail.stops[-1].extracted_value
            if isinstance(passcode_val, (int, float)):
                trail.passcode = int(passcode_val) % 10

    async def _reverbalize(self, trail: Trail) -> str | None:
        """Re-generate riddle.  Returns new riddle text or ``None``.

        Tries :meth:`TrailVerbalizer.verbalize_with_validation` first; falls
        back to plain :meth:`~TrailVerbalizer.verbalize` if validation fails.
        Returns ``None`` only when both approaches fail.
        """
        if self._verbalizer is None:
            return None

        new_riddle = await self._verbalizer.verbalize_with_validation(trail)
        if new_riddle:
            logger.info("Riddle regenerated for trail %s", trail.trail_id)
            return new_riddle

        # Fallback: unvalidated verbalization
        new_riddle = await self._verbalizer.verbalize(trail)
        if new_riddle:
            logger.warning(
                "Used unvalidated riddle for trail %s (validation failed)",
                trail.trail_id,
            )
            return new_riddle

        logger.error(
            "Failed to re-verbalize trail %s", trail.trail_id,
        )
        return None
