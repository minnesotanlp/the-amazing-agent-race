"""Augment existing trails with diamond (fork-merge) patterns.

A diamond consists of:
  - A source page stop (extracts a location / country)
  - Two parallel branch tool stops (both depend on the source, NOT on each other)
  - A merge tool stop (combines both branch values via python_execute_code)

The agent-facing clue sequence remains linear. The diamond exists only in the
data-dependency DAG tracked via ``Stop.depends_on``.
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass
from typing import Any

from trail.augmenter import (
    _country_from_stop,
    _location_from_stop,
)
from trail.builder import (
    _build_compute_stop,
    _build_country_area_chain,
    _build_country_population_chain,
    _build_geocode_elevation_chain,
    _build_geocode_weather_chain,
    _build_geocode_weather_precipitation_chain,
    _build_nearby_poi_count_chain,
    _build_place_rating_chain,
    _is_geocodable,
    _POI_TYPES,
    _value_to_number,
)
from trail.golden import _execute_tool_chain
from trail.models import Bridge, Stop, Trail, TrailDifficultyConfig
from trail.verbalizer import TrailVerbalizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Diamond type definitions
# ---------------------------------------------------------------------------

MERGE_OPS = ["add", "abs_diff"]


@dataclass
class _DiamondSpec:
    """Specification for one diamond type."""

    name: str
    source_type: str  # "location", "country"
    branch_a_desc: str
    branch_b_desc: str

    def build_chains(
        self, source_stop: Stop, rng: random.Random,
    ) -> tuple[tuple[str, list[dict]], tuple[str, list[dict]]] | None:
        """Return ((desc_a, chain_a), (desc_b, chain_b)) or None."""
        raise NotImplementedError


class _LocationElevationPoi(_DiamondSpec):
    def __init__(self):
        super().__init__("elevation_poi", "location", "elevation", "nearby_poi")

    def build_chains(self, source_stop, rng):
        loc = _location_from_stop(source_stop)
        if not loc or not _is_geocodable(loc):
            return None
        poi_type = rng.choice(_POI_TYPES)
        return (
            (f"elevation of {loc}", _build_geocode_elevation_chain(loc)),
            (f"count of {poi_type} near {loc}", _build_nearby_poi_count_chain(loc, poi_type)),
        )


class _LocationElevationRating(_DiamondSpec):
    def __init__(self):
        super().__init__("elevation_rating", "location", "elevation", "place_rating")

    def build_chains(self, source_stop, rng):
        loc = _location_from_stop(source_stop)
        if not loc or not _is_geocodable(loc):
            return None
        return (
            (f"elevation of {loc}", _build_geocode_elevation_chain(loc)),
            (f"Google rating of {loc}", _build_place_rating_chain(loc)),
        )


class _CountryPopulationArea(_DiamondSpec):
    def __init__(self):
        super().__init__("country_pop_area", "country", "population", "area")

    def build_chains(self, source_stop, rng):
        country = _country_from_stop(source_stop)
        if not country:
            return None
        return (
            (f"population of {country}", _build_country_population_chain(country)),
            (f"area of {country} in km\u00b2", _build_country_area_chain(country)),
        )


class _LocationWeatherPrecip(_DiamondSpec):
    def __init__(self):
        super().__init__("weather_precip", "location", "temperature", "precipitation")

    def build_chains(self, source_stop, rng):
        loc = _location_from_stop(source_stop)
        if not loc or not _is_geocodable(loc):
            return None
        date = "2024-01-15"
        return (
            (f"temperature in {loc} on {date}", _build_geocode_weather_chain(loc, date)),
            (f"precipitation in {loc} on {date}", _build_geocode_weather_precipitation_chain(loc, date)),
        )


DIAMOND_SPECS: list[_DiamondSpec] = [
    _LocationElevationPoi(),
    _LocationElevationRating(),
    _CountryPopulationArea(),
    _LocationWeatherPrecip(),
]

DIAMONDS_BY_DIFFICULTY: dict[str, tuple[int, int]] = {
    "easy": (1, 1),
    "medium": (1, 2),
    "hard": (2, 3),
    "extreme": (3, 5),
}


# ---------------------------------------------------------------------------
# DiamondAugmenter
# ---------------------------------------------------------------------------


class DiamondAugmenter:
    """Augments trails by inserting diamond (fork-merge) patterns."""

    def __init__(
        self,
        tool_registry,
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
        num_diamonds: int | None = None,
        skip_reverbalize: bool = False,
    ) -> Trail | None:
        """Add diamond patterns to an existing trail.

        Works on a deep copy. Returns the augmented trail or ``None`` on failure.
        """
        trail = copy.deepcopy(trail)

        difficulty = trail.difficulty.level if trail.difficulty else "easy"
        if num_diamonds is None:
            lo, hi = DIAMONDS_BY_DIFFICULTY.get(difficulty, (1, 2))
            num_diamonds = self._rng.randint(lo, hi)

        # Find eligible source stops
        eligible = self._find_eligible_sources(trail)
        if not eligible:
            logger.warning("Trail %s: no eligible source stops for diamonds", trail.trail_id)
            return None

        num_diamonds = min(num_diamonds, len(eligible))
        self._rng.shuffle(eligible)

        # Build diamonds
        # Each insertion is (insert_after_index, [branch_a, branch_b, merge])
        insertions: list[tuple[int, list[Stop]]] = []
        used_source_indices: set[int] = set()

        for source_idx in eligible:
            if len(insertions) >= num_diamonds:
                break

            source_stop = trail.stops[source_idx]

            # Skip if already used
            if source_idx in used_source_indices:
                continue

            diamond_stops = await self._try_build_diamond(source_stop)
            if diamond_stops is None:
                continue

            insertions.append((source_idx, diamond_stops))
            used_source_indices.add(source_idx)

        if not insertions:
            logger.warning("Trail %s: could not build any diamonds", trail.trail_id)
            return None

        # Insert diamond stops into the trail
        has_compute = trail.stops and trail.stops[-1].stop_type == "compute"
        content_stops = [s for s in trail.stops if s.stop_type != "compute"]

        # Sort descending so later inserts don't shift earlier indices
        insertions.sort(key=lambda x: x[0], reverse=True)
        for insert_after, diamond_stops in insertions:
            for i, stop in enumerate(diamond_stops):
                content_stops.insert(insert_after + 1 + i, stop)

        # Reindex with depends_on support
        trail.stops = self._reindex_and_rebuild_with_deps(
            content_stops, self._rng, has_compute,
        )

        # Update metadata
        self._update_trail_metadata(trail)

        # Re-verbalize
        if not skip_reverbalize:
            if self._verbalizer is None:
                logger.warning(
                    "Trail %s: no verbalizer provided", trail.trail_id,
                )
                return None

            new_riddle = await self._reverbalize(trail)
            if new_riddle is None:
                logger.error(
                    "Trail %s: re-verbalization failed", trail.trail_id,
                )
                return None
            trail.riddle = new_riddle

        # Validate
        from trail.validator import validate_trail as _validate_trail
        validation = _validate_trail(trail)
        if not validation.is_valid:
            logger.error(
                "Trail %s: validation failed after diamond augmentation: %s",
                trail.trail_id,
                "; ".join(str(i) for i in validation.critical_issues),
            )
            return None

        logger.info(
            "Trail %s: added %d diamond(s), passcode=%d",
            trail.trail_id, len(insertions), trail.passcode,
        )
        return trail

    # ------------------------------------------------------------------
    # Internal: find eligible source stops
    # ------------------------------------------------------------------

    def _find_eligible_sources(self, trail: Trail) -> list[int]:
        """Find page stop indices that could serve as diamond sources."""
        eligible = []
        # Track existing tool types per source to avoid duplicates
        existing_tools: set[int] = set()
        for i, stop in enumerate(trail.stops):
            if stop.stop_type == "tool":
                existing_tools.add(i)

        for i, stop in enumerate(trail.stops):
            if stop.stop_type != "page":
                continue
            # Skip the last few stops (need room for insertions before compute)
            if i >= len(trail.stops) - 2:
                continue

            loc = _location_from_stop(stop)
            country = _country_from_stop(stop)
            if loc and _is_geocodable(loc):
                eligible.append(i)
            elif country:
                eligible.append(i)

        return eligible

    # ------------------------------------------------------------------
    # Internal: build a single diamond
    # ------------------------------------------------------------------

    async def _try_build_diamond(
        self, source_stop: Stop,
    ) -> list[Stop] | None:
        """Try to build a diamond (branch_a, branch_b, merge) from a source stop.

        Returns [branch_a, branch_b, merge] or None.
        """
        specs = list(DIAMOND_SPECS)
        self._rng.shuffle(specs)

        for spec in specs:
            chains = spec.build_chains(source_stop, self._rng)
            if chains is None:
                continue

            (desc_a, chain_a), (desc_b, chain_b) = chains

            # Execute both branches
            val_a = await self._execute_and_validate(chain_a, desc_a, source_stop)
            if val_a is None:
                continue

            val_b = await self._execute_and_validate(chain_b, desc_b, source_stop)
            if val_b is None:
                continue

            # Build merge stop
            merge_stop = self._build_merge_stop(val_a, val_b, desc_a, desc_b, source_stop)
            if merge_stop is None:
                continue

            # Build branch stops
            branch_a = self._build_branch_stop(
                chain_a, val_a, desc_a, source_stop,
            )
            branch_b = self._build_branch_stop(
                chain_b, val_b, desc_b, source_stop,
            )

            # Set depends_on markers (will be remapped during reindex)
            # Use negative sentinel values that encode the relative position
            # -1 = "depends on source", will be resolved during reindexing
            branch_a._diamond_source_idx = source_stop.index
            branch_b._diamond_source_idx = source_stop.index
            merge_stop._diamond_branch_a_sentinel = True
            merge_stop._diamond_branch_b_sentinel = True

            return [branch_a, branch_b, merge_stop]

        return None

    async def _execute_and_validate(
        self,
        chain: list[dict[str, Any]],
        desc: str,
        source_stop: Stop,
    ) -> float | None:
        """Execute a tool chain and return numeric value, or None on failure."""
        try:
            value, records = await _execute_tool_chain(chain, self._tool_registry)
        except Exception as e:
            logger.debug("Chain %s failed: %s", desc, e)
            return None

        if value is None:
            logger.debug("Chain %s returned None for %s", desc, source_stop.page_url)
            return None

        try:
            numeric = float(value)
        except (ValueError, TypeError):
            logger.debug("Chain %s returned non-numeric: %s", desc, value)
            return None

        if numeric == 0:
            logger.debug("Chain %s returned zero, skipping", desc)
            return None

        return numeric

    def _build_branch_stop(
        self,
        chain: list[dict[str, Any]],
        value: float,
        desc: str,
        source_stop: Stop,
    ) -> Stop:
        """Create a tool stop for one branch of a diamond."""
        return Stop(
            index=-1,
            stop_type="tool",
            page_url=source_stop.page_url,
            extraction_target=f"Look up the {desc}",
            extraction_section=None,
            extracted_value=value,
            extracted_value_type="number",
            bridge=Bridge(bridge_type="tool_call", tool_chain=chain),
            reasoning=f"Diamond branch ({desc}) from {source_stop.page_url}",
        )

    def _build_merge_stop(
        self,
        val_a: float,
        val_b: float,
        desc_a: str,
        desc_b: str,
        source_stop: Stop,
    ) -> Stop | None:
        """Create a merge tool stop that combines two branch values."""
        ops = list(MERGE_OPS)
        self._rng.shuffle(ops)

        for op in ops:
            if op == "add":
                merged = int(val_a) + int(val_b)
                code = f"result = int({val_a}) + int({val_b})\nprint(result)"
                expr = f"({int(val_a)} + {int(val_b)})"
            elif op == "abs_diff":
                merged = abs(int(val_a) - int(val_b))
                code = f"result = abs(int({val_a}) - int({val_b}))\nprint(result)"
                expr = f"abs({int(val_a)} - {int(val_b)})"
            else:
                continue

            if merged == 0:
                continue

            chain = [{
                "tool_name": "python_execute_code",
                "arguments": {"code": code},
                "expected_output": None,
                "output_key": "merged",
            }]

            return Stop(
                index=-1,
                stop_type="tool",
                page_url=source_stop.page_url,
                extraction_target=f"Combine the {desc_a} and the {desc_b}: {expr}",
                extraction_section=None,
                extracted_value=merged,
                extracted_value_type="number",
                bridge=Bridge(bridge_type="tool_call", tool_chain=chain),
                reasoning=f"Diamond merge: {desc_a} {op} {desc_b}",
            )

        return None

    # ------------------------------------------------------------------
    # Internal: reindex with depends_on
    # ------------------------------------------------------------------

    @staticmethod
    def _reindex_and_rebuild_with_deps(
        content_stops: list[Stop],
        rng: random.Random,
        add_compute: bool = True,
    ) -> list[Stop]:
        """Re-index stops, fix references, populate depends_on, rebuild compute."""

        # Build old-index -> new-index mapping
        old_to_new: dict[int, int] = {}

        for new_idx, stop in enumerate(content_stops):
            if stop.index >= 0 and stop.stop_type != "reason":
                old_to_new[stop.index] = new_idx
            stop.index = new_idx

        # Fix reason_source_stop references
        for stop in content_stops:
            if stop.stop_type == "reason" and stop.reason_source_stop is not None:
                if stop.reason_source_stop in old_to_new:
                    stop.reason_source_stop = old_to_new[stop.reason_source_stop]
                else:
                    stop.reason_source_stop = max(0, stop.index - 1)

        # Populate depends_on for ALL stops
        for i, stop in enumerate(content_stops):
            # Diamond branch stops: depend on their source page
            if hasattr(stop, "_diamond_source_idx"):
                source_new = old_to_new.get(stop._diamond_source_idx, max(0, i - 1))
                stop.depends_on = [source_new]
                delattr(stop, "_diamond_source_idx")
            # Diamond merge stops: depend on the two branches before them
            elif hasattr(stop, "_diamond_branch_a_sentinel"):
                # The two branches are at i-2 and i-1
                stop.depends_on = [i - 2, i - 1]
                delattr(stop, "_diamond_branch_a_sentinel")
                delattr(stop, "_diamond_branch_b_sentinel")
            # Reason stops: depend on their source
            elif stop.stop_type == "reason" and stop.reason_source_stop is not None:
                stop.depends_on = [stop.reason_source_stop]
            # First stop: root
            elif i == 0:
                stop.depends_on = []
            # All other stops: depend on predecessor
            elif not stop.depends_on:
                stop.depends_on = [i - 1]

        if not add_compute:
            return content_stops

        # Build value dicts for compute stop
        numeric_values: dict[int, float] = {}
        all_values: dict[int, tuple[Any, str]] = {}

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

        for src_idx in reason_source_indices:
            numeric_values.pop(src_idx, None)

        compute_stop = _build_compute_stop(
            len(content_stops), numeric_values, all_values, rng,
        )
        if compute_stop is None:
            logger.warning("Failed to rebuild compute stop")
            return content_stops

        # Set compute stop depends_on from its referenced_stops
        if compute_stop.bridge and compute_stop.bridge.referenced_stops:
            compute_stop.depends_on = list(compute_stop.bridge.referenced_stops)
        else:
            compute_stop.depends_on = [len(content_stops) - 1]

        content_stops.append(compute_stop)
        return content_stops

    # ------------------------------------------------------------------
    # Internal: metadata, re-verbalization
    # ------------------------------------------------------------------

    @staticmethod
    def _update_trail_metadata(trail: Trail) -> None:
        """Update trail difficulty counts and passcode."""
        if trail.difficulty:
            tool_count = sum(1 for s in trail.stops if s.stop_type == "tool")
            reason_count = sum(1 for s in trail.stops if s.stop_type == "reason")
            trail.difficulty.tool_stop_count = tool_count
            trail.difficulty.reason_stop_count = reason_count
            trail.difficulty.depth = len(trail.stops)

        compute = [s for s in trail.stops if s.stop_type == "compute"]
        if compute:
            try:
                trail.passcode = int(float(compute[-1].extracted_value)) % 10
            except (ValueError, TypeError):
                pass

    async def _reverbalize(self, trail: Trail) -> str | None:
        """Re-generate the riddle using the verbalizer."""
        if self._verbalizer is None:
            return None
        try:
            return await self._verbalizer.verbalize(trail)
        except Exception as e:
            logger.error("Re-verbalization failed: %s", e)
            return None
