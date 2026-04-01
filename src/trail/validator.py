"""Unified trail validation module.

Consolidates all static validation, integrity checking, and numeric comparison
logic into a single module.  All workflows (generation, augmentation, repair,
evaluation) import from here.

Usage::

    from trail.validator import validate_trail, validate_trail_dict
    result = validate_trail(trail)
    if not result.is_valid:
        for issue in result.critical_issues:
            print(issue)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal
from urllib.parse import unquote

from trail.models import (
    Bridge,
    DIFFICULTY_CONFIGS,
    Stop,
    Trail,
    TrailDifficultyConfig,
    trail_from_json,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ValidationIssue:
    """A single validation finding."""

    category: str
    severity: Literal["critical", "warning"]
    message: str
    stop_index: int | None = None

    def __str__(self) -> str:
        loc = f"stop {self.stop_index}" if self.stop_index is not None else "trail"
        return f"[{self.severity}] {self.category} ({loc}): {self.message}"


@dataclass
class ValidationResult:
    """Aggregated result of running all checks on a trail."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if no critical issues were found."""
        return not any(i.severity == "critical" for i in self.issues)

    @property
    def critical_issues(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "critical"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CITATION_PATTERNS = [
    re.compile(r"\[\[\d+\]\]\(#cite_note"),  # [[1]](#cite_note-...)
    re.compile(r"\[\d+\]"),                    # [1], [2], etc.
    re.compile(r"\[note \d+\]"),               # [note 1]
]

STOCK_TOOLS = {"stock_historical_price", "stock_volume"}

ZERO_SUSPICIOUS_TOOLS = {
    "stock_historical_price",
    "stock_volume",
    "crypto_historical_price",
    "crypto_volume",
    "countries_population",
    "countries_area",
    "maps_geocode",
    "maps_elevation",
}

VALID_TOOLS = {
    "fetch_webpage",
    "weather_historical",
    "weather_forecast",
    "python_generate_code",
    "python_execute_code",
    "search_query",
    "maps_geocode",
    "maps_reverse_geocode",
    "maps_search_places",
    "maps_place_details",
    "maps_distance_matrix",
    "maps_elevation",
    "maps_directions",
    "countries_population",
    "countries_area",
    "stock_historical_price",
    "stock_volume",
    "crypto_historical_price",
    "crypto_volume",
}

VALID_STOP_TYPES = {"page", "tool", "reason", "compute"}
VALID_BRIDGE_TYPES = {"link_follow", "search_query", "tool_call", "compute"}
VALID_VALUE_TYPES = {"number", "text", "url", "coords", "date"}

COMMON_WORDS = {
    "the", "and", "for", "that", "this", "with", "from", "have",
    "will", "been", "were", "are", "was", "has", "its", "can",
    "not", "but", "all", "one", "two", "new", "old", "may",
}


# ---------------------------------------------------------------------------
# Utility functions (canonical implementations)
# ---------------------------------------------------------------------------


def digital_root(n: int) -> int:
    """Compute digital root (Luhn mod 9)."""
    n = abs(n)
    if n == 0:
        return 0
    return ((n - 1) % 9) + 1


def is_weekend(date_str: str) -> bool:
    """Check if a date string (YYYY-MM-DD) falls on a weekend."""
    try:
        dt = date.fromisoformat(date_str)
        return dt.weekday() >= 5
    except (ValueError, TypeError):
        return False


def has_citation_artifacts(text: str) -> bool:
    """Check if text contains markdown citation artifacts."""
    if not text:
        return False
    for pattern in CITATION_PATTERNS:
        if pattern.search(text):
            return True
    return False


def values_close(a: Any, b: Any, tolerance: float = 0.01) -> bool:
    """Compare two values with tolerance for numbers.

    For numeric values, uses relative tolerance.  For non-numeric, falls
    back to stripped string equality.  Returns True if both are None.
    """
    if a is None or b is None:
        return a is None and b is None
    try:
        fa, fb = float(a), float(b)
        if fb == 0:
            return fa == 0
        return abs(fa - fb) / max(abs(fb), 1) < tolerance
    except (ValueError, TypeError):
        return str(a).strip() == str(b).strip()


# ---------------------------------------------------------------------------
# Number parsing (canonical, merged from evaluate.py & repair_samples.py)
# ---------------------------------------------------------------------------

_MAGNITUDES = {
    "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
}

_WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1_000_000,
    "billion": 1_000_000_000, "trillion": 1_000_000_000_000,
}


def parse_number(text: str) -> int | float | None:
    """Parse a number from a messy string, preserving int/float type.

    Handles: '$160', '4,369', '0.30 mm', '$40 billion', '92%', 'eleven',
    'over 2800', 'about 90%', '33⅓ rpm', '100–500 acres' (takes first number),
    '3,609 (2023)', '857,000 speakers', '1,300+', '1/12', '3,990 lb (1,810 kg)',
    'c. 80 million', '250 g/m2', '413.79 g·mol−1', 'more than 13,000'.
    """
    if not isinstance(text, str):
        return None

    s = text.strip()

    # Unicode fractions
    fraction_map = {"½": .5, "⅓": 1/3, "⅔": 2/3, "¼": .25, "¾": .75}

    # Strip leading qualifiers
    s = re.sub(
        r"^(about|approximately|around|over|nearly|more than|no more than"
        r"|at least|up to|~|>|<|≈)\s*",
        "", s, flags=re.I,
    )

    # Strip 'c.' / 'ca.' prefix (circa) and narrow no-break space
    s = re.sub(r"^c(?:a)?\.[\s\u2009]*", "", s)
    s = s.replace("\u2009", " ").replace("\u00a0", " ")

    # Strip currency symbols
    s = re.sub(r"^[\$£€¥₹₩]", "", s)

    # Strip trailing plus signs: '1,300+' → '1,300'
    s = re.sub(r"\+\s*$", "", s)

    # Strip parenthetical suffixes: '3,609 (2023)' or '3,990 lb (1,810 kg)'
    paren_match = re.match(r"^([^(]+)\(", s)
    if paren_match:
        s = paren_match.group(1).strip()

    # Strip trailing units and percent (expanded list)
    s_no_unit = re.sub(
        r"\s*(mm|cm|km|m|kg|lb|lbs|rpm|mph|kph|ft|in|oz|acres?|mi|°[CF]?|%"
        r"|speakers?|languages?|minutes?|hours?|days?|years?"
        r"|g/m2|g·mol−1|mol|g|ha|km²|m²|sq\s*mi|sq\s*km)\s*$",
        "", s, flags=re.I,
    )
    if s_no_unit:
        s = s_no_unit

    # Handle unicode fractions (e.g. "33⅓")
    for frac_char, frac_val in fraction_map.items():
        if frac_char in s:
            parts = s.split(frac_char)
            try:
                base = float(parts[0]) if parts[0].strip() else 0
                return base + frac_val
            except ValueError:
                pass

    # Handle simple fractions: '1/12' → evaluate as division
    frac_match = re.match(r"^(\d+)\s*/\s*(\d+)$", s.strip())
    if frac_match:
        num, den = int(frac_match.group(1)), int(frac_match.group(2))
        if den != 0:
            result = num / den
            return int(result) if result == int(result) else result

    # Handle range (take first number): "100–500" or "100-500"
    range_match = re.match(r"^([\d,.]+)\s*[–\-]\s*[\d,.]+", s)
    if range_match:
        s = range_match.group(1)

    # Handle magnitude words: "40 billion", "1.5 million"
    mag_match = re.match(r"^([\d,.]+)\s+(thousand|million|billion|trillion)", s, re.I)
    if mag_match:
        try:
            base = float(mag_match.group(1).replace(",", ""))
            mult = _MAGNITUDES[mag_match.group(2).lower()]
            result = base * mult
            return int(result) if result == int(result) else result
        except (ValueError, KeyError):
            pass

    # Handle comma-separated numbers: "4,369" → 4369
    s = s.replace(",", "")

    # Try direct parse
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        pass

    # Fallback: strip everything after the last digit
    tail_strip = re.match(r"^([\d]+(?:[.,]\d+)*)", s)
    if tail_strip:
        cleaned = tail_strip.group(1).replace(",", "")
        try:
            if "." in cleaned:
                return float(cleaned)
            return int(cleaned)
        except ValueError:
            pass

    # Try word numbers
    word = text.strip().lower()
    if word in _WORD_NUMS:
        return _WORD_NUMS[word]

    return None


def normalize_numeric(value: Any) -> float | None:
    """Extract a number from any value as a float.

    Thin wrapper around :func:`parse_number` that always returns float.
    Accepts int, float, or string input.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    result = parse_number(value)
    return float(result) if result is not None else None


# ---------------------------------------------------------------------------
# Title leak detection
# ---------------------------------------------------------------------------


def _is_specific_title(title: str) -> bool:
    """Check if a title is specific enough to be a meaningful leak."""
    if len(title) < 6:
        return False
    words = set(title.lower().split())
    if words <= COMMON_WORDS:
        return False
    return True


def _title_from_url(page_url: str) -> str:
    """Extract a Wikipedia title from a URL."""
    if "/wiki/" not in page_url:
        return ""
    raw = page_url.split("/wiki/")[-1].replace("_", " ")
    return unquote(raw)


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------


def check_structure(trail: Trail) -> list[ValidationIssue]:
    """Check basic trail structure: stops exist, sequential indices, compute stop."""
    issues: list[ValidationIssue] = []

    if not trail.stops:
        issues.append(ValidationIssue(
            "no_stops", "critical", "trail has no stops",
        ))
        return issues

    # Last stop should be compute
    if trail.stops[-1].stop_type != "compute":
        issues.append(ValidationIssue(
            "no_compute_stop", "critical",
            f"last stop is '{trail.stops[-1].stop_type}', expected 'compute'",
        ))

    # Sequential indices
    for i, stop in enumerate(trail.stops):
        if stop.index != i:
            issues.append(ValidationIssue(
                "index_mismatch", "warning",
                f"stop['index']={stop.index}, expected={i}",
                stop_index=i,
            ))

    # Valid stop types
    for i, stop in enumerate(trail.stops):
        if stop.stop_type not in VALID_STOP_TYPES:
            issues.append(ValidationIssue(
                "invalid_stop_type", "critical",
                f"stop_type='{stop.stop_type}'",
                stop_index=i,
            ))

    return issues


def check_passcode(trail: Trail) -> list[ValidationIssue]:
    """Check passcode validity and consistency with compute stop."""
    issues: list[ValidationIssue] = []

    if not isinstance(trail.passcode, int) or trail.passcode < 0 or trail.passcode > 9:
        issues.append(ValidationIssue(
            "invalid_passcode", "critical",
            f"passcode={trail.passcode}",
        ))
        return issues

    # Recompute from compute stop
    if trail.stops and trail.stops[-1].stop_type == "compute":
        extracted = trail.stops[-1].extracted_value
        if extracted is not None:
            try:
                expected = int(float(extracted)) % 10
                if expected != trail.passcode:
                    issues.append(ValidationIssue(
                        "passcode_mismatch", "critical",
                        f"stored={trail.passcode}, recomputed_from_last_stop={expected}",
                    ))
            except (ValueError, TypeError):
                pass

    return issues


def check_stop_counts(trail: Trail) -> list[ValidationIssue]:
    """Check stop counts against difficulty config ranges (warning-level)."""
    issues: list[ValidationIssue] = []

    level = trail.difficulty.level if trail.difficulty else None
    if level is None or level not in DIFFICULTY_CONFIGS:
        return issues

    config = DIFFICULTY_CONFIGS[level]
    num_stops = len(trail.stops)
    depth_lo, depth_hi = config.depth_range

    tool_count = sum(1 for s in trail.stops if s.stop_type == "tool")
    reason_count = sum(1 for s in trail.stops if s.stop_type == "reason")

    # Allow +1 tolerance for compute stop
    if num_stops < depth_lo or num_stops > depth_hi + 1:
        issues.append(ValidationIssue(
            "stop_count_out_of_range", "warning",
            f"difficulty={level}, stops={num_stops}, expected={depth_lo}-{depth_hi}",
        ))

    tool_lo, tool_hi = config.tool_stops_range
    if tool_count < tool_lo or tool_count > tool_hi:
        issues.append(ValidationIssue(
            "tool_stop_count_out_of_range", "warning",
            f"difficulty={level}, tool_stops={tool_count}, expected={tool_lo}-{tool_hi}",
        ))

    reason_lo, reason_hi = config.reason_stops_range
    if reason_count < reason_lo or reason_count > reason_hi:
        issues.append(ValidationIssue(
            "reason_stop_count_out_of_range", "warning",
            f"difficulty={level}, reason_stops={reason_count}, expected={reason_lo}-{reason_hi}",
        ))

    return issues


def check_riddle(trail: Trail) -> list[ValidationIssue]:
    """Check riddle: existence, length, title leaks, clue count."""
    issues: list[ValidationIssue] = []

    riddle = trail.riddle or ""
    if not riddle or len(riddle) < 50:
        issues.append(ValidationIssue(
            "missing_riddle", "critical",
            f"riddle length={len(riddle)}",
        ))
        return issues

    # Clue count vs stop count
    clue_nums = re.findall(r"^\d+\.", riddle, re.MULTILINE)
    if clue_nums and len(clue_nums) != len(trail.stops):
        issues.append(ValidationIssue(
            "riddle_clue_count_mismatch", "warning",
            f"riddle has {len(clue_nums)} clues but trail has {len(trail.stops)} stops",
        ))

    # Title leaks: seed title
    seed_title = trail.seed_title or ""
    if seed_title and _is_specific_title(seed_title):
        clean_title = re.sub(r"\s*\([^)]+\)\s*$", "", seed_title).strip()
        if len(clean_title) >= 6 and clean_title.lower() in riddle.lower():
            issues.append(ValidationIssue(
                "title_leak_in_riddle", "warning",
                f"seed_title='{seed_title}' found verbatim in riddle",
            ))

    # Title leaks: page titles in stops
    for i, stop in enumerate(trail.stops):
        page_url = stop.page_url or ""
        if page_url:
            title = _title_from_url(page_url)
            if title and _is_specific_title(title):
                clean_title = re.sub(r"\s*\([^)]+\)\s*$", "", title).strip()
                if len(clean_title) >= 6 and clean_title.lower() in riddle.lower():
                    issues.append(ValidationIssue(
                        "title_leak_in_riddle", "warning",
                        f"page title '{title}' found verbatim in riddle",
                        stop_index=i,
                    ))

    return issues


def check_values(trail: Trail) -> list[ValidationIssue]:
    """Check extracted values: null, zero, NaN/Inf, type mismatch, duplicates."""
    issues: list[ValidationIssue] = []
    from collections import Counter

    # Per-stop checks
    for i, stop in enumerate(trail.stops):
        ev = stop.extracted_value
        ev_type = stop.extracted_value_type or ""

        # Null value
        if ev is None:
            issues.append(ValidationIssue(
                "null_extracted_value", "critical",
                f"stop_type={stop.stop_type}, extracted_value is None",
                stop_index=i,
            ))

        # NaN / Inf
        if isinstance(ev, float) and (math.isnan(ev) or math.isinf(ev)):
            issues.append(ValidationIssue(
                "nan_or_inf_value", "critical",
                f"extracted_value={ev}",
                stop_index=i,
            ))

        # Zero tool value (suspicious)
        if stop.stop_type == "tool" and stop.bridge.tool_chain:
            primary_tool = stop.bridge.tool_chain[0].get("tool_name", "")
            if primary_tool in ZERO_SUSPICIOUS_TOOLS and ev is not None and ev == 0:
                issues.append(ValidationIssue(
                    "zero_tool_value", "critical",
                    f"tool={primary_tool}, extracted_value=0 (likely API failure)",
                    stop_index=i,
                ))

        # Value type mismatch
        if ev_type == "number" and ev is not None:
            if not isinstance(ev, (int, float)):
                try:
                    float(ev)
                except (ValueError, TypeError):
                    issues.append(ValidationIssue(
                        "value_type_mismatch", "warning",
                        f"type='number' but value={ev!r} ({type(ev).__name__})",
                        stop_index=i,
                    ))

        # Invalid value type
        if ev_type and ev_type not in VALID_VALUE_TYPES:
            issues.append(ValidationIssue(
                "invalid_value_type", "warning",
                f"extracted_value_type='{ev_type}'",
                stop_index=i,
            ))

    # Duplicate extracted values (only flag > 100 to avoid natural small-int duplication)
    numeric_values: list[tuple[int, Any]] = []
    for i, stop in enumerate(trail.stops):
        ev = stop.extracted_value
        if stop.stop_type == "compute":
            continue
        if isinstance(ev, (int, float)) and ev != 0 and abs(ev) > 100:
            numeric_values.append((i, ev))
    value_counts = Counter(v for _, v in numeric_values)
    for val, count in value_counts.items():
        if count > 1:
            dup_indices = [idx for idx, v in numeric_values if v == val]
            issues.append(ValidationIssue(
                "duplicate_extracted_value", "warning",
                f"value={val} appears {count} times at stops {dup_indices}",
                stop_index=dup_indices[0],
            ))

    return issues


def check_bridges(trail: Trail) -> list[ValidationIssue]:
    """Check bridge type validity and required fields."""
    issues: list[ValidationIssue] = []

    for i, stop in enumerate(trail.stops):
        bt = stop.bridge.bridge_type
        bridge = stop.bridge

        if bt and bt not in VALID_BRIDGE_TYPES:
            issues.append(ValidationIssue(
                "invalid_bridge_type", "critical",
                f"bridge_type='{bt}'",
                stop_index=i,
            ))

        # Required fields per bridge type
        if bt == "link_follow" and not bridge.target_url:
            issues.append(ValidationIssue(
                "missing_bridge_field", "warning",
                "link_follow bridge missing target_url",
                stop_index=i,
            ))
        elif bt == "search_query" and not bridge.search_query:
            issues.append(ValidationIssue(
                "missing_bridge_field", "warning",
                "search_query bridge missing search_query",
                stop_index=i,
            ))
        elif bt == "tool_call" and not bridge.tool_chain:
            issues.append(ValidationIssue(
                "missing_bridge_field", "warning",
                "tool_call bridge missing tool_chain",
                stop_index=i,
            ))
        elif bt == "compute" and stop.stop_type == "compute":
            if not bridge.expression and not bridge.expression_code:
                issues.append(ValidationIssue(
                    "missing_bridge_field", "warning",
                    "compute stop missing expression/expression_code",
                    stop_index=i,
                ))

    return issues


def check_tool_chains(trail: Trail) -> list[ValidationIssue]:
    """Check tool chains: valid names, non-empty on tool stops, weekend dates."""
    issues: list[ValidationIssue] = []

    for i, stop in enumerate(trail.stops):
        tool_chain = stop.bridge.tool_chain

        # Empty tool chain on tool stops
        if stop.stop_type == "tool" and not tool_chain:
            issues.append(ValidationIssue(
                "empty_tool_chain", "critical",
                "stop_type=tool but tool_chain is empty",
                stop_index=i,
            ))

        for step in tool_chain:
            tool_name = step.get("tool_name", "")

            # Invalid tool name
            if tool_name and tool_name not in VALID_TOOLS:
                issues.append(ValidationIssue(
                    "invalid_tool_name", "critical",
                    f"tool_name='{tool_name}' not in registry",
                    stop_index=i,
                ))

            # Stock weekend dates
            if tool_name in STOCK_TOOLS:
                tool_date = step.get("arguments", {}).get("date", "")
                if is_weekend(tool_date):
                    try:
                        day_name = date.fromisoformat(tool_date).strftime("%A")
                    except (ValueError, TypeError):
                        day_name = "?"
                    issues.append(ValidationIssue(
                        "stock_weekend_date", "critical",
                        f"tool={tool_name}, date={tool_date} (weekday={day_name})",
                        stop_index=i,
                    ))

    return issues


def check_citations(trail: Trail) -> list[ValidationIssue]:
    """Check for citation artifacts in tool arguments and extracted values."""
    issues: list[ValidationIssue] = []

    for i, stop in enumerate(trail.stops):
        # In tool arguments
        for step in stop.bridge.tool_chain:
            args = step.get("arguments", {})
            for arg_name, arg_value in args.items():
                if isinstance(arg_value, str) and has_citation_artifacts(arg_value):
                    issues.append(ValidationIssue(
                        "citation_in_tool_arg", "critical",
                        f"tool={step.get('tool_name')}, arg={arg_name}, value={arg_value!r}",
                        stop_index=i,
                    ))

        # In extracted values
        ev = stop.extracted_value
        if isinstance(ev, str) and has_citation_artifacts(ev):
            issues.append(ValidationIssue(
                "citation_in_value", "critical",
                f"extracted_value={ev!r}",
                stop_index=i,
            ))

    return issues


def check_reason_stops(trail: Trail) -> list[ValidationIssue]:
    """Check reason stop source references."""
    issues: list[ValidationIssue] = []

    for i, stop in enumerate(trail.stops):
        if stop.stop_type != "reason":
            continue
        source = stop.reason_source_stop
        if source is not None:
            if not isinstance(source, int) or source < 0 or source >= i:
                issues.append(ValidationIssue(
                    "invalid_reason_source", "warning",
                    f"reason_source_stop={source}, must be 0..{i - 1}",
                    stop_index=i,
                ))

    return issues


def check_depends_on(trail: Trail) -> list[ValidationIssue]:
    """Check that depends_on references are valid (earlier stops only)."""
    issues: list[ValidationIssue] = []

    for i, stop in enumerate(trail.stops):
        for dep in stop.depends_on:
            if not isinstance(dep, int) or dep < 0 or dep >= i:
                issues.append(ValidationIssue(
                    "invalid_depends_on", "critical",
                    f"depends_on contains {dep}, must be 0..{i - 1}",
                    stop_index=i,
                ))

    return issues


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def validate_trail(trail: Trail) -> ValidationResult:
    """Run all static checks on a Trail object.

    Returns a :class:`ValidationResult` with all issues found.
    """
    issues: list[ValidationIssue] = []
    issues.extend(check_structure(trail))
    issues.extend(check_passcode(trail))
    issues.extend(check_stop_counts(trail))
    issues.extend(check_riddle(trail))
    issues.extend(check_values(trail))
    issues.extend(check_bridges(trail))
    issues.extend(check_tool_chains(trail))
    issues.extend(check_citations(trail))
    issues.extend(check_reason_stops(trail))
    issues.extend(check_depends_on(trail))
    return ValidationResult(issues=issues)


def validate_trail_dict(data: dict) -> ValidationResult:
    """Validate from a raw JSON dict.

    Deserializes via :func:`trail_from_json`, then delegates to
    :func:`validate_trail`.
    """
    try:
        trail = trail_from_json(data)
    except Exception as exc:
        return ValidationResult(issues=[
            ValidationIssue("invalid_json", "critical", f"Invalid trail data: {exc}"),
        ])
    return validate_trail(trail)
