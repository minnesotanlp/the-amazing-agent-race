"""Fact extraction engine for trail stops.

Two-tier extraction:
  Tier 1 — Deterministic: Parse structured infobox fields directly from PageInfo.
  Tier 2 — LLM-assisted: Use an LLM to identify extractable facts from prose text.

Each extracted fact is validated for unambiguity via multi-trial re-extraction.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from trail.models import PageInfo, ValueType

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ExtractableFact:
    """A fact that can be extracted from a page as a trail stop."""

    page_url: str
    description: str  # NL extraction target, e.g. "the elevation in the Geography section"
    section: str | None  # Section name where the fact lives
    value: Any  # The actual value
    value_type: ValueType
    confidence: float  # 0.0-1.0, how unambiguous this extraction is
    outgoing_links: list[str] = field(default_factory=list)  # Wikipedia URLs reachable from this fact
    source_tier: Literal["infobox", "prose"] = "infobox"


# ---------------------------------------------------------------------------
# Tier 1: Deterministic infobox extraction
# ---------------------------------------------------------------------------

# Map of infobox key patterns to (value_type, description_template)
_INFOBOX_PATTERNS: list[tuple[re.Pattern, ValueType, str]] = [
    (re.compile(r"elevation", re.I), "number", "the elevation listed in the infobox"),
    (re.compile(r"population", re.I), "number", "the population listed in the infobox"),
    (re.compile(r"area", re.I), "number", "the area listed in the infobox"),
    (re.compile(r"founded|established|inception", re.I), "number", "the founding/establishment year in the infobox"),
    (re.compile(r"coordinates|coord", re.I), "coords", "the coordinates listed in the infobox"),
    (re.compile(r"^capital$", re.I), "text", "the capital listed in the infobox"),
    (re.compile(r"^country$", re.I), "text", "the country listed in the infobox"),
    (re.compile(r"^language|official.lang", re.I), "text", "the official language listed in the infobox"),
    (re.compile(r"height|altitude", re.I), "number", "the height listed in the infobox"),
    (re.compile(r"^length$", re.I), "number", "the length listed in the infobox"),
    (re.compile(r"^capacity|seats$", re.I), "number", "the capacity listed in the infobox"),
    (re.compile(r"^(gdp|GDP)", re.I), "number", "the GDP listed in the infobox"),
    (re.compile(r"^opened$", re.I), "number", "the opening year listed in the infobox"),
    (re.compile(r"^architect$", re.I), "text", "the architect listed in the infobox"),
    (re.compile(r"^location$", re.I), "text", "the location listed in the infobox"),
    (re.compile(r"^(date|start.date|end.date)$", re.I), "date", "the date listed in the infobox"),
]


def _extract_infobox_facts(page: PageInfo) -> list[ExtractableFact]:
    """Extract facts from structured infobox fields (high confidence)."""
    facts: list[ExtractableFact] = []

    for ifield in page.infobox:
        for pattern, value_type, desc_template in _INFOBOX_PATTERNS:
            if not pattern.search(ifield.key):
                continue

            value: Any = ifield.value
            if value_type == "number":
                if ifield.numeric_value is not None:
                    value = ifield.numeric_value
                else:
                    continue  # Skip non-numeric fields that should be numeric
            else:
                # For non-numeric types, skip empty or pipe-only values
                val_str = str(value).strip().strip("|").strip()
                if not val_str:
                    continue

            # Build a specific description with a clean key
            clean_key = (
                ifield.key.split(":")[0].strip()
                if "geohack" in ifield.key
                else ifield.key
            )
            description = f"{desc_template} (field: '{clean_key}')"

            # Gather outgoing links from the value text
            link_urls = _extract_links_from_text(ifield.value, page)

            facts.append(
                ExtractableFact(
                    page_url=page.url,
                    description=description,
                    section="infobox",
                    value=value,
                    value_type=value_type,
                    confidence=1.0,
                    outgoing_links=link_urls,
                    source_tier="infobox",
                )
            )
            break  # Only match first pattern per field

    return facts


def _extract_links_from_text(text: str, page: PageInfo) -> list[str]:
    """Find Wikipedia URLs mentioned in a text value (by matching link texts)."""
    urls = []
    for link in page.outgoing_links:
        if link.text.lower() in text.lower():
            urls.append(link.target_url)
    return urls


# ---------------------------------------------------------------------------
# Tier 2: LLM-assisted prose extraction
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """\
You are a fact extractor for a scavenger-hunt puzzle benchmark.
Given a Wikipedia article, identify specific, unambiguous facts that could be
extracted by someone reading the page.

Rules:
- Each fact must have exactly ONE correct answer.
- Prefer numeric facts (counts, years, measurements).
- Prefer facts from specific sections (Geography, History, Demographics).
- Avoid subjective or interpretive facts.
- Avoid facts that require external knowledge beyond this page.
- For each fact, provide:
  1. "description": A natural-language extraction question (e.g. "the number of host cities listed in the Venues section")
  2. "section": The section name where the fact lives (or null if in the opening paragraph)
  3. "value": The exact value as it appears or can be derived from the text
  4. "value_type": One of "number", "text", "url", "date"

Output a JSON array of objects. Output ONLY the JSON array, no markdown fencing."""

_EXTRACTION_USER_TEMPLATE = """\
Article URL: {url}
Article title: {title}

Article text (first 12000 characters):
{text}"""


async def _llm_extract_facts(
    page: PageInfo,
    llm_client: OpenAI,
    model: str,
    *,
    max_facts: int = 8,
) -> list[ExtractableFact]:
    """Use an LLM to identify extractable facts from page prose."""
    text = page.raw_markdown[:12000]
    if not text.strip():
        return []

    user_msg = _EXTRACTION_USER_TEMPLATE.format(
        url=page.url,
        title=page.title,
        text=text,
    )

    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = (response.choices[0].message.content or "").strip()
    except Exception:
        logger.warning("LLM fact extraction failed for %s", page.url, exc_info=True)
        return []

    # Parse JSON response
    facts = _parse_llm_facts(raw, page, max_facts)
    return facts


def _parse_llm_facts(
    raw: str, page: PageInfo, max_facts: int
) -> list[ExtractableFact]:
    """Parse the LLM's JSON output into ExtractableFact objects."""
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        items = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM extraction output as JSON")
        return []

    if not isinstance(items, list):
        return []

    facts: list[ExtractableFact] = []
    for item in items[:max_facts]:
        if not isinstance(item, dict):
            continue
        description = item.get("description", "")
        section = item.get("section")
        value = item.get("value")
        value_type = item.get("value_type", "text")

        if not description or value is None:
            continue
        if value_type not in ("number", "text", "url", "date", "coords"):
            value_type = "text"

        # Assign confidence based on value type
        if value_type == "number":
            confidence = 0.8
        elif value_type == "date":
            confidence = 0.75
        else:
            confidence = 0.6

        facts.append(
            ExtractableFact(
                page_url=page.url,
                description=str(description),
                section=section,
                value=value,
                value_type=value_type,
                confidence=confidence,
                outgoing_links=_extract_links_from_text(str(value), page),
                source_tier="prose",
            )
        )

    return facts


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_VALIDATION_SYSTEM_PROMPT = """\
You are verifying a fact extraction from a Wikipedia page.
Given the article text and an extraction question, provide the answer.
Output ONLY the answer value — no explanation, no formatting."""


async def _validate_single_extraction(
    page: PageInfo,
    fact: ExtractableFact,
    llm_client: OpenAI,
    model: str,
) -> Any:
    """Run one LLM extraction trial and return the extracted value."""
    text = page.raw_markdown[:12000]
    user_msg = (
        f"Article: {page.title}\n\n"
        f"Text:\n{text}\n\n"
        f"Question: What is {fact.description}?"
    )

    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _VALIDATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = (response.choices[0].message.content or "").strip()
        return _normalize_value(raw, fact.value_type)
    except Exception:
        return None


def _normalize_value(raw: str, value_type: ValueType) -> Any:
    """Normalize an extracted value for comparison."""
    if value_type == "number":
        cleaned = re.sub(r"[^\d.\-+]", "", raw.replace(",", ""))
        try:
            val = float(cleaned)
            return int(val) if val == int(val) else val
        except (ValueError, OverflowError):
            return raw
    return raw.strip()


def _values_match(a: Any, b: Any, value_type: ValueType) -> bool:
    """Compare two extracted values for equality."""
    if a is None or b is None:
        return False
    if value_type == "number":
        from trail.validator import values_close
        return values_close(a, b)
    return str(a).strip().lower() == str(b).strip().lower()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


class FactExtractor:
    """Identifies and validates extractable facts from Wikipedia pages."""

    def __init__(self, llm_client: OpenAI, model: str):
        self._llm = llm_client
        self._model = model

    async def extract_facts(
        self,
        page: PageInfo,
        *,
        max_facts: int = 10,
        include_prose: bool = True,
    ) -> list[ExtractableFact]:
        """Identify candidate extraction targets from a page.

        Tier 1 (infobox) facts are always included.
        Tier 2 (prose) facts are added if include_prose is True.
        Results are sorted by confidence descending.
        """
        facts = _extract_infobox_facts(page)

        if include_prose and len(facts) < max_facts:
            remaining = max_facts - len(facts)
            prose_facts = await _llm_extract_facts(
                page, self._llm, self._model, max_facts=remaining
            )
            facts.extend(prose_facts)

        # Sort by confidence (highest first), then prefer numeric types
        facts.sort(
            key=lambda f: (
                f.confidence,
                1 if f.value_type == "number" else 0,
            ),
            reverse=True,
        )
        return facts[:max_facts]

    async def validate_extraction(
        self,
        page: PageInfo,
        fact: ExtractableFact,
        *,
        num_trials: int = 3,
    ) -> float:
        """Re-extract the fact multiple times. Return consistency score (0.0-1.0).

        For infobox facts (confidence=1.0), skip validation.
        For prose facts, run num_trials LLM extractions and check agreement.
        """
        if fact.source_tier == "infobox" and fact.confidence >= 1.0:
            return 1.0

        matches = 0
        for _ in range(num_trials):
            extracted = await _validate_single_extraction(
                page, fact, self._llm, self._model
            )
            if _values_match(extracted, fact.value, fact.value_type):
                matches += 1

        consistency = matches / num_trials
        return consistency
