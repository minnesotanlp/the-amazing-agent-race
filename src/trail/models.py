"""Data models for the trail generation pipeline.

Defines the core structures: PageInfo (Wikipedia knowledge graph nodes),
Stop/Bridge/Trail (puzzle structure), and difficulty configs.

Simplified from the old models.py: no TrailBundle, no branch fields.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Wikipedia knowledge graph
# ---------------------------------------------------------------------------


@dataclass
class InfoboxField:
    """A single key-value pair from a Wikipedia infobox."""

    key: str
    value: str
    numeric_value: float | None = None


@dataclass
class WikiLink:
    """An outgoing link from a Wikipedia page."""

    text: str
    target_title: str
    target_url: str
    section: str | None = None


@dataclass
class PageInfo:
    """Cached Wikipedia page metadata."""

    url: str
    title: str
    infobox: list[InfoboxField] = field(default_factory=list)
    outgoing_links: list[WikiLink] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)
    first_paragraph: str = ""
    page_length: int = 0
    pageviews: int | None = None
    fetch_timestamp: str = ""
    raw_markdown: str = ""


# ---------------------------------------------------------------------------
# Trail structures
# ---------------------------------------------------------------------------

BridgeType = Literal["link_follow", "search_query", "tool_call", "compute"]
StopType = Literal["page", "tool", "reason", "compute"]
ValueType = Literal["number", "text", "url", "coords", "date"]
DifficultyLevel = Literal["easy", "medium", "hard", "extreme"]
ExtractionDifficulty = Literal["infobox", "prose", "cross_section"]
BridgeObscurity = Literal["direct_link", "search", "multi_hop"]


@dataclass
class Bridge:
    """How one stop connects to the next."""

    bridge_type: BridgeType
    target_url: str | None = None
    search_query: str | None = None
    expected_result_url: str | None = None
    tool_chain: list[dict[str, Any]] = field(default_factory=list)
    expression: str | None = None
    expression_code: str | None = None
    referenced_stops: list[int] = field(default_factory=list)


@dataclass
class Stop:
    """A single step in a trail."""

    index: int
    stop_type: StopType
    page_url: str | None
    extraction_target: str
    extraction_section: str | None
    extracted_value: Any
    extracted_value_type: ValueType
    bridge: Bridge
    reasoning: str = ""
    # Reason stop fields
    reason_transform: str | None = None
    reason_source_stop: int | None = None
    reason_code: str | None = None
    # DAG dependency tracking (for compositional / diamond patterns)
    depends_on: list[int] = field(default_factory=list)


@dataclass
class TrailDifficulty:
    """Difficulty metadata for a trail."""

    level: DifficultyLevel
    depth: int
    tool_stop_count: int
    reason_stop_count: int = 0
    extraction_difficulty: ExtractionDifficulty = "infobox"
    bridge_obscurity: BridgeObscurity = "direct_link"


@dataclass
class Trail:
    """A complete scavenger-hunt trail producing one passcode digit."""

    trail_id: str
    seed_url: str
    seed_title: str
    stops: list[Stop] = field(default_factory=list)
    passcode: int = -1
    difficulty: TrailDifficulty | None = None
    riddle: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Difficulty configuration
# ---------------------------------------------------------------------------


@dataclass
class TrailDifficultyConfig:
    """Configuration for trail generation at a specific difficulty level."""

    level: DifficultyLevel
    depth_range: tuple[int, int]
    tool_stops_range: tuple[int, int]
    reason_stops_range: tuple[int, int] = (0, 0)
    extraction_types: list[str] = field(default_factory=list)
    bridge_types: list[str] = field(default_factory=list)
    crawl_depth: int = 2


DIFFICULTY_CONFIGS: dict[str, TrailDifficultyConfig] = {
    "easy": TrailDifficultyConfig(
        level="easy",
        depth_range=(3, 6),
        tool_stops_range=(1, 2),
        reason_stops_range=(1, 2),
        extraction_types=["infobox", "prose"],
        bridge_types=["link_follow", "search_query", "tool_call"],
        crawl_depth=1,
    ),
    "medium": TrailDifficultyConfig(
        level="medium",
        depth_range=(7, 12),
        tool_stops_range=(2, 4),
        reason_stops_range=(2, 3),
        extraction_types=["infobox", "prose", "cross_section"],
        bridge_types=["link_follow", "search_query", "tool_call"],
        crawl_depth=2,
    ),
    "hard": TrailDifficultyConfig(
        level="hard",
        depth_range=(13, 16),
        tool_stops_range=(4, 5),
        reason_stops_range=(3, 4),
        extraction_types=["infobox", "prose", "cross_section"],
        bridge_types=["link_follow", "search_query", "tool_call"],
        crawl_depth=3,
    ),
    "extreme": TrailDifficultyConfig(
        level="extreme",
        depth_range=(17, 21),
        tool_stops_range=(5, 7),
        reason_stops_range=(4, 6),
        extraction_types=["infobox", "prose", "cross_section"],
        bridge_types=["link_follow", "search_query", "tool_call"],
        crawl_depth=3,
    ),
}


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_value(obj: Any) -> Any:
    """Convert dataclass instances to dicts recursively for JSON serialization."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _serialize_value(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_serialize_value(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialize_value(v) for k, v in obj.items()}
    return obj


def trail_to_json(trail: Trail) -> str:
    """Serialize a Trail to a JSON string."""
    return json.dumps(_serialize_value(trail), indent=2, ensure_ascii=False)


def trail_from_json(data: str | dict) -> Trail:
    """Deserialize a Trail from a JSON string or dict."""
    if isinstance(data, str):
        data = json.loads(data)
    assert isinstance(data, dict)

    stops = []
    for s in data.get("stops", []):
        bridge_data = s.get("bridge", {})
        bridge = Bridge(
            bridge_type=bridge_data.get("bridge_type", "link_follow"),
            target_url=bridge_data.get("target_url"),
            search_query=bridge_data.get("search_query"),
            expected_result_url=bridge_data.get("expected_result_url"),
            tool_chain=bridge_data.get("tool_chain", []),
            expression=bridge_data.get("expression"),
            expression_code=bridge_data.get("expression_code"),
            referenced_stops=bridge_data.get("referenced_stops", []),
        )
        stops.append(
            Stop(
                index=s.get("index", 0),
                stop_type=s.get("stop_type", "page"),
                page_url=s.get("page_url"),
                extraction_target=s.get("extraction_target", ""),
                extraction_section=s.get("extraction_section"),
                extracted_value=s.get("extracted_value"),
                extracted_value_type=s.get("extracted_value_type", "text"),
                bridge=bridge,
                reasoning=s.get("reasoning", ""),
                reason_transform=s.get("reason_transform"),
                reason_source_stop=s.get("reason_source_stop"),
                reason_code=s.get("reason_code"),
                depends_on=s.get("depends_on", []),
            )
        )

    diff_data = data.get("difficulty")
    difficulty = None
    if diff_data:
        difficulty = TrailDifficulty(
            level=diff_data.get("level", "easy"),
            depth=diff_data.get("depth", 0),
            tool_stop_count=diff_data.get("tool_stop_count", 0),
            reason_stop_count=diff_data.get("reason_stop_count", 0),
            extraction_difficulty=diff_data.get("extraction_difficulty", "infobox"),
            bridge_obscurity=diff_data.get("bridge_obscurity", "direct_link"),
        )

    return Trail(
        trail_id=data.get("trail_id", ""),
        seed_url=data.get("seed_url", ""),
        seed_title=data.get("seed_title", ""),
        stops=stops,
        passcode=data.get("passcode", -1),
        difficulty=difficulty,
        riddle=data.get("riddle", ""),
        metadata=data.get("metadata", {}),
        generated_at=data.get("generated_at", ""),
    )
