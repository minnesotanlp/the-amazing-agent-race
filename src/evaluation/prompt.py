"""Build agent prompts from single-trail puzzle JSON files."""

from __future__ import annotations

import math
from typing import Any, Sequence


INTRO = (
    "This is a scavenger-hunt puzzle. Start from the seed URL, follow the"
    " riddle clues in order, and use the available tools to gather information"
    " and compute the answer. The final answer is a single digit (0-9)."
)

ANSWER_FORMAT = (
    "When you are done, output your answer on its own line in exactly this"
    " format:\n\n"
    "Final Answer: X\n\n"
    "where X is a single digit (0-9). Do not add commentary after the answer."
)


def build_tool_lines(tool_specs: dict[str, Any]) -> list[str]:
    """Build human-readable tool descriptions from ToolRegistry specs."""
    lines: list[str] = []
    for name in sorted(tool_specs):
        spec = tool_specs[name]
        desc = ""
        if hasattr(spec, "description"):
            desc = spec.description or ""
        elif isinstance(spec, dict):
            desc = spec.get("description", "")
        avail = ""
        if hasattr(spec, "availability") and not spec.availability.is_available:
            avail = f" (unavailable: {spec.availability.reason})"
        display = f": {desc}" if desc else ""
        lines.append(f"  - {name}{display}{avail}")
    return lines


def build_puzzle_prompt(
    trail_data: dict[str, Any],
    tool_lines: Sequence[str],
    *,
    sample_id: str = "",
) -> dict[str, Any]:
    """Build an agent prompt from a single-trail JSON dict.

    Returns dict with:
        prompt: str         - the full prompt text
        expected_passcode: int  - the expected single-digit answer
        metadata: dict      - scoring/diagnostic info
    """
    seed_url = trail_data.get("seed_url", "")
    riddle = (trail_data.get("riddle") or "").strip()
    passcode = trail_data.get("passcode", -1)
    difficulty = trail_data.get("difficulty", {})
    stops = trail_data.get("stops", [])

    # Compute step limit: ~1.5x the number of stops (min 10)
    step_limit = max(10, math.ceil(len(stops) * 3))

    sections: list[str] = []

    def add(header: str, body: str) -> None:
        sections.append(f"# {header}\n\n{body}")

    header = f"Puzzle {sample_id}" if sample_id else "Puzzle"
    add(header, INTRO)
    add("Seed URL", seed_url or "(no seed URL provided)")
    add("Riddle", riddle or "(no riddle provided)")

    if tool_lines:
        add("Available Tools", "\n".join(tool_lines))

    add("Instructions", ANSWER_FORMAT)

    prompt = "\n\n".join(sections)

    metadata = {
        "seed_url": seed_url,
        "sample_id": sample_id,
        "step_limit": step_limit,
        "difficulty": difficulty.get("level", "unknown") if isinstance(difficulty, dict) else str(difficulty),
        "num_stops": len(stops),
        "num_tool_stops": sum(1 for s in stops if s.get("stop_type") == "tool"),
        "num_reason_stops": sum(1 for s in stops if s.get("stop_type") == "reason"),
    }

    return {
        "prompt": prompt,
        "expected_passcode": passcode,
        "metadata": metadata,
    }
