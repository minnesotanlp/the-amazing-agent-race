#!/usr/bin/env python3
"""CLI wrapper for trail augmentation.

Adds tool stops and/or reason stops to existing trail puzzle samples,
rebuilds compute stops, and re-verbalizes riddles.

Usage:
    uv run python scripts/augment_stops.py data/trail_puzzles_v4 --add-tool-stops
    uv run python scripts/augment_stops.py data/trail_puzzles_v4 --add-tool-stops --add-reason-stops
    uv run python scripts/augment_stops.py data/trail_puzzles_v4 --add-tool-stops --difficulty easy
    uv run python scripts/augment_stops.py data/trail_puzzles_v4 --add-reason-stops --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project src is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from mcp_servers.registry import ToolRegistry
from trail.augmenter import TrailAugmenter
from trail.models import DIFFICULTY_CONFIGS, trail_from_json, trail_to_json
from trail.verbalizer import TrailVerbalizer
from trail.wiki_graph import WikiGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def _find_trail_files(
    data_dir: Path,
    difficulty: str | None = None,
) -> list[Path]:
    """Find all trail JSON files in the data directory."""
    files: list[Path] = []
    if difficulty:
        diff_dir = data_dir / difficulty
        if diff_dir.is_dir():
            files.extend(sorted(diff_dir.glob("*.json")))
        else:
            logger.warning("Difficulty directory not found: %s", diff_dir)
    else:
        for diff_dir in sorted(data_dir.iterdir()):
            if diff_dir.is_dir() and diff_dir.name in DIFFICULTY_CONFIGS:
                files.extend(sorted(diff_dir.glob("*.json")))
    return files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment trail puzzles by adding tool/reason stops.",
    )
    parser.add_argument(
        "data_dir", type=Path,
        help="Root data directory (e.g., data/trail_puzzles_v4)",
    )
    parser.add_argument(
        "--add-tool-stops", action="store_true",
        help="Add tool stops to samples below difficulty minimum",
    )
    parser.add_argument(
        "--add-reason-stops", action="store_true",
        help="Add reason stops to samples below difficulty minimum",
    )
    parser.add_argument(
        "--difficulty", choices=list(DIFFICULTY_CONFIGS.keys()),
        help="Only process trails of this difficulty level",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be changed without writing files",
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="LLM model for re-verbalization (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--skip-reverbalize", action="store_true",
        help="Skip riddle re-generation (WARNING: inconsistent riddles)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if not args.add_tool_stops and not args.add_reason_stops:
        parser.error("At least one of --add-tool-stops or --add-reason-stops required")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        logger.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    rng = random.Random(args.seed)

    # Initialize tool registry
    logger.info("Initializing ToolRegistry...")
    tool_registry = ToolRegistry()
    available = tool_registry.available_tools()
    logger.info("Loaded %d tools", len(available))

    # LLM client (only needed for non-dry-run, non-skip-reverbalize)
    llm = None
    if not args.dry_run and not args.skip_reverbalize:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        llm = OpenAI()

    # Find files grouped by difficulty directory
    files = _find_trail_files(data_dir, args.difficulty)
    if not files:
        logger.warning("No trail JSON files found in %s", data_dir)
        sys.exit(0)

    files_by_dir: dict[Path, list[Path]] = defaultdict(list)
    for f in files:
        files_by_dir[f.parent].append(f)

    logger.info("Found %d trail file(s)", len(files))

    # Dummy fetch for wiki_graph (only need cached pages)
    async def _dummy_fetch(url: str) -> str:
        return ""

    # Process each difficulty directory
    totals = {"tool_added": 0, "reason_added": 0, "files_modified": 0, "files_failed": 0}

    for diff_dir, dir_files in sorted(files_by_dir.items()):
        # Create verbalizer with wiki_graph for this difficulty
        verbalizer: TrailVerbalizer | None = None
        if llm is not None:
            cache_dir = diff_dir / ".wiki_cache"
            wiki_graph: WikiGraph | None = None
            if cache_dir.exists():
                wiki_graph = WikiGraph(cache_dir=cache_dir, fetch_fn=_dummy_fetch)
                logger.info(
                    "Loaded wiki_graph for %s (%d cached pages)",
                    diff_dir.name, len(wiki_graph._pages),
                )
            else:
                logger.warning("No wiki_cache for %s", diff_dir.name)
            verbalizer = TrailVerbalizer(
                llm_client=llm, model=args.model, wiki_graph=wiki_graph,
            )

        augmenter = TrailAugmenter(
            tool_registry=tool_registry,
            verbalizer=verbalizer,
            rng=rng,
        )

        for filepath in dir_files:
            try:
                raw = filepath.read_text(encoding="utf-8")
                trail = trail_from_json(raw)
            except Exception as exc:
                logger.error("Failed to load %s: %s", filepath, exc)
                continue

            logger.info(
                "Processing %s (trail=%s, stops=%d)",
                filepath.name, trail.trail_id, len(trail.stops),
            )

            # Compute what's needed
            needed = augmenter.compute_needed(trail)
            tool_needed = needed["tool_stops_needed"] if args.add_tool_stops else 0
            reason_needed = needed["reason_stops_needed"] if args.add_reason_stops else 0

            if tool_needed <= 0 and reason_needed <= 0:
                logger.info("No changes needed for %s", filepath.name)
                continue

            if args.dry_run:
                logger.info(
                    "[DRY RUN] %s would add +%d tool, +%d reason stops",
                    filepath.name, tool_needed, reason_needed,
                )
                totals["tool_added"] += tool_needed
                totals["reason_added"] += reason_needed
                totals["files_modified"] += 1
                continue

            # Run augmentation
            original_count = len(trail.stops)
            result = await augmenter.augment(
                trail,
                add_tool_stops=tool_needed,
                add_reason_stops=reason_needed,
                skip_reverbalize=args.skip_reverbalize,
            )

            if result is None:
                logger.error("Augmentation failed for %s — skipping", filepath.name)
                totals["files_failed"] += 1
                continue

            if result is trail:
                logger.info("No changes for %s", filepath.name)
                continue

            # Save
            filepath.write_text(trail_to_json(result), encoding="utf-8")

            added_tool = sum(1 for s in result.stops if s.stop_type == "tool") - \
                         sum(1 for s in trail.stops if s.stop_type == "tool")
            added_reason = sum(1 for s in result.stops if s.stop_type == "reason") - \
                           sum(1 for s in trail.stops if s.stop_type == "reason")
            # Since we deepcopy, count by comparing stop counts
            new_stops = len(result.stops) - original_count

            totals["tool_added"] += max(added_tool, 0)
            totals["reason_added"] += max(added_reason, 0)
            totals["files_modified"] += 1

            logger.info(
                "Saved %s: %d -> %d stops, passcode=%d [integrity OK]",
                filepath.name, original_count, len(result.stops), result.passcode,
            )

    # Summary
    prefix = "[DRY RUN] " if args.dry_run else ""
    logger.info(
        "%sDone. Processed %d files: %d modified, %d failed, "
        "+%d tool stops, +%d reason stops",
        prefix, len(files), totals["files_modified"], totals["files_failed"],
        totals["tool_added"], totals["reason_added"],
    )


if __name__ == "__main__":
    asyncio.run(main())
