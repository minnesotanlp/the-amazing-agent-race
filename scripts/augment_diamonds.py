#!/usr/bin/env python3
"""Add diamond (fork-merge) patterns to trail puzzles.

Reads existing v4 trails and augments them with diamond compositionality
patterns, saving to a new output directory (default: data/trail_puzzles_v5).

Usage:
    uv run python scripts/augment_diamonds.py data/trail_puzzles_v4
    uv run python scripts/augment_diamonds.py data/trail_puzzles_v4 --output-dir data/trail_puzzles_v5
    uv run python scripts/augment_diamonds.py data/trail_puzzles_v4 --difficulty easy --dry-run
    uv run python scripts/augment_diamonds.py data/trail_puzzles_v4 --max-diamonds 3
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
from trail.diamond_augmenter import DiamondAugmenter
from trail.models import DIFFICULTY_CONFIGS, trail_from_json, trail_to_json
from trail.verbalizer import TrailVerbalizer
from trail.wiki_graph import WikiGraph

logger = logging.getLogger(__name__)


def _find_trail_files(data_dir: Path, difficulty: str | None = None) -> list[Path]:
    """Find all trail JSON files in the data directory."""
    files: list[Path] = []
    if difficulty:
        diff_dir = data_dir / difficulty
        if diff_dir.is_dir():
            files.extend(sorted(diff_dir.glob("*.json")))
    else:
        for diff_dir in sorted(data_dir.iterdir()):
            if diff_dir.is_dir() and diff_dir.name in DIFFICULTY_CONFIGS:
                files.extend(sorted(diff_dir.glob("*.json")))
    return files


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add diamond (fork-merge) patterns to trail puzzles.",
    )
    parser.add_argument(
        "data_dir", type=Path,
        help="Source data directory (e.g., data/trail_puzzles_v4)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: data/trail_puzzles_v5)",
    )
    parser.add_argument(
        "--difficulty", choices=list(DIFFICULTY_CONFIGS.keys()),
        help="Only process trails of this difficulty level",
    )
    parser.add_argument(
        "--max-diamonds", type=int, default=None,
        help="Override max number of diamonds per trail",
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

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        logger.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    output_dir = args.output_dir or (_PROJECT_ROOT / "data" / "trail_puzzles_v5")

    rng = random.Random(args.seed)

    # Initialize tool registry
    logger.info("Initializing ToolRegistry...")
    tool_registry = ToolRegistry()
    available = tool_registry.available_tools()
    logger.info("Loaded %d tools", len(available))

    # LLM client
    llm = None
    if not args.dry_run and not args.skip_reverbalize:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        llm = OpenAI()

    # Find files
    files = _find_trail_files(data_dir, args.difficulty)
    if not files:
        logger.warning("No trail JSON files found in %s", data_dir)
        sys.exit(0)

    files_by_dir: dict[Path, list[Path]] = defaultdict(list)
    for f in files:
        files_by_dir[f.parent].append(f)

    logger.info("Found %d trail file(s)", len(files))

    async def _dummy_fetch(url: str) -> str:
        return ""

    totals = {"diamonds_added": 0, "files_modified": 0, "files_failed": 0, "files_skipped": 0}

    for diff_dir, dir_files in sorted(files_by_dir.items()):
        difficulty_name = diff_dir.name
        out_diff_dir = output_dir / difficulty_name
        out_diff_dir.mkdir(parents=True, exist_ok=True)

        # Create verbalizer
        verbalizer: TrailVerbalizer | None = None
        if llm is not None:
            cache_dir = diff_dir / ".wiki_cache"
            wiki_graph: WikiGraph | None = None
            if cache_dir.exists():
                wiki_graph = WikiGraph(cache_dir=cache_dir, fetch_fn=_dummy_fetch)
                logger.info(
                    "Loaded wiki_graph for %s (%d cached pages)",
                    difficulty_name, len(wiki_graph._pages),
                )
            verbalizer = TrailVerbalizer(
                llm_client=llm, model=args.model, wiki_graph=wiki_graph,
            )

        augmenter = DiamondAugmenter(
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

            original_count = len(trail.stops)
            logger.info(
                "Processing %s (trail=%s, stops=%d)",
                filepath.name, trail.trail_id, original_count,
            )

            if args.dry_run:
                eligible = augmenter._find_eligible_sources(trail)
                logger.info(
                    "[DRY RUN] %s: %d eligible source stops for diamonds",
                    filepath.name, len(eligible),
                )
                if eligible:
                    totals["files_modified"] += 1
                continue

            result = await augmenter.augment(
                trail,
                num_diamonds=args.max_diamonds,
                skip_reverbalize=args.skip_reverbalize,
            )

            if result is None:
                logger.warning("Diamond augmentation failed for %s — skipping", filepath.name)
                totals["files_failed"] += 1
                continue

            # Save to output directory
            out_path = out_diff_dir / filepath.name
            out_path.write_text(trail_to_json(result), encoding="utf-8")

            new_stops = len(result.stops) - original_count
            diamonds = new_stops // 3  # each diamond adds 3 stops

            totals["diamonds_added"] += diamonds
            totals["files_modified"] += 1

            logger.info(
                "Saved %s: %d -> %d stops (+%d diamonds), passcode=%d",
                filepath.name, original_count, len(result.stops),
                diamonds, result.passcode,
            )

    prefix = "[DRY RUN] " if args.dry_run else ""
    logger.info(
        "%sDone. Processed %d files: %d modified, %d failed, %d skipped, "
        "+%d total diamonds",
        prefix, len(files), totals["files_modified"], totals["files_failed"],
        totals["files_skipped"], totals["diamonds_added"],
    )


if __name__ == "__main__":
    asyncio.run(main())
