#!/usr/bin/env python3
"""Fix title leaks in riddles by calling the LLM to rewrite leaked titles.

Scans all samples, detects Wikipedia title mentions in riddles,
and uses the same LLM correction prompt from the verbalizer to fix them.

Usage:
    uv run python scripts/fix_title_leaks.py data/trail_puzzles_v4
    uv run python scripts/fix_title_leaks.py data/trail_puzzles_v4 --difficulty easy
    uv run python scripts/fix_title_leaks.py data/trail_puzzles_v4 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

_CORRECTION_SYSTEM = """\
You are editing a scavenger-hunt riddle to remove direct Wikipedia article \
title references. The solver should have to DEDUCE which page to visit — \
naming the exact title makes navigation trivially easy.

For each flagged clue, rewrite ONLY that clue so the destination is described \
indirectly — via a historical fact, a relationship, a riddle, or a descriptive \
phrase — without naming the article title.

Rules:
- Keep the same clue number and overall structure
- Keep any extraction questions or tool-lookup questions unchanged
- Only change the part that names the Wikipedia title
- The rewritten clue must still unambiguously lead to the same page
- Output the COMPLETE riddle with the corrected clues in place"""


def _detect_title_leaks(
    data: dict, riddle: str
) -> list[tuple[int, str, str]]:
    """Detect Wikipedia titles mentioned verbatim in the riddle.

    Returns list of (stop_index, title, page_url) tuples.
    """
    leaked: list[tuple[int, str, str]] = []

    # Check seed title
    seed_title = data.get("seed_title", "")
    if seed_title and len(seed_title) >= 6:
        clean = re.sub(r"\s*\([^)]+\)\s*$", "", seed_title).strip()
        if len(clean) >= 6 and clean.lower() in riddle.lower():
            seed_url = data.get("seed_url", "")
            leaked.append((-1, seed_title, seed_url))

    # Check page titles from stops
    for stop in data.get("stops", []):
        page_url = stop.get("page_url", "") or ""
        if not page_url or "/wiki/" not in page_url:
            continue
        raw_title = page_url.split("/wiki/")[-1]
        title = unquote(raw_title).replace("_", " ")
        if len(title) < 6:
            continue
        clean = re.sub(r"\s*\([^)]+\)\s*$", "", title).strip()
        if len(clean) >= 6 and clean.lower() in riddle.lower():
            leaked.append((stop.get("index", -1), title, page_url))

    # Deduplicate (seed title and stop 0 may overlap)
    seen = set()
    unique = []
    for idx, title, url in leaked:
        key = title.lower()
        if key not in seen:
            seen.add(key)
            unique.append((idx, title, url))

    return unique


def _correct_titles(
    llm: OpenAI, model: str, riddle: str, leaked: list[tuple[int, str, str]]
) -> str:
    """Call LLM to rewrite riddle with indirect references."""
    clue_list = "\n".join(
        f"- Clue referencing stop {idx}: names \"{title}\" directly "
        f"(page: {url})"
        for idx, title, url in leaked
    )

    user_msg = (
        f"Here is the riddle:\n\n{riddle}\n\n"
        f"The following clues name Wikipedia article titles directly and "
        f"need to be rewritten with indirect descriptions:\n{clue_list}\n\n"
        f"Rewrite the COMPLETE riddle with those clues fixed. Keep "
        f"everything else exactly the same."
    )

    response = llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _CORRECTION_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
    )
    corrected = (response.choices[0].message.content or "").strip()
    return corrected if corrected else riddle


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix title leaks in riddles")
    parser.add_argument("data_dir", help="Root data directory")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "extreme"])
    parser.add_argument("--dry-run", action="store_true", help="Count leaks without fixing")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model for correction")
    parser.add_argument("--max-retries", type=int, default=3, help="Max correction retries per sample")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-8s %(message)s")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    llm = OpenAI() if not args.dry_run else None

    difficulties = [args.difficulty] if args.difficulty else ["easy", "medium", "hard", "extreme"]
    total_with_leaks = 0
    total_fixed = 0
    total_failed = 0

    for diff in difficulties:
        diff_dir = data_dir / diff
        if not diff_dir.exists():
            continue

        leak_count = 0
        fixed_count = 0
        failed_count = 0

        for filepath in sorted(diff_dir.glob("sample_*.json")):
            if not filepath.exists():
                continue
            try:
                with open(filepath) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            riddle = data.get("riddle", "")
            if not riddle:
                continue

            leaked = _detect_title_leaks(data, riddle)
            if not leaked:
                continue

            leak_count += 1
            total_with_leaks += 1

            if args.dry_run:
                titles = [t for _, t, _ in leaked]
                logger.debug(f"  {diff}/{filepath.name}: {len(leaked)} leaks — {titles}")
                continue

            # Try to fix with retries
            current_riddle = riddle
            success = False
            for attempt in range(args.max_retries):
                try:
                    corrected = _correct_titles(llm, args.model, current_riddle, leaked)
                    # Verify the fix
                    remaining = _detect_title_leaks(data, corrected)
                    if not remaining:
                        data["riddle"] = corrected
                        with open(filepath, "w") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                            f.write("\n")
                        fixed_count += 1
                        total_fixed += 1
                        success = True
                        break
                    else:
                        # Still has leaks — retry with the corrected version
                        current_riddle = corrected
                        leaked = remaining
                except Exception as exc:
                    logger.warning(f"  {filepath.name} attempt {attempt+1} failed: {exc}")

            if not success:
                failed_count += 1
                total_failed += 1
                # Accept best effort — use last corrected version if it's different
                if current_riddle != riddle:
                    remaining_after = _detect_title_leaks(data, current_riddle)
                    if len(remaining_after) < len(_detect_title_leaks(data, riddle)):
                        data["riddle"] = current_riddle
                        with open(filepath, "w") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                            f.write("\n")
                        logger.info(f"  {filepath.name}: partial fix (reduced leaks)")

        print(f"  {diff}: {leak_count} with leaks, {fixed_count} fixed, {failed_count} failed")

    print(f"\n{'=' * 50}")
    print(f"Total with leaks: {total_with_leaks}")
    if not args.dry_run:
        print(f"Total fixed:      {total_fixed}")
        print(f"Total failed:     {total_failed}")


if __name__ == "__main__":
    main()
