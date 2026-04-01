#!/usr/bin/env python3
"""Programmatically repair fixable quality issues in trail puzzle samples.

Fixes:
1. passcode_mismatch — set passcode = int(last_stop.extracted_value) % 10
2. value_type_mismatch — convert string numbers to actual int/float
3. missing_bridge_field (link_follow) — set target_url = stop's page_url
4. stock_weekend_date — shift to prior Friday
5. citation_in_tool_arg / citation_in_value — clean citation artifacts

Usage:
    uv run python scripts/repair_samples.py data/trail_puzzles_v4
    uv run python scripts/repair_samples.py data/trail_puzzles_v4 --dry-run
    uv run python scripts/repair_samples.py data/trail_puzzles_v4 --difficulty easy
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

# Ensure project src is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from trail.validator import parse_number


# ---------------------------------------------------------------------------
# Repair functions
# ---------------------------------------------------------------------------

_CITATION_PATTERNS_FULL = [
    (re.compile(r"\[\[\d+\]\]\(#cite_note[^)]*\)"), ""),   # [[1]](#cite_note-...)
    (re.compile(r"\[(\d+|note \d+)\]"), ""),                 # [1], [note 1]
]


def _clean_citations(text: str) -> str:
    """Remove citation artifacts from text."""
    for pattern, replacement in _CITATION_PATTERNS_FULL:
        text = pattern.sub(replacement, text)
    return text.strip()


_try_parse_number = parse_number  # alias for backward compatibility


def _adjust_to_trading_day(date_str: str) -> str:
    """Shift weekend dates to prior Friday."""
    try:
        dt = date.fromisoformat(date_str)
    except (ValueError, TypeError):
        return date_str
    weekday = dt.weekday()
    if weekday == 5:  # Saturday → Friday
        dt = dt - timedelta(days=1)
    elif weekday == 6:  # Sunday → Friday
        dt = dt - timedelta(days=2)
    return dt.isoformat()


_STOCK_TOOLS = {"stock_historical_price", "stock_volume"}


def repair_sample(filepath: Path, dry_run: bool = False) -> dict[str, int]:
    """Repair a single sample. Returns counts of fixes applied."""
    fixes: dict[str, int] = {}

    try:
        with open(filepath) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    modified = False
    stops = data.get("stops", [])

    # --- Fix 1: Passcode mismatch ---
    if stops and stops[-1].get("stop_type") == "compute":
        last_val = stops[-1].get("extracted_value")
        if last_val is not None:
            try:
                expected_passcode = int(float(last_val)) % 10
                if data.get("passcode") != expected_passcode:
                    fixes["passcode_fixed"] = 1
                    if not dry_run:
                        data["passcode"] = expected_passcode
                        modified = True
            except (ValueError, TypeError):
                pass

    for i, stop in enumerate(stops):
        extracted_value = stop.get("extracted_value")
        extracted_value_type = stop.get("extracted_value_type", "")
        bridge = stop.get("bridge", {})
        tool_chain = bridge.get("tool_chain", [])

        # --- Fix 2: Value type mismatch (string → number) ---
        if extracted_value_type == "number" and isinstance(extracted_value, str):
            new_val = _try_parse_number(extracted_value)
            if new_val is not None:
                fixes["value_type_fixed"] = fixes.get("value_type_fixed", 0) + 1
                if not dry_run:
                    stop["extracted_value"] = new_val
                    modified = True

        # --- Fix 3: Missing target_url on link_follow bridges ---
        if (bridge.get("bridge_type") == "link_follow"
                and not bridge.get("target_url")
                and stop.get("page_url")):
            fixes["target_url_fixed"] = fixes.get("target_url_fixed", 0) + 1
            if not dry_run:
                bridge["target_url"] = stop["page_url"]
                modified = True

        # --- Fix 4: Stock weekend dates ---
        for step in tool_chain:
            tool_name = step.get("tool_name", "")
            if tool_name in _STOCK_TOOLS:
                args = step.get("arguments", {})
                tool_date = args.get("date", "")
                if tool_date:
                    try:
                        dt = date.fromisoformat(tool_date)
                        if dt.weekday() >= 5:
                            new_date = _adjust_to_trading_day(tool_date)
                            fixes["weekend_date_fixed"] = fixes.get("weekend_date_fixed", 0) + 1
                            if not dry_run:
                                args["date"] = new_date
                                modified = True
                    except (ValueError, TypeError):
                        pass

        # --- Fix 5: Citation artifacts in tool arguments ---
        for step in tool_chain:
            args = step.get("arguments", {})
            for arg_name, arg_value in list(args.items()):
                if isinstance(arg_value, str):
                    cleaned = _clean_citations(arg_value)
                    if cleaned != arg_value:
                        fixes["citation_in_arg_fixed"] = fixes.get("citation_in_arg_fixed", 0) + 1
                        if not dry_run:
                            args[arg_name] = cleaned
                            modified = True

        # --- Fix 6: Citation artifacts in extracted values ---
        if isinstance(extracted_value, str):
            cleaned = _clean_citations(extracted_value)
            if cleaned != extracted_value:
                fixes["citation_in_value_fixed"] = fixes.get("citation_in_value_fixed", 0) + 1
                if not dry_run:
                    stop["extracted_value"] = cleaned
                    modified = True

    # After fixing value types, re-check passcode (values may have changed)
    if not dry_run and modified and stops and stops[-1].get("stop_type") == "compute":
        last_val = stops[-1].get("extracted_value")
        if last_val is not None:
            try:
                data["passcode"] = int(float(last_val)) % 10
            except (ValueError, TypeError):
                pass

    # Write back
    if modified and not dry_run:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

    return fixes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Repair trail puzzle samples")
    parser.add_argument("data_dir", help="Root data directory")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "extreme"])
    parser.add_argument("--dry-run", action="store_true", help="Report fixes without applying")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    difficulties = [args.difficulty] if args.difficulty else ["easy", "medium", "hard", "extreme"]
    total_fix_counts: dict[str, int] = {}
    total_files_fixed = 0

    for diff in difficulties:
        diff_dir = data_dir / diff
        if not diff_dir.exists():
            continue

        files_fixed = 0
        for filepath in sorted(diff_dir.glob("sample_*.json")):
            fixes = repair_sample(filepath, dry_run=args.dry_run)
            if fixes:
                files_fixed += 1
                total_files_fixed += 1
                for k, v in fixes.items():
                    total_fix_counts[k] = total_fix_counts.get(k, 0) + v

        print(f"  {diff}: {files_fixed} files {'would be ' if args.dry_run else ''}fixed")

    print(f"\n{'=' * 50}")
    action = "would apply" if args.dry_run else "applied"
    print(f"Total files {action}: {total_files_fixed}")
    if total_fix_counts:
        print(f"\nFix breakdown:")
        for fix_type, count in sorted(total_fix_counts.items(), key=lambda x: -x[1]):
            print(f"  {fix_type}: {count}")

    if args.dry_run and total_files_fixed > 0:
        print(f"\nRe-run without --dry-run to apply fixes.")


if __name__ == "__main__":
    main()
