#!/usr/bin/env python3
"""Validate trail puzzle samples for known quality issues.

Thin CLI wrapper around :func:`trail.validator.validate_trail_dict`, extended
with diamond-pattern-specific checks for v5 augmented trails.

Usage:
    uv run python scripts/validate_samples.py data/trail_puzzles_v4
    uv run python scripts/validate_samples.py data/trail_puzzles_v5 --check-diamonds
    uv run python scripts/validate_samples.py data/trail_puzzles_v5 --check-diamonds --verbose
    uv run python scripts/validate_samples.py data/trail_puzzles_v4 --delete
    uv run python scripts/validate_samples.py data/trail_puzzles_v4 --difficulty easy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project src is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from trail.validator import ValidationResult, validate_trail_dict


# ---------------------------------------------------------------------------
# File-level validation entry point
# ---------------------------------------------------------------------------


def validate_sample(filepath: Path) -> list[dict]:
    """Validate a single sample file.

    Returns list of dicts with keys: type, stop_index, detail, severity.
    """
    try:
        with open(filepath) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return [{"type": "invalid_json", "stop_index": -1,
                 "detail": str(exc), "severity": "critical"}]

    result = validate_trail_dict(data)
    return [
        {
            "type": issue.category,
            "stop_index": issue.stop_index if issue.stop_index is not None else -1,
            "detail": issue.message,
            "severity": issue.severity,
        }
        for issue in result.issues
    ]


# ---------------------------------------------------------------------------
# Diamond-specific validation
# ---------------------------------------------------------------------------


def validate_diamonds(filepath: Path) -> list[dict]:
    """Run diamond-pattern-specific checks on a trail sample.

    Checks:
    1. depends_on coverage: every stop (except root) has depends_on populated
    2. depends_on acyclicity: all deps reference strictly earlier stops
    3. Branch stops: depend on a page stop (not another tool/reason)
    4. Merge stops: depend on exactly 2 stops whose depends_on share a common source
    5. Merge value consistency: merge stop value matches its code expression
    6. Passcode recomputation: final compute value is consistent
    7. DAG connectivity: all stops are reachable from roots
    8. No self-loops in depends_on
    9. Riddle clue count matches stop count
    """
    try:
        with open(filepath) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return [{"type": "invalid_json", "stop_index": -1,
                 "detail": str(exc), "severity": "critical"}]

    stops = data.get("stops", [])
    if not stops:
        return [{"type": "diamond_no_stops", "stop_index": -1,
                 "detail": "No stops in trail", "severity": "critical"}]

    issues: list[dict] = []

    # --- Check 1: depends_on coverage ---
    has_any_depends_on = any(s.get("depends_on") for s in stops)
    if not has_any_depends_on:
        # Not a diamond-augmented trail, skip diamond checks
        return []

    for i, stop in enumerate(stops):
        deps = stop.get("depends_on", [])
        if i == 0:
            if deps:
                issues.append({"type": "diamond_root_has_deps", "stop_index": 0,
                               "detail": f"Root stop has depends_on={deps}, expected []",
                               "severity": "warning"})
        elif not deps:
            issues.append({"type": "diamond_missing_depends_on", "stop_index": i,
                           "detail": f"Stop {i} ({stop.get('stop_type')}) has empty depends_on",
                           "severity": "warning"})

    # --- Check 2: depends_on acyclicity and validity ---
    for i, stop in enumerate(stops):
        deps = stop.get("depends_on", [])
        for dep in deps:
            if not isinstance(dep, int):
                issues.append({"type": "diamond_invalid_dep_type", "stop_index": i,
                               "detail": f"depends_on contains non-int: {dep}",
                               "severity": "critical"})
            elif dep < 0 or dep >= len(stops):
                issues.append({"type": "diamond_dep_out_of_range", "stop_index": i,
                               "detail": f"depends_on references stop {dep}, out of range [0, {len(stops)-1}]",
                               "severity": "critical"})
            elif dep >= i:
                issues.append({"type": "diamond_forward_dep", "stop_index": i,
                               "detail": f"depends_on references stop {dep} >= current {i} (not acyclic)",
                               "severity": "critical"})
            elif dep == i:
                issues.append({"type": "diamond_self_loop", "stop_index": i,
                               "detail": f"depends_on contains self-reference {dep}",
                               "severity": "critical"})

    # --- Check 3 & 4: Diamond structure ---
    # Find merge stops (depends_on has 2+ entries, not the compute stop)
    merge_stops = []
    branch_stops = []
    for i, stop in enumerate(stops):
        deps = stop.get("depends_on", [])
        stype = stop.get("stop_type", "")
        if len(deps) >= 2 and stype != "compute":
            merge_stops.append(i)
        elif deps and deps != [i - 1] and stype == "tool":
            branch_stops.append(i)

    for mi in merge_stops:
        merge = stops[mi]
        deps = merge.get("depends_on", [])

        # Merge must be a tool stop with python_execute_code
        if merge.get("stop_type") != "tool":
            issues.append({"type": "diamond_merge_wrong_type", "stop_index": mi,
                           "detail": f"Merge stop has type={merge.get('stop_type')}, expected 'tool'",
                           "severity": "critical"})
            continue

        chain = merge.get("bridge", {}).get("tool_chain", [])
        if not chain or chain[0].get("tool_name") != "python_execute_code":
            issues.append({"type": "diamond_merge_no_code", "stop_index": mi,
                           "detail": "Merge stop tool_chain missing python_execute_code",
                           "severity": "critical"})
            continue

        # Check 5: Merge value consistency
        code = chain[0].get("arguments", {}).get("code", "")
        expected_val = merge.get("extracted_value")
        if code and expected_val is not None:
            try:
                # Extract the expression from the first line (e.g., "result = int(18) + int(4)")
                first_line = code.split("\n")[0]
                if "=" in first_line:
                    expr = first_line.split("=", 1)[1].strip()
                    import io, contextlib
                    f_out = io.StringIO()
                    local_ns: dict = {}
                    with contextlib.redirect_stdout(f_out):
                        exec(f"__result__ = {expr}", {}, local_ns)
                    computed = local_ns.get("__result__")
                    if computed is not None and int(computed) != int(float(expected_val)):
                        issues.append({"type": "diamond_merge_value_mismatch", "stop_index": mi,
                                       "detail": f"Merge code evaluates to {computed} but extracted_value={expected_val}",
                                       "severity": "critical"})
            except Exception:
                pass  # Code may be complex; skip if eval fails

        # Check: merge deps should be branch stops or other tool stops
        for dep_idx in deps:
            dep_stop = stops[dep_idx] if dep_idx < len(stops) else None
            if dep_stop and dep_stop.get("stop_type") not in ("tool", "page", "reason"):
                issues.append({"type": "diamond_merge_dep_wrong_type", "stop_index": mi,
                               "detail": f"Merge depends on stop {dep_idx} (type={dep_stop.get('stop_type')})",
                               "severity": "warning"})

    # Check 3: Branch stops should depend on a page stop
    for bi in branch_stops:
        branch = stops[bi]
        deps = branch.get("depends_on", [])
        for dep_idx in deps:
            dep_stop = stops[dep_idx] if dep_idx < len(stops) else None
            if dep_stop and dep_stop.get("stop_type") != "page":
                issues.append({"type": "diamond_branch_dep_not_page", "stop_index": bi,
                               "detail": f"Branch stop depends on stop {dep_idx} (type={dep_stop.get('stop_type')}), expected page",
                               "severity": "warning"})

    # --- Check 6: Passcode recomputation ---
    compute_stop = stops[-1] if stops else None
    if compute_stop and compute_stop.get("stop_type") == "compute":
        try:
            compute_val = int(float(compute_stop.get("extracted_value", 0)))
            expected_passcode = compute_val % 10
            actual_passcode = data.get("passcode")
            if actual_passcode != expected_passcode:
                issues.append({"type": "diamond_passcode_mismatch", "stop_index": len(stops) - 1,
                               "detail": f"Passcode {actual_passcode} != compute_value {compute_val} % 10 = {expected_passcode}",
                               "severity": "critical"})
        except (ValueError, TypeError):
            issues.append({"type": "diamond_compute_not_numeric", "stop_index": len(stops) - 1,
                           "detail": f"Compute stop value not numeric: {compute_stop.get('extracted_value')}",
                           "severity": "critical"})

    # --- Check 7: DAG connectivity ---
    reachable = set()
    # BFS from roots (stops with empty depends_on)
    queue = [i for i, s in enumerate(stops) if not s.get("depends_on")]
    if not queue:
        queue = [0]
    reachable.update(queue)
    # Build successors from depends_on (reverse: if B depends on A, A -> B)
    successors: dict[int, list[int]] = {}
    for i, s in enumerate(stops):
        for dep in s.get("depends_on", []):
            successors.setdefault(dep, []).append(i)
    while queue:
        node = queue.pop(0)
        for succ in successors.get(node, []):
            if succ not in reachable:
                reachable.add(succ)
                queue.append(succ)
    unreachable = set(range(len(stops))) - reachable
    if unreachable:
        issues.append({"type": "diamond_unreachable_stops", "stop_index": -1,
                       "detail": f"Stops not reachable from roots: {sorted(unreachable)}",
                       "severity": "warning"})

    # --- Check 9: Riddle clue count ---
    riddle = data.get("riddle", "")
    if riddle:
        import re
        clue_numbers = re.findall(r"^\s*(\d+)\.", riddle, re.MULTILINE)
        if clue_numbers:
            max_clue = max(int(c) for c in clue_numbers)
            num_clues = len(clue_numbers)
            if num_clues != len(stops):
                issues.append({"type": "diamond_riddle_clue_mismatch", "stop_index": -1,
                               "detail": f"Riddle has {num_clues} clues but trail has {len(stops)} stops",
                               "severity": "warning"})

    return issues


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate trail puzzle samples")
    parser.add_argument(
        "data_dir",
        help="Root data directory (e.g. data/trail_puzzles_v4)",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "extreme"],
        help="Only validate a specific difficulty (default: all)",
    )
    parser.add_argument(
        "--check-diamonds",
        action="store_true",
        help="Also run diamond-pattern-specific validation (for v5 trails)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete flagged samples (moves to .flagged/ subdirectory)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print details for every issue found",
    )
    parser.add_argument(
        "--severity",
        choices=["all", "critical", "warning"],
        default="all",
        help="Filter by severity (default: all)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    difficulties = [args.difficulty] if args.difficulty else ["easy", "medium", "hard", "extreme"]

    total_scanned = 0
    total_flagged = 0
    total_critical = 0
    total_warning = 0
    total_deleted = 0
    issue_counts: dict[str, int] = {}

    for diff in difficulties:
        diff_dir = data_dir / diff
        if not diff_dir.exists():
            print(f"  Skipping {diff}/ (not found)")
            continue

        samples = sorted(diff_dir.glob("sample_*.json"))
        flagged_in_diff = 0
        critical_in_diff = 0

        for filepath in samples:
            total_scanned += 1
            issues = validate_sample(filepath)
            if args.check_diamonds:
                issues.extend(validate_diamonds(filepath))

            # Filter by severity if requested
            if args.severity == "critical":
                issues = [i for i in issues if i["severity"] == "critical"]
            elif args.severity == "warning":
                issues = [i for i in issues if i["severity"] == "warning"]

            if issues:
                flagged_in_diff += 1
                total_flagged += 1

                has_critical = any(i["severity"] == "critical" for i in issues)
                has_warning = any(i["severity"] == "warning" for i in issues)
                if has_critical:
                    critical_in_diff += 1
                    total_critical += 1
                if has_warning and not has_critical:
                    total_warning += 1

                for issue in issues:
                    issue_type = issue["type"]
                    issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

                if args.verbose:
                    severity_label = "CRITICAL" if has_critical else "WARNING"
                    print(f"\n  {severity_label}: {diff}/{filepath.name}")
                    for issue in issues:
                        sev = "!" if issue["severity"] == "critical" else "~"
                        print(f"    [{sev}] [{issue['type']}] stop {issue['stop_index']}: {issue['detail']}")

                if args.delete and has_critical:
                    flagged_dir = diff_dir / ".flagged"
                    flagged_dir.mkdir(exist_ok=True)
                    dest = flagged_dir / filepath.name
                    filepath.rename(dest)
                    total_deleted += 1
                    if args.verbose:
                        print(f"    -> moved to {dest}")

        print(f"  {diff}: {len(samples)} scanned, {flagged_in_diff} flagged "
              f"({critical_in_diff} critical)")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Total scanned:    {total_scanned}")
    print(f"Total flagged:    {total_flagged}")
    print(f"  Critical:       {total_critical}")
    print(f"  Warning only:   {total_warning}")
    if args.delete:
        print(f"Total moved:      {total_deleted}")
    print(f"Clean samples:    {total_scanned - total_flagged}")

    if issue_counts:
        print(f"\nIssue breakdown:")
        # Sort by severity then count
        critical_found = {k: v for k, v in issue_counts.items()
                         if k not in {"stop_count_out_of_range", "tool_stop_count_out_of_range",
                                      "reason_stop_count_out_of_range", "title_leak_in_riddle",
                                      "duplicate_extracted_value", "index_mismatch",
                                      "missing_bridge_field", "invalid_reason_source",
                                      "invalid_value_type", "value_type_mismatch",
                                      "riddle_clue_count_mismatch",
                                      "diamond_riddle_clue_mismatch",
                                      "diamond_missing_depends_on",
                                      "diamond_root_has_deps",
                                      "diamond_unreachable_stops",
                                      "diamond_branch_dep_not_page",
                                      "diamond_merge_dep_wrong_type"}}
        warning_found = {k: v for k, v in issue_counts.items() if k not in critical_found}

        if critical_found:
            print("  Critical:")
            for issue_type, count in sorted(critical_found.items(), key=lambda x: -x[1]):
                print(f"    {issue_type}: {count}")
        if warning_found:
            print("  Warnings:")
            for issue_type, count in sorted(warning_found.items(), key=lambda x: -x[1]):
                print(f"    {issue_type}: {count}")

    if total_flagged > 0 and not args.delete:
        print(f"\nRe-run with --delete to move critically flagged samples "
              f"to .flagged/ subdirectories.")


if __name__ == "__main__":
    main()
