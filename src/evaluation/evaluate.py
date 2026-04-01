"""Evaluate agents on The Amazing Agent Race (AAR) trail puzzles.

Usage:
    uv run python src/evaluation/evaluate.py \
        --data-dir data/trail_puzzles/easy \
        --model gpt-4o \
        --output-dir results/

    uv run python src/evaluation/evaluate.py \
        --data-dir data/trail_puzzles \
        --model gpt-4o-mini \
        --max-samples 5 \
        --output-dir results/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Ensure project src on path
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import re

from openai import OpenAI

from evaluation.agent import AgentResult, ReActAgent
from evaluation.prompt import build_puzzle_prompt, build_tool_lines
from trail.golden import GoldenExecutor
from trail.models import trail_from_json
from trail.validator import normalize_numeric, values_close
from mcp_servers.registry import ToolRegistry

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Hybrid value comparison
# ---------------------------------------------------------------------------

def compare_values(
    golden: Any,
    agent_value: Any,
    *,
    tolerance: float = 0.01,
) -> tuple[bool, str]:
    """Hybrid comparison of golden vs agent value.

    Returns (match: bool, method: str) where method describes how the
    match was determined.
    """
    # Convert to strings for exact comparison
    golden_str = str(golden).strip() if golden is not None else ""
    agent_str = str(agent_value).strip() if agent_value is not None else ""

    # Exact string match
    if golden_str and agent_str and golden_str == agent_str:
        return (True, "exact_match")

    # Numeric comparison
    golden_num = normalize_numeric(golden)
    agent_num = normalize_numeric(agent_value)

    if golden_num is not None and agent_num is not None:
        # Relative tolerance check
        if golden_num == 0:
            match = abs(agent_num) <= tolerance
        else:
            match = abs(golden_num - agent_num) / abs(golden_num) <= tolerance
        return (match, "numeric_match" if match else "numeric_mismatch")

    # Case-insensitive string match
    if golden_str.lower() == agent_str.lower():
        return (True, "normalized_match")

    # No deterministic match — needs LLM judge
    return (False, "needs_llm_judge")


def llm_judge_values(
    golden_value: Any,
    agent_value: Any,
    context: dict[str, Any],
    llm: OpenAI,
    model: str,
) -> tuple[bool, str]:
    """Use an LLM to judge whether agent_value matches golden_value.

    *context* should include keys like ``extraction_target``, ``page_url``,
    and ``expected_value_type`` so the judge has enough information.

    Returns (match: bool, reasoning: str).
    """
    system_prompt = (
        "You are a judge comparing values extracted during a scavenger-hunt puzzle evaluation.\n"
        "The GOLDEN value is the expected correct answer from the puzzle's answer key.\n"
        "The AGENT value is what an AI agent extracted during its solving attempt.\n\n"
        "Determine if the agent's value is semantically equivalent to the golden value.\n"
        "Consider: numeric equivalence (e.g., \"3,609\" vs 3609), unit variations, \n"
        "rounding differences, and format differences.\n\n"
        "Respond with JSON: {\"match\": true/false, \"reasoning\": \"brief explanation\"}"
    )

    user_message_parts = [
        f"GOLDEN value: {golden_value}",
        f"AGENT value: {agent_value}",
    ]
    if context.get("extraction_target"):
        user_message_parts.append(f"Extraction target: {context['extraction_target']}")
    if context.get("page_url"):
        user_message_parts.append(f"Source page: {context['page_url']}")
    if context.get("expected_value_type"):
        user_message_parts.append(f"Expected value type: {context['expected_value_type']}")

    user_message = "\n".join(user_message_parts)

    try:
        response = llm.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            max_tokens=256,
        )
        content = response.choices[0].message.content or ""
        # Parse JSON from response (handle markdown fences)
        json_match = re.search(r"\{[^}]+\}", content)
        if json_match:
            result = json.loads(json_match.group())
            return (bool(result.get("match", False)), result.get("reasoning", ""))
        return (False, f"Failed to parse LLM judge response: {content[:200]}")
    except Exception as exc:
        logger.warning("LLM judge call failed: %s", exc)
        return (False, f"LLM judge error: {exc}")


def compute_per_stop_metrics(
    trail_data: dict[str, Any],
    agent_result: AgentResult,
    llm: OpenAI | None = None,
    model: str = "gpt-4o",
) -> dict[str, Any]:
    """Compute per-stop value-match metrics using hybrid comparison.

    For each stop with a golden ``extracted_value``, tries to find the agent's
    corresponding value from its tool calls, then compares using
    :func:`compare_values` with an optional LLM judge fallback.

    Returns a dict with per-stop details and an overall ``value_accuracy``.
    """
    stops = trail_data.get("stops", [])
    if not stops:
        return {"stop_details": [], "value_accuracy": None}

    # Index agent tool call results by tool name for lookup
    agent_tool_results: dict[str, Any] = {}
    for call in agent_result.tool_calls:
        name = call.get("tool_name", "")
        result = call.get("result")
        if name and result is not None:
            agent_tool_results[name] = result

    stop_details: list[dict[str, Any]] = []
    matched = 0
    evaluated = 0

    for i, stop in enumerate(stops):
        golden_value = stop.get("extracted_value")
        if golden_value is None:
            continue

        stop_type = stop.get("stop_type", "")
        detail: dict[str, Any] = {
            "stop_index": i,
            "stop_type": stop_type,
            "golden_value": golden_value,
            "agent_value": None,
            "match": False,
            "method": "no_agent_value",
        }

        # Try to find the agent's value for this stop
        agent_value = None
        if stop_type == "tool":
            chain = stop.get("bridge", {}).get("tool_chain", [])
            if chain:
                last_tool = chain[-1].get("tool_name", "")
                agent_value = agent_tool_results.get(last_tool)
        elif stop_type == "page":
            page_url = stop.get("page_url", "")
            # Check if agent fetched this page
            for call in agent_result.tool_calls:
                if (call.get("tool_name") == "fetch_webpage"
                        and call.get("arguments", {}).get("url") == page_url):
                    agent_value = call.get("result")
                    break

        if agent_value is not None:
            detail["agent_value"] = agent_value
            evaluated += 1

            match, method = compare_values(golden_value, agent_value)

            # LLM judge fallback
            if method == "needs_llm_judge" and llm is not None:
                context = {
                    "extraction_target": stop.get("extraction_target", ""),
                    "page_url": stop.get("page_url", ""),
                    "expected_value_type": stop.get("extracted_value_type", ""),
                }
                match, reasoning = llm_judge_values(
                    golden_value, agent_value, context, llm, model,
                )
                method = f"llm_judge: {reasoning}"

            detail["match"] = match
            detail["method"] = method
            if match:
                matched += 1
        else:
            evaluated += 1  # count as evaluated but unmatched

        stop_details.append(detail)

    return {
        "stop_details": stop_details,
        "value_accuracy": matched / evaluated if evaluated else None,
        "stops_evaluated": evaluated,
        "stops_matched": matched,
    }


# ---------------------------------------------------------------------------
# Trail metrics
# ---------------------------------------------------------------------------


def compute_trail_metrics(
    trail_data: dict[str, Any],
    agent_result: AgentResult,
    llm: OpenAI | None = None,
    model: str = "gpt-4o",
) -> dict[str, Any]:
    """Compare agent actions to golden trail for fine-grained metrics.

    When *llm* and *model* are provided, per-stop value comparison uses a
    hybrid approach (exact / numeric / LLM judge).
    """
    stops = trail_data.get("stops", [])
    if not stops:
        return {}

    # Collect agent's fetched URLs and tool names
    agent_urls: set[str] = set()
    agent_tool_names: list[str] = []
    for call in agent_result.tool_calls:
        name = call.get("tool_name", "")
        if name == "fetch_webpage":
            url = call.get("arguments", {}).get("url", "")
            if url:
                agent_urls.add(url)
        agent_tool_names.append(name)

    # Page stop completion
    page_stops = [s for s in stops if s.get("stop_type") == "page"]
    pages_visited = sum(
        1 for s in page_stops
        if s.get("page_url") and s["page_url"] in agent_urls
    )

    # Tool stop completion
    tool_stops = [s for s in stops if s.get("stop_type") == "tool"]
    tools_completed = 0
    for s in tool_stops:
        chain = s.get("bridge", {}).get("tool_chain", [])
        expected = [step.get("tool_name", "") for step in chain]
        if expected and all(t in agent_tool_names for t in expected):
            tools_completed += 1

    reason_stops = [s for s in stops if s.get("stop_type") == "reason"]

    metrics: dict[str, Any] = {
        "total_page_stops": len(page_stops),
        "pages_visited": pages_visited,
        "page_visit_rate": pages_visited / len(page_stops) if page_stops else None,
        "total_tool_stops": len(tool_stops),
        "tools_completed": tools_completed,
        "tool_completion_rate": tools_completed / len(tool_stops) if tool_stops else None,
        "total_reason_stops": len(reason_stops),
        "agent_steps": agent_result.steps,
        "agent_tool_calls": len(agent_result.tool_calls),
        "hit_step_limit": agent_result.hit_step_limit,
    }

    # Per-stop value comparison (hybrid: exact / numeric / LLM judge)
    per_stop = compute_per_stop_metrics(trail_data, agent_result, llm=llm, model=model)
    metrics["per_stop"] = per_stop

    return metrics


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_trail_files(
    data_dir: Path,
    *,
    max_samples: int | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Load trail JSON files from a directory (recursively).

    Returns list of (sample_id, trail_data) tuples.
    """
    json_files: list[Path] = []

    # Check for difficulty subdirectories
    subdirs = [d for d in sorted(data_dir.iterdir()) if d.is_dir() and not d.name.startswith(".")]
    if subdirs:
        for subdir in subdirs:
            json_files.extend(sorted(subdir.glob("*.json")))
    else:
        json_files = sorted(data_dir.glob("*.json"))

    trails: list[tuple[str, dict[str, Any]]] = []
    for path in json_files:
        if path.name.startswith("_") or path.name.startswith("."):
            continue

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load %s: %s", path, exc)
            continue

        # Must have stops and a riddle to be evaluable
        if "stops" not in data or not data.get("riddle"):
            continue

        # Build sample_id from path
        if path.parent != data_dir:
            difficulty = path.parent.name
            sample_id = f"{difficulty}/{path.stem}"
        else:
            # Infer difficulty from trail data if available
            diff_data = data.get("difficulty", {})
            difficulty = diff_data.get("level", "") if isinstance(diff_data, dict) else ""
            sample_id = f"{difficulty}/{path.stem}" if difficulty else path.stem

        trails.append((sample_id, data))

        if max_samples is not None and len(trails) >= max_samples:
            break

    logger.info("Loaded %d trail puzzles from %s", len(trails), data_dir)
    return trails


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


async def evaluate_trails(
    trails: list[tuple[str, dict[str, Any]]],
    agent: ReActAgent,
    tool_lines: list[str],
    llm: OpenAI | None = None,
    model: str = "gpt-4o",
) -> list[dict[str, Any]]:
    """Run agent on each trail and collect results."""
    results: list[dict[str, Any]] = []

    for idx, (sample_id, trail_data) in enumerate(trails, start=1):
        logger.info("[%d/%d] Evaluating %s", idx, len(trails), sample_id)

        # Build prompt
        prompt_info = build_puzzle_prompt(
            trail_data, tool_lines, sample_id=sample_id,
        )
        prompt = prompt_info["prompt"]
        expected = prompt_info["expected_passcode"]
        metadata = prompt_info["metadata"]

        # Run agent
        start = time.monotonic()
        agent_result = await agent.solve(
            prompt,
            max_steps=metadata.get("step_limit", 25),
        )
        elapsed_ms = (time.monotonic() - start) * 1000

        # Score
        correct = agent_result.answer == expected if agent_result.answer is not None else False

        # Trail metrics (with hybrid per-stop comparison when llm is available)
        trail_metrics = compute_trail_metrics(trail_data, agent_result, llm=llm, model=model)

        result = {
            "sample_id": sample_id,
            "expected_passcode": expected,
            "predicted_passcode": agent_result.answer,
            "correct": correct,
            "steps": agent_result.steps,
            "hit_step_limit": agent_result.hit_step_limit,
            "tool_calls_count": len(agent_result.tool_calls),
            "elapsed_ms": round(elapsed_ms),
            "trail_metrics": trail_metrics,
            "metadata": metadata,
            "tool_calls": agent_result.tool_calls,
        }
        results.append(result)

        status = "CORRECT" if correct else "WRONG"
        logger.info(
            "  %s — predicted=%s expected=%s steps=%d time=%.1fs",
            status, agent_result.answer, expected,
            agent_result.steps, elapsed_ms / 1000,
        )

        # Brief delay between puzzles to avoid rate limits
        if idx < len(trails):
            await asyncio.sleep(1)

    return results


def compute_summary(results: list[dict[str, Any]], model: str) -> dict[str, Any]:
    """Aggregate evaluation results into a summary."""
    total = len(results)
    correct = sum(1 for r in results if r["correct"])

    # Per-difficulty breakdown
    by_difficulty: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        diff = r["metadata"].get("difficulty", "unknown")
        by_difficulty[diff]["total"] += 1
        if r["correct"]:
            by_difficulty[diff]["correct"] += 1

    difficulty_accuracy = {
        diff: stats["correct"] / stats["total"] if stats["total"] else 0
        for diff, stats in sorted(by_difficulty.items())
    }

    # Aggregate trail metrics
    page_visit_rates = [
        r["trail_metrics"]["page_visit_rate"]
        for r in results
        if r["trail_metrics"].get("page_visit_rate") is not None
    ]
    tool_completion_rates = [
        r["trail_metrics"]["tool_completion_rate"]
        for r in results
        if r["trail_metrics"].get("tool_completion_rate") is not None
    ]
    avg_steps = sum(r["steps"] for r in results) / total if total else 0
    step_limit_hits = sum(1 for r in results if r["hit_step_limit"])

    return {
        "model": model,
        "total_puzzles": total,
        "correct": correct,
        "accuracy": correct / total if total else 0,
        "accuracy_by_difficulty": difficulty_accuracy,
        "avg_page_visit_rate": sum(page_visit_rates) / len(page_visit_rates) if page_visit_rates else None,
        "avg_tool_completion_rate": sum(tool_completion_rates) / len(tool_completion_rates) if tool_completion_rates else None,
        "avg_steps": round(avg_steps, 1),
        "step_limit_hits": step_limit_hits,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def refresh_golden_answers(
    trails: list[tuple[str, dict[str, Any]]],
    registry: ToolRegistry,
    data_dir: Path,
) -> int:
    """Re-run golden execution on all trails and update passcodes in-place.

    Tool-stop values (geocode, elevation, weather) are refreshed from live
    APIs. Page-stop values are verified via deterministic extraction — if a
    Wikipedia edit changed the value, extraction fails and the trail is
    skipped (logged as a warning). Compute expressions use hardcoded values
    and will only change if their tool-stop inputs change.

    Returns the number of trails whose passcode changed.
    """
    executor = GoldenExecutor(tool_registry=registry)
    changed = 0

    for idx, (sample_id, trail_data) in enumerate(trails, start=1):
        logger.info("[refresh %d/%d] %s", idx, len(trails), sample_id)

        trail = trail_from_json(json.dumps(trail_data))
        old_passcode = trail.passcode

        try:
            result = await executor.execute_trail(trail)
        except Exception as e:
            logger.warning("  refresh failed for %s: %s", sample_id, e)
            continue

        if not result.success:
            errors = [sr.error for sr in result.stop_results if sr.error]
            logger.warning("  refresh execution failed for %s: %s", sample_id, errors[:2])
            continue

        new_passcode = result.computed_passcode

        # Update in-memory trail_data with fresh values
        for sr in result.stop_results:
            stop = trail_data["stops"][sr.stop_index]
            if stop.get("stop_type") == "tool" and sr.actual_value is not None:
                stop["extracted_value"] = sr.actual_value
            elif stop.get("stop_type") == "compute" and sr.actual_value is not None:
                stop["extracted_value"] = sr.actual_value

        trail_data["passcode"] = new_passcode

        if old_passcode != new_passcode:
            changed += 1
            logger.info("  CHANGED — %s: passcode %s → %s", sample_id, old_passcode, new_passcode)

            # Also update the source JSON file
            parts = sample_id.split("/")
            if len(parts) == 2:
                json_path = data_dir / parts[0] / f"{parts[1]}.json"
            else:
                json_path = data_dir / f"{parts[0]}.json"

            if json_path.exists():
                json_path.write_text(
                    json.dumps(trail_data, indent=2, ensure_ascii=False) + "\n"
                )

    return changed


async def run_evaluation(args: argparse.Namespace) -> None:
    load_dotenv()

    # Init components
    llm = OpenAI()
    model = args.model
    registry = ToolRegistry()
    agent = ReActAgent(llm, model, registry, max_steps=args.max_steps)

    # Tool descriptions for prompt
    tool_lines = build_tool_lines(registry.available_tools())

    # Load puzzles
    data_dir = Path(args.data_dir)
    trails = load_trail_files(data_dir, max_samples=args.max_samples)
    if not trails:
        logger.error("No trail puzzles found in %s", data_dir)
        return

    # Refresh golden answers if requested
    if args.refresh_golden:
        logger.info("Refreshing golden answers for %d trails...", len(trails))
        start = time.monotonic()
        num_changed = await refresh_golden_answers(trails, registry, data_dir)
        elapsed = time.monotonic() - start
        logger.info(
            "Golden refresh complete in %.1fs: %d/%d passcodes changed",
            elapsed, num_changed, len(trails),
        )

    # Run evaluation (pass llm + model for hybrid per-stop comparison)
    results = await evaluate_trails(trails, agent, tool_lines, llm=llm, model=model)

    # Summary
    summary = compute_summary(results, model)

    logger.info("=" * 60)
    logger.info("Evaluation Results (%s)", model)
    logger.info("=" * 60)
    logger.info("  Accuracy: %d/%d (%.1f%%)", summary["correct"], summary["total_puzzles"], summary["accuracy"] * 100)
    for diff, acc in summary["accuracy_by_difficulty"].items():
        diff_total = sum(1 for r in results if r["metadata"].get("difficulty") == diff)
        diff_correct = sum(1 for r in results if r["metadata"].get("difficulty") == diff and r["correct"])
        logger.info("    %s: %d/%d (%.1f%%)", diff, diff_correct, diff_total, acc * 100)
    if summary["avg_page_visit_rate"] is not None:
        logger.info("  Avg page visit rate: %.1f%%", summary["avg_page_visit_rate"] * 100)
    if summary["avg_tool_completion_rate"] is not None:
        logger.info("  Avg tool completion rate: %.1f%%", summary["avg_tool_completion_rate"] * 100)
    logger.info("  Avg steps: %.1f", summary["avg_steps"])
    logger.info("  Step limit hits: %d", summary["step_limit_hits"])

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = model.replace("/", "_")
    output_path = output_dir / f"eval_{model_tag}_{timestamp}.json"

    output = {
        "summary": summary,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Results saved to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate agents on AAR trail puzzles",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing trail puzzle JSON files (supports subdirs)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model to use for the agent (default: gpt-4o)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of puzzles to evaluate",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=25,
        help="Maximum agent steps per puzzle (default: 25)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--refresh-golden",
        action="store_true",
        help="Re-run golden tool chains before evaluation to get fresh expected answers",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
