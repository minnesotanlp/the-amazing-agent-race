"""CLI entry point for single-trail pipeline generation (steps 1-4).

Pipeline:
  1. Initialize components (WikiGraph, FactExtractor, TrailBuilder, etc.)
  2. Resolve seed URL(s) — fixed, from file, or random Wikipedia articles
  3. Per seed: crawl -> build_trail -> golden_execute -> verbalize -> validate -> save
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote

from dotenv import load_dotenv

# Ensure project src on path
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openai import OpenAI

from trail.builder import TrailBuilder
from trail.extractor import FactExtractor
from trail.golden import GoldenExecutor
from trail.models import (
    DIFFICULTY_CONFIGS,
    Trail,
    TrailDifficultyConfig,
    trail_to_json,
)
from trail.validator import validate_trail as validate_trail_static
from trail.verbalizer import TrailVerbalizer
from trail.wiki_graph import WikiGraph

from mcp_servers.registry import ToolRegistry

logger = logging.getLogger(__name__)

_WIKI_PREFIX = "https://en.wikipedia.org/wiki/"

# Title patterns that make bad seeds (prefix or contains patterns)
_BAD_TITLE_PATTERNS = re.compile(
    r"^(List of |Lists of |Index of |Outline of |Glossary of |"
    r"Timeline of |Comparison of |Bibliography of |"
    r"History of |Deaths in |"
    r"Template:|Category:|Portal:|Draft:|Wikipedia:|Module:|"
    r"File:|Help:|Talk:|User:|Special:|"
    r".*\(disambiguation\)|"
    r".*\(journal\)|.*\(magazine\)|"
    r".*\(TV series\)|.*\(TV programme\)|"
    r".*\(season \d|"
    r".*\(film series\)|"
    # Year/number-only articles ("1987", "1987 in music", "2020s")
    r"\d{4}\b|"
    # ISO standards, technical specs
    r"ISO \d|IEEE \d|RFC \d)",
    re.IGNORECASE,
)

# Patterns suggesting a fictional/in-universe or highly niche topic
_FICTIONAL_PATTERNS = re.compile(
    r"\(fictional\)|"
    r"\(comics\)|"
    r"\(character\)|"
    r"\(video game\)|"
    r"\(album\)|"
    r"\(song\)|"
    r"\(single\)|"
    r"\(EP\)",
    re.IGNORECASE,
)

# Taxonomy-heavy infobox keys — if most fields are these, the page is
# a species article with little useful numeric/geographic data
_TAXONOMY_KEYS = frozenset({
    "kingdom", "phylum", "subphylum", "class", "order", "suborder",
    "infraorder", "family", "subfamily", "superfamily", "genus",
    "subgenus", "species", "subspecies", "domain", "clade",
    "subdivision", "binomial", "binomial name", "authority",
    "type species", "type genus", "tribe", "subtribe",
})


def _is_bad_title(title: str) -> bool:
    """Check if a title is unsuitable as a seed article."""
    if _BAD_TITLE_PATTERNS.search(title):
        return True
    if _FICTIONAL_PATTERNS.search(title):
        return True
    # Very short titles are often stubs or disambiguation-like
    if len(title) <= 2:
        return True
    # Titles that are purely numeric
    if title.strip().isdigit():
        return True
    return False


def _is_taxonomy_heavy(page) -> bool:
    """Check if an article's infobox is dominated by biological taxonomy."""
    if not page.infobox:
        return False
    tax_count = sum(
        1 for f in page.infobox
        if f.key.lower().strip().rstrip(":") in _TAXONOMY_KEYS
    )
    return tax_count >= 4 and tax_count / len(page.infobox) >= 0.4


def _first_paragraph_is_english(page) -> bool:
    """Quick heuristic: check that article content is primarily English.

    Catches articles about topics in non-Latin scripts where the English
    Wikipedia article is a thin translation stub.
    Falls back to raw_markdown if first_paragraph is missing or too short.
    """
    text = page.first_paragraph
    # If first_paragraph extraction failed (boilerplate, empty, etc.),
    # sample from raw markdown instead
    if not text or len(text) < 50:
        raw = getattr(page, "raw_markdown", "")
        if raw:
            # Skip first 500 chars (likely navigation/boilerplate) and
            # sample a chunk from the article body
            text = raw[500:2500]
    if not text or len(text) < 50:
        # Can't determine — give benefit of the doubt
        return True
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text) >= 0.7


# ---------------------------------------------------------------------------
# Fetch function adapter
# ---------------------------------------------------------------------------


async def _make_fetch_fn(tool_registry: ToolRegistry):
    """Create an async fetch function that uses the fetch_webpage MCP tool."""
    spec = tool_registry.available_tools().get("fetch_webpage")
    if spec is None or spec.executor is None:
        raise RuntimeError("fetch_webpage tool not available in ToolRegistry")

    async def fetch(url: str) -> str:
        result = await spec.executor({"url": url})
        return result.output_text if hasattr(result, "output_text") else str(result)

    return fetch


# ---------------------------------------------------------------------------
# Random seed discovery
# ---------------------------------------------------------------------------


async def _fetch_random_batch(batch_size: int = 20) -> list[dict]:
    """Fetch a batch of random article titles from the MediaWiki API."""
    import httpx

    _API_URL = "https://en.wikipedia.org/w/api.php"
    _HEADERS = {
        "User-Agent": "AARBenchmark/1.0 (academic research; https://github.com)",
        "Accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(headers=_HEADERS, timeout=15) as client:
            resp = await client.get(_API_URL, params={
                "action": "query",
                "list": "random",
                "rnnamespace": "0",  # main namespace only
                "rnlimit": str(batch_size),
                "format": "json",
            })
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning("Failed to fetch random articles from Wikipedia API: %s", e)
        return []

    return data.get("query", {}).get("random", [])


def _has_numeric_infobox_fields(page) -> bool:
    """Check if the page has at least one infobox field with a numeric value."""
    for f in page.infobox:
        if f.numeric_value is not None:
            return True
        # Also check if the value string contains digits (dates, measurements, etc.)
        if any(c.isdigit() for c in f.value):
            return True
    return False


async def _discover_random_seed(
    fetch_fn,
    wiki_graph: WikiGraph,
    *,
    max_batches: int = 3,
    batch_size: int = 20,
    min_page_length: int = 5000,
    min_links: int = 15,
    min_sections: int = 4,
    min_infobox_fields: int = 3,
) -> str | None:
    """Use the Wikipedia API to discover a random quality article.

    Uses the MediaWiki API (action=query&list=random) which is more
    bot-friendly than hitting Special:Random directly. Retries with
    new batches if no candidate passes all quality gates.

    Quality criteria (all required):
    - Not a list, disambiguation, template, timeline, or meta page
    - Title is not too short or purely numeric
    - Sufficient page length (≥5000 chars, rejects stubs and short articles)
    - Enough outgoing links (≥15, ensures crawl can expand meaningfully)
    - Enough sections (≥4, indicates article depth)
    - Has an infobox with ≥3 fields (ensures extractable structured data)
    - Has at least one numeric infobox field (enables tool stops)

    Returns the canonical Wikipedia URL, or None if all batches exhausted.
    """
    total_checked = 0
    best_fallback: str | None = None
    best_fallback_score = 0

    for batch_num in range(max_batches):
        random_pages = await _fetch_random_batch(batch_size)
        if not random_pages:
            logger.warning("Wikipedia API returned no random pages (batch %d)", batch_num + 1)
            continue

        for page_info in random_pages:
            total_checked += 1
            title = page_info.get("title", "")

            # Title-level filtering (fast, no network)
            if _is_bad_title(title):
                logger.debug("Random #%d: bad title: %s", total_checked, title)
                continue

            # Build URL from title
            url_title = title.replace(" ", "_")
            candidate_url = f"{_WIKI_PREFIX}{url_title}"

            try:
                # Fetch and parse the page
                await wiki_graph.crawl(
                    candidate_url, max_depth=0, max_pages=1,
                    include_pageviews=False,
                )
                page = wiki_graph.get_page(candidate_url)
                if page is None:
                    logger.debug("Random #%d: failed to parse: %s", total_checked, title)
                    continue

                # --- Quality gates ---

                if page.page_length < min_page_length:
                    logger.debug(
                        "Random #%d: too short (%d chars): %s",
                        total_checked, page.page_length, title,
                    )
                    continue

                n_links = len(page.outgoing_links)
                if n_links < min_links:
                    logger.debug(
                        "Random #%d: too few links (%d): %s",
                        total_checked, n_links, title,
                    )
                    continue

                n_sections = len(page.sections)
                if n_sections < min_sections:
                    logger.debug(
                        "Random #%d: too few sections (%d): %s",
                        total_checked, n_sections, title,
                    )
                    continue

                # Content quality gates

                if not _first_paragraph_is_english(page):
                    logger.debug(
                        "Random #%d: non-English or thin first paragraph: %s",
                        total_checked, title,
                    )
                    continue

                if _is_taxonomy_heavy(page):
                    logger.debug(
                        "Random #%d: taxonomy-heavy species article: %s",
                        total_checked, title,
                    )
                    continue

                n_infobox = len(page.infobox)
                if n_infobox < min_infobox_fields:
                    logger.debug(
                        "Random #%d: too few infobox fields (%d): %s",
                        total_checked, n_infobox, title,
                    )
                    # Track as fallback — pages without rich infoboxes can
                    # still work, just less ideal for tool stops
                    score = page.page_length + n_links * 100 + n_sections * 50
                    if score > best_fallback_score:
                        best_fallback = candidate_url
                        best_fallback_score = score
                    continue

                has_numeric = _has_numeric_infobox_fields(page)
                if not has_numeric:
                    logger.debug(
                        "Random #%d: no numeric infobox fields: %s",
                        total_checked, title,
                    )
                    # Still a reasonable fallback
                    score = page.page_length + n_links * 100 + n_sections * 50
                    if score > best_fallback_score:
                        best_fallback = candidate_url
                        best_fallback_score = score
                    continue

                # All gates passed
                logger.info(
                    "Random seed discovered (#%d): %s "
                    "(%d chars, %d links, %d sections, %d infobox fields)",
                    total_checked, title, page.page_length,
                    n_links, n_sections, n_infobox,
                )
                return candidate_url

            except Exception as e:
                logger.debug("Random #%d failed: %s", total_checked, e)
                continue

        logger.debug(
            "Batch %d/%d exhausted, %d candidates checked so far",
            batch_num + 1, max_batches, total_checked,
        )

    # If no perfect candidate found, use best fallback
    if best_fallback:
        logger.info(
            "No perfect random seed found after %d candidates; "
            "using best fallback: %s",
            total_checked, best_fallback,
        )
        return best_fallback

    logger.warning(
        "Failed to discover a quality random seed after %d candidates "
        "across %d batches",
        total_checked, max_batches,
    )
    return None


# ---------------------------------------------------------------------------
# Post-crawl graph quality check
# ---------------------------------------------------------------------------


def _check_graph_quality(
    wiki_graph: WikiGraph,
    difficulty: TrailDifficultyConfig,
) -> str | None:
    """Check if a crawled graph has enough diversity for trail building.

    Returns None if the graph passes, or an error reason string if it fails.
    This catches cases where the seed page looked promising but its
    neighborhood is too sparse or narrow to build a coherent trail.
    """
    all_pages = wiki_graph.pages()
    infobox_pages = wiki_graph.pages_with_infobox()

    # Need enough total pages for the planned route
    min_stops = difficulty.depth_range[0]
    if len(all_pages) < min_stops:
        return (
            f"too few pages in graph ({len(all_pages)}) "
            f"for depth range {difficulty.depth_range}"
        )

    # Need enough infobox pages for fact extraction on page stops
    # Rule of thumb: at least half the planned non-tool stops should have
    # infobox pages available (the planner strongly prefers them)
    min_infobox_pages = max(3, min_stops // 2)
    if len(infobox_pages) < min_infobox_pages:
        return (
            f"too few infobox pages ({len(infobox_pages)}, "
            f"need ≥{min_infobox_pages})"
        )

    # Check for topic diversity: if most pages share the exact same first
    # 2 words in their title, the graph is probably a narrow cluster
    # (e.g., "Municipality of X", "Barangay Y", "Parish of Z")
    if len(all_pages) >= 8:
        prefixes: dict[str, int] = {}
        for p in all_pages:
            prefix = " ".join(p.title.split()[:2]).lower()
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
        most_common_count = max(prefixes.values()) if prefixes else 0
        if most_common_count / len(all_pages) > 0.5:
            most_common = max(prefixes, key=lambda k: prefixes[k])
            return (
                f"narrow topic cluster: {most_common_count}/{len(all_pages)} "
                f"pages share prefix '{most_common}'"
            )

    # Need some numeric facts across the graph for compute stop
    numeric_count = 0
    for p in infobox_pages:
        for f in p.infobox:
            if f.numeric_value is not None or any(c.isdigit() for c in f.value):
                numeric_count += 1
                break
    if numeric_count < 2:
        return (
            f"too few pages with numeric infobox data ({numeric_count})"
        )

    return None


# ---------------------------------------------------------------------------
# Seed resolution
# ---------------------------------------------------------------------------


def _load_seed_urls_from_file(path: str) -> list[str]:
    """Load seed URLs from a text file (one per line, # comments allowed)."""
    urls = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    if not urls:
        raise ValueError(f"No URLs found in {path}")
    return urls


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


async def _generate_one_sample(
    sample_id: str,
    seed_url: str,
    difficulty: TrailDifficultyConfig,
    wiki_graph: WikiGraph,
    trail_builder: TrailBuilder,
    golden_executor: GoldenExecutor,
    verbalizer: TrailVerbalizer,
    args: argparse.Namespace,
    avoid_pages: set[str] | None = None,
) -> Trail | None:
    """Generate one trail sample. Returns trail JSON dict or None on failure."""

    # Build trail
    trail = await trail_builder.build_trail(
        seed_url=seed_url,
        difficulty=difficulty,
        avoid_pages=avoid_pages,
    )
    if trail is None:
        logger.warning("Failed to build trail for %s", sample_id)
        return None

    # Golden execution
    golden_result = await golden_executor.execute_trail(trail)
    if not golden_result.success:
        logger.warning(
            "Golden execution failed for trail %s: %s",
            trail.trail_id,
            [sr.error for sr in golden_result.stop_results if sr.error],
        )
        return None

    # Verbalization (riddle generation)
    if not args.skip_riddle:
        riddle = await verbalizer.verbalize_with_validation(trail)
        if riddle is None:
            logger.warning(
                "Validated riddle failed for %s, using unvalidated fallback",
                trail.trail_id,
            )
            riddle = await verbalizer.verbalize(trail)
            # Even for fallback riddles, verify the passcode matches
            if riddle:
                passcode_ok = await verbalizer.verify_passcode(trail, riddle)
                if not passcode_ok:
                    logger.warning(
                        "Fallback riddle for %s failed passcode verification — "
                        "rejecting trail",
                        trail.trail_id,
                    )
                    return None
        trail.riddle = riddle
        if riddle:
            logger.info("Generated riddle for %s (%d chars)", trail.trail_id, len(riddle))

    # Validation
    if not args.skip_validation:
        validation = await golden_executor.validate_trail(
            trail, num_trials=args.validation_trials,
        )
        if not validation.all_consistent:
            logger.warning(
                "Validation failed for trail %s: %s",
                trail.trail_id, validation.issues,
            )
            return None

    # Attach metadata
    all_pages = wiki_graph.pages()
    trail.metadata.update({
        "sample_id": sample_id,
        "pages_crawled": len(all_pages),
    })

    return trail


async def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the trail generation pipeline."""
    load_dotenv()

    # Initialize LLM client
    llm = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")

    # Initialize tool registry
    tool_registry = ToolRegistry()
    fetch_fn = await _make_fetch_fn(tool_registry)

    difficulty = DIFFICULTY_CONFIGS.get(args.difficulty)
    if difficulty is None:
        logger.error("Unknown difficulty level: %s", args.difficulty)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / ".wiki_cache"

    # Resolve seed URLs
    if args.random_seeds:
        seed_mode = "random"
        seed_urls = []  # Will be discovered per-sample
    elif args.seed_urls_file:
        seed_mode = "file"
        seed_urls = _load_seed_urls_from_file(args.seed_urls_file)
        logger.info("Loaded %d seed URLs from %s", len(seed_urls), args.seed_urls_file)
    else:
        seed_mode = "fixed"
        seed_urls = [args.seed_url]

    stats = {
        "samples_attempted": 0,
        "samples_generated": 0,
        "golden_failures": 0,
        "build_failures": 0,
        "seed_failures": 0,
    }

    # Shared crawl for fixed seed mode (all samples use same graph)
    shared_graph = None
    if seed_mode == "fixed":
        shared_graph = WikiGraph(cache_dir=cache_dir, fetch_fn=fetch_fn)
        logger.info(
            "Crawling from %s (depth=%d, max_pages=%d)...",
            seed_urls[0], difficulty.crawl_depth, args.max_pages,
        )
        crawl_start = time.monotonic()
        await shared_graph.crawl(
            seed_urls[0],
            max_depth=difficulty.crawl_depth,
            max_pages=args.max_pages,
            include_pageviews=not args.skip_pageviews,
        )
        logger.info(
            "Crawl complete in %.1fs: %d pages (%d with infobox)",
            time.monotonic() - crawl_start,
            len(shared_graph.pages()),
            len(shared_graph.pages_with_infobox()),
        )

    # Auto-detect starting index from existing files
    if args.start_index is not None:
        start_index = args.start_index
    else:
        existing = sorted(output_dir.glob("sample_*.json"))
        if existing:
            # Parse highest existing index and start after it
            last_name = existing[-1].stem  # e.g. "sample_042"
            try:
                start_index = int(last_name.split("_")[1]) + 1
            except (IndexError, ValueError):
                start_index = len(existing)
            logger.info(
                "Found %d existing samples, starting from sample_%03d",
                len(existing), start_index,
            )
        else:
            start_index = 0

    # Retry budget: scale with difficulty since harder trails have higher failure rates
    min_depth = difficulty.depth_range[0]
    if min_depth >= 17:
        retry_multiplier = 10
    elif min_depth >= 13:
        retry_multiplier = 5
    else:
        retry_multiplier = 3
    max_attempts = args.num_samples * retry_multiplier
    seed_idx = 0  # tracks round-robin position for file mode

    # Track pages used across samples to improve diversity
    used_pages: set[str] = set()

    # Pre-populate from existing samples in output dir
    for existing_file in output_dir.glob("sample_*.json"):
        try:
            with open(existing_file) as f:
                existing_trail = json.load(f)
            for stop in existing_trail.get("stops", []):
                page_url = stop.get("page_url")
                if page_url:
                    used_pages.add(page_url)
        except Exception:
            pass
    if used_pages:
        logger.info("Pre-loaded %d used pages from existing samples", len(used_pages))

    while stats["samples_generated"] < args.num_samples and stats["samples_attempted"] < max_attempts:
        sample_num = start_index + stats["samples_generated"]
        sample_id = f"sample_{sample_num:03d}"
        logger.info(
            "--- Generating %s (attempt %d/%d) ---",
            sample_id, stats["samples_attempted"] + 1, max_attempts,
        )
        stats["samples_attempted"] += 1

        # Resolve seed for this sample
        if seed_mode == "random":
            # Each random sample gets its own graph
            wiki_graph = WikiGraph(cache_dir=cache_dir, fetch_fn=fetch_fn)
            seed_url = await _discover_random_seed(fetch_fn, wiki_graph)
            if seed_url is None:
                logger.warning("Failed to find random seed for %s, skipping", sample_id)
                stats["seed_failures"] += 1
                continue

            # Crawl from the discovered seed
            logger.info("Crawling from %s (depth=%d, max_pages=%d)...",
                        seed_url, difficulty.crawl_depth, args.max_pages)
            crawl_start = time.monotonic()
            await wiki_graph.crawl(
                seed_url,
                max_depth=difficulty.crawl_depth,
                max_pages=args.max_pages,
                include_pageviews=not args.skip_pageviews,
            )
            logger.info(
                "Crawl complete in %.1fs: %d pages (%d with infobox)",
                time.monotonic() - crawl_start,
                len(wiki_graph.pages()),
                len(wiki_graph.pages_with_infobox()),
            )

            # Check graph quality before expensive LLM planning
            graph_issue = _check_graph_quality(wiki_graph, difficulty)
            if graph_issue:
                logger.warning(
                    "Graph from %s rejected: %s — retrying with new seed",
                    seed_url, graph_issue,
                )
                stats["seed_failures"] += 1
                continue

        elif seed_mode == "file":
            seed_url = seed_urls[seed_idx % len(seed_urls)]
            seed_idx += 1
            # Each distinct seed gets its own graph
            wiki_graph = WikiGraph(cache_dir=cache_dir, fetch_fn=fetch_fn)
            logger.info("Crawling from %s (depth=%d, max_pages=%d)...",
                        seed_url, difficulty.crawl_depth, args.max_pages)
            crawl_start = time.monotonic()
            await wiki_graph.crawl(
                seed_url,
                max_depth=difficulty.crawl_depth,
                max_pages=args.max_pages,
                include_pageviews=not args.skip_pageviews,
            )
            logger.info(
                "Crawl complete in %.1fs: %d pages (%d with infobox)",
                time.monotonic() - crawl_start,
                len(wiki_graph.pages()),
                len(wiki_graph.pages_with_infobox()),
            )

            # Check graph quality
            graph_issue = _check_graph_quality(wiki_graph, difficulty)
            if graph_issue:
                logger.warning(
                    "Graph from %s rejected: %s — skipping",
                    seed_url, graph_issue,
                )
                stats["seed_failures"] += 1
                continue

        else:  # fixed
            seed_url = seed_urls[0]
            wiki_graph = shared_graph

        # Build components for this graph
        fact_extractor = FactExtractor(llm_client=llm, model=model)
        trail_builder = TrailBuilder(
            wiki_graph=wiki_graph,
            fact_extractor=fact_extractor,
            llm_client=llm,
            model=model,
            tool_registry=tool_registry,
        )
        golden_executor = GoldenExecutor(tool_registry=tool_registry)
        verbalizer = TrailVerbalizer(llm_client=llm, model=model, wiki_graph=wiki_graph)

        trail = await _generate_one_sample(
            sample_id, seed_url, difficulty,
            wiki_graph, trail_builder, golden_executor, verbalizer,
            args, avoid_pages=used_pages if used_pages else None,
        )

        if trail is None:
            stats["build_failures"] += 1
            continue

        # Static validation before saving
        validation_result = validate_trail_static(trail)
        if not validation_result.is_valid:
            logger.warning(
                "Trail %s failed static validation: %s",
                trail.trail_id,
                [str(i) for i in validation_result.critical_issues],
            )
            stats["build_failures"] += 1
            continue
        if validation_result.warnings:
            logger.info(
                "Trail %s has %d validation warning(s)",
                trail.trail_id, len(validation_result.warnings),
            )

        # Optional: add diamond compositionality patterns
        if args.compositional:
            from trail.diamond_augmenter import DiamondAugmenter
            diamond_aug = DiamondAugmenter(
                tool_registry=tool_registry,
                verbalizer=verbalizer,
            )
            diamond_trail = await diamond_aug.augment(
                trail,
                num_diamonds=args.max_diamonds,
            )
            if diamond_trail is not None:
                trail = diamond_trail
                logger.info(
                    "Diamond augmentation: %d -> %d stops",
                    len(trail.stops) - 3, len(trail.stops),  # approximate
                )
            else:
                logger.warning(
                    "Diamond augmentation failed for %s; saving linear trail",
                    trail.trail_id,
                )

        # Save trail
        output_path = output_dir / f"{sample_id}.json"
        with open(output_path, "w") as f:
            f.write(trail_to_json(trail))
        logger.info(
            "Saved %s (passcode: %d, %d stops, seed: %s)",
            output_path.name, trail.passcode, len(trail.stops), seed_url,
        )
        stats["samples_generated"] += 1

        # Track pages from this trail for diversity in subsequent samples
        for stop in trail.stops:
            if stop.page_url:
                used_pages.add(stop.page_url)

    if stats["samples_generated"] < args.num_samples:
        logger.warning(
            "Only generated %d/%d samples after %d attempts",
            stats["samples_generated"], args.num_samples, stats["samples_attempted"],
        )

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("Generation complete:")
    logger.info("  Samples attempted:   %d", stats["samples_attempted"])
    logger.info("  Samples generated:   %d", stats["samples_generated"])
    logger.info("  Build/golden failures: %d", stats["build_failures"])
    if seed_mode == "random":
        logger.info("  Seed failures:       %d", stats["seed_failures"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate single-trail scavenger-hunt puzzles",
    )

    # Seed source (mutually exclusive)
    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument(
        "--seed-url",
        help="Wikipedia URL to start crawling from (single fixed seed)",
    )
    seed_group.add_argument(
        "--seed-urls-file",
        help="Text file with one seed URL per line (round-robin across samples)",
    )
    seed_group.add_argument(
        "--random-seeds",
        action="store_true",
        help="Discover a random Wikipedia article as seed for each sample",
    )

    parser.add_argument(
        "--difficulty",
        choices=list(DIFFICULTY_CONFIGS.keys()),
        default="medium",
        help="Difficulty level (default: medium)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of puzzle samples to generate (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/trail_puzzles",
        help="Output directory for generated puzzles (default: data/trail_puzzles)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=50,
        help="Maximum pages to crawl (default: 50)",
    )
    parser.add_argument(
        "--validation-trials",
        type=int,
        default=3,
        help="Number of golden execution trials for validation (default: 3)",
    )
    parser.add_argument(
        "--skip-riddle",
        action="store_true",
        help="Skip riddle generation (faster, no verbalization LLM calls)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip multi-trial validation (faster, less reliable)",
    )
    parser.add_argument(
        "--skip-pageviews",
        action="store_true",
        help="Skip Wikimedia pageview fetching (faster crawl)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Starting sample index (default: auto-detect from existing files)",
    )
    parser.add_argument(
        "--compositional",
        action="store_true",
        help="Add diamond (fork-merge) patterns after generation for compositional DAG structure",
    )
    parser.add_argument(
        "--max-diamonds",
        type=int,
        default=None,
        help="Max diamonds per trail when --compositional is used (default: auto by difficulty)",
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

    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
