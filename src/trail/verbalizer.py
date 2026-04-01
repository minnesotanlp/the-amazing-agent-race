"""Trail verbalizer: converts structured trails into QA-style puzzle riddles.

Instead of step-by-step instructions ("Visit page X, extract Y"), produces
a chain of cryptic clues where:
- Navigation clues are questions whose ANSWER is the next page to visit
- Extraction clues describe what to find obliquely, not by field name
- Tool clues are framed as natural curiosity, not tool invocations

Uses round-trip validation to ensure the riddle is solvable.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

from trail.models import Stop, Trail

if TYPE_CHECKING:
    from openai import OpenAI
    from trail.wiki_graph import WikiGraph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_VERBALIZE_SYSTEM = """\
You are a puzzle master crafting a cryptic scavenger-hunt riddle. You will \
receive a structured trail of stops through Wikipedia pages and tool lookups. \
Your job is to turn it into a chain of clues that is engaging and non-trivial \
to solve.

CRITICAL RULES:

NAVIGATION — NEVER use exact Wikipedia article titles (except the seed page \
in clue 1). Instead, describe each destination through a question, riddle, \
or indirect description so the solver must deduce which page to visit.
  GOOD: "Seek the engineer whose surname adorns this iron lattice."
  GOOD: "What peak did Tenzing and Hillary conquer in 1953? Go there."
  GOOD: "This tower's creator also designed a famous canal lock in Panama — \
who was he?"
  BAD:  "Visit the Wikipedia page for Gustave Eiffel."
  BAD:  "Go to the Mount Everest article."
  BAD:  "Find the article about Mount Everett." (names the exact title)
  BAD:  "Seek the Berkshire peak known as Mount Everett." (names exact title)

EXTRACTION — Never name the exact infobox field or section. Describe what to \
find indirectly, as a question that requires reading the page.
  GOOD: "How tall does this iron giant stand, in meters?"
  GOOD: "In what year did this visionary breathe his last?"
  GOOD: "How many souls call this land their home?"
  BAD:  "Find the height field in the infobox."
  BAD:  "Look at the population listed in the infobox."

TOOL LOOKUPS — Frame tool-based lookups as natural questions about the world, \
not as tool calls.
  GOOD: "How high above the sea does the ground sit beneath this peak?"
  GOOD: "If you drove from here to there, how many minutes would the journey take?"
  GOOD: "How many museums stand within sight of this landmark?"
  GOOD: "What star rating do travelers give this place?"
  GOOD: "How much snow blanketed this spot on the day it was founded?"
  BAD:  "Look up the elevation using a maps tool."
  BAD:  "Call the distance API between A and B."

REASONING TRANSFORMS — Some clues apply an analytical operation to a \
previously discovered value (e.g., "find the next prime", "count the digits", \
"what is the Scrabble score"). Frame these as engaging reasoning challenges, \
not as code to execute. The solver should understand what to compute.
  GOOD: "Before moving on, find the next prime number after the count you just found."
  GOOD: "Convert that year to a Roman numeral — how many characters does it take to write?"
  GOOD: "Take the population figure and sum its digits."
  GOOD: "How many vowels does the name you uncovered contain?"
  BAD:  "Run the digit_sum function on the value."
  BAD:  "Execute the next_prime algorithm."
  BAD:  "Apply a binary 1-bit count to the number."
Make it clear the reasoning step transforms the PREVIOUS clue's answer — \
the solver must apply the operation to get a new number for later use.

MERGE STEPS — Some clues combine results from two earlier clues. These are \
tool calls that merge two independently obtained values. Frame the combination \
naturally, referencing the two source clues by number:
  GOOD: "Add the elevation you found in clue 3 to the count of nearby parks from clue 4."
  GOOD: "Take the absolute difference between your results from clues 5 and 6."
  BAD:  "Compute merge(clue 3, clue 4)."
  BAD:  "Run python_execute_code to add 90 and 15." (leaks values)

FORMULA — State the final arithmetic explicitly, referencing earlier clues \
by their EXACT clue number, NEVER by raw numeric values. \
The solver must compute the formula using their own discovered values.
  GOOD: "Take the absolute difference between clue 2 and clue 4, modulo 10."
  GOOD: "Compute the digital root of (clue 1 + clue 3 + clue 5)."
  BAD:  "Compute abs(174) % 10." (leaks the value from clue 1)
  BAD:  "Compute abs(len(\"211\")) % 10." (leaks the extracted value)
  BAD:  "Take the number of characters in the value from clue 3." \
(character counting is forbidden — use only numeric values directly)

VOICE & VARIETY — Each clue must use a DIFFERENT sentence structure. Vary \
openings, moods, and rhetorical devices across the riddle. Do NOT let \
multiple clues start the same way.

  Openings to rotate among (use each at most once):
    • Imperative:      "Seek…", "Track down…", "Unearth…", "Pinpoint…"
    • Declarative:     "There exists a page where…", "A city once crowned \
the empire's heart — find it."
    • Interrogative:   "What fortress withstood a 53-day siege in 1453?"
    • Conditional:     "If you follow the link mentioning Ravenna, you'll \
land on a page about…"
    • Relative clause:  "The article whose opening line mentions syncopation \
holds your next answer."
    • Narrative:        "In 1453, a great city fell. Its Wikipedia entry \
records how many days the siege lasted."
    • Metaphor/persona: "The iron lady of Paris knows her own height — \
read it from her page."
    • Aside/parenthetical: "One detail hides in the contents sidebar \
(count the sub-headings under Governance)."

  AVOID these repetitive patterns:
    ✗ Starting every clue with "Begin at…" / "From that article…" / \
"On that page…"
    ✗ Using the same question form ("How many…") in consecutive clues
    ✗ Back-to-back imperative verbs ("Find…", "Follow…", "Seek…")
    ✗ Echoing the same transition phrase ("From there, move to…")

  AIM for a mix: at least 3 different sentence types across the riddle.

STRUCTURE:
- Number each clue sequentially: 1, 2, 3, etc.
- CRITICAL: Clue N corresponds EXACTLY to stop N in the trail data below. \
Stop 1 = Clue 1, Stop 2 = Clue 2, etc. Do NOT renumber, merge, or skip clues.
- The first clue should set the scene with the seed page (you may name it).
- Each subsequent clue should flow from the previous one.
- The final clue states the arithmetic and that the answer is a single digit (0-9).
- In the final clue, reference values ONLY by "clue N" where N matches the stop \
number. Use ONLY the numeric value from each referenced clue — never count \
characters, letters, or digits in a value.
- Do NOT reveal any actual values — the solver must discover them.
- Aim for {target_length} words total.

Output ONLY the riddle text, no preamble or commentary."""

_VERBALIZE_USER = """\
Trail to verbalize:

Theme: {theme}

{stops_context}

Final computation: {computation}
The computation uses values from these clues (1-based clue numbers): {referenced_clues}

IMPORTANT: Each stop above maps to a clue with the SAME number. \
Stop 1 = Clue 1, Stop 2 = Clue 2, etc. In the final clue, reference \
values using "clue N" where N matches the stop number shown above.

Write the riddle as a chain of numbered clues."""

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

_ROUNDTRIP_SYSTEM = """\
Read this scavenger-hunt riddle carefully. Extract the step-by-step plan a \
solver would need to follow to arrive at the final answer.

For each step, identify:
- page_or_topic: The Wikipedia page, place, or topic to investigate
- what_to_find: What specific information to extract or look up
- tool_or_method: What kind of lookup is needed (e.g., "read Wikipedia page", \
"elevation lookup", "weather data", "distance calculation", "place search", \
"country data", "arithmetic", etc.)
- result_used_for: How this step's result feeds into later steps or the final answer

Output a JSON array of step objects. Output ONLY the JSON array, no fencing."""


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------


def _find_link_text(
    wiki_graph: WikiGraph | None, from_url: str, to_url: str
) -> str | None:
    """Find the anchor text of a link from one page to another."""
    if wiki_graph is None:
        return None
    page = wiki_graph.get_page(from_url)
    if page is None:
        return None
    for link in page.outgoing_links:
        if link.target_url == to_url:
            return link.text
    return None


def _get_page_summary(wiki_graph: WikiGraph | None, url: str) -> str:
    """Get a short summary of a page for context."""
    if wiki_graph is None:
        return ""
    page = wiki_graph.get_page(url)
    if page is None:
        return ""
    return page.first_paragraph[:300] if page.first_paragraph else ""


def _build_stop_context(
    stop: Stop,
    index: int,
    next_stop: Stop | None,
    wiki_graph: WikiGraph | None,
) -> str:
    """Build rich context for one stop, including page info and connection to next."""
    clue_num = index + 1
    lines = [f"--- Stop {clue_num} / Clue {clue_num} (type: {stop.stop_type}) ---"]

    if stop.page_url:
        if wiki_graph:
            page = wiki_graph.get_page(stop.page_url)
            if page:
                lines.append(f"Page title: {page.title}")
                if page.first_paragraph:
                    lines.append(f"Page intro: {page.first_paragraph[:300]}")
            else:
                lines.append(f"Page URL: {stop.page_url}")
        else:
            lines.append(f"Page URL: {stop.page_url}")

    lines.append(f"What to extract: {stop.extraction_target}")
    lines.append(f"Value type: {stop.extracted_value_type}")
    if stop.extraction_section:
        lines.append(f"Found in section: {stop.extraction_section}")

    # Tool chain details
    if stop.stop_type == "tool":
        tool_names = [tc["tool_name"] for tc in stop.bridge.tool_chain]
        lines.append(f"Tools used: {', '.join(tool_names)}")
        # Include key arguments for context
        for tc in stop.bridge.tool_chain:
            args = tc.get("arguments", {})
            # Filter out __from_previous
            visible_args = {k: v for k, v in args.items() if not k.startswith("__from_previous")}
            if visible_args:
                lines.append(f"  {tc['tool_name']} args: {visible_args}")

    # Merge stop (diamond pattern): depends_on has 2+ entries
    if stop.depends_on and len(stop.depends_on) >= 2:
        dep_clues = [d + 1 for d in stop.depends_on]
        lines.append(f"MERGE STEP: combines values from clues {dep_clues}")
        lines.append(
            "Write a clue that tells the solver to combine their results "
            f"from clues {dep_clues[0]} and {dep_clues[1]}."
        )

    # Reason stop
    if stop.stop_type == "reason":
        source_clue = (stop.reason_source_stop or 0) + 1
        lines.append(f"Transform type: {stop.reason_transform}")
        lines.append(f"Transforms value from: Clue {source_clue}")
        lines.append(f"Hint for riddle: {stop.extraction_target}")

    # Compute stop
    if stop.stop_type == "compute":
        lines.append(f"Expression: {stop.bridge.expression}")
        lines.append(f"Expression code: {stop.bridge.expression_code}")
        ref_clues = [i + 1 for i in (stop.bridge.referenced_stops or [])]
        lines.append(f"Referenced clues (1-based): {ref_clues}")

    # Navigation to next stop
    if next_stop and stop.stop_type == "page":
        bridge = stop.bridge
        if bridge.bridge_type == "link_follow" and next_stop.page_url:
            link_text = _find_link_text(wiki_graph, stop.page_url or "", next_stop.page_url)
            next_summary = _get_page_summary(wiki_graph, next_stop.page_url)
            lines.append(f"Navigation to next: follow link")
            if link_text:
                lines.append(f"  Link anchor text on this page: \"{link_text}\"")
            if next_summary:
                lines.append(f"  Next page intro: {next_summary[:200]}")
        elif bridge.bridge_type == "search_query":
            lines.append(f"Navigation to next: search")
            if bridge.search_query:
                lines.append(f"  Search query: \"{bridge.search_query}\"")
            if next_stop.page_url:
                next_summary = _get_page_summary(wiki_graph, next_stop.page_url)
                if next_summary:
                    lines.append(f"  Next page intro: {next_summary[:200]}")

    return "\n".join(lines)


def _build_full_context(trail: Trail, wiki_graph: WikiGraph | None) -> str:
    """Build the full context string for all stops."""
    parts = []
    for i, stop in enumerate(trail.stops):
        next_stop = trail.stops[i + 1] if i + 1 < len(trail.stops) else None
        parts.append(_build_stop_context(stop, i, next_stop, wiki_graph))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# TrailVerbalizer
# ---------------------------------------------------------------------------


class TrailVerbalizer:
    """Converts trails into QA-style puzzle riddles."""

    def __init__(
        self,
        llm_client: OpenAI,
        model: str,
        wiki_graph: WikiGraph | None = None,
    ):
        self._llm = llm_client
        self._model = model
        self._wiki_graph = wiki_graph

    def _collect_forbidden_titles(self, trail: Trail) -> list[str]:
        """Collect Wikipedia article titles that must NOT appear in the riddle.

        Skip stop 0 (seed page is allowed to be named).
        """
        titles = []
        for stop in trail.stops:
            if stop.index == 0:
                continue
            if not stop.page_url:
                continue
            raw_title = stop.page_url.split("/wiki/")[-1]
            title = unquote(raw_title).replace("_", " ")
            if len(title) > 3:  # skip very short titles that may be common words
                titles.append(title)
        return titles

    async def verbalize(self, trail: Trail) -> str:
        """Convert a trail into a QA-style riddle."""
        stops_context = _build_full_context(trail, self._wiki_graph)

        # Determine target length based on difficulty
        num_stops = len(trail.stops)
        if num_stops <= 4:
            target_length = "100-200"
        elif num_stops <= 7:
            target_length = "200-350"
        else:
            target_length = "300-500"

        # Get computation details
        if trail.stops and trail.stops[-1].stop_type == "compute":
            comp_stop = trail.stops[-1]
            computation = comp_stop.bridge.expression or "See last stop"
            referenced_0based = comp_stop.bridge.referenced_stops or []
        else:
            computation = "No explicit computation stop"
            referenced_0based = []

        # Convert 0-based stop indices to 1-based clue numbers
        referenced_clues = [i + 1 for i in referenced_0based]

        theme = trail.metadata.get("theme", "")

        system_prompt = _VERBALIZE_SYSTEM.format(target_length=target_length)

        # Build forbidden titles list to inject into user prompt
        forbidden_titles = self._collect_forbidden_titles(trail)
        forbidden_block = ""
        if forbidden_titles:
            titles_list = "\n".join(f"  - \"{t}\"" for t in forbidden_titles)
            forbidden_block = (
                f"\n\nFORBIDDEN TITLES — these exact strings must NEVER appear "
                f"in the riddle (use indirect descriptions instead):\n{titles_list}"
            )

        user_msg = _VERBALIZE_USER.format(
            theme=theme,
            stops_context=stops_context,
            computation=computation,
            referenced_clues=referenced_clues,
        ) + forbidden_block

        try:
            response = self._llm.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
            )
            riddle = (response.choices[0].message.content or "").strip()
            return riddle
        except Exception:
            logger.error(
                "Verbalization failed for trail %s", trail.trail_id, exc_info=True
            )
            return ""

    def _detect_direct_titles(self, trail: Trail, riddle: str) -> list[tuple[int, str, str]]:
        """Detect stops whose exact Wikipedia title appears in the riddle.

        Returns list of (stop_index, title, page_url) for offending stops.
        Skips stop 0 (seed page is allowed to be named).
        """
        leaked = []
        for stop in trail.stops:
            if stop.index == 0:
                continue  # seed page may be named
            if not stop.page_url:
                continue
            # Extract title from URL
            raw_title = stop.page_url.split("/wiki/")[-1]
            title = unquote(raw_title).replace("_", " ")
            # Check for exact match (case-insensitive)
            if re.search(re.escape(title), riddle, re.IGNORECASE):
                leaked.append((stop.index, title, stop.page_url))
        return leaked

    async def _correct_direct_titles(
        self, trail: Trail, riddle: str, leaked: list[tuple[int, str, str]]
    ) -> str:
        """Ask LLM to rewrite clues that directly name Wikipedia titles."""
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

        try:
            response = self._llm.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _CORRECTION_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
            )
            corrected = (response.choices[0].message.content or "").strip()
            if corrected:
                return corrected
        except Exception:
            logger.warning(
                "Title correction failed for trail %s", trail.trail_id,
                exc_info=True,
            )
        return riddle  # fallback to original

    async def verbalize_with_validation(
        self,
        trail: Trail,
        *,
        max_attempts: int = 5,
        min_alignment: float = 0.7,
    ) -> str | None:
        """Verbalize and validate via round-trip. Return None if all attempts fail.

        If passcode verification consistently yields the same wrong digit across
        multiple high-alignment riddles, we adopt the riddle-implied passcode
        (the riddle IS what the solver sees, so its implied answer is correct).
        """
        # Track (riddle, alignment, implied_passcode) for passcode-adoption fallback
        good_riddles: list[tuple[str, float, int]] = []

        for attempt in range(max_attempts):
            riddle = await self.verbalize(trail)
            if not riddle:
                continue

            # Correct any direct title references (retry up to 3 times)
            leaked = self._detect_direct_titles(trail, riddle)
            correction_failed = False
            for corr_attempt in range(3):
                if not leaked:
                    break
                titles = [t for _, t, _ in leaked]
                logger.info(
                    "Trail %s attempt %d, correction %d: fixing %d direct title(s): %s",
                    trail.trail_id, attempt + 1, corr_attempt + 1, len(leaked), titles,
                )
                riddle = await self._correct_direct_titles(trail, riddle, leaked)
                leaked = self._detect_direct_titles(trail, riddle)

            if leaked:
                logger.warning(
                    "Trail %s attempt %d: %d title(s) still leak after 3 corrections, "
                    "rejecting this attempt: %s",
                    trail.trail_id, attempt + 1, len(leaked),
                    [t for _, t, _ in leaked],
                )
                correction_failed = True

            if correction_failed:
                continue  # try a fresh verbalization

            alignment = await self._check_roundtrip(trail, riddle)
            logger.info(
                "Trail %s verbalization attempt %d: alignment=%.2f",
                trail.trail_id,
                attempt + 1,
                alignment,
            )

            if alignment < min_alignment:
                continue

            # Verify the riddle's computation matches the golden passcode
            implied = await self._compute_implied_passcode(trail, riddle)
            if implied is not None and implied == trail.passcode:
                logger.info(
                    "Passcode verification PASSED for %s (passcode=%d)",
                    trail.trail_id, trail.passcode,
                )
                return riddle

            if implied is not None:
                logger.warning(
                    "Passcode verification FAILED for %s: "
                    "riddle implies %d but golden says %d",
                    trail.trail_id, implied, trail.passcode,
                )
                good_riddles.append((riddle, alignment, implied))
            else:
                logger.warning(
                    "Trail %s attempt %d: alignment ok but passcode verification failed",
                    trail.trail_id, attempt + 1,
                )

        # Passcode adoption: if ≥2 high-alignment riddles consistently imply the
        # same alternative passcode, adopt it — the riddle defines the puzzle.
        if len(good_riddles) >= 2:
            from collections import Counter
            implied_counts = Counter(p for _, _, p in good_riddles)
            most_common_passcode, count = implied_counts.most_common(1)[0]
            if count >= 2:
                # Pick the highest-alignment riddle with this passcode
                best_riddle = max(
                    ((r, a) for r, a, p in good_riddles if p == most_common_passcode),
                    key=lambda x: x[1],
                )[0]
                logger.info(
                    "Trail %s: adopting riddle-implied passcode %d "
                    "(was %d, %d/%d riddles agreed)",
                    trail.trail_id, most_common_passcode,
                    trail.passcode, count, len(good_riddles),
                )
                trail.passcode = most_common_passcode
                return best_riddle

        logger.warning(
            "Trail %s: failed to produce verified riddle after %d attempts",
            trail.trail_id,
            max_attempts,
        )
        return None

    async def verify_passcode(self, trail: Trail, riddle: str) -> bool:
        """Verify that the riddle's final computation matches the golden passcode."""
        implied = await self._compute_implied_passcode(trail, riddle)
        if implied is not None and implied == trail.passcode:
            logger.info(
                "Passcode verification PASSED for %s (passcode=%d)",
                trail.trail_id, trail.passcode,
            )
            return True
        if implied is not None:
            logger.warning(
                "Passcode verification FAILED for %s: "
                "riddle implies %d but golden says %d",
                trail.trail_id, implied, trail.passcode,
            )
        return False

    async def _compute_implied_passcode(
        self, trail: Trail, riddle: str
    ) -> int | None:
        """Extract the single-digit answer the riddle implies.

        Returns the digit (0-9) the verification LLM computes, or None on error.
        """
        if not riddle or not trail.stops:
            return None

        # Build a clue-number → value mapping for all non-compute stops
        value_map_lines = []
        for stop in trail.stops:
            if stop.stop_type == "compute":
                continue
            clue_num = stop.index + 1
            val = stop.extracted_value
            vtype = stop.extracted_value_type or "unknown"
            value_map_lines.append(
                f"  Clue {clue_num}: value = {val!r} (type: {vtype})"
            )
        value_map = "\n".join(value_map_lines)

        verify_prompt = (
            "You are verifying a scavenger-hunt puzzle. Below is the riddle "
            "and the actual values discovered at each clue.\n\n"
            f"RIDDLE:\n{riddle}\n\n"
            f"DISCOVERED VALUES:\n{value_map}\n\n"
            "TASK: Read the FINAL clue's arithmetic instructions carefully. "
            "Using ONLY the discovered values above (matched by clue number), "
            "compute the final single-digit answer (0-9).\n\n"
            "IMPORTANT RULES:\n"
            "- Use the NUMERIC value from each referenced clue directly\n"
            "- Do NOT count characters, letters, or digits in any value\n"
            "- 'modulo 10' means: take the remainder when dividing by 10\n"
            "- 'digital root' means: sum all digits repeatedly until single digit\n"
            "- 'absolute difference' means: |a - b| (always positive)\n"
            "- Show your arithmetic step by step, then give the final digit\n\n"
            "Output your work, then on the LAST line output ONLY a single digit (0-9)."
        )

        try:
            response = self._llm.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "user", "content": verify_prompt},
                ],
            )
            raw = (response.choices[0].message.content or "").strip()
            # Extract the final digit (last line should be the answer)
            last_line = raw.strip().split("\n")[-1].strip()
            digits_last = [c for c in last_line if c.isdigit()]
            if digits_last:
                return int(digits_last[-1])
            # Fallback: last digit in entire response
            all_digits = [c for c in raw if c.isdigit()]
            if not all_digits:
                logger.warning(
                    "Passcode verification returned no digit: %r", raw
                )
                return None
            return int(all_digits[-1])
        except Exception:
            logger.warning(
                "Passcode verification error for %s",
                trail.trail_id, exc_info=True,
            )
            return None

    async def _check_roundtrip(self, trail: Trail, riddle: str) -> float:
        """Check alignment between a riddle and the original trail."""
        try:
            response = self._llm.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _ROUNDTRIP_SYSTEM},
                    {"role": "user", "content": riddle},
                ],
            )
            raw = (response.choices[0].message.content or "").strip()
        except Exception:
            logger.warning("Round-trip check failed", exc_info=True)
            return 0.0

        steps = _parse_roundtrip_steps(raw)
        if not steps:
            return 0.0

        return _compute_alignment(trail, steps)


# ---------------------------------------------------------------------------
# Round-trip parsing & alignment
# ---------------------------------------------------------------------------


def _parse_roundtrip_steps(raw: str) -> list[dict[str, Any]]:
    """Parse the LLM's round-trip extraction output."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return []


# Tool type keyword mapping for alignment checking
_TOOL_TYPE_KEYWORDS: dict[str, list[str]] = {
    "elevation": ["elevation", "altitude", "height above sea"],
    "weather": ["weather", "temperature", "precipitation", "rain"],
    "snowfall": ["snow", "snowfall"],
    "sunshine": ["sunshine", "sunlight", "daylight duration"],
    "distance": ["distance", "how far"],
    "directions": ["driving time", "duration", "travel time", "drive"],
    "places": ["nearby", "poi", "how many.*near", "count.*near", "museums", "restaurants", "parks"],
    "rating": ["rating", "star", "review"],
    "population": ["population", "people", "inhabitants"],
    "area": ["area", "square kilometer", "km²", "sq km", "land area"],
    "conversion": ["convert", "conversion", "unit"],
    "computation": ["arithmetic", "comput", "calculat", "days between", "date", "formula", "modulo", "digit"],
    "reasoning": ["transform", "digital root", "prime", "divisor", "scrabble",
                   "vowel", "binary", "roman numeral", "leap year",
                   "day of week", "reverse", "alphabetical position", "digit sum"],
}


def _extract_tool_types_from_trail(trail: Trail) -> set[str]:
    """Identify tool types used in the trail."""
    types: set[str] = set()
    for s in trail.stops:
        if s.stop_type == "tool":
            for tc in s.bridge.tool_chain:
                name = tc.get("tool_name", "")
                if "elevation" in name:
                    types.add("elevation")
                elif "weather" in name:
                    args = tc.get("arguments", {})
                    select = args.get("select", [])
                    if any("snowfall" in str(sel) for sel in select):
                        types.add("snowfall")
                    elif any("sunshine" in str(sel) for sel in select):
                        types.add("sunshine")
                    elif any("precipitation" in str(sel) for sel in select):
                        types.add("weather")
                    else:
                        types.add("weather")
                elif "distance" in name:
                    types.add("distance")
                elif "directions" in name:
                    types.add("directions")
                elif "search_places" in name:
                    output_key = tc.get("output_key", "")
                    if "rating" in output_key:
                        types.add("rating")
                    else:
                        types.add("places")
                elif "countries_population" in name:
                    types.add("population")
                elif "countries_area" in name:
                    types.add("area")
                elif "python" in name:
                    args = tc.get("arguments", {})
                    code = str(args.get("code", ""))
                    if "convert" in code.lower() or "* " in code:
                        types.add("conversion")
                    else:
                        types.add("computation")
        elif s.stop_type == "reason":
            types.add("reasoning")
        elif s.stop_type == "compute":
            types.add("computation")
    return types


def _extract_tool_types_from_steps(steps: list[dict[str, Any]]) -> set[str]:
    """Identify tool types mentioned in round-trip extracted steps."""
    types: set[str] = set()
    for step in steps:
        tool = str(step.get("tool_or_method", "")).lower()
        extraction = str(step.get("what_to_find", "")).lower()
        combined = tool + " " + extraction

        for type_name, keywords in _TOOL_TYPE_KEYWORDS.items():
            if any(kw in combined for kw in keywords):
                types.add(type_name)

    return types


def _compute_alignment(trail: Trail, steps: list[dict[str, Any]]) -> float:
    """Compute alignment score between a trail and extracted round-trip steps.

    Checks:
    - Number of steps (within ±2)
    - Topic/page references
    - Tool types implied
    - Final formula presence
    """
    score = 0.0
    total = 4.0

    # 1. Step count alignment
    trail_steps = len(trail.stops)
    extracted_steps = len(steps)
    if abs(trail_steps - extracted_steps) <= 2:
        score += 1.0
    elif abs(trail_steps - extracted_steps) <= 4:
        score += 0.5

    # 2. Topic references (relaxed — check for page titles or topic keywords,
    #    not exact URLs, since QA clues won't name pages directly)
    trail_topics: set[str] = set()
    for s in trail.stops:
        if s.page_url:
            # Extract page title from URL
            title = s.page_url.split("/wiki/")[-1].replace("_", " ").lower()
            trail_topics.add(title)

    extracted_topics: set[str] = set()
    for step in steps:
        topic = str(step.get("page_or_topic", "")).lower()
        if topic:
            extracted_topics.add(topic)

    if trail_topics and extracted_topics:
        # Check how many trail topics are mentioned (substring match)
        matches = 0
        for tt in trail_topics:
            for et in extracted_topics:
                if tt in et or et in tt or any(w in et for w in tt.split() if len(w) > 3):
                    matches += 1
                    break
        score += min(1.0, matches / max(len(trail_topics), 1))
    elif not trail_topics:
        score += 1.0

    # 3. Tool type alignment
    trail_tool_types = _extract_tool_types_from_trail(trail)
    extracted_tool_types = _extract_tool_types_from_steps(steps)

    if trail_tool_types:
        overlap = len(trail_tool_types & extracted_tool_types) / len(trail_tool_types)
        score += overlap
    else:
        score += 1.0

    # 4. Final formula presence
    has_formula = any(
        any(
            kw in str(step.get("tool_or_method", "")).lower()
            or kw in str(step.get("what_to_find", "")).lower()
            or kw in str(step.get("result_used_for", "")).lower()
            for kw in ["arithmetic", "comput", "mod", "digit", "formula", "passcode", "final"]
        )
        for step in steps
    )
    if has_formula:
        score += 1.0

    return score / total
