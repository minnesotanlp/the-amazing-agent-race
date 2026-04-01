"""Wikipedia knowledge graph: BFS crawler, infobox parser, obscurity scorer.

Crawls Wikipedia pages starting from a seed URL, extracts structured metadata
(infobox fields, outgoing links, section headings), and caches results to disk.
Uses the existing fetch_webpage MCP tool (via a user-supplied fetch function)
so the content seen by the crawler matches what agents see during evaluation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence
from urllib.parse import quote, unquote, urlparse

from trail.models import InfoboxField, PageInfo, WikiLink

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WIKI_PREFIX = "https://en.wikipedia.org/wiki/"
_EXCLUDED_NAMESPACES = {
    "Special:",
    "File:",
    "Category:",
    "Talk:",
    "Help:",
    "Wikipedia:",
    "Template:",
    "Portal:",
    "Module:",
    "Draft:",
    "User:",
    "MediaWiki:",
}
_BORING_LINK_PATTERNS = re.compile(
    r"^\d{3,4}$|^\d{1,2}(st|nd|rd|th)_century$|^List_of_|^Index_of_",
    re.IGNORECASE,
)

# Extremely generic pages that are linked from everywhere but make poor
# trail stops — no specific facts, no useful infobox data.
_SKIP_CRAWL_TITLES = frozenset({
    "Main Page", "Wikipedia", "English language", "United States",
    "United Kingdom", "Mathematics", "Physics", "Computer science",
    "Science", "History", "Geography", "Philosophy", "Religion",
    "Music", "Art", "Literature", "Education", "Technology",
    "Engineering", "Medicine", "Biology", "Chemistry",
    "Economics", "Sociology", "Psychology", "Politics",
    "Europe", "Asia", "Africa", "North America", "South America",
    "Latin", "Greek language", "French language", "German language",
    "Spanish language", "Chinese language", "Japanese language",
    "Pi", "Geometry", "Trigonometry", "Algebra",
    "Free content", "Encyclopedia", "Internet",
    "World War I", "World War II",
})

# Sections whose links we deprioritize (navigational, not content-rich)
_DEPRIORITIZED_SECTIONS = {
    "see also",
    "references",
    "external links",
    "further reading",
    "notes",
    "bibliography",
    "sources",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _title_from_url(url: str) -> str:
    """Extract the article title from a Wikipedia URL."""
    path = urlparse(url).path
    if path.startswith("/wiki/"):
        return unquote(path[len("/wiki/") :]).replace("_", " ")
    return url


def _normalize_wiki_url(url: str) -> str:
    """Normalize a Wikipedia URL to a canonical form."""
    parsed = urlparse(url)
    if parsed.fragment:
        url = url.split("#")[0]
    return url


def _is_wikipedia_article_url(url: str) -> bool:
    """Check if a URL points to an English Wikipedia article (not a namespace page)."""
    if not url.startswith(_WIKI_PREFIX):
        return False
    title = url[len(_WIKI_PREFIX) :]
    for ns in _EXCLUDED_NAMESPACES:
        if title.startswith(ns):
            return False
    return True


def _is_boring_link(title: str) -> bool:
    """Check if a link target is a year, decade, or overly broad category."""
    cleaned = title.replace(" ", "_")
    return bool(_BORING_LINK_PATTERNS.match(cleaned))


# ---------------------------------------------------------------------------
# Infobox parsing
# ---------------------------------------------------------------------------


def _fix_coords_in_key(key: str, value: str) -> tuple[str, str]:
    """Fix cases where coordinate data leaks into the infobox key.

    Wikipedia renders coordinates as markdown links inside table cells,
    which causes the entire coordinate text (including geohack URL) to
    be captured as the key instead of the value.
    """
    if "geohack.toolforge.org" not in key and not re.search(
        r"\d+°\d+[′']\d+[″\"]", key
    ):
        return key, value

    # Extract decimal lat/lon from the key text
    coord_match = re.search(r"(-?\d+\.\d+)[;,]\s*(-?\d+\.\d+)", key)
    if coord_match:
        value = f"{coord_match.group(1)}, {coord_match.group(2)}"
        # Clean up the key to just the label (e.g., "Coordinates")
        key = key.split(":")[0].strip()
        return key, value

    return key, value


def _strip_wiki_chrome(markdown: str) -> str:
    """Strip Wikipedia navigation chrome (sidebar, TOC, language links).

    Wikipedia pages fetched via markdownify include thousands of characters
    of navigation, sidebar, and table-of-contents content before the actual
    article.  We skip to the article body so that infobox parsing can work
    within a reasonable search window.
    """
    # Best marker: "From Wikipedia, the free encyclopedia" appears right
    # before the article body on virtually all Wikipedia pages.
    marker = "From Wikipedia, the free encyclopedia"
    idx = markdown.find(marker)
    if idx >= 0:
        return markdown[idx + len(marker) :]

    # Fallback: first H1 heading (article title)
    h1_match = re.search(r"^# .+$", markdown, re.MULTILINE)
    if h1_match:
        return markdown[h1_match.start() :]

    return markdown


def parse_infobox(markdown: str) -> list[InfoboxField]:
    """Extract key-value pairs from a Wikipedia infobox rendered as markdown.

    Wikipedia infoboxes typically render as a table or a series of bold key / value
    lines near the top of the article.  We attempt several heuristic patterns.
    """
    # Strip navigation chrome so the search window covers actual article content
    article = _strip_wiki_chrome(markdown)
    search_window = article[:12000]

    fields: list[InfoboxField] = []
    seen_keys: set[str] = set()

    # Pattern 1: Markdown table rows  "| Key | Value |"
    table_row_re = re.compile(
        r"^\|\s*\*?\*?([^|*]+?)\*?\*?\s*\|\s*(.+?)\s*\|?\s*$", re.MULTILINE
    )
    for match in table_row_re.finditer(search_window):
        key = match.group(1).strip().strip("*").strip()
        value = match.group(2).strip()
        if not key or not value or key.startswith("---"):
            continue
        key, value = _fix_coords_in_key(key, value)
        # Skip fields with empty or pipe-only values
        cleaned_val = value.strip().strip("|").strip()
        if not cleaned_val:
            continue
        key_lower = key.lower()
        if key_lower in seen_keys:
            continue
        seen_keys.add(key_lower)
        numeric = _try_parse_numeric(value)
        fields.append(InfoboxField(key=key, value=value, numeric_value=numeric))

    # Pattern 2: Bold key followed by value on same line  "**Key**: Value" or
    # "**Key** Value"
    bold_kv_re = re.compile(
        r"\*\*([^*]+?)\*\*\s*:?\s+(.+?)(?:\n|$)", re.MULTILINE
    )
    for match in bold_kv_re.finditer(search_window):
        key = match.group(1).strip()
        value = match.group(2).strip()
        if not key or not value:
            continue
        key, value = _fix_coords_in_key(key, value)
        cleaned_val = value.strip().strip("|").strip()
        if not cleaned_val:
            continue
        key_lower = key.lower()
        if key_lower in seen_keys:
            continue
        seen_keys.add(key_lower)
        numeric = _try_parse_numeric(value)
        fields.append(InfoboxField(key=key, value=value, numeric_value=numeric))

    return fields


def _try_parse_numeric(text: str) -> float | None:
    """Attempt to parse a numeric value from a string like '2,240 m' or '8,849'."""
    # Remove markdown links
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Try to find a number with optional commas and decimal
    match = re.search(r"[-+]?[\d,]+\.?\d*", cleaned)
    if not match:
        return None
    num_str = match.group(0).replace(",", "")
    try:
        val = float(num_str)
        return int(val) if val == int(val) else val
    except (ValueError, OverflowError):
        return None


# ---------------------------------------------------------------------------
# Link extraction
# ---------------------------------------------------------------------------


def extract_wiki_links(markdown: str) -> list[WikiLink]:
    """Extract outgoing Wikipedia links from markdown content."""
    links: list[WikiLink] = []
    seen_urls: set[str] = set()

    current_section: str | None = None
    heading_re = re.compile(r"^#{1,6}\s+(.+)", re.MULTILINE)
    link_re = re.compile(
        r"\[([^\]]+)\]\((/wiki/(?:[^()\s\"#]|\([^()]*\))+)(?:#[^\s)\"]*)?(?:\s[^)]*)?\)"
    )

    # Build a section map by character position
    section_starts: list[tuple[int, str]] = []
    for m in heading_re.finditer(markdown):
        section_starts.append((m.start(), m.group(1).strip()))

    for m in link_re.finditer(markdown):
        anchor_text = m.group(1).strip()
        path = m.group(2)
        url = f"https://en.wikipedia.org{path}"
        url = _normalize_wiki_url(url)

        if not _is_wikipedia_article_url(url):
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)

        title = _title_from_url(url)
        if _is_boring_link(title):
            continue

        # Determine the section this link is in
        section = None
        for start_pos, sec_name in reversed(section_starts):
            if m.start() >= start_pos:
                section = sec_name
                break

        links.append(
            WikiLink(
                text=anchor_text,
                target_title=title,
                target_url=url,
                section=section,
            )
        )

    return links


def extract_sections(markdown: str) -> list[str]:
    """Extract section headings from markdown."""
    heading_re = re.compile(r"^#{1,6}\s+(.+)", re.MULTILINE)
    return [m.group(1).strip() for m in heading_re.finditer(markdown)]


# Short boilerplate lines that appear in fetched Wikipedia markdown before
# the actual article content.  Any line whose stripped text matches one of
# these (case-insensitive) is skipped.
_BOILERPLATE_LINES = frozenset({
    "main menu", "move to sidebar", "hide", "navigation", "search",
    "personal tools", "contents", "toggle the table of contents",
    "article", "talk", "from wikipedia, the free encyclopedia",
    "appearance", "general", "not logged in",
})

# Substrings that indicate a navigation / boilerplate line.
_BOILERPLATE_SUBSTRINGS = (
    "Jump to content",
    "- Wikipedia",
    "[Jump to",
    "From Wikipedia",
    "Toggle the table of contents",
    "move to sidebar",
    "Main menu",
    "Toggle ",
    "Create account",
)


def extract_first_paragraph(markdown: str) -> str:
    """Extract the first substantial paragraph from a Wikipedia article."""
    # Strip navigation chrome so we start from actual article content
    article = _strip_wiki_chrome(markdown)
    lines = article.split("\n")
    paragraph_lines: list[str] = []
    started = False

    for line in lines:
        stripped = line.strip()
        # Skip headings, blank lines, images, tables at the start
        if not started:
            if (
                not stripped
                or stripped.startswith("#")
                or stripped.startswith("|")
                or stripped.startswith("![")
                or stripped.startswith("---")
            ):
                continue
            # Skip Wikipedia boilerplate / navigation lines
            if stripped.lower() in _BOILERPLATE_LINES:
                continue
            if any(sub in stripped for sub in _BOILERPLATE_SUBSTRINGS):
                continue
            # Skip navigation menu lists (e.g., "* [Main page]...")
            if stripped.startswith("* [") or stripped.startswith("- ["):
                continue
            # Skip very short non-content lines (nav fragments like "hide",
            # single-word menu items, breadcrumb pieces)
            if len(stripped) < 20 and not stripped.startswith("**"):
                continue
            started = True

        if started:
            if not stripped:
                if paragraph_lines:
                    break
                continue
            if stripped.startswith("#"):
                break
            paragraph_lines.append(stripped)

    return " ".join(paragraph_lines)[:2000]


# ---------------------------------------------------------------------------
# Pageview fetching
# ---------------------------------------------------------------------------


async def fetch_pageviews(title: str, *, days: int = 30) -> int | None:
    """Fetch monthly pageview count from the Wikimedia REST API.

    Returns None on any failure (missing httpx, rate limit, network error, etc.).
    """
    try:
        import httpx as _httpx
    except ImportError:
        logger.debug("httpx not installed; skipping pageview fetch")
        return None

    encoded_title = quote(title.replace(" ", "_"), safe="")
    end = datetime.now(timezone.utc)
    start = datetime(end.year, end.month, 1, tzinfo=timezone.utc)
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")

    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"en.wikipedia/all-access/all-agents/{encoded_title}/daily/{start_str}/{end_str}"
    )

    try:
        async with _httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                url,
                headers={"User-Agent": "AARBot/1.0 (research benchmark)"},
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            items = data.get("items", [])
            return sum(item.get("views", 0) for item in items)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# PageInfo construction
# ---------------------------------------------------------------------------


def build_page_info(url: str, markdown: str, pageviews: int | None = None) -> PageInfo:
    """Build a PageInfo from a fetched markdown page."""
    title = _title_from_url(url)
    infobox = parse_infobox(markdown)
    outgoing_links = extract_wiki_links(markdown)
    sections = extract_sections(markdown)
    first_paragraph = extract_first_paragraph(markdown)

    return PageInfo(
        url=url,
        title=title,
        infobox=infobox,
        outgoing_links=outgoing_links,
        sections=sections,
        first_paragraph=first_paragraph,
        page_length=len(markdown),
        pageviews=pageviews,
        fetch_timestamp=datetime.now(timezone.utc).isoformat(),
        raw_markdown=markdown,
    )


# ---------------------------------------------------------------------------
# WikiGraph
# ---------------------------------------------------------------------------


class WikiGraph:
    """Crawls and caches Wikipedia pages as a local knowledge graph."""

    def __init__(
        self,
        cache_dir: Path,
        fetch_fn: Callable[[str], Awaitable[str]],
    ):
        """
        Args:
            cache_dir: Directory to persist crawled PageInfo as JSON.
            fetch_fn: Async function that fetches a URL and returns markdown.
                      In practice, wraps fetch_webpage from the existing MCP tool.
        """
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._fetch_fn = fetch_fn
        self._pages: dict[str, PageInfo] = {}  # url -> PageInfo
        self._load_cache()

    def _load_cache(self) -> None:
        """Load any previously cached pages from disk."""
        index_path = self._cache_dir / "_index.json"
        if not index_path.exists():
            return
        try:
            with open(index_path) as f:
                index = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        for url, filename in index.get("pages", {}).items():
            page_path = self._cache_dir / filename
            if not page_path.exists():
                continue
            try:
                with open(page_path) as f:
                    data = json.load(f)
                page = _page_info_from_dict(data)
                self._pages[url] = page
            except (json.JSONDecodeError, OSError, KeyError):
                continue

        logger.info("Loaded %d cached pages from %s", len(self._pages), self._cache_dir)

    def _save_page(self, page: PageInfo) -> None:
        """Save a single PageInfo to the cache directory."""
        filename = f"{_url_hash(page.url)}.json"
        page_path = self._cache_dir / filename
        with open(page_path, "w") as f:
            json.dump(_page_info_to_dict(page), f, ensure_ascii=False, indent=2)

    def _save_index(self) -> None:
        """Save the URL-to-filename index."""
        index = {
            "pages": {url: f"{_url_hash(url)}.json" for url in self._pages},
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "page_count": len(self._pages),
        }
        with open(self._cache_dir / "_index.json", "w") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    async def crawl(
        self,
        seed_url: str,
        *,
        max_depth: int = 2,
        max_pages: int = 50,
        include_pageviews: bool = True,
    ) -> None:
        """BFS crawl from seed_url, following Wikipedia links up to max_depth hops.

        Pages already in the cache are reused unless force_refresh is needed.
        """
        seed_url = _normalize_wiki_url(seed_url)
        queue: deque[tuple[str, int]] = deque()  # (url, depth)
        visited: set[str] = set()

        queue.append((seed_url, 0))
        visited.add(seed_url)
        pages_fetched = 0

        while queue and pages_fetched < max_pages:
            url, depth = queue.popleft()

            if url not in self._pages:
                try:
                    logger.info("Fetching [depth=%d] %s", depth, url)
                    markdown = await self._fetch_fn(url)
                    pv = None
                    if include_pageviews:
                        title = _title_from_url(url)
                        pv = await fetch_pageviews(title)
                    page = build_page_info(url, markdown, pageviews=pv)
                    self._pages[url] = page
                    self._save_page(page)
                    pages_fetched += 1
                except Exception:
                    logger.warning("Failed to fetch %s, skipping", url, exc_info=True)
                    continue
            else:
                page = self._pages[url]

            # Enqueue neighbors if within depth
            if depth < max_depth:
                for link in page.outgoing_links:
                    link_url = _normalize_wiki_url(link.target_url)
                    if link_url in visited:
                        continue
                    if not _is_wikipedia_article_url(link_url):
                        continue
                    # Skip links from deprioritized sections (but don't hard-block)
                    section_lower = (link.section or "").lower()
                    if section_lower in _DEPRIORITIZED_SECTIONS:
                        continue
                    # Skip overly generic / navigational pages
                    link_title = _title_from_url(link_url)
                    if link_title in _SKIP_CRAWL_TITLES:
                        continue
                    if _is_boring_link(link_title):
                        continue
                    visited.add(link_url)
                    queue.append((link_url, depth + 1))

        self._save_index()
        logger.info(
            "Crawl complete: %d pages total (%d new), max_depth=%d",
            len(self._pages),
            pages_fetched,
            max_depth,
        )

    def get_page(self, url: str) -> PageInfo | None:
        """Return cached PageInfo for a URL, or None if not crawled."""
        return self._pages.get(_normalize_wiki_url(url))

    def pages(self) -> list[PageInfo]:
        """Return all crawled pages."""
        return list(self._pages.values())

    def neighbors(self, url: str) -> list[PageInfo]:
        """Return PageInfo for all pages linked from the given URL that are in the graph."""
        page = self.get_page(url)
        if not page:
            return []
        result = []
        for link in page.outgoing_links:
            neighbor = self.get_page(link.target_url)
            if neighbor:
                result.append(neighbor)
        return result

    def obscurity_score(self, url: str) -> float:
        """0.0 = very famous page, 1.0 = very obscure. Based on pageview count.

        Falls back to inverse page_length if pageviews unavailable.
        """
        page = self.get_page(url)
        if not page:
            return 0.5

        if page.pageviews is not None:
            # Typical Wikipedia pageview ranges:
            # Famous: 100k+/month, Medium: 1k-10k, Obscure: <500
            if page.pageviews <= 0:
                return 1.0
            import math

            score = 1.0 - min(1.0, math.log10(max(1, page.pageviews)) / 6.0)
            return max(0.0, min(1.0, score))

        # Fallback: shorter pages tend to be more obscure
        if page.page_length <= 0:
            return 0.5
        import math

        score = 1.0 - min(1.0, math.log10(max(1, page.page_length)) / 5.5)
        return max(0.0, min(1.0, score))

    def pages_with_infobox(self) -> list[PageInfo]:
        """Return pages that have at least one infobox field."""
        return [p for p in self._pages.values() if p.infobox]


# ---------------------------------------------------------------------------
# PageInfo serialization
# ---------------------------------------------------------------------------


def _page_info_to_dict(page: PageInfo) -> dict[str, Any]:
    return {
        "url": page.url,
        "title": page.title,
        "infobox": [
            {"key": f.key, "value": f.value, "numeric_value": f.numeric_value}
            for f in page.infobox
        ],
        "outgoing_links": [
            {
                "text": l.text,
                "target_title": l.target_title,
                "target_url": l.target_url,
                "section": l.section,
            }
            for l in page.outgoing_links
        ],
        "sections": page.sections,
        "first_paragraph": page.first_paragraph,
        "page_length": page.page_length,
        "pageviews": page.pageviews,
        "fetch_timestamp": page.fetch_timestamp,
        "raw_markdown": page.raw_markdown,
    }


def _page_info_from_dict(data: dict[str, Any]) -> PageInfo:
    return PageInfo(
        url=data["url"],
        title=data["title"],
        infobox=[
            InfoboxField(
                key=f["key"],
                value=f["value"],
                numeric_value=f.get("numeric_value"),
            )
            for f in data.get("infobox", [])
        ],
        outgoing_links=[
            WikiLink(
                text=l["text"],
                target_title=l["target_title"],
                target_url=l["target_url"],
                section=l.get("section"),
            )
            for l in data.get("outgoing_links", [])
        ],
        sections=data.get("sections", []),
        first_paragraph=data.get("first_paragraph", ""),
        page_length=data.get("page_length", 0),
        pageviews=data.get("pageviews"),
        fetch_timestamp=data.get("fetch_timestamp", ""),
        raw_markdown=data.get("raw_markdown", ""),
    )
