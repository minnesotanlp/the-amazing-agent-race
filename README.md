# The Amazing Agent Race (AAR)

This repository contains the code and the datasets for **The Amazing Agent Race (AAR)**, a benchmark featuring directed acyclic graph (DAG) puzzles with fork-merge tool chains. Agents must navigate Wikipedia, execute multi-step tool chains, and aggregate results into a verifiable single-digit answer (0--9).

Each puzzle (called a *leg*) provides:

- A **seed URL** pointing to a Wikipedia article
- A **clue envelope**: a natural-language riddle whose clues describe a sequence of steps without naming Wikipedia titles or tool names
- A **tool set** of 19 tools with schema descriptions
- A **step budget** of B = max(10, floor(1.5K)) where K is the number of pit stops

## Datasets

AAR releases two benchmark variants totaling **1,400 legs**:

### AAR-Linear (800 legs)

Sequential puzzles where each step depends on the previous one in a single chain.

| Level | Legs | Stops | RB | Det. | Tools |
|-------|------|-------|----|------|-------|
| Easy | 200 | 5.4 | 1.1 | 1.0 | 1.4 |
| Medium | 200 | 11.7 | 2.3 | 2.3 | 2.9 |
| Hard | 200 | 18.7 | 4.1 | 3.9 | 5.0 |
| Extreme | 200 | 24.3 | 5.4 | 5.3 | 6.6 |
| *All* | *800* | *15.0* | *3.2* | *3.1* | *4.0* |

### AAR-DAG (600 legs)

Compositional puzzles with DAG structure featuring fork-merge diamond patterns. These require agents to branch information into parallel API calls and merge results -- a non-linear dependency pattern that linear benchmarks leave untested.

| Level | Legs | Stops | RB | Det. | Tools |
|-------|------|-------|----|------|-------|
| Easy | 100 | 8.3 | 3.6 | 1.0 | 4.8 |
| Medium | 150 | 15.4 | 6.0 | 2.3 | 7.9 |
| Hard | 166 | 24.6 | 10.1 | 3.9 | 13.2 |
| Extreme | 184 | 33.0 | 14.1 | 5.2 | 18.2 |
| *All* | *600* | *22.1* | *9.2* | *3.4* | *12.0* |

**Stops**: mean pit stops per leg. **RB**: roadblocks. **Det.**: detours. **Tools**: tool invocations in the golden trace.

### Format

Each puzzle is a JSON file representing a single leg (trail). Puzzles are procedurally generated from random Wikipedia seed articles (sampled from the top 100,000 most-viewed English pages) and validated with live API calls across four difficulty levels. Every leg passes the full quality pipeline: tool-chain pre-validation, golden execution, diamond augmentation (DAG only), and round-trip clue-envelope validation.

## Leg Structure

A leg is a directed acyclic graph (DAG) of **pit stops**, each producing a typed value:

1. **Route info** (`route_info`): Navigate to a Wikipedia page and extract a fact (e.g., a numeric infobox field, a date from prose).
2. **Roadblock** (`roadblock`): Execute a multi-step tool chain, e.g., geocode a location then query the elevation API.
3. **Detour** (`detour`): Apply an analytical transform to a prior value, e.g., `next_prime(v)`, `digit_sum(v)`.
4. **Finish line** (`finish_line`): Aggregate values from earlier stops via arithmetic to produce the final answer y* in {0,...,9}.

Transitions are typed (`link_follow`, `search_query`, `tool_call`, `compute`), and values are typed (`number`, `text`, `coords`, `date`), enabling type-aware argument passing between stops.

### Diamond Patterns (DAG)

AAR-DAG legs use **diamond patterns** (fork-merge) to create non-linear structure. A diamond has a **source stop** (extract a geocodable entity), two **branch stops** (independent tool chains on the same entity, e.g., elevation and POI count), and a **merge stop** (combines branch outputs). Diamond count scales with difficulty (1 for easy up to 3--5 for extreme) across four types (`elevation x POI`, `elevation x rating`, `population x area`, `temperature x precipitation`), guaranteeing every instance is a true DAG.

### Difficulty Levels

Difficulty is controlled through four levels that independently vary five parameters:

| Level | Pit Stops | Roadblocks | Detours | Diamonds | Extraction | Crawl |
|-------|-----------|------------|---------|----------|------------|-------|
| Easy | 3--6 | 1--2 | 1--2 | 1 | infobox, prose | 1 |
| Medium | 7--12 | 2--4 | 2--3 | 1--2 | + cross-section | 2 |
| Hard | 13--16 | 4--5 | 3--4 | 2--3 | + cross-section | 3 |
| Extreme | 17--21 | 5--7 | 4--6 | 3--5 | + cross-section | 3 |

After diamond augmentation, each diamond adds 3 stops (two branches + merge), so final pit-stop counts exceed the configured range.

## Tool Set

AAR provides **19 tools** across eight categories, designed for composability and temporal dynamism (stock/crypto tools return live data):

| Category | Tool | Description |
|----------|------|-------------|
| Fetch & Search | `fetch_webpage` | Fetch and parse web content |
| | `web_search` | Google search via Serper API |
| Google Maps | `maps_geocode` | Address to coordinates |
| | `maps_reverse_geocode` | Coordinates to address |
| | `maps_search_places` | Search nearby places |
| | `maps_place_details` | Place metadata and ratings |
| | `maps_distance_matrix` | Driving distances |
| | `maps_elevation` | Elevation at coordinates |
| | `maps_directions` | Directions and duration |
| Weather | `weather_historical` | Historical weather data |
| | `weather_forecast` | Weather forecasts |
| Code | `python_execute_code` | Run Python code |
| | `python_generate_code` | LLM-generated Python |
| Countries | `countries_population` | Population data |
| | `countries_area` | Area in km^2 |
| Stocks | `stock_historical_price` | Closing price on a date |
| | `stock_volume` | Trading volume on a date |
| Crypto | `crypto_historical_price` | Crypto closing price on a date |
| | `crypto_volume` | 24h trading volume on a date |

Each tool returns values in a canonical unit (elevation in meters, distance in km, temperature in degrees C). Tool outputs longer than 8,000 characters are truncated.

## Metrics

### Primary Metrics

1. **Finish-line accuracy (FA)**: Whether the agent's single-digit answer matches the golden finish-line code. This is the primary success metric.
2. **Pit-stop visit rate (PVR)**: Fraction of golden `route_info` pit stops for which the agent fetched the correct Wikipedia URL, measuring navigation quality.
3. **Roadblock completion rate (RCR)**: Fraction of golden `roadblock` pit stops for which the agent invoked all expected tools in the chain, measuring tool-use competence.

### Supplementary Indicators

- **Intermediate value rate**: Fraction of golden pit stops whose expected value appears in the agent's tool results (partial credit).
- **Average steps**: Mean number of LLM turns per leg (lower is more efficient).
- **Step-limit hit rate**: Fraction of legs where the agent exhausted its step budget.

### Scoring

- **Reward 1.0** = correct passcode digit
- **Reward 0.0** = incorrect or no answer
- Aggregate metric: **mean** across all tasks (accuracy)

## Evaluation via Harbor

All evaluations run through [Harbor](https://github.com/harbor-framework/harbor), an open-source agent evaluation framework that orchestrates trials in containerized Docker environments. Each agent receives the same Docker environment with a command-line tool executor (`tools.py`), the clue envelope as a Markdown instruction file, and internet access for web fetching. The agent must write its single-digit answer to `/app/answer.txt`.

The `harbor-adapter/` directory contains the AAR adapter, built on top of Harbor at commit [`48ae2ba`](https://github.com/harbor-framework/harbor/tree/48ae2ba). See `harbor-adapter/README.md` for setup and usage instructions.

### Evaluation Environment

- **Docker container**: Python 3.11, 10,240 MB memory, internet access enabled
- **Timeout**: 600 seconds per trial (uniform across difficulty levels)
- **Answer format**: Agent writes a single digit (0--9) to `/app/answer.txt`
- **Temperature**: 0 for deterministic outputs (where supported)

## Quality Assurance

Every leg satisfies six invariants:

1. **Solvability**: golden executor produces y* (dry-run at generation time)
2. **API stability**: cached traces and page snapshots for reproducibility
3. **Input cleanliness**: geocodability filtering
4. **Clue-envelope integrity**: round-trip alignment >= 0.7, no direct Wikipedia titles
5. **Contamination resistance**: clue paraphrasing replaces titles with circumlocutions, roadblock answers depend on live APIs, detour transforms produce values absent from Wikipedia, and finish-line codes use modular arithmetic
6. **Inter-instance diversity**: mean pairwise Jaccard similarity of 0.0005 across 10K sampled pairs

## Citation

Paper under review.
