# AAR Harbor Adapter

Converts **The Amazing Agent Race (AAR)** multi-step scavenger-hunt puzzles into [Harbor](https://github.com/vector-institute/harbor) task format for standardized agent evaluation in containerized Docker environments.

Built on Harbor commit [`48ae2ba`](https://github.com/vector-institute/harbor/tree/48ae2ba).

## Overview

AAR is a benchmark featuring directed acyclic graph (DAG) puzzles with fork-merge tool chains.
Agents must navigate Wikipedia, execute multi-step tool chains, and aggregate results into a verifiable single-digit answer (0--9).
Each puzzle (called a *leg*) provides:

- A **seed URL** pointing to a Wikipedia article
- A **clue envelope**: a natural-language riddle whose clues describe a sequence of steps without naming Wikipedia titles or tool names
- A **tool set** of 19 tools with schema descriptions
- A **step budget** of B = max(10, floor(1.5K)) where K is the number of pit stops

## Dataset

AAR releases two benchmark variants totaling **1,400 legs**:

| Variant | Level | Legs | Stops | RB | Det. | Tools |
|---------|-------|------|-------|----|------|-------|
| AAR-Linear | Easy | 200 | 5.4 | 1.1 | 1.0 | 1.4 |
| | Medium | 200 | 11.7 | 2.3 | 2.3 | 2.9 |
| | Hard | 200 | 18.7 | 4.1 | 3.9 | 5.0 |
| | Extreme | 200 | 24.3 | 5.4 | 5.3 | 6.6 |
| | *All* | *800* | *15.0* | *3.2* | *3.1* | *4.0* |
| AAR-DAG | Easy | 100 | 8.3 | 3.6 | 1.0 | 4.8 |
| | Medium | 150 | 15.4 | 6.0 | 2.3 | 7.9 |
| | Hard | 166 | 24.6 | 10.1 | 3.9 | 13.2 |
| | Extreme | 184 | 33.0 | 14.1 | 5.2 | 18.2 |
| | *All* | *600* | *22.1* | *9.2* | *3.4* | *12.0* |

**Stops**: mean pit stops per leg. **RB**: roadblocks. **Det.**: detours. **Tools**: tool invocations in the golden trace.

### Leg Structure

A leg is a DAG of **pit stops**, each producing a typed value:

1. **Route info** (`route_info`): Navigate to a Wikipedia page and extract a fact (e.g., a numeric infobox field, a date from prose).
2. **Roadblock** (`roadblock`): Execute a multi-step tool chain, e.g., geocode a location then query the elevation API.
3. **Detour** (`detour`): Apply an analytical transform to a prior value, e.g., `next_prime(v)`, `digit_sum(v)`.
4. **Finish line** (`finish_line`): Aggregate values from earlier stops via arithmetic to produce the final answer y* in {0,...,9}.

AAR-DAG legs use **diamond patterns** (fork-merge) to create non-linear structure: a source stop extracts a geocodable entity, two branch stops run independent tool chains (e.g., elevation and POI count), and a merge stop combines outputs.

### Difficulty Levels

| Level | Pit Stops | Roadblocks | Detours | Diamonds | Extraction | Crawl |
|-------|-----------|------------|---------|----------|------------|-------|
| Easy | 3--6 | 1--2 | 1--2 | 1 | infobox, prose | 1 |
| Medium | 7--12 | 2--4 | 2--3 | 1--2 | + cross-section | 2 |
| Hard | 13--16 | 4--5 | 3--4 | 2--3 | + cross-section | 3 |
| Extreme | 17--21 | 5--7 | 4--6 | 3--5 | + cross-section | 3 |

## Tool Set

AAR provides **19 tools** across eight categories:

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

Agents interact with tools via CLI:

```bash
python3 /app/tools.py <tool_name> '<json_arguments>'
python3 /app/tools.py fetch_webpage '{"url": "https://en.wikipedia.org/wiki/Mount_Everest"}'
python3 /app/tools.py maps_geocode '{"address": "Mount Everest"}'
python3 /app/tools.py maps_elevation '{"locations": [{"latitude": 27.98, "longitude": 86.92}]}'
python3 /app/tools.py python_execute_code '{"code": "print(8848 % 10)"}'
python3 /app/tools.py --list  # list all tools
```

Tool outputs longer than 8,000 characters are truncated. All tool invocations are logged to `/app/tool_log.jsonl` for the verifier.

## Setup

### Prerequisites

| Variable | Required | Used by |
|----------|----------|---------|
| `GOOGLE_API_KEY` | Yes | Geocoding, elevation, distance, directions, places tools |
| `OPENAI_API_KEY` | No | Only if agent uses OpenAI models |
| `SERPER_API_KEY` | No | Web search tool |

### Generate Harbor Tasks

```bash
cd harbor-adapter

# Generate from AAR-Linear puzzles
python run_adapter.py \
    --data-dir /path/to/this-repo/data/aar-linear \
    --output-dir /path/to/harbor/datasets/aar

# Generate from AAR-DAG puzzles
python run_adapter.py \
    --data-dir /path/to/this-repo/data/aar-dag \
    --output-dir /path/to/harbor/datasets/aar

# Specific difficulty only
python run_adapter.py \
    --data-dir /path/to/this-repo/data/aar-linear \
    --difficulty easy \
    --output-dir /path/to/harbor/datasets/aar

# Limit number of tasks
python run_adapter.py \
    --data-dir /path/to/this-repo/data/aar-linear \
    --limit 20 \
    --output-dir /path/to/harbor/datasets/aar
```

### Run Evaluation

```bash
# Run with config file
harbor run -c adapters/aar/aar.yaml

# Run with Claude Code agent
harbor run \
    -p datasets/aar \
    -a claude-code \
    -m anthropic/claude-sonnet-4-6 \
    -n 4

# Run specific difficulty
harbor run \
    -p datasets/aar \
    --task-names "easy-*" \
    -a claude-code \
    -m anthropic/claude-sonnet-4-6

# Run with different agents
harbor run -p datasets/aar -a codex -m openai/gpt-5.4
harbor run -p datasets/aar -a mini-swe-agent -m openai/gpt-5.4
```

## How It Works

Each generated Harbor task directory contains:

```
<task_id>/
├── instruction.md          # Riddle, seed URL, tool docs, answer format
├── task.toml               # Harbor metadata (difficulty, timeout, env config)
├── environment/
│   ├── Dockerfile          # Python 3.11 container with tool dependencies
│   ├── requirements.txt    # httpx, markdownify, yfinance, etc.
│   ├── tools.py            # CLI wrapper: dispatches tool calls, logs to JSONL
│   └── tool_implementations.py  # 19 tool implementations (Google Maps, weather, etc.)
├── solution/
│   └── solve.sh            # Oracle: writes the correct passcode digit
└── tests/
    ├── test.sh             # Launches the Python verifier
    ├── verify.py           # Partial-credit scoring (see Metrics below)
    ├── expected_answer.txt # Ground-truth passcode digit
    └── trail.json          # Full golden trail (agents don't see this)
```

### Evaluation Environment

- **Docker container**: Python 3.11, 10,240 MB memory, internet access enabled
- **Timeout**: 600 seconds per trial (uniform across difficulty levels)
- **Answer format**: Agent writes a single digit (0--9) to `/app/answer.txt`
- **Temperature**: 0 for deterministic outputs (where supported)

## Metrics

AAR reports three primary metrics and several supplementary indicators:

### Primary Metrics

1. **Finish-line accuracy (FA)**: Whether the agent's single-digit answer matches the golden finish-line code. This is the primary success metric (reported as `reward` in `reward.json`).
2. **Pit-stop visit rate (PVR)**: Fraction of golden `route_info` pit stops for which the agent fetched the correct Wikipedia URL, measuring navigation quality.
3. **Roadblock completion rate (RCR)**: Fraction of golden `roadblock` pit stops for which the agent invoked all expected tools in the chain, measuring tool-use competence.

### Supplementary Indicators

- **Intermediate value rate**: Fraction of golden pit stops whose expected value appears in the agent's tool results (partial credit).
- **Average steps**: Mean number of LLM turns per leg.
- **Step-limit hit rate**: Fraction of legs where the agent exhausted its step budget.

### Scoring

- **Reward 1.0** = correct passcode digit
- **Reward 0.0** = incorrect or no answer
- Aggregate metric: **mean** across all tasks (accuracy)

### Verifier Output

The verifier (`tests/verify.py`) writes `reward.json` with detailed fields including: `status`, `agent_answer`, `expected_answer`, `total_tool_calls`, `valid_tool_calls`, `right_tool_calls`, `pages_visited`, `tools_used`, `intermediate_correct`, `intermediate_total`, `intermediate_rate`, and `ground_truth_steps`.

## Analyzing Results

Use `analyze_results.py` to produce summary tables from Harbor job directories:

```bash
# Table summary
python analyze_results.py jobs/<job_dir>

# Breakdown by difficulty
python analyze_results.py jobs/<job_dir> --by-difficulty

# Per-trial details
python analyze_results.py jobs/<job_dir> --per-trial

# CSV or JSON output
python analyze_results.py jobs/<job_dir> --format csv
python analyze_results.py jobs/<job_dir> --format json
```
