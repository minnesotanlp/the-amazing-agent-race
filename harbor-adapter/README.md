# The Amazing Agent Race (AAR) Adapter

Converts The Amazing Agent Race (AAR) multi-step scavenger-hunt puzzles into Harbor task format.

## Overview

The Amazing Agent Race (AAR) is a benchmark for evaluating LLM agents on multi-step scavenger-hunt puzzles that require:
- **Web navigation** — reading Wikipedia pages and following links
- **Tool use** — geocoding, elevation lookups, distance calculations, weather data, code execution
- **Arithmetic reasoning** — combining extracted values into a final computation

Each puzzle provides a riddle with ordered clues. The agent must follow the clues, use tools to gather data, and compute a single-digit passcode (0-9).

## Dataset

- **400 puzzles** across 4 difficulty levels (100 each)
- **Easy**: 3-6 stops, 1-2 tool stops
- **Medium**: 7-12 stops, 2-4 tool stops
- **Hard**: 13-16 stops, 4-5 tool stops
- **Extreme**: 17-21 stops, 5-7 tool stops

## Setup

### Prerequisites

The following API keys must be set as environment variables:

| Variable | Required | Used by |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Geocoding, elevation, distance, directions tools |
| `OPENAI_API_KEY` | No | Only if agent uses OpenAI models |
| `SERPER_API_KEY` | No | Web search tool |

### Generate Harbor Tasks

```bash
cd adapters/aar

# Generate all 400 tasks
python run_adapter.py \
    --data-dir /path/to/the-amazing-agent-race/data/trail_puzzles \
    --output-dir ../../datasets/aar

# Generate only easy tasks
python run_adapter.py \
    --data-dir /path/to/the-amazing-agent-race/data/trail_puzzles \
    --difficulty easy \
    --output-dir ../../datasets/aar

# Generate a subset
python run_adapter.py \
    --data-dir /path/to/the-amazing-agent-race/data/trail_puzzles \
    --limit 20 \
    --output-dir ../../datasets/aar
```

### Run Evaluation

```bash
# Run with Claude Code agent
harbor run \
    -p datasets/aar \
    -a claude-code \
    -m anthropic/claude-sonnet-4-6 \
    -n 4

# Run with config file
harbor run -c adapters/aar/aar.yaml

# Run specific difficulty
harbor run \
    -p datasets/aar \
    --task-names "easy-*" \
    -a claude-code \
    -m anthropic/claude-sonnet-4-6

# Run with different agents
harbor run -p datasets/aar -a codex -m openai/gpt-4o
harbor run -p datasets/aar -a gemini-cli -m google/gemini-2.5-pro
harbor run -p datasets/aar -a openhands -m anthropic/claude-sonnet-4-6
```

## How It Works

Each Harbor task includes:

- **`instruction.md`** — The riddle, seed URL, tool documentation, and answer format
- **`environment/`** — Docker container with Python + tool CLI (`tools.py`)
- **`tests/test.sh`** — Verifies the agent's answer against the expected passcode
- **`solution/solve.sh`** — Oracle that writes the correct answer

### Tool Access

Agents interact with tools via CLI:

```bash
python3 /app/tools.py fetch_webpage '{"url": "https://en.wikipedia.org/wiki/Mount_Everest"}'
python3 /app/tools.py maps_geocode '{"address": "Mount Everest"}'
python3 /app/tools.py maps_elevation '{"locations": [{"latitude": 27.98, "longitude": 86.92}]}'
python3 /app/tools.py python_execute_code '{"code": "print(8848 % 10)"}'
python3 /app/tools.py --list  # list all tools
```

### Scoring

- **Reward 1** = correct passcode digit
- **Reward 0** = incorrect or no answer
- Aggregate metric: **mean** across all tasks (accuracy)
