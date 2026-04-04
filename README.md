# The Amazing Agent Race (AAR)
This repository contains the code and the datasets for **The Amazing Agent Race (AAR)**, a benchmark for evaluating LLM agents on multi-step scavenger-hunt puzzles that require web navigation, tool use, and arithmetic reasoning.

## Datasets

### aar-linear (800 legs)

Sequential puzzles where each step depends on the previous one in a single chain. Each leg requires an agent to navigate Wikipedia pages, execute tool calls (geocoding, elevation, distance, etc.), and compute a final numeric answer.

| Difficulty | Count |
|------------|-------|
| Easy       | 200   |
| Medium     | 200   |
| Hard       | 200   |
| Extreme    | 200   |

### aar-dag (600 legs)

Compositional puzzles with directed acyclic graph (DAG) structure featuring fork-merge tool chains. These require agents to branch information into parallel API calls and merge results -- a non-linear dependency pattern that linear benchmarks leave untested.

| Difficulty | Count |
|------------|-------|
| Easy       | 100   |
| Medium     | 150   |
| Hard       | 166   |
| Extreme    | 184   |

## Format

Each puzzle is a JSON file representing a single leg (trail). Puzzles are procedurally generated from Wikipedia seed pages and validated with live API calls across four difficulty levels.

## Harbor Adapter

The `harbor-adapter/` directory contains a [Harbor](https://github.com/vector-institute/harbor) adapter that converts AAR puzzles into Harbor task format for standardized agent evaluation.

This adapter was built on top of Harbor at commit [`48ae2ba`](https://github.com/vector-institute/harbor/tree/48ae2ba). To use it:

1. Clone Harbor and checkout the base commit:
   ```bash
   git clone https://github.com/vector-institute/harbor.git
   cd harbor
   git checkout 48ae2ba
   ```

2. Copy the adapter into Harbor:
   ```bash
   cp -r /path/to/this-repo/harbor-adapter harbor/adapters/aar
   ```

3. Generate Harbor tasks from AAR puzzles:
   ```bash
   cd adapters/aar
   python run_adapter.py \
       --data-dir /path/to/this-repo/data/trail_puzzles \
       --output-dir ../../datasets/aar
   ```

4. Run evaluation:
   ```bash
   harbor run -c adapters/aar/aar.yaml
   ```

See `harbor-adapter/README.md` for full details on configuration, supported agents, and result analysis.

## Citation

Paper under review.
