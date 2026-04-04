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

## Citation

Paper under review.
