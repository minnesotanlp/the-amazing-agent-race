#!/usr/bin/env bash
# Batch-generate trail puzzles across all difficulty levels.
#
# Usage:
#   ./scripts/batch_generate.sh                    # defaults: 10 samples per difficulty, random seeds
#   ./scripts/batch_generate.sh --seed-file data/seed_urls.txt  # use curated seed list
#   ./scripts/batch_generate.sh --random 20        # 20 random-seed samples per difficulty
#   ./scripts/batch_generate.sh --only easy,medium  # specific difficulties
#
# Requires: uv, .env with OPENAI_API_KEY and GOOGLE_API_KEY

set -euo pipefail
cd "$(dirname "$0")/.."

SAMPLES=10
DIFFICULTIES="easy medium hard extreme"
MODE="random"   # "file" or "random"
MAX_PAGES=50
SEED_FILE="data/seed_urls.txt"
OUTPUT_BASE="data/trail_puzzles"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --random)
            MODE="random"
            if [[ $# -gt 1 && "$2" =~ ^[0-9]+$ ]]; then
                SAMPLES="$2"; shift
            fi
            shift ;;
        --samples)
            SAMPLES="$2"; shift 2 ;;
        --only)
            DIFFICULTIES="${2//,/ }"; shift 2 ;;
        --max-pages)
            MAX_PAGES="$2"; shift 2 ;;
        --seed-file)
            SEED_FILE="$2"; shift 2 ;;
        --output)
            OUTPUT_BASE="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

echo "=== AAR Batch Generation ==="
echo "  Mode:         $MODE"
echo "  Samples:      $SAMPLES per difficulty"
echo "  Difficulties: $DIFFICULTIES"
echo "  Max pages:    $MAX_PAGES"
echo "  Output:       $OUTPUT_BASE"
echo ""

TOTAL_OK=0
TOTAL_FAIL=0

for difficulty in $DIFFICULTIES; do
    echo "========================================"
    echo " Generating $difficulty ($SAMPLES samples)"
    echo "========================================"

    OUTDIR="$OUTPUT_BASE/$difficulty"
    mkdir -p "$OUTDIR"

    SEED_ARG=""
    if [[ "$MODE" == "random" ]]; then
        SEED_ARG="--random-seeds"
    else
        SEED_ARG="--seed-urls-file $SEED_FILE"
    fi

    if uv run python src/trail/generate.py \
        $SEED_ARG \
        --difficulty "$difficulty" \
        --num-samples "$SAMPLES" \
        --output-dir "$OUTDIR" \
        --max-pages "$MAX_PAGES" \
        --skip-pageviews; then
        TOTAL_OK=$((TOTAL_OK + 1))
    else
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        echo "WARNING: $difficulty generation had errors"
    fi

    echo ""
done

echo "=== Batch complete ==="
echo "  Successful runs: $TOTAL_OK"
echo "  Failed runs:     $TOTAL_FAIL"

# Summary of generated files
echo ""
echo "Generated puzzles:"
for difficulty in $DIFFICULTIES; do
    count=$(find "$OUTPUT_BASE/$difficulty" -name "*.json" -not -name "_*" -not -name ".*" 2>/dev/null | wc -l | tr -d ' ')
    echo "  $difficulty: $count files"
done
