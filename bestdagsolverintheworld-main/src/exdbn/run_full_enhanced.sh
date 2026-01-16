#!/bin/bash
# =================================================
# EXDBN Full Pipeline Script (Enhanced)
# Supports CLI overrides for lambda, max-degree, sample size, and workers
# =================================================

set -e  # Exit immediately if any command fails

# --------------------------
# Default parameters
# --------------------------
CONFIG="./src/exdbn/config.yaml"
LAMBDA1=""
LAMBDA2=""
MAX_DEGREES=()
SAMPLE_SIZES=()
NUM_WORKERS=""

# --------------------------
# Parse CLI arguments
# --------------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --lambda1)
      LAMBDA1="$2"
      shift 2
      ;;
    --lambda2)
      LAMBDA2="$2"
      shift 2
      ;;
    --max-degree)
      MAX_DEGREES+=("$2")
      shift 2
      ;;
    --sample-size)
      SAMPLE_SIZES+=("$2")
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# Convert arrays to repeated --max-degree / --sample-size flags
MAX_DEG_FLAGS=()
for deg in "${MAX_DEGREES[@]}"; do
  MAX_DEG_FLAGS+=(--max-degrees "$deg")  # <-- plural, matches Typer
done

SAMPLE_SIZE_FLAGS=()
for n in "${SAMPLE_SIZES[@]}"; do
  SAMPLE_SIZE_FLAGS+=(--sample-sizes "$n")
done

# --------------------------
# Step 1: Generate datasets
# --------------------------
echo "[INFO] Generating datasets..."
uv run exdbn generate all --config "$CONFIG"

# --------------------------
# Step 2: Run static experiments
# --------------------------
echo "[INFO] Running static experiments..."
uv run exdbn run static --config "$CONFIG" \
   ${LAMBDA1:+--lambda1 $LAMBDA1} \
   ${LAMBDA2:+--lambda2 $LAMBDA2} \
   "${MAX_DEG_FLAGS[@]}" \
   "${SAMPLE_SIZE_FLAGS[@]}" \
   ${NUM_WORKERS:+--num-workers $NUM_WORKERS}

# --------------------------
# Step 3: Run dynamic experiments
# --------------------------
echo "[INFO] Running dynamic experiments..."
uv run exdbn run dynamic --config "$CONFIG" \
   ${LAMBDA1:+--lambda1 $LAMBDA1} \
   ${LAMBDA2:+--lambda2 $LAMBDA2} \
   "${MAX_DEG_FLAGS[@]}" \
   "${SAMPLE_SIZE_FLAGS[@]}" \
   ${NUM_WORKERS:+--num-workers $NUM_WORKERS}

# --------------------------
# Step 4: Invoke EXDBN in static mode
# --------------------------
echo "[INFO] Invoking EXDBN in static mode..."
uv run invoke exdbn --mode=static

echo "[DONE] EXDBN full pipeline completed!"
