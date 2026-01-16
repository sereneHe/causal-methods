#!/bin/bash
# =================================================
# EXDBN Full Pipeline Script
# 1️⃣ Generate datasets
# 2️⃣ Run static experiments
# 3️⃣ Run dynamic experiments
# 4️⃣ Invoke EXDBN in static mode
# =================================================

set -e  # Exit immediately if a command fails

# Step 1: Generate datasets
echo "[INFO] Generating datasets..."
uv run exdbn generate all --out datasets/syntheticdata

# Step 2: Run static experiments
echo "[INFO] Running static experiments..."
uv run exdbn run static --max-degree 3 --max-degree 5

# Step 3: Run dynamic experiments
echo "[INFO] Running dynamic experiments..."
uv run exdbn run dynamic --lambda1 0.5

# Step 4: Invoke EXDBN in static mode
echo "[INFO] Invoking EXDBN in static mode..."
uv run invoke exdbn --mode=static

echo "[DONE] EXDBN full pipeline completed!"
