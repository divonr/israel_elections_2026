#!/usr/bin/env bash
# Robustness check: re-run the back-test, model selection, final factors and the 2026 forecast with the
# MERGED historical poll set (English Wikipedia + polls found only in the Hebrew-Wikipedia source, step 04b),
# inside an isolated copy of the project so the main outputs are untouched.
# Usage: bash robustness_merged_polls.sh <scratch_dir>
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$1/forecast_k26"
rm -rf "$DST"
mkdir -p "$DST/data" "$DST/outputs/tables" "$DST/outputs/figures"
cp -r "$SRC/code" "$DST/code"
cp -r "$SRC/data/processed" "$DST/data/processed"
cp -r "$SRC/data/research" "$DST/data/research"
cp "$SRC/outputs/tables/clustering_validation.csv" "$DST/outputs/tables/"
cp "$DST/data/processed/hist_polls_merged.csv" "$DST/data/processed/hist_polls.csv"
cd "$DST/code"
python -P 06_poll_error_analysis.py > ../outputs/r06.txt 2>&1 && echo "06 ok"
python -P 06b_model_selection.py > ../outputs/r06b.txt 2>&1 && echo "06b ok"
python -P 06c_final_factors.py > ../outputs/r06c.txt 2>&1 && echo "06c ok"
N_SIM=20000 TAG=main python -P 07_forecast_2026.py > ../outputs/r07.txt 2>&1 && echo "07 ok"
echo "robustness done"
