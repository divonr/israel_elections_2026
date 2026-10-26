#!/usr/bin/env bash
# Reproducibility check: rebuild EVERYTHING from the raw downloads in an isolated copy and compare the
# final forecast with the submitted one. (Step 01, the downloads, is skipped: the raw files are copied.)
# Usage: bash reproduce_from_raw.sh <scratch_dir>
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
BASE="$(dirname "$SRC")"
DST_BASE="$1/repro"
DST="$DST_BASE/forecast_k26"
rm -rf "$DST_BASE"
mkdir -p "$DST/data/processed" "$DST/outputs/tables" "$DST/outputs/logs" "$DST/report"
cp -r "$SRC/code" "$DST/code"
rm -rf "$DST/code/__pycache__"
cp -r "$SRC/data/raw" "$DST/data/raw"
cp -r "$SRC/data/research" "$DST/data/research"
cp -r "$BASE/instinct" "$DST_BASE/instinct"          # the AI agents' original files (inputs of step 04)
cp -r "$BASE/perplexity" "$DST_BASE/perplexity"
cd "$DST/code"
for s in 02_parse_results 03_parse_wiki_polls 04_build_poll_datasets 04b_merge_hewiki_polls 05_clustering 06_poll_error_analysis \
         06b_model_selection 06c_final_factors 06d_calibration 06e_pollster_profiles; do
  python -P "$s.py" > "../outputs/logs/log_$s.txt" 2>&1 && echo "$s ok"
done
N_SIM=40000 TAG=main python -P 07_forecast_2026.py > ../outputs/logs/log_07.txt 2>&1 && echo "07 ok"
python -P 07b_firm_heterogeneity.py > ../outputs/logs/log_07b.txt 2>&1 && echo "07b ok"
python -P 08_turnout.py > ../outputs/logs/log_08.txt 2>&1 && echo "08 ok"
TAG=main python -P 09_build_report.py > ../outputs/logs/log_09.txt 2>&1 && echo "09 ok"
python -P - "$SRC" "$DST" <<'EOF'
import sys, json, pandas as pd
src, dst = sys.argv[1], sys.argv[2]
a = pd.read_csv(f"{src}/outputs/tables/forecast_2026_main.csv").set_index("id").FORECAST_SEATS
b = pd.read_csv(f"{dst}/outputs/tables/forecast_2026_main.csv").set_index("id").FORECAST_SEATS
ta = json.load(open(f"{src}/outputs/tables/turnout_forecast.json"))["turnout_median"]
tb = json.load(open(f"{dst}/outputs/tables/turnout_forecast.json"))["turnout_median"]
print("seat vector identical:", bool((a == b).all()), "| sum |diff| =", int((a - b).abs().sum()))
print("turnout submitted %.2f vs reproduced %.2f" % (ta, tb))
print(pd.DataFrame({"submitted": a, "reproduced": b}).to_string())
EOF
echo "reproduction done"
