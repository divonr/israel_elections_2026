#!/usr/bin/env bash
# Sensitivity grid for the 2026 forecast: re-runs step 07 under 16 alternative settings, one TAG each.
# Step 09 collects every outputs/tables/forecast_2026_sens_*.csv into the report's sensitivity table.
# The "house" variant uses the carry-over slope measured by step 06 (house_effect_persistence.csv).
# Usage: bash run_sensitivity_grid.sh [N_SIM]   (default 8000 simulations per variant)
set -uo pipefail
cd "$(dirname "$0")"
N=${1:-8000}
HC=$(python -P -c "import pandas as pd; print(round(float(pd.read_csv('../outputs/tables/house_effect_persistence.csv').carryover_slope.iloc[0]), 2))")
run() {
  local tag=$1; shift
  if env N_SIM="$N" TAG="sens_$tag" "$@" python -P 07_forecast_2026.py > "../outputs/logs/log_07_sens_$tag.txt" 2>&1; then
    echo "sens_$tag ok"
  else
    echo "sens_$tag FAILED (see outputs/logs/log_07_sens_$tag.txt)"
  fi
}
run none CORRECTION=none
run m1s CORRECTION=m1s
run comp CORRECTION=comp
run win6 WINDOW_DAYS=6
run win14 WINDOW_DAYS=14
run dis0 DISAGREE_SCALE=0
run dis1 DISAGREE_SCALE=1
run fam15 FAMILY_SD=0.15
run house HOUSE_CARRY="$HC"
run rawpct RAW_MODE=raw
run noCh14 EXCLUDE_FIRMS=NextData_Filber
run noDRI EXCLUDE_FIRMS=DRI
run noRight EXCLUDE_FIRMS=NextData_Filber,DirectPolls,DRI
run onlyRight EXCLUDE_FIRMS=Midgam,Kantar,Lazar_Panels,MaagarMochot,Ch13_Madad2026,Tatika
run halfOpaque FIRM_WEIGHTS=NextData_Filber:0.5,DRI:0.5
run rightOneVote FIRM_WEIGHTS=NextData_Filber:0.3333,DirectPolls:0.3333,DRI:0.3333
echo "sensitivity grid done (house carry-over = $HC)"
