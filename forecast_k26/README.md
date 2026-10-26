# Knesset 26 seat & turnout forecast (election 27.10.2026)

Systematic-track submission for the Department of Statistics & Data Science forecasting competition.
Score = sum of absolute seat errors (tie-break: national turnout). **Start here.** (Hebrew: `README_he.md`.)

## Final answer

| List (ballot letters) | Seats |
|---|---|
| Likud (מחל) | 23 |
| Yashar! with Eisenkot (דרך) | 22 |
| Together - Bennett/Lapid (רק) | 13 |
| Joint List - Hadash-Ta'al-Balad (ודם) | 10 |
| Shas (שס) | 9 |
| The Democrats (אמת) | 8 |
| Yisrael Beiteinu (ל) | 8 |
| United Torah Judaism (ג) | 8 |
| Otzma Yehudit (ב) | 7 |
| Religious Zionism & Zehut (ט) | 6 |
| Ra'am (עם) | 6 |
| Hendel/Zelicha (די), Amcha Israel (ך), Blue & White (withdrew), any other list | 0 |

**National turnout: 71.2%** (80% range 69.0-73.3%).
Machine-readable: `outputs/tables/forecast_2026_main.csv` (`FORECAST_SEATS`), `outputs/tables/turnout_forecast.json`.

## What is where

| Path | Contents |
|---|---|
| `report/` | **Deliverables.** `index.html` = full report (self-contained page), `Knesset26_forecast_report.pdf` = same as PDF, `method_description.md` = method text for the form (<= 500 words), `submission.md` = submission card, `report_data.json` = all numbers shown in the report. Hebrew: `*_he.md`, and `report_he.md` = full results report in Hebrew. |
| `code/` | The pipeline (steps 01-09 below), `config.py`, `report_template.html`, and two shell scripts: `run_sensitivity_grid.sh`, `reproduce_from_raw.sh`. |
| `data/raw/` | Downloads: CEC results K16-K25 (`results/`), English Wikipedia poll pages (`wiki/`), the verified second poll source (`hewiki/polls_<election>.csv`). |
| `data/research/` | Hand-collected 2026 facts: lists, surplus agreements, CEC raw poll percentages, turnout inputs (with sources in `auxiliary_2026.md`). |
| `data/processed/` | Intermediate data produced by the scripts (ballot tables, poll datasets, voter segments in `clusters/`, model parameters, simulation draws). |
| `outputs/tables/` | Every result table (back-tests, tests, calibration, pollster profiles, forecasts incl. 16 sensitivity variants). |
| `outputs/logs/` | Console output of the final run of each step. |
| `docs/DECISION_LOG.md` | Chronological log of every decision, problem and fix (Hebrew: `DECISION_LOG_he.md`). |
| `_archive_not_main/` | Kept but not part of the final pipeline: superseded runs, old logs, the pre-fix clusters, and the earlier results that used the English-Wikipedia history only (`results_v8_english_wikipedia_history_only/`, same seat vector). See its README. |

The 2026 poll collections of the two AI agents (`../instinct/`, `../perplexity/`) are read by step 04 from the parent folder.

## Method in one paragraph

Official ballot-box results 2003-2022 are clustered into voter segments (k-means, GMM, Ward, NMF; K = 4-14; ballot-box
and settlement level). Past polls (516 polls, 8 elections, from English Wikipedia plus an independently verified
second source) are compared with the true results to measure how polls err: an industry-wide bias by party family
and a model of each poll as a re-weighting of the true segments. The candidate corrections are compared in a
leave-one-election-out back-test on the competition metric (raw poll average 20.0 seats of error per election; the
selected correction 18.75). The 2026 polls are averaged one vote per firm, corrected, and 40,000 elections are simulated
through the exact allocation law (3.25% threshold, Bader-Ofer, surplus agreements); the single guess is the
L1-optimal feasible seat vector. Turnout comes from a structural model (domestic electorate x sector turnout).
Details: `report/method_description.md` and the report.

## Pipeline (run in order from `code/`, each with `python -P <script>`)

| Step | Script | What it does | Main outputs |
|---|---|---|---|
| 01 | `01_download_data.py` | Downloads official CEC ballot-box and settlement results for Knesset 16-25 (2003-2022) and the Wikipedia opinion-polling pages | `data/raw/` |
| 02 | `02_parse_results.py` | Parses all CEC formats into one long table; flags double-envelope rows | `data/processed/ballots_*.parquet`, `national_totals.csv` |
| 03 | `03_parse_wiki_polls.py` | Rowspan/colspan-aware parser of the Wikipedia seat-projection tables | `wiki_polls_long.csv` |
| 04 | `04_build_poll_datasets.py` | Historical polls mapped to ballot letters, party families and pollster lineages; 2026 polls merged with the two AI agents' collections and cross-checked | `hist_polls.csv`, `polls_2026.csv`, `polls_2026_crosscheck.csv` |
| 04b | `04b_merge_hewiki_polls.py` | Second, independent historical poll source (Hebrew Wikipedia + archived originals, final 30 days, each election extracted and re-verified by separate research agents): one-to-one matching to the English-Wikipedia polls and addition of the polls English Wikipedia lacks | `hist_polls_merged.csv` (used by step 06 and later), `outputs/tables/hewiki_crosscheck.csv` |
| 05 | `05_clustering.py` | Voter segments from true results: k-means, Gaussian mixture, Ward, NMF; K=4-14; ballot-box and settlement level; internal validity + bootstrap stability | `data/processed/clusters/`, `clustering_validation.csv` |
| 06 | `06_poll_error_analysis.py` | Industry bias by party family (tests), house-effect persistence (permutation test), segment-composition model (PyTorch), leave-one-election-out back-test of all methods | `industry_*`, `house_effect_*`, `backtest_*` |
| 06b | `06b_model_selection.py` | Friedman/Nemenyi tests, nested (honest) selection, ensembles | `model_selection_*` |
| 06c | `06c_final_factors.py` | Selects the final correction by back-test score among pre-specified candidates (E2 selected: shrunk industry bias + segment-composition model averaged over NMF segmentations), its 2026 factors, and the out-of-sample error model | `data/processed/model_params_final.json` |
| 06d | `06d_calibration.py` | Out-of-sample calibration of the simulation (interval coverage, Brier score) | `calibration_*` |
| 06e | `06e_pollster_profiles.py` | Pollster profiles: pollster x family bias matrix, bias+slope and calibration-regression matrices (back-tested), pollster x segment representation matrix; jackknife SEs | `pollster_*`, `backtest_slope_*` |
| 07 | `07_forecast_2026.py` | 2026 poll average, corrections, Monte Carlo with the exact allocation law and surplus agreements, L1-optimal seat vector | `forecast_2026_<TAG>.csv` |
| 07b | `07b_firm_heterogeneity.py` | Cochran Q / I² test: is the 2026 disagreement between firms larger than sampling error? | `firm_heterogeneity.csv` |
| 08 | `08_turnout.py` | Structural turnout model (register abroad, sector turnout) | `turnout_forecast.json` |
| 09 | `09_build_report.py` | Builds the report page from the output tables | `report/index.html` |

Final run order used: 04 -> 04b -> 06 -> 06b -> 06c -> 06d -> 06e -> 07 (`N_SIM=40000 TAG=main`) -> 07b ->
`run_sensitivity_grid.sh` -> 08 -> 09 (01-03 and 05 do not depend on the polls and were run before).

`config.py` holds the party/letter/family mappings, official seat results, pollster lineages, the seat-allocation law
(`allocate()`: 3.25% of all valid votes, quota floor, Bader-Ofer remainder with surplus-agreement pairs; it reproduces
the official 2022 seats exactly) and the seat-to-vote conversion `seats_to_shares()` used for history and 2026 alike.

Requirements: Python 3.11+, pandas, numpy, scipy, statsmodels, scikit-learn, torch, beautifulsoup4, lxml, xlrd,
pyarrow, requests.

## Sensitivity and checks

* `run_sensitivity_grid.sh [N_SIM]`: the 16 sensitivity variants of the report. `07_forecast_2026.py` reads the
  environment variables `CORRECTION` (selected|e2|m1s|comp|none), `WINDOW_DAYS`, `HALF_LIFE`, `DISAGREE_SCALE`,
  `FAMILY_SD`, `HOUSE_CARRY`, `EXCLUDE_FIRMS`, `FIRM_WEIGHTS`, `RAW_MODE`, `N_SIM`, `TAG`.
* `reproduce_from_raw.sh <scratch_dir>`: rebuilds everything from the raw downloads (02 -> 09) in an isolated copy and
  compares the seat vector and turnout with the submitted ones (last check: identical, sum of seat differences 0).
* `HIST_SOURCE=enwiki` makes step 06 and everything after it use the English-Wikipedia history only.
