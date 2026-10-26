# PROGRESS / CHECKPOINT LOG (resume from here after a crash)

Competition: predict seats per list + national turnout, 26th Knesset (election 27.10.2026),
deadline 26.10.2026 23:59. Score = sum |seat errors|; tie-break turnout. Work in English only.

## Done (as of 07:40, 26.10)
1. `code/01_download_data.py` -> raw CEC ballot/settlement results K16-K25 (data/raw/results), Wikipedia poll pages (data/raw/wiki).
2. `code/02_parse_results.py` -> data/processed/ballots_long.parquet, ballots_meta.parquet, national_totals.csv, national_summary.csv (turnout checks OK).
3. `code/03_parse_wiki_polls.py` -> wiki_polls_long.csv, wiki_poll_events.csv.
4. `code/config.py` -> party letters/families/actual seats, pollster lineages, allocation (validated: +-1 seat = surplus agreements).
5. `code/04_build_poll_datasets.py` -> hist_polls.csv (2009,2015,2019a,2019b,2020,2021,2022 final tables), actual_results.csv, polls_2026.csv (Wikipedia + agents; 48 matched polls, 0 seat discrepancies; 7 agent-only polls appended).
6. Research subagent -> data/research/auxiliary_2026.md, surplus_agreements_2026.csv, lists_2026.csv.

## Key facts
- Blue & White withdrew 19.10 (CEC confirmed 21.10) -> 0 seats. Achi, HaKahal withdrew (negligible).
- Surplus pairs 2026: Together-YB, Joint-Ra'am, RZP_Zehut-Likud, UTJ-Shas, Democrats-Yashar.
- Eligible voters 2026: 7,331,981. Arab turnout expected higher than 2022 (53%).
- Huge pollster divergence: Ch14 (Filber/NextData) Likud ~30, bloc 62; mainstream Likud 18-21, bloc 46-51; i24 (Direct Polls) 26/58; DRI 28/60.

## Done (as of ~07:50)
7. `code/05_clustering.py` (resumable) -> data/processed/clusters/*, outputs/tables/clustering_validation.csv (kmeans/gmm/ward/nmf x K4-14 x ballot/settlement).
8. `code/config.py` allocate() now = exact legal procedure (quota floor + Bader-Ofer with surplus pairs as one list); reproduces 2022 exactly. `total=1.0` must be passed when shares are fractions of all valid votes.
9. `code/06_poll_error_analysis.py` -> industry bias (NATREL +20% p=.034; SHAS -11.5% p=.038), house-effect persistence weak (r=.12, perm p=.052, carry-over .24), LOEO backtest (outputs/tables/backtest_*). `code/06b_model_selection.py` -> Friedman/Nemenyi, nested selection, ensembles.
10. DECISION: final correction = E2 = 0.5*(-EB-shrunk industry family bias) + 0.5*(composition model averaged over NMF configs, industry variant, 2022 segment base). No pollster-specific house effects. Equal weight per firm. (Backtest: raw 22.5, M1s 19.75, E2 20.0 best mean rank.)
11. Background workflow wf_8dbd9651-05a (research+verify: polls, methodology, turnout, sector polls) launched ~07:35.

## Done (as of ~08:30)
12. `code/06c_final_factors.py` -> data/processed/model_params_final.json (E2 factors, LOEO residual SDs). `code/06d_calibration.py` -> outputs/tables/calibration_*.
13. `code/07_forecast_2026.py` uses model_params_final.json; env vars CORRECTION, WINDOW_DAYS, HALF_LIFE, DISAGREE_SCALE, FAMILY_SD, EXCLUDE_FIRMS, N_SIM, TAG. l1_optimal restricted to {0} U [4,120], sum 120.
14. Sensitivity grid done (outputs/tables/forecast_2026_sens_*.csv): robust; only firm-weighting matters (no-right firms: Likud 20; only-right: Likud 29).
15. BUG FIXED ~08:25: comp_multipliers NaN for family absent in base election (YB 2013) -> chain 06/06b/06c/06d re-running (task b8gry0uwe). Must re-check model selection after it finishes, then rerun 07 with N_SIM=50000 TAG=main.
16. Workflow wf_00f21fb5-c03 = adversarial code review (read-only). Workflow wf_8dbd9651-05a = research+verify.
- Preliminary forecast (pre-fix, test): Likud 23, Yashar 21, Together 13, Dems 9, YB 8, Otzma 7, Shas 9, UTJ 8, RZP_Zehut 6, Joint 10, Raam 6, Hendel 0, Amcha 0, B&W 0.

## Done (as of ~08:15)
17. Chain re-run after NaN fix: E2 now outright best in backtest (19.25 vs raw 22.5; M1s 19.75; nested-best-M3 22.25). Calibration: 80% coverage 87.6%, Brier near threshold 0.124 vs 0.172.
18. MAIN FORECAST (outputs/tables/forecast_2026_main.csv, N_SIM=40000): Likud 23, Yashar 21, Together 13, Dems 9, YB 8, Otzma 7, Shas 9, UTJ 8, RZP_Zehut 6, Joint 10, Raam 6, Hendel 0 (P pass .33), Amcha 0 (.21), B&W 0. Expected |err| 18.8.
19. Preliminary turnout 70.8% (code/08_turnout.py; inputs data/research/turnout_inputs.json, A_2022 pending research).
20. Report: code/09_build_report.py + code/report_template.html -> report/index.html (dry run rendered OK). Method text report/method_description.md (485 words). README.md written.

## Crash #2 (~08:25 -> resumed 11:04)
21. L1 optimality verified by local search (best transfer delta +0.049 -> no improvement). Unconstrained medians sum 116; cost of 120-coherence = 0.24 expected seats.
22. Sensitivity re-run with feasible optimizer: m1s,none,noDRI,noCh14,noRight,onlyRight,win6,win14 done; dis1,dis0,fam15,house re-running (b3k6jm32r).
23. Workflows resumed: research w8v124dd3 (run wf_8dbd9651-05a), review wzwx1y9yz (run wf_00f21fb5-c03).
24. Report builder fixed: NaN->null (valid JSON), NMF K8 ballot segment illustration with lift-based names.
NEXT: finalize turnout inputs from research -> 08_turnout.py -> 09_build_report.py TAG=main -> publish artifact (report/index.html) -> apply any confirmed review findings.

## Old next steps (done)
- 05 clustering of ballot boxes (k-means/GMM/Ward/NMF) + validation.
- 06 historical poll error analysis: industry & pollster house effects by family, persistence tests, composition (cluster) model, leave-one-election-out backtest.
- 07 2026 forecast: corrected average, Monte Carlo with allocation + surplus pairs, L1-optimal integer seats.
- 08 turnout model. 09 report + 500-word method text.

## After crash #2 (11:04-11:30)
25. Crash zeroed code/report_template.html and report/method_description.md (NUL bytes) and emptied sens_win6/win14 CSVs. Template RECOVERED from report/index.html (data blob -> placeholder); method text rewritten; win6/win14 re-run. Use `powershell Write-VolumeCache C` after important writes.
26. Research (cached): all 7 agent-only polls confirmed vs CEC reports; no missing polls 16-23 Oct; pollster-methodology findings (NEXT DATA ownership/low response, Direct Polls track record) added to report text.
27. New sensitivity: FIRM_WEIGHTS env var; halfOpaque = identical to main; rightOneVote = Likud 22 / Yashar 22.
28. REPORT PUBLISHED: https://claude.ai/artifact/GZpvXudLDMjz3Z16XDRG9P (republish same file path to update).
NEXT: turnout finalization when research turnout agent returns; apply confirmed review findings; republish.
29. 04: date fixes (Kan, i24 final polls end 22 Oct). 07: merges data/research/cec_raw_pct_2026.csv if present (USE_CEC_RAW=1). Background agent extracting CEC raw % for 8 final polls. After it lands: rerun 07 TAG=main (N_SIM=40000) + sensitivity, 09, republish.
30. CONSISTENCY FIX: 07 now converts 2026 polls exactly like history (seats for passing lists; reported % for 0-seat lists); RAW_MODE=raw is a sensitivity. Ch16 CEC table excluded (unweighted). FINAL MAIN: Likud 23, Yashar 22, Together 13, Dems 8, YB 8, Otzma 7, Shas 9, UTJ 8, RZP 6, Joint 10, Raam 6, Hendel 0 (.31), Amcha 0 (.21). E|err| 18.67. Sensitivity grid re-running.
31. 07 writes firm_means_raw_<TAG>.csv (report reads _main). 07b_firm_heterogeneity.py: Cochran Q/I2 (Likud I2=.97, p<1e-50; Together .90; YB .81; Amcha .80; Shas .73; Hendel .62).
32. Sensitivity grid (15 variants) re-run in consistent mode: all methodological variants within one moved seat of main; only noRight/none/onlyRight differ more.
33. Report v3 published (same URL). Submission card + method text updated (Yashar 22, Dems 8).
PENDING: research turnout/sector agents (turnout inputs A_2022), code-review workflow verdicts.
34. TURNOUT FINAL: 08_turnout.py rebuilt (Arab/Druze-locality definition; A_2022 600k CBS; A_2026 810k; Arab 60%+-4; non-Arab +1+-2pp; flyers 35k) -> median 70.9% (68.8-73.0). Report/submission/method updated.
35. REVIEW FIXES started ~12:05: 02 unique ballot_uid (K20 dups), 05 NMF contribution-based labels; full re-cluster running; old clusters in data/processed/clusters_old.
36. REVIEW-DRIVEN FIXES (12:00-12:15), chain NOT yet re-run:
  - 04: misdated rows (monotone dates per table), joint-column sum, IH 22.10 Amcha 0 seats/2.7%.
  - 06: family_errors() = size-weighted per election-family error, excluding OTHER and 0-seat placeholder lists (used by M1/M1s/E2, tests, regression); M1 via per-poll renormalisation; M1r without OTHER.
  - 06b: E5 (composition-NMF only) added to compared classes (it scored 18.75 < E2 19.25 before fixes).
  - 06c: selects final correction (e2 | comp(=E5) | m1s) by backtest mean; residual SD from measured lists only. 07 CORRECTION default = "selected".
  - config.seats_to_shares(): D'Hondt-midpoint conversion (s+0.5)*T/(S+J/2), used by BOTH history (06) and 2026 (07). Validated vs CEC raw % (IH 22.10: Hendel 3.42 vs raw 3.7; Likud 16.35 vs 16.5).
  - Reason: calibration reliability showed P(pass) under-confident (pred .69 vs obs .78; Platt intercept +0.69 p=.09); seat/120 conversion put 4-seat lists below threshold.
  NEXT: after recluster finishes -> 06, 06b, 06c, 06d, 07 main, 07b, sensitivity, 09, republish.
37. 12:10 recluster done (NMF silhouettes .29-.32, ARI .95-.99). Running chain 06->06b->06c->06d->07 main->07b (task log outputs/chain2_log.txt).
38. 12:35 CLEAN CHAIN DONE. Backtest: raw 20.0, M1s 18.75, E2 18.75, E5(comp NMF) 17.0 (5W/0L, p=.0625) -> SELECTED comp. Calibration: Platt intercept .19 (p=.70), Brier .103 vs .165, coverage 87.6%. FORECAST: Likud 23, Yashar 21, Together 13, Dems 9, YB 8, Otzma 7, Shas 9, UTJ 8, RZP 6, Joint 10, Raam 6, Hendel 0 (.34), Amcha 0 (.23). Sensitivity re-running.
39. 12:45 FINAL: sensitivity grid (16 variants) robust; report v5 published (same URL) with limitations & review section. Turnout harmonised (Arab 62% +- 4.5, consistent with 16 Arab seats) -> 71.2%.
   FINAL ANSWER: Likud 23, Yashar 21, Together 13, Joint 10, Dems 9, Shas 9, YB 8, UTJ 8, Otzma 7, RZP_Zehut 6, Raam 6, Hendel 0, Amcha 0, B&W 0; turnout 71.2%.
40. 12:40 REVIEW clustering#1 (verified REAL): segment truth excluded double envelopes (~10% votes) -> composition model targeted geographic not official result (perfect-poll test gave nonzero corrections). FIX: load_profiles adds a double-envelope pseudo-segment per election (national minus geographic votes by family); w@P now equals official shares exactly. fit_composition infers K from profiles. clustering#3 (leakage) verified REAL but ZERO impact (per-fold refits give identical E2/E5 scores). Chain re-running (chain4).
41. 13:05 chain4 (envelope segment): backtest raw 20.0, M1s 18.75, E2 18.25 (5W/1L), E5 19.75 -> SELECTED e2. FORECAST: Likud 23, Yashar 22, Together 13, Joint 10, Shas 9, Dems 8, YB 8, UTJ 8, Otzma 7, RZP 6, Raam 6, Hendel 0 (.38), Amcha 0 (.22). Sensitivity re-running.
42. 13:10 FINAL v6 published (same URL). Sensitivity (16 variants) robust. Method text 459 words. Submission card final.
   FINAL ANSWER: Likud 23, Yashar 22, Together 13, Joint 10, Shas 9, Dems 8, YB 8, UTJ 8, Otzma 7, RZP_Zehut 6, Raam 6, Hendel 0, Amcha 0, B&W 0; turnout 71.2%.
   Code review workflow complete: all number-changing findings fixed; residual limitations documented in the report.
43. 13:30 Stop-hook gaps addressed: 06e_pollster_profiles.py (pollster x family bias matrix + jackknife, bias+slope M6 (worse 25.5), calibration regression M7 (19.0), pollster x segment matrix); report section 'Pollster profiles and weight matrices'. Hebrew-Wikipedia historical polls workflow running (wf_e5e7b617-85c).
44. 13:10 report v7 published (pollster section). PDF export report/Knesset26_forecast_report.pdf (16 pages). robustness_merged_polls.sh prepared; waiting for Hebrew-wiki workflow.
45. 13:15 User asked: Hebrew copies of final explanatory files (method_description, README, PROGRESS, submission) -> do via translate+verify workflow after final docs; GPU not needed (told user); at the end: Desktop summary file + notification (PushNotification + Windows balloon). Launched reproduce_from_raw.sh (scratch/repro) in background; Hebrew-wiki workflow still extracting.
46. 13:31 REPRODUCIBILITY: reproduce_from_raw.sh (02->09 from raw downloads, isolated copy) reproduced the submitted seat vector exactly (sum|diff|=0) and turnout 71.16%. Runtime ~17 min.
47. 14:30 Internet outage: workflow survived. hewiki preview: 191 polls, 167 matched (97% identical, 100% within 2 seats), 24 new (10 in final fortnight: 2013 7, 2015 3) -> rule met: adopt merged history. 06 load() prefers hist_polls_merged.csv (HIST_SOURCE=merged default); reproduce script includes 04b.
48. 16:30-16:45 Hebrew-source poll collection finished for all 8 elections (2009 44, 2013 48, 2015 51, 2019a 39, 2019b 28, 2020 29, 2021 31, 2022 35 polls).
   - Independent verifiers (high confidence each): 2013 7 rows fixed + 3 missing polls added; 2015 0; 2019a 3; 2019b 6; 2020 1; 2021 5; 2022 0.
   - Hebrew Wikipedia has NO poll table for 2009, 2013 or 2015; those files were built from archived primary sources (bhirot2009, IMRA, mako/nrg/Globes/Ynet/Haaretz/JPost archives, themadad etc.).
   - 2009 extractor spent ~3.5 h in 10-minute waits on Wayback Machine HTTP 429s; told to wrap up (file final, 8 rows fully primary-confirmed, 36 flagged UNCONFIRMED in notes). 2009 adds only 1 new poll (20 Jan, outside the 14-day window) -> no effect on the model.
   - Old workflow stopped (its 2009 slot never received the hand-back); 2009 verification re-launched as its own workflow (wf_bb264615-e5d).
   - FIX config.pollster_lineage: New Wave/Gal Hadash -> NewWave, Shvakim Panorama -> ShvakimPanorama, Shiluv Millward Brown -> Shiluv, TRI -> TRI, "Meno/Mano Geva" -> Midgam (both Wikipedia editions now get the same firm label; prevents duplicate merges in 04b).
   - FIX 04: drop non-poll reference rows ("Current Composition" of the outgoing Knesset, dated 2006, was counted as a 2009 poll; outside the window, so counts only).
   - NEW code/run_sensitivity_grid.sh (the 16-variant grid as a script; house variant reads the carry-over slope from step 06).
   - 09: history inventory counts the merged file (n_hist_polls, n_hist_polls_added_hewiki).
   - Rehearsal of the full chain on the merged history running in an isolated copy (scratch/prelim).
49. 16:44 STOPPING POINT (user shutting the computer down; resume on "continue").
   - Further fixes: config.pollster_lineage "Dahaf" checked before Midgam ("Dahaf (Mina Tzemach)" -> Dahaf, like English "Channel 2/Dahaf"); 04b matches the English date against BOTH Hebrew dates (fieldwork and publication), 3-4 day gaps only for near-identical tables. This removed 2 true duplicates (2009 Dialog/Haaretz 13 vs 16 Jan; 2013 Maagar Mochot 8 vs 11 Jan); the two Panels look-alikes (2013 Jan 18 Sof Hashavua, 2015 Mar 5 Maariv/103FM) are documented as separate polls and stay.
   - Rehearsal 04b result: 299 Hebrew-source polls, 272 matched (262 identical tables, mean |diff| 0.06 seats), 27 new (2009 1, 2013 19, 2015 6, 2021 1), of which 12 in the final fortnight (2013 8, 2015 3, 2021 1).
   - All background jobs stopped cleanly. Backup of all 8 Hebrew-source files: data/raw/hewiki_backup/ (polls_2009.csv = extractor's final, 16:36).
   - MAIN OUTPUTS ARE STILL THE SUBMITTABLE v8 FORECAST (English-Wikipedia history): Likud 23, Yashar 22, Together 13, Joint 10, Shas 9, Dems 8, YB 8, UTJ 8, Otzma 7, RZP_Zehut 6, Raam 6, others 0; turnout 71.2%. data/processed/hist_polls_merged.csv in MAIN is an old partial preview; do NOT run 06 in main before re-running 04 + 04b.
   RESUME PLAN: (1) relaunch verify:2009 (Workflow scriptPath .../workflows/scripts/k26-verify-2009-wf_bb264615-e5d.js, fresh run); it does not block the model (2009 adds no in-window poll). (2) In MAIN: 04 -> 04b -> 06 -> 06b -> 06c -> 06d -> 06e -> 07 (N_SIM=40000 TAG=main) -> 07b -> run_sensitivity_grid.sh -> 08 -> 09 (~25 min). (3) Update report template text (poll sources: + Hebrew/archival source, cross-check numbers), method text (<=500 words), README (04b, 06e, run_sensitivity_grid.sh, robustness/reproduce scripts, hewiki data), submission.md. (4) Republish report (same file path), PDF, rebuild both zips. (5) Hebrew copies (*_he.md) via translate+verify workflow. (6) Desktop summary file + notifications.
   - Note: TaskStop on a background bash leaves the inner script running; kill the process tree (ps -ef | grep prelim_run / python) before restarting. The empty 06 log in scratch/prelim came from that kill, not from a crash. All processes stopped at 16:44.
50. 17:04-17:20 FINAL RUN on the merged history (main project): 04 -> 04b -> 06 -> 06b -> 06c -> 06d -> 06e -> 07 -> 07b -> sensitivity grid -> 08 -> 09.
   - 04: 2026 polls file byte-identical to v8; 19 history polls relabelled to consistent firm lineages; 489 English-Wikipedia polls (the non-poll "Current Composition" row dropped). 04b: 516 polls in the merged history.
   - Industry bias: far right +20% (p = 0.023), Shas -10% (p = 0.057). House-effect persistence r = 0.16 (permutation p = 0.011), carry-over slope 0.29.
   - Backtest: raw 20.0, M1s 18.75, E2 18.75 (4W/2L), E5 19.0, E1/E3 18.5 (not pre-specified candidates), nested-best 20.5. 06c: tie E2 = M1s -> pre-specified primary E2 kept (tie rule now commented in code). Factors: Likud +4.2, Shas +7.0, UTJ +1.5, far right -12.0, YB -0.8, Centre +1.8, Left -5.3, Arab +6.2 (%).
   - Calibration: coverage 88.7%, Brier 0.115 vs 0.177, Platt intercept 0.39 (p = 0.38).
   - FORECAST IDENTICAL TO v8 (sum |diff| = 0): Likud 23, Yashar 22, Together 13, Joint 10, Shas 9, Dems 8, YB 8, UTJ 8, Otzma 7, RZP_Zehut 6, Raam 6, Hendel 0 (P pass .38), Amcha 0 (.22). The v8 outputs (English-only history) are kept in ../v8_backup as the HIST_SOURCE=enwiki comparison.
   - Method text updated (497 words), submission card, README (04b, run_sensitivity_grid.sh, checks), report template (second-source paragraph).
51. 17:20 2009 verification (wf_4779dd76-c44): 44 rows checked (23 in the final 14 days), 0 errors, high confidence; file unchanged -> no re-run needed. All 8 elections of the second source are now independently verified (22 rows fixed in total, 3 missing polls added). The verifier notes the claimed English-Wikipedia "9 in each Arab column" (Makor Rishon 5 Feb 2009) is a merged-cell parsing artefact, not an error.
   Report v9 published (same URL) with tie wording; PDF regenerated; Hebrew copies (method, submission, README, PROGRESS, full Hebrew results report report/report_he.md) produced and number-checked by independent reviewers.
52. 17:40 CLEAN-UP: superseded material moved to _archive_not_main/ (with README): old run logs, pre-fix clusters, clustering_validation_old, test simulations, robustness_merged_polls.sh, and the v8 results (English-Wikipedia history only). Current step logs in outputs/logs/ (scripts updated). PROGRESS.md -> docs/DECISION_LOG.md. README rewritten as the single entry point (final answer, folder map, pipeline, checks; Hebrew README_he.md). Identical hewiki backup and the stale full zip deleted; empty outputs/figures and unused config.OUTF removed. Report rebuilt (reproducibility paragraph lists 04b and points to docs/DECISION_LOG.md); PDF and code zip rebuilt.
