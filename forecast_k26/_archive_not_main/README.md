# Archive (kept, not part of the final pipeline)

Nothing here is needed to understand or reproduce the final forecast. It is kept for traceability.

| Folder | What it is |
|---|---|
| `results_v8_english_wikipedia_history_only/` | Tables and parameters of the earlier final run (report v8), which used the English-Wikipedia poll history only. Its seat vector is identical to the final one; it serves as the `HIST_SOURCE=enwiki` comparison. |
| `old_logs/` | Console logs of superseded runs during the day (before the code-review fixes, re-clustering and the merged history). |
| `old_clusters/clusters_before_review_fix/` | Voter segments before two review fixes (unique ballot IDs for 2015, contribution-based NMF labels). |
| `old_tables/` | `clustering_validation_old.csv`, the clustering validation of those old segments. |
| `old_simulations/` | Simulation draws of test runs. |
| `old_scripts/robustness_merged_polls.sh` | Earlier check that ran the model on the merged poll history in an isolated copy; obsolete now that the merged history is the default. |

# ארכיון (נשמר, אינו חלק מהפייפליין הסופי)

שום דבר כאן אינו נחוץ כדי להבין או לשחזר את התחזית הסופית; הקבצים נשמרו לצורך מעקב.
`results_v8_english_wikipedia_history_only` = תוצאות הריצה הקודמת (היסטוריית סקרים מוויקיפדיה האנגלית בלבד, אותו וקטור מנדטים);
`old_logs` = יומני ריצות שהוחלפו; `old_clusters` = הפלחים לפני תיקוני הסקירה; `old_tables`, `old_simulations` = טבלאות והגרלות ישנות;
`old_scripts` = סקריפט בדיקה שהתייתר.
