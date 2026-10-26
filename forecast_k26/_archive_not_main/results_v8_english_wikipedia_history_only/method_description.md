# Method description (competition form, systematic track)

**Model: poll aggregation corrected by historically-validated error models (industry bias plus a voter-segment composition model), Monte-Carlo seat allocation, and an L1-optimal single guess.**

1. **Data.** Official CEC ballot-box results 2003–2022 (≈100,000 boxes, 10 elections). Every 2009–2022 seat poll on Wikipedia (≈490 polls, 8 elections). 82 polls for 2026, from Wikipedia merged with two independent collections. The 48 shared polls agree exactly, and the rest were verified against CEC reports or outlets. Blue & White withdrew (19 Oct) and is set to 0. The five official surplus-vote agreements are applied.

2. **Voter segments.** Ballot boxes (and settlements) from all elections are pooled in a 9-family vote-share space and clustered with k-means, Gaussian mixtures, Ward and NMF (K = 4–14). Internal validity (silhouette, Calinski–Harabasz, Davies–Bouldin, BIC, bootstrap ARI) is reported. A double-envelope segment (soldiers, diplomats, etc.) completes the electorate.

3. **How polls err.** Each pollster's sample is modelled as a re-weighting of the true segments, π ∝ w·e^ρ, fitted with PyTorch to final-fortnight polls. Polls over-sample national-religious areas and under-sample Likud-heartland, Shas-periphery, Arab and Haredi areas. Industry bias by family is size-weighted and empirical-Bayes shrunk. Far-right lists are over-estimated ≈18% (p = 0.023) and Shas under-estimated ≈10% (p = 0.07). Seats are converted to votes at the midpoint of each list's Bader-Ofer interval, which reproduces the CEC raw percentages. Per-pollster bias, bias+slope and segment-representation matrices were estimated (jackknife SEs). House effects carry over only weakly (slope 0.24), and no pollster-specific matrix beat the industry-level correction out of sample.

4. **Model selection.** Leave-one-election-out backtest with the competition metric. Raw firm average: 20.0 seats error per election. Industry correction: 18.75. Composition model averaged over NMF segmentations: 19.75. Their ensemble: 18.25 (5 wins, 1 loss), and this is used. Choosing the single best clustering in-sample scores worse than raw (winner's curse), hence averaging. Friedman/Nemenyi tests are reported.

5. **2026.** Polls fielded 14–23 Oct are averaged within firm (4-day half-life), then one vote per firm across nine firms. Corrections: Arab lists and Shas +7%, Likud +4%, UTJ and Centre +2%, YB −1%, Left −6%, far right −12%. 40,000 elections are simulated with out-of-sample error sizes by list size, a family shock, and a shock along the main axis of the 2026 firm disagreement (Cochran Q for Likud p < 10⁻⁵⁰). Each draw is allocated by the exact law: 3.25% of valid votes, quota floor, Bader-Ofer with surplus pairs. Backtested 80% intervals cover 89% of outcomes, and threshold probabilities are calibrated (Brier 0.12 vs 0.18).

6. **Single guess.** Each list's optimal guess is its simulated median, adjusted to a feasible 120-seat vector (0 or ≥4 per list) by the smallest increase in expected error. The result is robust to dropping or down-weighting divergent firms.

7. **Turnout.** Registered voters abroad (≈810,000) cannot vote. Turnout is modelled as domestic electorate × sector turnout (Arab localities 62% ± 4.5, others +1 point vs 2022), calibrated on 2022. Median: 71.2%.
