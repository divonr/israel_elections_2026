"""
Step 07b - Is the 2026 disagreement between polling firms real, or sampling noise?

For each list: Cochran's Q test of heterogeneity across firms (final 10-day window), using each firm's
time-weighted mean vote share and a binomial standard error with the firm's effective sample size
(sum of its polls' n, deflated by a design effect of 1.5 for weighting). I^2 = (Q - df) / Q is the share of
between-firm variation not explained by sampling error. A large, significant Q justifies treating
"which firms are right" as a separate uncertainty term in the simulation (step 07).
Output: outputs/tables/firm_heterogeneity.csv
"""
import os
import runpy
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as CF  # noqa: E402

F = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "07_forecast_2026.py"), run_name="lib")
DEFF = 1.5
N_FIX = {"2026-10-23_Kan 11": 1106, "2026-10-22_Channel 16": 5000, "2026-10-23_i24 News": 5124}  # CEC/outlet-verified


def main():
    T = F["build_poll_table"]()
    _, firm, _ = F["aggregate"](T, lambda lin, l: 1.0)
    p = pd.read_csv(os.path.join(CF.PROC, "polls_2026.csv"))
    n = p.groupby("poll_id").sample_size.apply(lambda s: pd.to_numeric(s, errors="coerce").max())
    for k, v in N_FIX.items():
        n[k] = v
    T["n"] = T.poll_id.map(n).fillna(500)
    n_eff = T.groupby("lineage").n.sum() / DEFF
    rows = []
    for l in F["LISTS"]:
        pf = firm[l].clip(1e-4)
        w = n_eff.reindex(pf.index) / (pf * (1 - pf))
        pbar = np.sum(w * pf) / np.sum(w)
        Q = float(np.sum(w * (pf - pbar) ** 2))
        df = len(pf) - 1
        rows.append(dict(list=CF.NAMES_2026[l], id=l, firm_min=100 * pf.min(), firm_max=100 * pf.max(),
                         Q=Q, df=df, p_value=float(stats.chi2.sf(Q, df)), I2=max(0.0, (Q - df) / Q) if Q > 0 else 0.0))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(CF.OUTT, "firm_heterogeneity.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
