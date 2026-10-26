"""
Step 08 - National turnout forecast (official definition: voters / all registered eligible voters).

Structural decomposition with ONE consistent sector definition (Arab & Druze localities, as in the
standard Rudnitzky series), calibrated on 2022 and projected to 2026 by Monte Carlo:

  2022:  t_J = (V_2022 - t_A,2022 * ArabLoc_2022) / (E_2022 - A_2022 - ArabLoc_2022)
         (non-Arab turnout among voters living in Israel; double envelopes counted with them)
  2026:  D = E_2026 - A_2026                      (eligible voters living in Israel)
         V = (1 - s) * D * (t_J + delta) + s * D * t_A,2026 + flyers
         T = V / E_2026
  E = register, A = registered voters living abroad (no absentee voting), s = Arab-locality share of D.

The point forecast is the median (it minimises expected absolute error, the competition tie-breaker).
Inputs, distributions and sources: data/research/turnout_inputs.json.
Outputs: outputs/tables/turnout_forecast.json, turnout_scenarios.csv, turnout_inputs.csv.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as CF  # noqa: E402

RNG = np.random.default_rng(7)
N = 200000
INPUTS = json.load(open(os.path.join(CF.ROOT, "data", "research", "turnout_inputs.json"), encoding="utf-8"))


def draw(spec, n):
    if spec["dist"] == "fixed":
        return np.full(n, float(spec["value"]))
    if spec["dist"] == "normal":
        return RNG.normal(spec["mean"], spec["sd"], n)
    raise ValueError(spec)


def project(E22, V22, A22, AL22, tA22, E26, A26, s26, tA26, dJ, fly):
    tJ22 = (V22 - tA22 * AL22) / (E22 - A22 - AL22)
    D26 = E26 - A26
    V26 = (1 - s26) * D26 * np.clip(tJ22 + dJ, 0.5, 0.97) + s26 * D26 * np.clip(tA26, 0.3, 0.9) + fly
    return 100 * V26 / E26, V26, tJ22


def main():
    I = INPUTS
    args = {k: draw(I[k], N) for k in ["E_2022", "V_2022", "A_2022", "arab_loc_eligible_2022", "arab_turnout_2022",
                                       "E_2026", "A_2026", "arab_loc_share_domestic_2026", "arab_turnout_2026",
                                       "jewish_turnout_change", "expat_flyers"]}
    T26, V26, tJ22 = project(*args.values())
    out = dict(turnout_median=float(np.median(T26)), turnout_mean=float(T26.mean()),
               p10=float(np.percentile(T26, 10)), p90=float(np.percentile(T26, 90)),
               voters_median=float(np.median(V26)), nonarab_domestic_turnout_2022=float(np.median(tJ22)),
               domestic_turnout_2022=float(np.median(I["V_2022"]["value"] / (I["E_2022"]["value"] - args["A_2022"]))))
    sc = []
    for a22 in (570000, 600000, 630000):
        for ta in (0.55, 0.60, 0.65):
            for dj in (-0.01, 0.0, 0.01, 0.02):
                t, _, _ = project(I["E_2022"]["value"], I["V_2022"]["value"], a22, I["arab_loc_eligible_2022"]["value"],
                                  I["arab_turnout_2022"]["value"], I["E_2026"]["value"], I["A_2026"]["mean"],
                                  I["arab_loc_share_domestic_2026"]["mean"], ta, dj, I["expat_flyers"]["mean"])
                sc.append(dict(abroad_2022=a22, arab_turnout_2026=ta, nonarab_change=dj, turnout_2026=float(t)))
    pd.DataFrame(sc).to_csv(os.path.join(CF.OUTT, "turnout_scenarios.csv"), index=False)
    pd.DataFrame([{"input": k, **v} for k, v in I.items()]).to_csv(os.path.join(CF.OUTT, "turnout_inputs.csv"),
                                                                  index=False, encoding="utf-8-sig")
    json.dump(out, open(os.path.join(CF.OUTT, "turnout_forecast.json"), "w"), indent=1)
    print("historical official turnout:", CF.OFFICIAL_TURNOUT)
    print({k: round(v, 4) for k, v in out.items()})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
