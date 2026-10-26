"""
Step 06c - Final correction factors for 2026 + empirical error model for the simulation.

Correction (selected in 06b): E2 = 0.5 * (-EB-shrunk industry family bias)
                                 + 0.5 * (composition-model factor averaged over all NMF segmentations)
 * industry bias and composition parameters are fitted on ALL 8 elections (2009-2022);
 * the composition factor uses the 2022 (K25) segment sizes and family profiles as the base.

Error model: leave-one-election-out residuals of the E2-corrected poll average,
   resid = log(actual share) - log(corrected poll share), for every list that ran;
 -> family-level election shocks (mean residual per family x election) and list-level residuals,
    with SD by list size; plus an empirical check of threshold-crossing calibration.
Output: data/processed/model_params_final.json, outputs/tables/e2_loeo_residuals.csv
"""
import json
import os
import runpy
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as CF  # noqa: E402

warnings.filterwarnings("ignore")
A = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "06_poll_error_analysis.py"),
                   run_name="lib")
FAM = A["FAM_NO_OTHER"]
NMF_CFGS = [("nmf", K, u) for K in (4, 6, 8) for u in ("ballot", "settlement")]


def poll_families(P):
    out = []
    for pid, gg in P.groupby("poll"):
        kk = gg.knesset.iloc[0]
        fam = gg.assign(family=[CF.PARTIES[kk].get(l, ("", "OTHER"))[1] for l in gg.letters]) \
            .groupby("family").share.sum().reindex(FAM, fill_value=1e-4).values
        out.append((gg.lineage.iloc[0], kk, fam / fam.sum()))
    return out


def e2_logfactors(P, D, base_k, exclude=None):
    """family -> log correction factor (E2), fitted on elections != exclude, base segments of base_k."""
    Pt = P if exclude is None else P[P.knesset != exclude]
    Dt = D if exclude is None else D[D.knesset != exclude]
    m1s = A["eb_shrink"](A["family_errors"](Dt))
    pf = poll_families(Pt)
    lineages = sorted({x[0] for x in pf if not x[0].startswith("Other")})
    comps = []
    for (method, K, unit) in NMF_CFGS:
        prof = A["load_profiles"](method, K, unit)
        r0, _ = A["fit_composition"](pf, prof, lineages, K, lam=0.1, lam_ind=0.01, iters=150)
        base = prof[base_k]
        comps.append(-np.log(A["comp_multipliers"](r0, base.weight.values, base[FAM].values)))
    comp = dict(zip(FAM, np.mean(comps, 0)))
    comp_sd = dict(zip(FAM, np.std(comps, 0)))
    e2 = {f: 0.5 * (-m1s.get(f, 0.0)) + 0.5 * comp[f] for f in FAM}
    return e2, {f: -m1s.get(f, 0.0) for f in FAM}, comp, comp_sd


def select_method():
    """Pick the final correction among the pre-specified candidates by mean LOEO backtest score
    (outputs/tables/model_selection_summary.csv from 06b)."""
    ms = pd.read_csv(os.path.join(CF.OUTT, "model_selection_summary.csv")).set_index("method")
    cand = {"e2": "E2_M1s+compNMF", "comp": "E5_compNMF_only", "m1s": "M1s_industry_family_shrunk"}
    scores = {k: float(ms.loc[v, "mean"]) for k, v in cand.items() if v in ms.index}
    best = min(scores, key=scores.get)  # ties go to the pre-specified primary model, in the order e2, comp, m1s
    return best, scores


def main():
    h, a = A["load"]()
    P = A["poll_matrix"](h)
    D, _, _ = A["industry_bias"](P, a)
    seats_of = D.set_index(["knesset", "letters"]).poll_seats
    selected, scores = select_method()
    print("selection by backtest mean:", scores, "->", selected)
    # ---- final factors for 2026 (all elections, 2022 base)
    e2, m1s, comp, comp_sd = e2_logfactors(P, D, base_k=25)
    final_f = {"e2": e2, "comp": comp, "m1s": m1s}[selected]
    print("E2 log-factors:", {k: round(v, 3) for k, v in e2.items()})
    print("  M1s part:", {k: round(v, 3) for k, v in m1s.items()})
    print("  composition part (=E5):", {k: round(v, 3) for k, v in comp.items()}, "sd across NMF cfgs:",
          {k: round(v, 3) for k, v in comp_sd.items()})
    # ---- LOEO residuals of the SELECTED correction
    rows = []
    for k in sorted(P.knesset.unique()):
        f_e2, f_m1s, f_comp, _ = e2_logfactors(P, D, base_k=k - 1, exclude=k)
        f_k = {"e2": f_e2, "comp": f_comp, "m1s": f_m1s}[selected]

        def corr(lin, fam):
            return 1.0 if fam == "OTHER" else float(np.exp(f_k[fam]))
        sh = A["pollster_average"](P, k, corr)
        raw = A["pollster_average"](P, k)
        act = a[a.knesset == k].set_index("letters").share
        for l, v in sh.items():
            if l in act.index and act[l] > 0:
                rows.append(dict(knesset=k, letters=l, family=CF.PARTIES[k].get(l, ("", "OTHER"))[1],
                                 poll_seats=float(seats_of.get((k, l), np.nan)),
                                 corrected=v, raw=raw[l], actual=act[l],
                                 resid=np.log(act[l]) - np.log(v), resid_raw=np.log(act[l]) - np.log(raw[l]),
                                 seats_actual=CF.ACTUAL_SEATS[k].get(l, 0)))
        print(f"  LOEO residuals K{k}", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(CF.OUTT, "e2_loeo_residuals.csv"), index=False, encoding="utf-8-sig")
    # error model: measured lists only (0-seat placeholders carry no poll measurement), no OTHER
    Rm = R[(R.family != "OTHER") & (R.poll_seats > 0.05)].copy()
    fam_shock = Rm.groupby(["knesset", "family"]).resid.mean()
    Rm["size"] = pd.cut(Rm.corrected, [0, 0.04, 0.08, 0.15, 1.0])
    sd_size = Rm.groupby("size").resid.std()
    n_size = Rm.groupby("size").resid.count()
    print("total resid sd by size (measured lists):", sd_size.round(3).to_dict(), "n:", n_size.to_dict())
    print("raw vs corrected mean |log err|:", Rm.resid_raw.abs().mean().round(4), Rm.resid.abs().mean().round(4))
    params = dict(selected=selected, selection_scores=scores, e2_logfactor=e2, m1s_logfactor=m1s,
                  comp_logfactor=comp, comp_logfactor_sd=comp_sd, final_logfactor=final_f,
                  family_shock_sd=float(fam_shock.std()),
                  total_resid_sd_by_size={str(k): float(v) for k, v in sd_size.items()},
                  n_by_size={str(k): int(v) for k, v in n_size.items()}, n_resid=int(len(Rm)))
    json.dump(params, open(os.path.join(CF.PROC, "model_params_final.json"), "w"), indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
