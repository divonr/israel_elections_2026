"""
Step 09 - Build the self-contained HTML report (report/index.html) from the output tables.
All numbers on the page are read from outputs/tables and data/processed; charts are drawn in the
browser from embedded JSON (no external data requests).
"""
import json
import os
import runpy
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as CF  # noqa: E402

TAG = os.environ.get("TAG", "main")
T = CF.OUTT
REPORT = os.path.join(CF.ROOT, "report")
os.makedirs(REPORT, exist_ok=True)

LETTERS = {"LIKUD": "מחל", "YASHAR": "דרך", "TOGETHER": "רק", "DEMOCRATS": "אמת", "YB": "ל", "OTZMA": "ב",
           "SHAS": "שס", "UTJ": "ג", "RZP_ZEHUT": "ט", "JOINT": "ודם", "RAAM": "עם", "HENDEL_ZELICHA": "די",
           "AMCHA": "ך", "BLUE_WHITE": "כן"}
FAMILY_LABEL = {"LIKUD": "Likud", "SHAS": "Shas (Sephardi Haredi)", "UTJ": "UTJ (Ashkenazi Haredi)",
                "NATREL": "Religious-Zionist / far right", "YB": "Yisrael Beiteinu", "CENTER": "Centre lists",
                "LEFT": "Left (Labor/Meretz/Democrats)", "ARAB": "Arab lists"}
FIRM_LABEL = {"Midgam": "Midgam (Channel 12)", "Kantar": "Kantar (Kan 11, Israel Hayom)",
              "Lazar_Panels": "Lazar / Panel4All (Maariv, Walla)", "MaagarMochot": "Maagar Mochot (Channel 16)",
              "Ch13_Madad2026": "Midgam Project / Madad (Channel 13)", "Tatika": "Tatika (Zman Yisrael)",
              "DirectPolls": "Direct Polls (i24 News)", "NextData_Filber": "Filber / NEXT DATA (Channel 14)",
              "DRI": "DRI (CEC report only)"}
RIGHT_LEANING = {"DirectPolls", "NextData_Filber", "DRI"}


def clean(o):
    """Recursively replace NaN/inf with None and numpy scalars with Python numbers (valid JSON)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def rd(path, **kw):
    p = os.path.join(T, path)
    return pd.read_csv(p, **kw) if os.path.exists(p) else None


def segment_illustration(method="nmf", K=8, unit="ballot"):
    """Profiles (2022) of one illustrative segmentation + industry representation ratios."""
    A = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "06_poll_error_analysis.py"),
                       run_name="lib")
    FAM = A["FAM_NO_OTHER"]
    h, a = A["load"]()
    P = A["poll_matrix"](h)
    prof = A["load_profiles"](method, K, unit)
    pf = []
    for pid, gg in P.groupby("poll"):
        kk = gg.knesset.iloc[0]
        fam = gg.assign(family=[CF.PARTIES[kk].get(l, ("", "OTHER"))[1] for l in gg.letters]) \
            .groupby("family").share.sum().reindex(FAM, fill_value=1e-4).values
        pf.append((gg.lineage.iloc[0], kk, fam / fam.sum()))
    lineages = sorted({x[0] for x in pf if not x[0].startswith("Other")})
    r0, _ = A["fit_composition"](pf, prof, lineages, K, lam=0.1, lam_ind=0.01, iters=150)
    base = prof[25]
    pi = base.weight.values * np.exp(r0)
    pi = pi / pi.sum()
    segs = []
    nat = (base[FAM].mul(base.weight, axis=0)).sum() / base.weight.sum()
    lift_name = {"ARAB": "Arab localities", "UTJ": "Haredi areas", "SHAS": "Shas-strong periphery",
                 "NATREL": "National-religious areas", "YB": "Russian-speaking areas", "LEFT": "Centre-left urban",
                 "CENTER": "Secular centre", "LIKUD": "Likud heartland"}
    short = {f: FAMILY_LABEL[f].split(" (")[0].replace("Religious-Zionist / far right", "Far right") for f in FAM}
    for c in base.index:
        row = base.loc[c]
        shares = {f: float(row[f]) for f in FAM}
        lift = {f: shares[f] / max(float(nat[f]), 1e-6) for f in FAM}
        top = sorted(shares.items(), key=lambda kv: -kv[1])[:2]
        key = top[0][0] if top[0][1] >= 0.40 else max(lift, key=lift.get)  # dominant family, else most over-represented
        name = f"{lift_name[key]} ({', '.join(f'{short[f]} {100 * v:.0f}%' for f, v in top)})"
        if c == K:
            name = f"Double envelopes: soldiers, diplomats, etc. ({', '.join(f'{short[f]} {100 * v:.0f}%' for f, v in top)})"
        segs.append(dict(segment=int(c), name=name, weight=float(row["weight"]), poll_weight=float(pi[c]),
                         ratio=float(pi[c] / row["weight"]), shares=shares))
    segs.sort(key=lambda s: -s["weight"])
    mult = A["comp_multipliers"](r0, base.weight.values, base[FAM].values)
    return dict(method=method, K=K, unit=unit, families=FAM, family_labels=[FAMILY_LABEL[f] for f in FAM],
                segments=segs, multipliers=dict(zip(FAM, [float(x) for x in mult])))


def main():
    fc = rd(f"forecast_2026_{TAG}.csv")
    summ = json.load(open(os.path.join(T, f"forecast_2026_{TAG}_summary.json")))
    fc = fc[fc.id.notna()]
    surplus = [("TOGETHER", "YB"), ("JOINT", "RAAM"), ("RZP_ZEHUT", "LIKUD"), ("UTJ", "SHAS"), ("DEMOCRATS", "YASHAR")]
    raw_sh = {r.id: r.raw_poll_avg_share for r in fc.itertuples() if not pd.isna(r.raw_poll_avg_share)}
    raw_seats = CF.allocate(raw_sh, 0.0325, 120, surplus, total=100.0)
    forecast = []
    for r in fc.itertuples():
        forecast.append(dict(id=r.id, name=r.list, letters=LETTERS.get(r.id, ""),
                             raw=None if pd.isna(r.raw_poll_avg_share) else round(r.raw_poll_avg_share, 2),
                             corrected=None if pd.isna(r.corrected_share) else round(r.corrected_share, 2),
                             median=None if pd.isna(r.seats_median) else float(r.seats_median),
                             mean=None if pd.isna(r.seats_mean) else round(float(r.seats_mean), 2),
                             p10=None if pd.isna(r.p10) else float(r.p10), p90=None if pd.isna(r.p90) else float(r.p90),
                             p_pass=None if pd.isna(r.p_pass_threshold) else round(float(r.p_pass_threshold), 3),
                             seats=int(r.FORECAST_SEATS), raw_seats=raw_seats.get(r.id)))
    turnout = json.load(open(os.path.join(T, "turnout_forecast.json")))
    firm_raw = pd.read_csv(os.path.join(T, f"firm_means_raw_{TAG}.csv"), index_col=0)
    firms = [dict(id=f, label=FIRM_LABEL.get(f, f), right=f in RIGHT_LEANING,
                  values={k: round(float(v), 2) for k, v in firm_raw.loc[f].items()}) for f in firm_raw.index]
    fam_tests = rd("industry_bias_by_family_tests.csv")
    fam_tests = fam_tests[fam_tests.family != "OTHER"]
    final = json.load(open(os.path.join(CF.PROC, "model_params_final.json")))
    bias = [dict(family=r.family, label=FAMILY_LABEL[r.family], pct=round(float(r.pct_bias), 1), p_t=float(r.p_t),
                 p_w=float(r.p_wilcoxon), e2_pct=round(100 * (np.exp(final["final_logfactor"][r.family]) - 1), 1))
            for r in fam_tests.itertuples()]
    ms = rd("model_selection_summary.csv")
    fr_cls = rd("model_selection_friedman_model_classes.csv", index_col=0)
    fr_alg = rd("model_selection_friedman_cluster_algorithms.csv", index_col=0)
    fr_k = rd("model_selection_friedman_cluster_K.csv", index_col=0)
    bt = rd("backtest_method_comparison.csv", index_col=0)
    by_el = rd("model_selection_scores_by_election.csv", index_col=0)
    val = rd("clustering_validation.csv")
    pers = rd("house_effect_persistence.csv").iloc[0].to_dict()
    calib = rd("calibration_summary.csv").iloc[0].to_dict()
    sens = {}
    for f in sorted(os.listdir(T)):
        if f.startswith("forecast_2026_sens_") and f.endswith(".csv"):
            d = pd.read_csv(os.path.join(T, f)).set_index("id")
            sens[f[len("forecast_2026_"):-4]] = {k: int(v) for k, v in d.FORECAST_SEATS.items() if k != "BLUE_WHITE"}
    het = rd("firm_heterogeneity.csv")
    main_firms = ["Midgam", "Kantar", "Lazar_Panels", "MaagarMochot", "DirectPolls", "Smith", "Dialog_Fuchs"]
    firm_names = {"Midgam": "Midgam (Mano Geva)", "Kantar": "Kantar / Teleseker", "Lazar_Panels": "Lazar / Panels Politics",
                  "MaagarMochot": "Maagar Mochot", "DirectPolls": "Direct Polls", "Smith": "Smith",
                  "Dialog_Fuchs": "Dialog / Camil Fuchs"}
    bm = rd("pollster_bias_matrix.csv")
    bias_matrix = [dict(lineage=firm_names.get(r.lineage, r.lineage), family=r.family, pct=round(float(r.bias_pct), 1),
                        se=round(100 * float(r.se), 1), n=int(r.n_elections))
                   for r in bm[bm.lineage.isin(main_firms)].itertuples()] if bm is not None else []
    sp = rd("pollster_segment_profiles.csv")
    seg_prof = []
    if sp is not None:
        for r in sp[sp.lineage.isin(main_firms)].itertuples():
            seg_prof.append(dict(lineage=firm_names.get(r.lineage, r.lineage), segment=int(r.segment),
                                 ratio=round(float(r.ratio), 3), n=int(r.n_elections)))
    ss = rd("backtest_slope_summary.csv")
    slope_summary = ss.to_dict("records") if ss is not None else []
    calib_par = rd("calibration_regression_params.csv")
    calib_d = float(calib_par[calib_par.lineage == "INDUSTRY"].d.iloc[0]) if calib_par is not None else None
    slope_par = rd("slope_model_params.csv")
    slope_b = float(slope_par[slope_par.lineage == "INDUSTRY"].b.iloc[0]) if slope_par is not None else None
    seg = segment_illustration()
    polls = pd.read_csv(os.path.join(CF.PROC, "polls_2026.csv"))
    # same history file as step 06 (merged English + Hebrew source unless HIST_SOURCE=enwiki)
    merged = os.path.join(CF.PROC, "hist_polls_merged.csv")
    use_merged = os.path.exists(merged) and os.environ.get("HIST_SOURCE", "merged") == "merged"
    hist = pd.read_csv(merged if use_merged else os.path.join(CF.PROC, "hist_polls.csv"))
    n_added = int(hist[hist.get("poll_source", pd.Series(index=hist.index, dtype=object)) == "hewiki"]
                  .groupby(["election", "row_idx"]).ngroups) if use_merged else 0
    meta = pd.read_parquet(os.path.join(CF.PROC, "ballots_meta.parquet"))
    inventory = dict(n_ballots=int(len(meta)), n_elections_results=int(meta.knesset.nunique()),
                     n_hist_polls=int(hist.groupby(["election", "row_idx"]).ngroups),
                     n_hist_polls_added_hewiki=n_added,
                     n_hist_elections=int(hist.election.nunique()),
                     n_polls_2026=int(polls.poll_id.nunique()), n_polls_window=int(summ["n_polls"]),
                     n_firms_window=int(summ["n_firms"]))
    data = dict(forecast=forecast, summary=summ, turnout=turnout, firms=firms, bias=bias,
                model_selection=ms.to_dict("records") if ms is not None else [],
                friedman_classes=fr_cls.reset_index().to_dict("records"),
                friedman_algos=fr_alg.reset_index().to_dict("records"),
                friedman_k=fr_k.reset_index().rename(columns={"index": "K"}).to_dict("records"),
                backtest=bt.reset_index().rename(columns={"index": "method"}).to_dict("records"),
                scores_by_election=by_el.reset_index().to_dict("records"),
                clustering=val.round(4).to_dict("records"), persistence=pers, calibration=calib,
                sensitivity=sens, segments=seg, inventory=inventory, e2=final["final_logfactor"],
                selected=final.get("selected", "e2"), selection_scores=final.get("selection_scores", {}),
                resid_sd=final.get("total_resid_sd_by_size", {}),
                heterogeneity=het.to_dict("records") if het is not None else [],
                pollster_bias=bias_matrix, pollster_segments=seg_prof, slope_summary=slope_summary,
                calib_d=calib_d, slope_b=slope_b)
    data = clean(data)
    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_template.html"), encoding="utf-8").read()
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, allow_nan=False))
    open(os.path.join(REPORT, "index.html"), "w", encoding="utf-8").write(html)
    json.dump(data, open(os.path.join(REPORT, "report_data.json"), "w", encoding="utf-8"), ensure_ascii=False,
              indent=1, allow_nan=False)
    print("report written:", os.path.join(REPORT, "index.html"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
