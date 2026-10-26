"""
Step 06e - Pollster profiles and pollster "weight matrices" (bias, slope, segment representation).

For every pollster lineage with final-fortnight polls in >= 2 of the 8 back-test elections:

 (1) Pollster x family BIAS matrix (direct house effects): mean over elections of
     log(pollster's family share) - log(official family share), size-weighted over measured lists,
     with a jackknife (leave-one-election-out) standard error.
 (2) BIAS + SLOPE matrix (the "matrix weight" idea): per pollster h and family F
         log(poll share of list j) = a_hF + b_h * log(true share of list j) + e
     estimated by ridge-penalised least squares around the industry line (a_F, b); deviations of each
     pollster are shrunk toward the industry. b < 1 means the pollster compresses differences between
     lists (over-states small lists, under-states large ones). Correction inverts the line:
         log(v_hat) = (log v - a_hF) / b_h.
     Back-tested leave-one-election-out exactly like every other method (competition score).
 (3) Pollster x SEGMENT representation matrix from the composition model (step 06) with pollster
     deviations: pi_hk / w_k for each voter segment (NMF K=8 ballot-box segmentation + the double-envelope
     segment), and the family biases it implies, averaged over the six NMF segmentations; jackknife SEs.

Outputs (outputs/tables): pollster_bias_matrix.csv, pollster_segment_profiles.csv,
pollster_comp_family_bias.csv, slope_model_params.csv, backtest_slope_models.csv
"""
import os
import runpy
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as CF  # noqa: E402

warnings.filterwarnings("ignore")
A = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "06_poll_error_analysis.py"),
                   run_name="lib")
FAM = A["FAM_NO_OTHER"]
NMF_CFGS = [("nmf", K, u) for K in (4, 6, 8) for u in ("ballot", "settlement")]
MAIN_FIRMS = ["Midgam", "Kantar", "Lazar_Panels", "MaagarMochot", "DirectPolls", "Smith", "Dialog_Fuchs",
              "Geocartography", "MidgamProject"]


def pollster_list_table(P, a):
    """pollster x election x list: mean poll share, actual share, family, measured flag."""
    rows = []
    for (e, k), g in P.groupby(["election", "knesset"]):
        act = a[a.knesset == k].set_index("letters").share
        pm = g.groupby(["lineage", "letters"]).agg(share=("share", "mean"), seats=("seats", "mean")).reset_index()
        for r in pm.itertuples():
            if r.letters in act.index and act[r.letters] > 0:
                rows.append(dict(election=e, knesset=k, lineage=r.lineage, letters=r.letters,
                                 family=CF.PARTIES[k].get(r.letters, ("", "OTHER"))[1], poll=r.share,
                                 true=act[r.letters], seats=r.seats))
    T = pd.DataFrame(rows)
    T = T[(T.family != "OTHER") & (T.seats > 0.05) & ~T.lineage.str.startswith("Other")]
    return T


# ------------------------------------------------------------------ (1) bias matrix
def bias_matrix(T):
    fe = T.groupby(["lineage", "election", "family"])[["poll", "true"]].sum()
    fe = (np.log(fe.poll) - np.log(fe.true)).rename("err").reset_index()
    out = []
    for (h, f), g in fe.groupby(["lineage", "family"]):
        n = g.election.nunique()
        if n < 2:
            continue
        m = g.err.mean()
        jk = np.array([g[g.election != e].err.mean() for e in g.election.unique()])
        se = np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2))
        out.append(dict(lineage=h, family=f, n_elections=n, bias_pct=100 * (np.exp(m) - 1), log_bias=m, se=se,
                        t=m / se if se > 0 else np.nan))
    return pd.DataFrame(out)


# ------------------------------------------------------------- (2) bias + slope
def fit_slope(T, lam_a=5.0, lam_b=50.0):
    """Ridge least squares: y = a_F + da_hF + (b + db_h) x. Returns dict of parameters."""
    T = T.copy()
    T["y"] = np.log(T.poll)
    T["x"] = np.log(T.true)
    lin = sorted(T.lineage.unique())
    fams = FAM
    nF, nH = len(fams), len(lin)
    # columns: a_F (nF), b (1), da_hF (nH*nF), db_h (nH)
    p = nF + 1 + nH * nF + nH
    X = np.zeros((len(T), p))
    fi = T.family.map({f: i for i, f in enumerate(fams)}).values
    hi = T.lineage.map({h: i for i, h in enumerate(lin)}).values
    X[np.arange(len(T)), fi] = 1.0
    X[:, nF] = T.x.values
    X[np.arange(len(T)), nF + 1 + hi * nF + fi] = 1.0
    X[np.arange(len(T)), nF + 1 + nH * nF + hi] = T.x.values
    pen = np.zeros(p)
    pen[nF + 1: nF + 1 + nH * nF] = lam_a
    pen[nF + 1 + nH * nF:] = lam_b
    beta = np.linalg.solve(X.T @ X + np.diag(pen), X.T @ T.y.values)
    aF = dict(zip(fams, beta[:nF]))
    b = beta[nF]
    da = {(h, f): beta[nF + 1 + i * nF + j] for i, h in enumerate(lin) for j, f in enumerate(fams)}
    db = {h: beta[nF + 1 + nH * nF + i] for i, h in enumerate(lin)}
    return dict(aF=aF, b=b, da=da, db=db, lineages=lin)


def fit_reverse(T, lam_a=5.0, lam_b=50.0):
    """Calibration (forecast-direction) regression: log(true) = c_F + dc_hF + (d + dd_h) log(poll)."""
    U = T.rename(columns={"poll": "true", "true": "poll"})
    return fit_slope(U, lam_a, lam_b)


def reverse_correction(par, pollster_specific=True):
    def corr_share(lin, fam, v):
        if fam == "OTHER":
            return v
        c = par["aF"].get(fam, 0.0)
        d = par["b"]
        if pollster_specific and lin in par["db"]:
            c += par["da"].get((lin, fam), 0.0)
            d += par["db"][lin]
        return float(np.exp(c + d * np.log(v)))
    return corr_share


def slope_correction(par, pollster_specific=True):
    def corr_share(lin, fam, v):
        if fam == "OTHER":
            return v
        a = par["aF"].get(fam, 0.0)
        b = par["b"]
        if pollster_specific and lin in par["db"]:
            a += par["da"].get((lin, fam), 0.0)
            b += par["db"][lin]
        b = max(b, 0.5)
        return float(np.exp((np.log(v) - a) / b))
    return corr_share


def apply_share_correction(P, k, fn):
    g = P[P.knesset == k].copy()
    g["family"] = [CF.PARTIES[k].get(l, ("", "OTHER"))[1] for l in g.letters]
    g["share0"] = g.share
    g["share"] = [fn(lin, fam, v) for lin, fam, v in zip(g.lineage, g.family, g.share)]
    g["share"] = g.share * g.groupby("poll").share0.transform("sum") / g.groupby("poll").share.transform("sum")
    return g.groupby(["lineage", "letters"]).share.mean().groupby("letters").mean()


def backtest_slope(P, a, T):
    res = []
    for k in sorted(P.knesset.unique()):
        Tt = T[T.knesset != k]
        raw = A["pollster_average"](P, k)
        res.append(dict(knesset=k, method="M0_raw_average", score=A["score"](A["seats_from_shares"](raw, k), k)))
        for lam in (2.0, 10.0, 50.0):
            rpar = fit_reverse(Tt, lam_a=lam, lam_b=10 * lam)
            for nm, ps in (("industry", False), ("pollster", True)):
                if nm == "industry" and lam != 10.0:
                    continue
                sh = apply_share_correction(P, k, reverse_correction(rpar, ps))
                res.append(dict(knesset=k, method=f"M7_calib_{nm}" + (f"_lam{lam:g}" if ps else ""),
                                score=A["score"](A["seats_from_shares"](sh, k), k)))
            par = fit_slope(Tt, lam_a=lam, lam_b=10 * lam)
            for nm, ps in (("industry", False), ("pollster", True)):
                if nm == "industry" and lam != 10.0:
                    continue
                sh = apply_share_correction(P, k, slope_correction(par, ps))
                res.append(dict(knesset=k, method=f"M6_slope_{nm}" + (f"_lam{lam:g}" if ps else ""),
                                score=A["score"](A["seats_from_shares"](sh, k), k)))
    R = pd.DataFrame(res)
    W = R.pivot_table(index="knesset", columns="method", values="score")
    base = W["M0_raw_average"]
    summ = []
    for m in W.columns:
        d = W[m] - base
        p = stats.wilcoxon(W[m], base).pvalue if (d != 0).sum() >= 3 else np.nan
        summ.append(dict(method=m, mean=W[m].mean(), wins=int((d < 0).sum()), losses=int((d > 0).sum()), p_wilcoxon=p))
    return R, pd.DataFrame(summ).sort_values("mean")


# ------------------------------------------------- (3) segment representation
def composition_profiles(P, D):
    pf_all = []
    for pid, gg in P.groupby("poll"):
        kk = gg.knesset.iloc[0]
        fam = gg.assign(family=[CF.PARTIES[kk].get(l, ("", "OTHER"))[1] for l in gg.letters]) \
            .groupby("family").share.sum().reindex(FAM, fill_value=1e-4).values
        pf_all.append((gg.lineage.iloc[0], kk, fam / fam.sum()))
    lineages = sorted({x[0] for x in pf_all if not x[0].startswith("Other")})
    seg_rows, fam_rows = [], []
    for (method, K, unit) in NMF_CFGS:
        prof = A["load_profiles"](method, K, unit)
        base = prof[25]
        w, Pk = base.weight.values, base[FAM].values

        def fit_and_extract(pf):
            r0, R = A["fit_composition"](pf, prof, lineages, K, lam=0.1, lam_ind=0.01, iters=150)
            out_seg, out_fam = {}, {}
            for h in lineages:
                rho = r0 + R[h]
                pi = w * np.exp(rho)
                pi = pi / pi.sum()
                out_seg[h] = pi / w
                out_fam[h] = np.log(A["comp_multipliers"](rho, w, Pk))
            return out_seg, out_fam
        seg, famb = fit_and_extract(pf_all)
        # jackknife over elections
        jk_seg, jk_fam = [], []
        for e in sorted(P.knesset.unique()):
            s_e, f_e = fit_and_extract([x for x in pf_all if x[1] != e])
            jk_seg.append(s_e)
            jk_fam.append(f_e)
        n = len(jk_seg)
        for h in lineages:
            n_el = P[P.lineage == h].knesset.nunique()
            js = np.array([j[h] for j in jk_seg])
            jf = np.array([j[h] for j in jk_fam])
            se_s = np.sqrt((n - 1) / n * ((js - js.mean(0)) ** 2).sum(0))
            se_f = np.sqrt((n - 1) / n * ((jf - jf.mean(0)) ** 2).sum(0))
            if method == "nmf" and K == 8 and unit == "ballot":
                for c in range(len(w)):
                    seg_rows.append(dict(lineage=h, n_elections=n_el, segment=int(base.index[c]),
                                         ratio=float(seg[h][c]), se_log=float(se_s[c] / max(seg[h][c], 1e-9))))
            for j, f in enumerate(FAM):
                fam_rows.append(dict(cfg=f"{method}_K{K}_{unit}", lineage=h, n_elections=n_el, family=f,
                                     log_bias=float(famb[h][j]), se=float(se_f[j])))
        print(f"  composition profiles {method} K{K} {unit} done", flush=True)
    S = pd.DataFrame(seg_rows)
    F = pd.DataFrame(fam_rows)
    Fa = F.groupby(["lineage", "family", "n_elections"]).agg(log_bias=("log_bias", "mean"), se=("se", "mean")).reset_index()
    Fa["bias_pct"] = 100 * (np.exp(Fa.log_bias) - 1)
    return S, Fa


def main():
    h, a = A["load"]()
    P = A["poll_matrix"](h)
    D, _, _ = A["industry_bias"](P, a)
    T = pollster_list_table(P, a)
    B = bias_matrix(T)
    B.to_csv(os.path.join(CF.OUTT, "pollster_bias_matrix.csv"), index=False)
    print("(1) pollster x family bias (main firms, %):")
    print(B[B.lineage.isin(MAIN_FIRMS)].pivot_table(index="lineage", columns="family", values="bias_pct").round(1).to_string())
    par = fit_slope(T)
    sp = [dict(lineage="INDUSTRY", b=par["b"], **{f"a_{f}": par["aF"][f] for f in FAM})]
    for hh in par["lineages"]:
        sp.append(dict(lineage=hh, b=par["b"] + par["db"][hh],
                       **{f"a_{f}": par["aF"][f] + par["da"][(hh, f)] for f in FAM}))
    SP = pd.DataFrame(sp)
    SP.to_csv(os.path.join(CF.OUTT, "slope_model_params.csv"), index=False)
    print("\n(2) bias+slope: industry slope b =", round(par["b"], 3))
    print(SP[SP.lineage.isin(["INDUSTRY"] + MAIN_FIRMS)][["lineage", "b"]].round(3).to_string(index=False))
    rpar = fit_reverse(T)
    print("calibration regression (truth on poll): industry slope d =", round(rpar["b"], 3),
          "| intercepts:", {f: round(v, 3) for f, v in rpar["aF"].items()})
    pd.DataFrame([dict(lineage="INDUSTRY", d=rpar["b"], **{f"c_{f}": rpar["aF"][f] for f in FAM})] +
                 [dict(lineage=hh, d=rpar["b"] + rpar["db"][hh],
                       **{f"c_{f}": rpar["aF"][f] + rpar["da"][(hh, f)] for f in FAM}) for hh in rpar["lineages"]])         .to_csv(os.path.join(CF.OUTT, "calibration_regression_params.csv"), index=False)
    R, Ssum = backtest_slope(P, a, T)
    R.to_csv(os.path.join(CF.OUTT, "backtest_slope_models.csv"), index=False)
    Ssum.to_csv(os.path.join(CF.OUTT, "backtest_slope_summary.csv"), index=False)
    print(Ssum.round(3).to_string(index=False))
    S, Fa = composition_profiles(P, D)
    S.to_csv(os.path.join(CF.OUTT, "pollster_segment_profiles.csv"), index=False)
    Fa.to_csv(os.path.join(CF.OUTT, "pollster_comp_family_bias.csv"), index=False)
    print("\n(3) composition-implied pollster family bias (%, avg of NMF cfgs):")
    print(Fa[Fa.lineage.isin(MAIN_FIRMS)].pivot_table(index="lineage", columns="family", values="bias_pct").round(1).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
