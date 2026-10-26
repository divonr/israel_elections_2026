"""
Step 06b - Formal model selection on the leave-one-election-out (LOEO) backtest.

1. Friedman test (non-parametric repeated-measures ANOVA, elections = blocks) comparing
   (a) the four clustering algorithms (scores averaged over K, unit and variant), and
   (b) the main model classes; Nemenyi critical difference for post-hoc comparison.
2. Ensembles that avoid "winner's curse" selection among 126 composition variants:
     E1 = geometric mean of the M1s (industry family, empirical-Bayes) factors and the
          composition factors AVERAGED over all clustering configurations (industry variant);
     E2 = same but averaging only NMF configurations;
     E3 = M1s + composition (all configs) + pollster deviations (lam 0.1).
3. A "honest" nested selection: for each held-out election, pick the best composition config
   using only the other 7 elections' scores, then score it on the held-out election.
Outputs: outputs/tables/model_selection_*.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as CF  # noqa: E402
import runpy  # noqa: E402

warnings.filterwarnings("ignore")
A = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "06_poll_error_analysis.py"),
                   run_name="lib")


def nemenyi_cd(k, n, alpha=0.05):
    q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}[k]
    return q * np.sqrt(k * (k + 1) / (6 * n))


def friedman(W, label):
    ranks = W.rank(axis=1)
    stat, p = stats.friedmanchisquare(*[W[c] for c in W.columns])
    cd = nemenyi_cd(W.shape[1], W.shape[0])
    out = pd.DataFrame({"mean_score": W.mean(), "mean_rank": ranks.mean()}).sort_values("mean_rank")
    out["friedman_chi2"] = stat
    out["friedman_p"] = p
    out["nemenyi_CD"] = cd
    out.to_csv(os.path.join(CF.OUTT, f"model_selection_friedman_{label}.csv"))
    print(f"\nFriedman ({label}): chi2={stat:.2f} p={p:.3f}  Nemenyi CD={cd:.2f}")
    print(out.round(3).to_string())
    return out


def ensembles():
    h, a = A["load"]()
    P = A["poll_matrix"](h)
    D, _, _ = A["industry_bias"](P, a)
    val = pd.read_csv(os.path.join(CF.OUTT, "clustering_validation.csv"))
    cfgs = [(r.method, int(r.K), r.unit) for r in val.itertuples()]
    FAM = A["FAM_NO_OTHER"]
    rows = []
    for k in sorted(P.knesset.unique()):
        train = D[D.knesset != k]
        trainP = P[P.knesset != k]
        m1s = A["eb_shrink"](A["family_errors"](train))
        logf_all, logf_nmf, logf_pol = [], [], []
        for (method, K, unit) in cfgs:
            prof = A["load_profiles"](method, K, unit)
            polls_fam = []
            for pid, gg in trainP.groupby("poll"):
                kk = gg.knesset.iloc[0]
                fam = gg.assign(family=[CF.PARTIES[kk].get(l, ("", "OTHER"))[1] for l in gg.letters]) \
                    .groupby("family").share.sum().reindex(FAM, fill_value=1e-4).values
                polls_fam.append((gg.lineage.iloc[0], kk, fam / fam.sum()))
            lineages = sorted({x[0] for x in polls_fam if not x[0].startswith("Other")})
            r0, R = A["fit_composition"](polls_fam, prof, lineages, K, lam=0.1, lam_ind=0.01, iters=150)
            base = prof[k - 1] if (k - 1) in prof else prof[k]
            w, Pk = base.weight.values, base[FAM].values
            lf = -np.log(A["comp_multipliers"](r0, w, Pk))
            logf_all.append(lf)
            if method == "nmf":
                logf_nmf.append(lf)
            logf_pol.append({lin: -np.log(A["comp_multipliers"](r0 + R[lin], w, Pk)) - lf for lin in R})
        comp_all = dict(zip(FAM, np.mean(logf_all, 0)))
        comp_nmf = dict(zip(FAM, np.mean(logf_nmf, 0)))
        pol_dev = {lin: np.mean([d[lin] for d in logf_pol if lin in d], 0) for lin in logf_pol[0]}

        def mk(comp, wt_m1=0.5, pol=False):
            def corr(lin, fam):
                if fam == "OTHER":
                    return 1.0
                f = wt_m1 * (-m1s.get(fam, 0.0)) + (1 - wt_m1) * comp[fam]
                if pol and lin in pol_dev:
                    f += pol_dev[lin][FAM.index(fam)]
                return np.exp(f)
            return corr
        for nm, corr in [("E1_M1s+compAll", mk(comp_all)), ("E2_M1s+compNMF", mk(comp_nmf)),
                         ("E3_M1s+compAll+pollster", mk(comp_all, pol=True)),
                         ("E4_compAll_only", mk(comp_all, 0.0)), ("E5_compNMF_only", mk(comp_nmf, 0.0))]:
            sh = A["pollster_average"](P, k, corr)
            rows.append(dict(knesset=k, method=nm, score=A["score"](A["seats_from_shares"](sh, k), k)))
        print(f"  ensembles K{k} done", flush=True)
    E = pd.DataFrame(rows)
    E.to_csv(os.path.join(CF.OUTT, "backtest_ensembles_long.csv"), index=False)
    return E


def nested_selection(L):
    """Pick the best M3 config on 7 elections, score on the 8th (honest estimate of 'choose best')."""
    W = L[L.method.str.startswith("M3")].pivot_table(index="knesset", columns="method", values="score")
    out = []
    for k in W.index:
        best = W.drop(index=k).mean().idxmin()
        out.append(dict(knesset=k, method="N_nested_best_M3", chosen=best, score=W.loc[k, best]))
    return pd.DataFrame(out)


def main():
    L = pd.read_csv(os.path.join(CF.OUTT, "backtest_scores_long.csv"))
    W = L.pivot_table(index="knesset", columns="method", values="score")
    # (a) clustering algorithms, averaged over K/unit/variant
    m3 = L[L.method.str.startswith("M3")].copy()
    m3["algo"] = m3.method.str.extract(r"_(kmeans|gmm|ward|nmf)_")[0]
    m3["unit"] = m3.method.str.extract(r"_(ballot|settlement)$")[0]
    m3["K"] = m3.method.str.extract(r"_K(\d+)_")[0].astype(int)
    friedman(m3.pivot_table(index="knesset", columns="algo", values="score", aggfunc="mean"), "cluster_algorithms")
    friedman(m3.pivot_table(index="knesset", columns="K", values="score", aggfunc="mean"), "cluster_K")
    # (b) nested selection + ensembles
    N = nested_selection(L)
    N.to_csv(os.path.join(CF.OUTT, "model_selection_nested_best_M3.csv"), index=False)
    E = ensembles()
    allL = pd.concat([L, N[["knesset", "method", "score"]], E])
    W2 = allL.pivot_table(index="knesset", columns="method", values="score")
    classes = ["M0_raw_average", "M1s_industry_family_shrunk", "M2h_pollster_house_half",
               "N_nested_best_M3", "E1_M1s+compAll", "E2_M1s+compNMF", "E3_M1s+compAll+pollster",
               "E4_compAll_only", "E5_compNMF_only"]
    F = friedman(W2[classes], "model_classes")
    base = W2["M0_raw_average"]
    tests = []
    for c in classes:
        d = W2[c] - base
        p = stats.wilcoxon(W2[c], base).pvalue if (d != 0).sum() >= 3 else np.nan
        tests.append(dict(method=c, mean=W2[c].mean(), mean_diff_vs_raw=d.mean(),
                          pct_improvement=-100 * d.mean() / base.mean(), wins=int((d < 0).sum()),
                          losses=int((d > 0).sum()), p_wilcoxon=p))
    T = pd.DataFrame(tests)
    T.to_csv(os.path.join(CF.OUTT, "model_selection_summary.csv"), index=False)
    W2[classes].to_csv(os.path.join(CF.OUTT, "model_selection_scores_by_election.csv"))
    pd.set_option("display.width", 250)
    print(T.round(3).to_string(index=False))
    print(W2[classes].to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
