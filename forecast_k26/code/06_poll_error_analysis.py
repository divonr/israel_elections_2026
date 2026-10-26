"""
Step 06 - How wrong were the polls historically, and which correction model predicts best?

Data  : final-window polls (published within WINDOW days of election day) for 8 elections
        (2009-2022) and the official results.
Space : vote shares. A poll's seats are converted to vote shares:
            v_j = seats_j/120 * (1 - B)  for lists with seats,  v_j = pct_j (or 1.8%) for lists at 0,
        where B is the poll's below-threshold mass (+1.5% for unlisted lists).
Error : delta = log(v_poll) - log(v_actual) per poll and list.

Analyses
 (A) Industry (all-pollster) bias by party family + "new list" + "small list" effects:
     OLS on election-list mean errors with election-clustered standard errors.
 (B) Pollster house effects (relative to the election-list mean) and their persistence across
     elections: leave-one-election-out correlation + permutation test.
 (C) Composition ("segment representation") model: a pollster's sample = re-weighting of the
     voter segments found in step 05:  y_hF = sum_k pi_hk p_kF,  pi_hk ~ w_k exp(rho_hk).
 (D) Leave-one-election-out backtest of forecast methods, scored exactly like the competition
     (sum of absolute seat errors), with paired tests between methods.
Outputs: outputs/tables/*.csv and data/processed/model_params.json (used by step 07).
"""
import itertools
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as CF  # noqa: E402

warnings.filterwarnings("ignore")
WINDOW = 14
FAM_NO_OTHER = [f for f in CF.FAMILIES if f != "OTHER"]
CL = os.path.join(CF.PROC, "clusters")
RNG = np.random.default_rng(2026)
# lists running for the first time (or a first-time joint list with a new leader/brand)
NEW_LISTS = {18: {"ב"}, 19: {"פה", "צפ", "נץ", "ץ"}, 20: {"כ", "קץ", "ודעם", "אמת"},
             21: {"פה", "טב", "נ", "ז", "נר", "דעם"}, 22: {"טב", "מרצ"}, 23: {"אמת"},
             24: {"ת", "ט", "ב", "יז"}, 25: {"כן", "ט"}}


# ----------------------------------------------------------------------------- data
def load():
    # merged history (English Wikipedia + polls found only in the verified Hebrew-Wikipedia source, step 04b)
    merged = os.path.join(CF.PROC, "hist_polls_merged.csv")
    use = merged if (os.path.exists(merged) and os.environ.get("HIST_SOURCE", "merged") == "merged") else \
        os.path.join(CF.PROC, "hist_polls.csv")
    h = pd.read_csv(use)
    a = pd.read_csv(os.path.join(CF.PROC, "actual_results.csv"))
    h = h[(h.days_before <= WINDOW)].copy()
    h["poll"] = h.election.astype(str) + "_" + h.row_idx.astype(str)
    return h, a


def poll_shares(df):
    """df: rows of one poll (letters, seats, pct) -> Series letters -> vote share (config.seats_to_shares)."""
    s = df.drop_duplicates("letters").set_index("letters")
    return CF.seats_to_shares(s.seats, s.pct).dropna()


def poll_matrix(h):
    rows = []
    for pid, g in h.groupby("poll"):
        v = poll_shares(g)
        for l, x in v.items():
            rows.append(dict(poll=pid, election=g.election.iloc[0], knesset=g.knesset.iloc[0],
                             lineage=g.lineage.iloc[0], days_before=g.days_before.iloc[0],
                             letters=l, share=x, seats=g.set_index("letters").seats.get(l)))
    return pd.DataFrame(rows)


def actual_share(a, k):
    g = a[a.knesset == k].set_index("letters")
    return g.share


# ------------------------------------------------------------------- (A) industry bias
def industry_bias(P, a):
    rows = []
    for (e, k), g in P.groupby(["election", "knesset"]):
        act = actual_share(a, k)
        # pollster-balanced mean poll share per list
        m = g.groupby(["lineage", "letters"]).share.mean().groupby("letters").mean()
        ms = g.groupby(["lineage", "letters"]).seats.mean().groupby("letters").mean()
        for l, v in m.items():
            if l not in act.index or act[l] <= 0:
                continue
            rows.append(dict(election=e, knesset=k, letters=l, name=CF.PARTIES[k].get(l, (l,))[0],
                             family=CF.PARTIES[k].get(l, ("", "OTHER"))[1], poll_share=v,
                             poll_seats=ms[l], actual_share=act[l],
                             actual_seats=CF.ACTUAL_SEATS[k].get(l, 0),
                             log_err=np.log(v) - np.log(act[l]),
                             new_list=int(l in NEW_LISTS.get(k, set())),
                             small=int(v < 0.05)))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(CF.OUTT, "industry_errors_by_list.csv"), index=False, encoding="utf-8-sig")
    D["seat_err"] = D.poll_seats - D.actual_seats
    Dm = D[(D.family != "OTHER") & (D.poll_seats > 0.05)]  # measured lists only (no 0-seat placeholders)
    fit = smf.ols("log_err ~ 0 + C(family) + new_list + small", data=Dm).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(Dm.election)[0]})
    tab = pd.DataFrame({"coef": fit.params, "se": fit.bse, "p": fit.pvalues})
    tab.to_csv(os.path.join(CF.OUTT, "industry_bias_regression.csv"))
    # per-family one-sample t-tests on election-level family errors (each election = 1 obs)
    fs = Dm.groupby(["election", "family"])[["poll_share", "actual_share"]].sum()
    fam = (np.log(fs.poll_share) - np.log(fs.actual_share)).rename("log_err").reset_index()
    ft = []
    for f, g in fam.groupby("family"):
        t, p = stats.ttest_1samp(g.log_err, 0) if len(g) > 2 else (np.nan, np.nan)
        w = stats.wilcoxon(g.log_err).pvalue if len(g) > 5 else np.nan
        ft.append(dict(family=f, n_elections=len(g), mean_log_err=g.log_err.mean(),
                       pct_bias=100 * (np.exp(g.log_err.mean()) - 1), t=t, p_t=p, p_wilcoxon=w))
    ft = pd.DataFrame(ft)
    ft.to_csv(os.path.join(CF.OUTT, "industry_bias_by_family_tests.csv"), index=False)
    print("\n(A) industry bias regression (log poll/actual; negative = polls UNDER-estimate)")
    print(tab.round(3).to_string())
    print(ft.round(3).to_string())
    return D, fit, fam


# ------------------------------------------------------------- (B) house effects
def house_effects(P, a):
    """relative house effect per pollster x family x election, persistence test."""
    rows = []
    for (e, k), g in P.groupby(["election", "knesset"]):
        g = g.copy()
        g["family"] = [CF.PARTIES[k].get(l, ("", "OTHER"))[1] for l in g.letters]
        fam = g.groupby(["poll", "lineage", "family"]).share.sum().reset_index()
        pm = fam.groupby(["lineage", "family"]).share.mean().reset_index()
        cons = pm.groupby("family").share.mean()
        act = a[a.knesset == k].groupby("family").share.sum()
        for _, r in pm.iterrows():
            if r.family == "OTHER" or r.share <= 0:
                continue
            rows.append(dict(election=e, lineage=r.lineage, family=r.family,
                             rel=np.log(r.share) - np.log(cons[r.family]),
                             abs_err=np.log(r.share) - np.log(act.get(r.family, np.nan))))
    H = pd.DataFrame(rows)
    H = H[~H.lineage.str.startswith("Other")]
    H.to_csv(os.path.join(CF.OUTT, "house_effects_by_election.csv"), index=False)
    # persistence: correlate this election's relative effect with the mean over OTHER elections
    pairs = []
    for (l, f), g in H.groupby(["lineage", "family"]):
        if g.election.nunique() < 3:
            continue
        for _, r in g.iterrows():
            others = g[g.election != r.election].rel.mean()
            pairs.append(dict(lineage=l, family=f, election=r.election, rel=r.rel, prior=others))
    Pp = pd.DataFrame(pairs)
    rho, p = stats.pearsonr(Pp.rel, Pp.prior)
    # permutation: shuffle pollster labels within election-family cells
    perm = []
    for _ in range(2000):
        Q = H.copy()
        Q["lineage"] = Q.groupby(["election", "family"]).lineage.transform(lambda s: RNG.permutation(s.values))
        pp = []
        for (l, f), g in Q.groupby(["lineage", "family"]):
            if g.election.nunique() < 3:
                continue
            m = g.rel.sum()
            n = len(g)
            pp.append(np.c_[g.rel.values, (m - g.rel.values) / (n - 1)])
        pp = np.vstack(pp)
        perm.append(np.corrcoef(pp[:, 0], pp[:, 1])[0, 1])
    p_perm = (np.sum(np.array(perm) >= rho) + 1) / (len(perm) + 1)
    # slope: how much of the historical house effect carries over (shrinkage factor)
    slope = smf.ols("rel ~ prior", data=Pp).fit()
    summary = dict(n_pairs=len(Pp), pearson_r=rho, p_param=p, p_permutation=p_perm,
                   carryover_slope=slope.params["prior"], slope_se=slope.bse["prior"])
    pd.DataFrame([summary]).to_csv(os.path.join(CF.OUTT, "house_effect_persistence.csv"), index=False)
    print("\n(B) house-effect persistence:", {k: round(v, 3) for k, v in summary.items()})
    # by family
    byf = Pp.groupby("family").apply(lambda g: pd.Series(dict(
        n=len(g), r=np.corrcoef(g.rel, g.prior)[0, 1] if len(g) > 3 else np.nan))).reset_index()
    byf.to_csv(os.path.join(CF.OUTT, "house_effect_persistence_by_family.csv"), index=False)
    print(byf.round(3).to_string())
    return H, summary


# --------------------------------------------------------- (C) composition model
_NAT_FAM = None


def national_family_votes():
    """Official national votes by knesset x family (including double envelopes)."""
    global _NAT_FAM
    if _NAT_FAM is None:
        a = pd.read_csv(os.path.join(CF.PROC, "actual_results.csv"))
        _NAT_FAM = a.pivot_table(index="knesset", columns="family", values="votes", aggfunc="sum", fill_value=0)
    return _NAT_FAM


def load_profiles(method, K, unit):
    """
    Segment profiles per knesset, reindexed to all K segments (an empty segment gets weight ~0 and the
    segment's average family mix over the other elections), PLUS one 'double-envelope' pseudo-segment
    (index K): official national votes minus the geographic ballot-box votes, by family (soldiers,
    diplomats, prisoners, hospitals ...). With it, sum_k w_k P_k equals the OFFICIAL national result, so an
    unbiased poll (rho = 0) reproduces the official shares that the backtest and the competition score.
    """
    fam_all = FAM_NO_OTHER + ["OTHER"]
    p = pd.read_csv(os.path.join(CL, f"profiles_{method}_K{K}_{unit}.csv"))
    mean_prof = p.groupby("cluster")[fam_all].mean()
    nat = national_family_votes()
    out = {}
    for k, g in p.groupby("knesset"):
        g = g.set_index("cluster").reindex(range(K))
        miss = g.weight.isna()
        g.loc[miss, fam_all] = mean_prof.reindex(g.index[miss]).values
        g["votes"] = g.votes.fillna(0.0)
        g[FAM_NO_OTHER] = g[FAM_NO_OTHER].fillna(1.0 / len(FAM_NO_OTHER))
        g["OTHER"] = g["OTHER"].fillna(0.0)
        geo_F = g[fam_all].mul(g.votes, axis=0).sum()
        nat_F = nat.loc[k].reindex(fam_all).fillna(0.0) if k in nat.index else geo_F
        env_F = (nat_F - geo_F).clip(lower=0.0)
        env = pd.Series({f: (env_F[f] / env_F.sum() if env_F.sum() > 0 else 1.0 / len(fam_all)) for f in fam_all})
        env["votes"] = float(env_F.sum())
        g.loc[K, fam_all + ["votes"]] = env[fam_all + ["votes"]].values
        tot = g.votes.sum()
        g["weight"] = (g.votes / tot).clip(lower=1e-6)
        out[k] = g
    return out


def comp_predict(rho, w, Pk):
    """rho: (K,), w: (K,), Pk: (K,F) family profiles -> poll family shares (F,)"""
    pi = w * np.exp(rho)
    pi = pi / pi.sum()
    y = pi @ Pk
    return y / y.sum()


def fit_composition(polls_fam, prof, lineages, K, lam=2.0, lam_ind=0.2, iters=300):
    """
    Segment-representation model fitted by PyTorch autograd (batched over all training polls).
    polls_fam: list of (lineage, knesset, y (F,)) ; prof: dict knesset -> profile df.
    Each poll i:  pi_i = softmax(log w_{e(i)} + rho0 + R[h(i)]),  yhat_i = pi_i @ P_{e(i)}.
    Loss: squared Hellinger distance sum_i ||sqrt(yhat_i) - sqrt(y_i)||^2
          + ridge(rho0) + ridge(R) (pollster deviations shrunk to the industry pattern).
    """
    import torch
    K = len(next(iter(prof.values())))  # segments incl. the double-envelope pseudo-segment
    H = len(lineages)
    li = {l: i for i, l in enumerate(lineages)}
    Wl, Pm, Y, Hi = [], [], [], []
    for l, k, y in polls_fam:
        g = prof[k]
        Wl.append(np.log(g.weight.values + 1e-9))
        Pm.append(g[FAM_NO_OTHER].values)
        Y.append(np.sqrt(y))
        Hi.append(li.get(l, H))  # unknown lineage -> extra row fixed at zero
    Wl = torch.tensor(np.array(Wl), dtype=torch.float64)
    Pm = torch.tensor(np.array(Pm), dtype=torch.float64)
    Y = torch.tensor(np.array(Y), dtype=torch.float64)
    Hi = torch.tensor(Hi)
    r0 = torch.zeros(K, dtype=torch.float64, requires_grad=True)
    R = torch.zeros(H + 1, K, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(H + 1, 1, dtype=torch.float64)
    mask[H] = 0.0
    opt = torch.optim.LBFGS([r0, R], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        rho = r0[None, :] + (R * mask)[Hi]
        pi = torch.softmax(Wl + rho, dim=1)
        yhat = torch.einsum("nk,nkf->nf", pi, Pm)
        yhat = yhat / yhat.sum(1, keepdim=True)
        loss = ((torch.sqrt(yhat + 1e-12) - Y) ** 2).sum() + lam_ind * (r0 ** 2).sum()             + lam * ((R * mask) ** 2).sum() + 10 * r0.mean() ** 2
        loss.backward()
        return loss
    opt.step(closure)
    r0n = r0.detach().numpy()
    Rn = (R * mask).detach().numpy()
    return r0n, {l: Rn[i] for l, i in li.items()}


def comp_multipliers(rho, w, Pk):
    """m_F = (pollster-weighted family share)/(true family share) for every family."""
    pi = w * np.exp(rho)
    pi = pi / pi.sum()
    true = w @ Pk
    m = (pi @ Pk) / np.where(true > 1e-4, true, 1.0)
    # a family absent from the base election (e.g. YB inside Likud-Beiteinu in 2013) gets no correction
    return np.where(true > 1e-4, m, 1.0)


# --------------------------------------------------------------- (D) backtest
def pollster_average(P, k, correct=None):
    """mean share per list: average within pollster, then across pollsters (optionally corrected)."""
    g = P[P.knesset == k].copy()
    g["family"] = [CF.PARTIES[k].get(l, ("", "OTHER"))[1] for l in g.letters]
    if correct is not None:
        g["share0"] = g.share
        g["share"] = [s * correct(lin, fam) for s, lin, fam in zip(g.share, g.lineage, g.family)]
        # renormalise so each poll keeps its original total (corrections only move shares between lists)
        g["share"] = g.share * g.groupby("poll").share0.transform("sum") / g.groupby("poll").share.transform("sum")
    return g.groupby(["lineage", "letters"]).share.mean().groupby("letters").mean()


def seats_from_shares(sh, k):
    alloc = CF.allocate(sh.to_dict(), CF.THRESHOLD[k], total=1.0)
    return pd.Series(alloc)


def score(pred, k):
    act = pd.Series(CF.ACTUAL_SEATS[k])
    idx = act.index.union(pred.index)
    return float((pred.reindex(idx).fillna(0) - act.reindex(idx).fillna(0)).abs().sum())


def backtest(P, a, D, cluster_cfgs):
    elections = sorted(P.knesset.unique())
    res = []
    for k in elections:
        train = D[D.knesset != k]
        trainP = P[P.knesset != k]
        # ---- M0 raw average
        m0 = pollster_average(P, k)
        res.append(dict(knesset=k, method="M0_raw_average", score=score(seats_from_shares(m0, k), k)))
        # ---- M0m median / trimmed mean across pollsters
        gk = P[P.knesset == k]
        pmk = gk.groupby(["lineage", "letters"]).share.mean().unstack()
        res.append(dict(knesset=k, method="M0m_median_pollster", score=score(seats_from_shares(pmk.median(), k), k)))
        res.append(dict(knesset=k, method="M0t_trimmed_mean_pollster",
                        score=score(seats_from_shares(pmk.apply(lambda c: stats.trim_mean(c.dropna(), 0.2)), k), k)))
        # ---- M0b last poll per pollster
        g = P[P.knesset == k]
        last = g[g.days_before == g.groupby("lineage").days_before.transform("min")]
        m0b = last.groupby(["lineage", "letters"]).share.mean().groupby("letters").mean()
        res.append(dict(knesset=k, method="M0b_last_poll_avg", score=score(seats_from_shares(m0b, k), k)))
        # ---- M1 industry family correction (shrunk mean of training family errors)
        fam_err = family_errors(train)
        mu = fam_err["mean"]
        shrink = eb_shrink(fam_err)
        for nm, corr_tab in [("M1_industry_family", mu), ("M1s_industry_family_shrunk", shrink)]:
            def corr(lin, fam, corr_tab=corr_tab):
                return float(np.exp(-corr_tab.get(fam, 0.0))) if fam != "OTHER" else 1.0
            sh = pollster_average(P, k, corr)  # per-poll renormalisation, as for every other method
            res.append(dict(knesset=k, method=nm, score=score(seats_from_shares(sh, k), k)))
        # ---- M1r regression correction (family + new + small)
        fitk = smf.ols("log_err ~ 0 + C(family) + new_list + small",
                       data=train[(train.family != "OTHER") & (train.poll_seats > 0.05)]).fit()
        dd = pd.DataFrame({"letters": m0.index, "family": [CF.PARTIES[k].get(l, ("", "OTHER"))[1] for l in m0.index],
                           "new_list": [int(l in NEW_LISTS.get(k, set())) for l in m0.index],
                           "small": (m0.values < 0.05).astype(int)})
        known = dd.family.isin(fitk.model.data.frame.family.unique())
        pr = np.zeros(len(dd))
        pr[known.values] = fitk.predict(dd[known])
        prs = pd.Series(pr, index=m0.index)
        gk = P[P.knesset == k].copy()
        gk["share0"] = gk.share
        gk["share"] = gk.share * np.exp(-gk.letters.map(prs).fillna(0))
        gk["share"] = gk.share * gk.groupby("poll").share0.transform("sum") / gk.groupby("poll").share.transform("sum")
        sh = gk.groupby(["lineage", "letters"]).share.mean().groupby("letters").mean()
        res.append(dict(knesset=k, method="M1r_industry_regression", score=score(seats_from_shares(sh, k), k)))
        # ---- M2 pollster-specific (industry + relative house effects from training, carried with slope)
        Ht = house_effects_quick(trainP, a)
        rel = Ht.groupby(["lineage", "family"]).rel.mean()
        for nm, c in [("M2_pollster_house_full", 1.0), ("M2h_pollster_house_half", 0.5)]:
            def corr(lin, fam, c=c):
                return np.exp(-mu.get(fam, 0) - c * rel.get((lin, fam), 0))
            sh = pollster_average(P, k, corr)
            res.append(dict(knesset=k, method=nm, score=score(seats_from_shares(sh, k), k)))
        # ---- M4 accuracy weighted average (inverse historical MSE of the pollster)
        acc = Ht.groupby("lineage").abs_err.apply(lambda s: np.mean(s ** 2))
        gk = P[P.knesset == k]
        pm = gk.groupby(["lineage", "letters"]).share.mean().unstack()
        wts = pd.Series({l: 1 / (acc.get(l, acc.median() if len(acc) else 1) + 0.002) for l in pm.index})
        sh = (pm.mul(wts, axis=0).sum() / pm.notna().mul(wts, axis=0).sum())
        res.append(dict(knesset=k, method="M4_accuracy_weighted", score=score(seats_from_shares(sh, k), k)))
        # ---- M3 composition models
        for (method, K, unit) in cluster_cfgs:
            prof = load_profiles(method, K, unit)
            polls_fam = []
            for pid, gg in trainP.groupby("poll"):
                kk = gg.knesset.iloc[0]
                fam = gg.assign(family=[CF.PARTIES[kk].get(l, ("", "OTHER"))[1] for l in gg.letters]) \
                    .groupby("family").share.sum().reindex(FAM_NO_OTHER, fill_value=1e-4).values
                polls_fam.append((gg.lineage.iloc[0], kk, fam / fam.sum()))
            lineages = sorted({x[0] for x in polls_fam if not x[0].startswith("Other")})
            K_eff = len(prof[k])  # K segments + the double-envelope pseudo-segment
            fits = {lam: fit_composition(polls_fam, prof, lineages, K_eff, lam=lam, lam_ind=0.01, iters=150)
                    for lam in (0.02, 0.1)}
            base = prof[k - 1] if (k - 1) in prof else prof[k]  # previous election's segments (no leakage)
            w, Pk = base.weight.values, base[FAM_NO_OTHER].values
            variants = [("industry", 0.1, False), ("pollster_lam0.02", 0.02, True), ("pollster_lam0.1", 0.1, True)]
            for nm, lam, use_h in variants:
                r0, R = fits[lam]

                def corr(lin, fam, use_h=use_h, r0=r0, R=R):
                    if fam == "OTHER":
                        return 1.0
                    rho = r0 + (R[lin] if (use_h and lin in R) else 0)
                    m = comp_multipliers(rho, w, Pk)
                    return 1.0 / m[FAM_NO_OTHER.index(fam)]
                sh = pollster_average(P, k, corr)
                res.append(dict(knesset=k, method=f"M3_comp_{nm}_{method}_K{K}_{unit}",
                                score=score(seats_from_shares(sh, k), k)))
        print(f"  backtest K{k} done")
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(CF.OUTT, "backtest_scores_long.csv"), index=False)
    return R


def family_errors(D):
    """Per election x family: log(sum of poll shares / sum of actual shares) over the family's lists,
    i.e. a size-weighted family error. Lists the polls put at 0 seats are excluded (their 'poll share' is
    a reported or placeholder number, not a measurement), as is the residual family OTHER.
    Returns a table family -> mean, std, count across elections."""
    X = D[(D.family != "OTHER") & (D.poll_seats > 0.05)]
    fe = X.groupby(["election", "family"])[["poll_share", "actual_share"]].sum()
    fe = np.log(fe.poll_share) - np.log(fe.actual_share)
    return fe.groupby("family").agg(["mean", "std", "count"])


def eb_shrink(fam_err):
    """Empirical-Bayes (normal-normal) shrinkage of family biases toward 0.
    prior var tau^2 by method of moments: var(means) - mean(se^2)."""
    se2 = (fam_err["std"] ** 2 / fam_err["count"]).fillna(fam_err["std"].pow(2).mean())
    tau2 = max(float(np.nanvar(fam_err["mean"]) - np.nanmean(se2)), 1e-6)
    return fam_err["mean"] * tau2 / (tau2 + se2)


def house_effects_quick(P, a):
    rows = []
    for (e, k), g in P.groupby(["election", "knesset"]):
        g = g.assign(family=[CF.PARTIES[k].get(l, ("", "OTHER"))[1] for l in g.letters])
        pm = g.groupby(["poll", "lineage", "family"]).share.sum().groupby(["lineage", "family"]).mean().reset_index()
        cons = pm.groupby("family").share.mean()
        act = a[a.knesset == k].groupby("family").share.sum()
        pm = pm[(pm.family != "OTHER") & (pm.share > 0)]
        pm["rel"] = np.log(pm.share) - np.log(pm.family.map(cons))
        pm["abs_err"] = np.log(pm.share) - np.log(pm.family.map(act))
        rows.append(pm)
    H = pd.concat(rows)
    return H[~H.lineage.str.startswith("Other")]


def compare_methods(R):
    W = R.pivot_table(index="knesset", columns="method", values="score")
    summ = pd.DataFrame({"mean_score": W.mean(), "median": W.median(), "sd": W.std()}).sort_values("mean_score")
    base = W["M0_raw_average"]
    tests = []
    for m in W.columns:
        d = W[m] - base
        p = stats.wilcoxon(W[m], base).pvalue if (d != 0).sum() >= 3 else np.nan
        # sign-flip permutation test of mean difference (exact over 2^8)
        n = len(d)
        flips = np.array(list(itertools.product([-1, 1], repeat=n)))
        perm = (flips * d.values).mean(1)
        p_perm = np.mean(np.abs(perm) >= abs(d.mean()) - 1e-12)
        tests.append(dict(method=m, mean_diff_vs_raw=d.mean(), wins=(d < 0).sum(), losses=(d > 0).sum(),
                          p_wilcoxon=p, p_signflip=p_perm))
    T = pd.DataFrame(tests).set_index("method")
    out = summ.join(T)
    out.to_csv(os.path.join(CF.OUTT, "backtest_method_comparison.csv"))
    W.to_csv(os.path.join(CF.OUTT, "backtest_scores_by_election.csv"))
    return out, W


def main():
    h, a = load()
    P = poll_matrix(h)
    P.to_csv(os.path.join(CF.PROC, "hist_poll_shares.csv"), index=False, encoding="utf-8-sig")
    D, fit, fam = industry_bias(P, a)
    H, pers = house_effects(P, a)
    val = pd.read_csv(os.path.join(CF.OUTT, "clustering_validation.csv"))
    cfgs = [(r.method, int(r.K), r.unit) for r in val.itertuples() if r.method in ("kmeans", "gmm", "ward", "nmf")]
    R = backtest(P, a, D, cfgs)
    out, W = compare_methods(R)
    pd.set_option("display.width", 250)
    print("\n(D) backtest (sum |seat error| per election, LOEO):")
    print(out.round(3).head(40).to_string())
    # parameters for the 2026 forecast (fit on ALL elections)
    fam_err = family_errors(D)
    Hall = house_effects_quick(P, a)
    params = dict(window=WINDOW,
                  industry_family_mu=fam_err["mean"].to_dict(), industry_family_sd=fam_err["std"].to_dict(),
                  industry_regression=fit.params.to_dict(),
                  house_rel=Hall.groupby(["lineage", "family"]).rel.mean().rename(lambda x: x).to_dict(),
                  persistence=pers)
    params["house_rel"] = {f"{a_}|{b_}": v for (a_, b_), v in params["house_rel"].items()}
    # residual error distribution of the corrected average (for the simulation): list-level
    D["resid"] = D.log_err - D.family.map(fam_err["mean"])
    params["list_resid_sd"] = float(D.resid.std())
    params["list_resid_sd_by_size"] = D.assign(sz=pd.cut(D.poll_share, [0, .05, .1, .2, 1])).groupby("sz").resid.std().astype(float).to_dict()
    params["list_resid_sd_by_size"] = {str(k): v for k, v in params["list_resid_sd_by_size"].items()}
    json.dump(params, open(os.path.join(CF.PROC, "model_params.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
