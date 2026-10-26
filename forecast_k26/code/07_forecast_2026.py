"""
Step 07 - Forecast the 26th Knesset (27.10.2026).

1. Poll window: polls whose fieldwork ended within WINDOW_DAYS of the last poll (23.10).
   Blue & White (withdrew 19.10, CEC-confirmed 21.10) is removed and the other lists renormalised.
2. Each poll -> vote shares (CEC-report raw % when available for the poll, otherwise seats->shares).
3. Corrections (parameters from step 06, chosen by the leave-one-election-out backtest):
     a. firm house effect from history x carry-over slope (measured 0.24), by party family;
     b. industry family bias (polls vs results, 2009-2022) x shrinkage factor.
4. Aggregation: time-decayed mean within firm, then equal-weight mean across firms
   (random-effects style; firms = independent "houses").
5. Monte Carlo (N_SIM draws): log-share errors = family shock (sd of historical election-level
   family errors) + list shock (historical residual sd by list size) + 2026 between-firm
   disagreement term; then threshold 3.25% + Bader-Ofer with the 2026 surplus agreements.
6. Point forecast minimising expected sum |seat error| (the competition score): per-list medians,
   then the greedy marginal-cost fix so seats sum to 120 (optimal for separable convex loss).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as CF  # noqa: E402

RNG = np.random.default_rng(27102026)
N_SIM = int(os.environ.get("N_SIM", 20000))
WINDOW_DAYS = int(os.environ.get("WINDOW_DAYS", 10))
HALF_LIFE = float(os.environ.get("HALF_LIFE", 4))
CORRECTION = os.environ.get("CORRECTION", "selected")  # selected (by 06c backtest) | e2 | m1s | comp | none
HOUSE_CARRY = float(os.environ.get("HOUSE_CARRY", 0.0))  # backtest: pollster house effects do not help
FAMILY_SD = float(os.environ.get("FAMILY_SD", 0.08))
EXCLUDE_FIRMS = [x for x in os.environ.get("EXCLUDE_FIRMS", "").split(",") if x]
# optional firm weights for sensitivity runs, e.g. "NextData_Filber:0.5,DRI:0.5" (default: one vote per firm)
FIRM_WEIGHTS = {k: float(v) for k, v in (x.split(":") for x in os.environ.get("FIRM_WEIGHTS", "").split(",") if x)}
DISAGREE_SCALE = float(os.environ.get("DISAGREE_SCALE", 0.5))
TAG = os.environ.get("TAG", "main")
SURPLUS = [("TOGETHER", "YB"), ("JOINT", "RAAM"), ("RZP_ZEHUT", "LIKUD"), ("UTJ", "SHAS"),
           ("DEMOCRATS", "YASHAR")]
LISTS = ["LIKUD", "YASHAR", "TOGETHER", "DEMOCRATS", "YB", "OTZMA", "SHAS", "UTJ", "RZP_ZEHUT",
         "JOINT", "RAAM", "HENDEL_ZELICHA", "AMCHA"]
OTHERS_SHARE = 0.012  # small lists not polled (HaTzibur HaHaredi, Noam, Israel First, ...)


RAW_MODE = os.environ.get("RAW_MODE", "seats")  # "seats" = same conversion as the historical calibration


def poll_vote_shares(g):
    """
    g: rows of one poll. Returns Series over LISTS of vote shares.
    Default ("seats"): exactly the conversion used for the 2009-2022 polls in step 06 (config.seats_to_shares:
    D'Hondt-midpoint shares for lists with seats, reported % for lists at 0 seats), so the historical
    corrections and error sizes apply unchanged.
    ("raw": use the CEC-report raw percentages when the poll has them - sensitivity only, because raw
    support of small lists is systematically above their seat-implied share, unlike the calibration data.)
    """
    s = g.set_index("party")
    s = s[~s.index.duplicated()]
    raw = s.pct_raw.reindex(LISTS)
    seats = s.seats.reindex(LISTS)
    if RAW_MODE == "raw" and raw.notna().sum() >= 10 and raw[seats > 0].notna().all():
        v = raw / 100.0
        v[v.isna()] = 0.015
        return v
    v = CF.seats_to_shares(seats, raw)  # identical to the historical calibration (step 06)
    v[v.isna()] = 0.015
    return v


def build_poll_table():
    p = pd.read_csv(os.path.join(CF.PROC, "polls_2026.csv"))
    p["date_end"] = pd.to_datetime(p.date_end)
    # raw national percentages read from the CEC poll reports filed after publication (if extracted)
    cec = os.path.join(CF.ROOT, "data", "research", "cec_raw_pct_2026.csv")
    if os.path.exists(cec) and os.environ.get("USE_CEC_RAW", "1") == "1":
        c = pd.read_csv(cec)
        c["pct"] = c.pct_weighted.fillna(c.pct_raw)
        c = c[c.party.isin(LISTS + ["BLUE_WHITE", "HAREDI_PUBLIC", "OTHER", "UNDECIDED"])]
        c = c[c.poll_id != "2026-10-22_Channel 16"]  # its CEC table is unweighted with 8% "don't know" in the base
        for pid, g in c.groupby("poll_id"):
            g = g.set_index("party").pct
            if g.reindex(LISTS).notna().sum() < 10:
                continue
            if "UNDECIDED" in g.index and str(c[c.poll_id == pid].base_includes_undecided.iloc[0]).lower() == "yes":
                g = g.drop("UNDECIDED") / (100 - g["UNDECIDED"]) * 100
            for party, v in g.items():
                p.loc[(p.poll_id == pid) & (p.party == party), "pct_raw"] = v
    last = p.date_end.max()
    p = p[p.date_end > last - pd.Timedelta(days=WINDOW_DAYS)]
    rows = []
    for pid, g in p.groupby("poll_id"):
        v = poll_vote_shares(g[g.party != "BLUE_WHITE"])
        rows.append(dict(poll_id=pid, lineage=g.lineage.iloc[0], publisher=g.publisher.iloc[0],
                         date_end=g.date_end.iloc[0], **v.to_dict()))
    T = pd.DataFrame(rows).sort_values("date_end")
    T["age_days"] = (last - T.date_end).dt.days
    T["w_time"] = 0.5 ** (T.age_days / HALF_LIFE)
    return T


def corrections(params, final):
    logf = {"selected": final["final_logfactor"], "e2": final["e2_logfactor"], "m1s": final["m1s_logfactor"],
            "comp": final["comp_logfactor"], "none": {}}[CORRECTION]
    house = params["house_rel"]

    def factor(lineage, party):
        fam = CF.FAMILY_2026[party]
        f = logf.get(fam, 0.0) if fam != "OTHER" else 0.0
        f -= HOUSE_CARRY * house.get(f"{lineage}|{fam}", 0.0)
        return float(np.exp(f))
    return factor


def aggregate(T, factor):
    C2 = T.copy()
    for l in LISTS:
        C2[l] = [v * factor(lin, l) for v, lin in zip(C2[l], C2.lineage)]
    tot0 = T[LISTS].sum(1)
    C2[LISTS] = C2[LISTS].div(C2[LISTS].sum(1), axis=0).mul(tot0, axis=0)
    firm = C2.groupby("lineage")[LISTS + ["w_time"]].apply(
        lambda g: pd.Series(np.average(g[LISTS], axis=0, weights=g.w_time), index=LISTS))
    w = np.array([FIRM_WEIGHTS.get(f, 1.0) for f in firm.index])
    return pd.Series(np.average(firm.values, axis=0, weights=w), index=LISTS), firm, C2


def simulate(center, firm, final, n=N_SIM):
    """
    log v_j = log center_j + family shock (sd FAMILY_SD, shared by lists of a family)
              + list shock (sd so that total sd = empirical LOEO residual sd for the list's size)
              + "which firms are right" shock along the leading axis of 2026 firm disagreement
                (scaled by DISAGREE_SCALE; the axis is estimated from the firm means by SVD)
              + small independent firm-disagreement noise.
    Then renormalise (unpolled small lists ~1.2%), threshold 3.25% of ALL valid votes, Bader-Ofer
    with the 2026 surplus agreements.
    """
    lists = LISTS
    mu = np.log(center[lists].values)
    fams = sorted({CF.FAMILY_2026[l] for l in lists})
    fam_idx = np.array([fams.index(CF.FAMILY_2026[l]) for l in lists])
    tot_sd = final["total_resid_sd_by_size"]

    def total_sd_of(v):
        for key, sd in tot_sd.items():
            lo, hi = [float(x) for x in key.strip("(]").split(",")]
            if lo < v <= hi:
                return sd
        return 0.2
    sd_tot = np.array([total_sd_of(v) for v in center[lists].values])
    sd_l = np.sqrt(np.maximum(sd_tot ** 2 - FAMILY_SD ** 2, 0.01 ** 2))
    lf = np.log(firm[lists].clip(lower=0.005))
    firm_dev = lf - lf.mean()
    U, S, Vt = np.linalg.svd(firm_dev.values, full_matrices=False)
    pc1 = Vt[0] * S[0] / np.sqrt(len(firm))  # 1-sd move along the main disagreement axis
    dis = firm_dev.std().values
    z_pc = RNG.standard_normal(n) * DISAGREE_SCALE
    E = (FAMILY_SD * RNG.standard_normal((n, len(fams)))[:, fam_idx]
         + sd_l[None, :] * RNG.standard_normal((n, len(lists)))
         + z_pc[:, None] * pc1[None, :]
         + 0.25 * dis[None, :] * RNG.standard_normal((n, len(lists))))
    V = np.exp(mu[None, :] + E)
    others = OTHERS_SHARE * np.exp(0.3 * RNG.standard_normal(n))
    V = V / V.sum(1, keepdims=True) * (1 - others)[:, None]
    seats = np.zeros_like(V, dtype=int)
    for i in range(n):
        a = CF.allocate(dict(zip(lists, V[i])), 0.0325, 120, SURPLUS, total=1.0)
        seats[i] = [a[l] for l in lists]
    return V, seats, pc1


def l1_optimal(seats, min_seats=4):
    """
    Integer seat vector summing to 120 that minimises E[sum_j |s_j - S_j|] (the competition score),
    restricted to feasible values s_j in {0} U {min_seats, ..., 120} (a list that passes the 3.25%
    threshold cannot win 1-3 seats).
    Start: per-list medians (the unconstrained optimum), projected to the feasible set; then move one
    "step" at a time - +/-1 seat, or a 0<->4 jump for a threshold list - choosing the step with the
    smallest expected-loss increase per seat until the total is 120.
    """
    n, J = seats.shape
    el = lambda j, x: float(np.mean(np.abs(seats[:, j] - x)))  # expected abs loss of guessing x
    s = np.round(np.median(seats, axis=0)).astype(int)
    for j in range(J):  # project 1..3 to the better of 0 / 4
        if 0 < s[j] < min_seats:
            s[j] = 0 if el(j, 0) <= el(j, min_seats) else min_seats
    while s.sum() != 120:
        best, best_cost = None, np.inf
        for j in range(J):
            if s.sum() < 120:
                new = min_seats if s[j] == 0 else s[j] + 1
                if new - s[j] > 120 - s.sum():
                    continue
            else:
                new = 0 if s[j] == min_seats else s[j] - 1
                if new < 0 or s[j] - new > s.sum() - 120:
                    continue
            cost = (el(j, new) - el(j, s[j])) / abs(new - s[j])
            if cost < best_cost:
                best, best_cost = (j, new), cost
        s[best[0]] = best[1]
    exp_loss = sum(el(j, s[j]) for j in range(J))
    unconstrained = sum(el(j, np.median(seats[:, j])) for j in range(J))
    return s, exp_loss, unconstrained


def main():
    params = json.load(open(os.path.join(CF.PROC, "model_params.json")))
    final = json.load(open(os.path.join(CF.PROC, "model_params_final.json")))
    T = build_poll_table()
    T = T[~T.lineage.isin(EXCLUDE_FIRMS)]
    factor = corrections(params, final)
    center, firm, Tc = aggregate(T, factor)
    raw_center, raw_firm, _ = aggregate(T, lambda lin, l: 1.0)
    V, S, pc1 = simulate(center, firm, final)
    point, exp_loss, exp_loss_unconstrained = l1_optimal(S)
    out = pd.DataFrame({
        "list": [CF.NAMES_2026[l] for l in LISTS], "id": LISTS,
        "raw_poll_avg_share": raw_center.values * 100, "corrected_share": center.values * 100,
        "sim_mean_share": V.mean(0) * 100,
        "seats_mean": S.mean(0), "seats_median": np.median(S, 0),
        "p10": np.percentile(S, 10, 0), "p90": np.percentile(S, 90, 0),
        "p_pass_threshold": (S > 0).mean(0), "FORECAST_SEATS": point})
    out = pd.concat([out, pd.DataFrame([{"list": "Blue & White (withdrew)", "id": "BLUE_WHITE", "FORECAST_SEATS": 0}])])
    os.makedirs(CF.OUTT, exist_ok=True)
    out.to_csv(os.path.join(CF.OUTT, f"forecast_2026_{TAG}.csv"), index=False, encoding="utf-8-sig")
    firm.mul(100).round(2).to_csv(os.path.join(CF.OUTT, f"firm_means_corrected_{TAG}.csv"))
    raw_firm.mul(100).round(2).to_csv(os.path.join(CF.OUTT, f"firm_means_raw_{TAG}.csv"))
    bloc = S[:, [LISTS.index(x) for x in ["LIKUD", "OTZMA", "SHAS", "UTJ", "RZP_ZEHUT"]]].sum(1)
    summary = dict(tag=TAG, n_polls=len(T), n_firms=T.lineage.nunique(), expected_abs_error=exp_loss,
                   expected_abs_error_unconstrained_medians=exp_loss_unconstrained,
                   p_coalition_61=float((bloc >= 61).mean()), bloc_median=float(np.median(bloc)),
                   disagreement_axis=dict(zip(LISTS, np.round(pc1, 3).tolist())),
                   settings=dict(WINDOW_DAYS=WINDOW_DAYS, HALF_LIFE=HALF_LIFE, CORRECTION=CORRECTION,
                                 HOUSE_CARRY=HOUSE_CARRY, FAMILY_SD=FAMILY_SD, DISAGREE_SCALE=DISAGREE_SCALE,
                                 EXCLUDE_FIRMS=EXCLUDE_FIRMS, FIRM_WEIGHTS=FIRM_WEIGHTS, RAW_MODE=RAW_MODE,
                                 N_SIM=N_SIM))
    json.dump(summary, open(os.path.join(CF.OUTT, f"forecast_2026_{TAG}_summary.json"), "w"), indent=1)
    np.save(os.path.join(CF.PROC, f"sim_seats_{TAG}.npy"), S)
    pd.set_option("display.width", 220)
    print(out.round(2).to_string(index=False))
    print(summary)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
