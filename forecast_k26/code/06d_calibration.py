"""
Step 06d - Out-of-sample calibration of the simulation (is the uncertainty honest?).

For each historical election k (2009-2022):
  * center  = E2-corrected pollster average, with correction factors fitted WITHOUT election k
              (from 06c's LOEO residual file: corrected shares already computed out of sample);
  * error sd by list size estimated from the OTHER elections' residuals only;
  * simulate N draws with step 07's historical error terms (family shock 0.08 + list shock; the 2026-only
    firm-disagreement terms are not part of this check), allocate seats
    with the election's threshold (no surplus agreements - unknown for some years);
  * record: coverage of actual seats by the 80% (p10-p90) interval, PIT values, the Brier score
    and log-loss of P(list passes threshold), and the expected vs realised sum |seat error| of the
    L1-optimal point forecast.
Output: outputs/tables/calibration_by_list.csv, calibration_summary.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as CF  # noqa: E402

RNG = np.random.default_rng(99)
N = 4000
FAMILY_SD = 0.08
BINS = [0, 0.04, 0.08, 0.15, 1.0]


def main():
    R = pd.read_csv(os.path.join(CF.OUTT, "e2_loeo_residuals.csv"))
    R = R[R.family != "OTHER"]
    rows, summ = [], []
    for k in sorted(R.knesset.unique()):
        g = R[R.knesset == k].reset_index(drop=True)
        other = R[(R.knesset != k) & (R.poll_seats > 0.05)]  # error sizes from measured lists only
        sd_by_bin = other.groupby(pd.cut(other.corrected, BINS)).resid.std()
        def sd_of(v):
            for iv, sd in sd_by_bin.items():
                if iv.left < v <= iv.right:
                    return sd
            return 0.2
        sd_tot = np.array([sd_of(v) for v in g.corrected])
        sd_l = np.sqrt(np.maximum(sd_tot ** 2 - FAMILY_SD ** 2, 1e-4))
        fams = sorted(g.family.unique())
        fi = np.array([fams.index(f) for f in g.family])
        mu = np.log(g.corrected.values)
        E = FAMILY_SD * RNG.standard_normal((N, len(fams)))[:, fi] + sd_l * RNG.standard_normal((N, len(g)))
        V = np.exp(mu + E)
        tot = g.corrected.sum()
        V = V / V.sum(1, keepdims=True) * tot  # keep the polled lists' total share
        S = np.array([[CF.allocate(dict(zip(g.letters, v)), CF.THRESHOLD[k], total=1.0)[l] for l in g.letters]
                      for v in V])
        act = g.seats_actual.values
        p10, p90 = np.percentile(S, 10, 0), np.percentile(S, 90, 0)
        med = np.median(S, 0)
        pit = (S < act).mean(0) + 0.5 * (S == act).mean(0)
        ppass = (S > 0).mean(0)
        passed = (act > 0).astype(float)
        for j, l in enumerate(g.letters):
            rows.append(dict(knesset=k, list=CF.PARTIES[k].get(l, (l,))[0], corrected_share=g.corrected[j],
                             actual_seats=act[j], median=med[j], p10=p10[j], p90=p90[j],
                             in80=bool(p10[j] <= act[j] <= p90[j]), pit=pit[j], p_pass=ppass[j], passed=passed[j]))
        exp_loss = np.mean(np.abs(S - med).sum(1))
        summ.append(dict(knesset=k, expected_sum_abs_err=exp_loss, realised_sum_abs_err=np.abs(med - act).sum()))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(CF.OUTT, "calibration_by_list.csv"), index=False, encoding="utf-8-sig")
    near = D[(D.corrected_share > 0.015) & (D.corrected_share < 0.06)]
    brier = np.mean((near.p_pass - near.passed) ** 2)
    brier_clim = np.mean((near.passed.mean() - near.passed) ** 2)
    eps = 1e-3
    ll = -np.mean(near.passed * np.log(near.p_pass.clip(eps, 1 - eps)) +
                  (1 - near.passed) * np.log((1 - near.p_pass).clip(eps, 1 - eps)))
    # reliability of P(pass) near the threshold: logistic (Platt) recalibration y ~ a + b*logit(p)
    import statsmodels.api as sm
    pc = near.p_pass.clip(0.01, 0.99)
    xl = np.log(pc / (1 - pc))
    try:
        pl = sm.Logit(near.passed.values, sm.add_constant(xl.values)).fit(disp=0)
        platt = dict(platt_intercept=float(pl.params[0]), platt_slope=float(pl.params[1]),
                     platt_p_intercept=float(pl.pvalues[0]))
    except Exception:  # noqa: BLE001
        platt = dict(platt_intercept=np.nan, platt_slope=np.nan, platt_p_intercept=np.nan)
    rel = near.assign(bin=pd.cut(near.p_pass, [0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0])).groupby("bin", observed=True)         .agg(n=("passed", "size"), mean_predicted=("p_pass", "mean"), observed=("passed", "mean"))
    rel.to_csv(os.path.join(CF.OUTT, "calibration_reliability.csv"))
    print(rel.round(3).to_string())
    S2 = pd.DataFrame(summ)
    out = dict(n_lists=len(D), coverage_80=D.in80.mean(), pit_mean=D.pit.mean(), pit_sd=D.pit.std(),
               n_near_threshold=len(near), brier_near_threshold=brier, brier_climatology=brier_clim,
               logloss_near_threshold=ll, mean_expected_err=S2.expected_sum_abs_err.mean(),
               mean_realised_err=S2.realised_sum_abs_err.mean(), mean_p_near=float(near.p_pass.mean()),
               observed_pass_near=float(near.passed.mean()), **platt)
    pd.DataFrame([out]).to_csv(os.path.join(CF.OUTT, "calibration_summary.csv"), index=False)
    S2.to_csv(os.path.join(CF.OUTT, "calibration_expected_vs_realised.csv"), index=False)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()})
    print(S2.round(1).to_string(index=False))
    print(near[["knesset", "list", "corrected_share", "p_pass", "passed"]].round(3).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
