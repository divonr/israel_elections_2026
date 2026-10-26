"""
Step 05 - Cluster the electorate into voting segments using the true results.

Units      : ballot boxes (K16-K25 pooled, ~100k unit-elections) and, as a variant, settlements.
Features   : vote shares of 9 party families (LIKUD, SHAS, UTJ, NATREL, YB, CENTER, LEFT, ARAB, OTHER),
             Hellinger-transformed (sqrt of shares) - the standard Euclidean embedding for compositions.
             Pooling all elections in one space gives segment definitions that are *the same across
             elections*, so a segment's size and party mix can be followed over time.
Methods    : K-means, Gaussian mixture (full covariance), Ward agglomerative (fit on a subsample,
             extended by nearest centroid), NMF soft archetypes (hard label = argmax loading).
Validation : silhouette (sample), Calinski-Harabasz, Davies-Bouldin, GMM BIC, bootstrap stability
             (adjusted Rand index between fits on two bootstrap resamples, evaluated on a common
             hold-out set). The downstream test (does the segmentation explain pollster errors?) is in
             step 06.
Outputs    : data/processed/clusters/{method}_K{k}_{unit}.parquet  (unit id, knesset, cluster)
             data/processed/clusters/profiles_{method}_K{k}_{unit}.csv (knesset, cluster, weight, family shares)
             outputs/tables/clustering_validation.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import NMF
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

warnings.filterwarnings("ignore")
CL = os.path.join(C.PROC, "clusters")
os.makedirs(CL, exist_ok=True)
os.makedirs(C.OUTT, exist_ok=True)
RNG = np.random.default_rng(26)
KS = [4, 6, 8, 10, 12, 14]


def family_matrix(unit="ballot"):
    lg = pd.read_parquet(os.path.join(C.PROC, "ballots_long.parquet"))
    meta = pd.read_parquet(os.path.join(C.PROC, "ballots_meta.parquet"))
    fam = {(k, l): f for k, d in C.PARTIES.items() for l, (_, f) in d.items()}
    lg["family"] = [fam.get((k, l), "OTHER") for k, l in zip(lg.knesset, lg.letters)]
    if unit == "ballot":
        X = lg.pivot_table(index=["ballot_uid", "knesset"], columns="family", values="votes",
                           aggfunc="sum", fill_value=0)
    else:
        lg = lg.merge(meta[["ballot_uid", "settlement_code", "settlement_name"]], on="ballot_uid")
        lg["unit"] = lg.settlement_code.fillna(-1).astype(int).astype(str)
        lg.loc[lg.settlement_code.isna(), "unit"] = "N_" + lg.settlement_name
        X = lg.pivot_table(index=["unit", "knesset"], columns="family", values="votes",
                           aggfunc="sum", fill_value=0)
    X = X.reindex(columns=C.FAMILIES, fill_value=0)
    X = X[X.sum(1) >= 50]
    return X


def hellinger(X):
    S = X.div(X.sum(1), axis=0).values
    return np.sqrt(S), S


def fit_predict(method, k, Z, S, fit_idx):
    if method == "kmeans":
        m = KMeans(k, n_init=4, random_state=0).fit(Z[fit_idx])
        return m.predict(Z), m
    if method == "gmm":
        m = GaussianMixture(k, covariance_type="full", random_state=0, max_iter=200,
                            reg_covar=1e-4).fit(Z[fit_idx])
        return m.predict(Z), m
    if method == "ward":
        sub = fit_idx if len(fit_idx) <= 12000 else RNG.choice(fit_idx, 12000, replace=False)
        lab = AgglomerativeClustering(k, linkage="ward").fit_predict(Z[sub])
        cent = np.vstack([Z[sub][lab == j].mean(0) for j in range(k)])
        d = ((Z[:, None, :] - cent[None]) ** 2).sum(-1)
        return d.argmin(1), cent
    if method == "nmf":
        m = NMF(k, init="nndsvda", random_state=0, max_iter=400).fit(S[fit_idx])
        W = m.transform(S)
        # hard label = archetype contributing the largest share of the unit's reconstructed vote
        # (W alone is scale-dependent: NMF can trade scale between W columns and H rows)
        contrib = W * m.components_.sum(axis=1)[None, :]
        return contrib.argmax(1), m
    raise ValueError(method)


def stability(method, k, Z, S, n_rep=3, n_sub=20000):
    n = len(Z)
    hold = RNG.choice(n, min(8000, n), replace=False)
    aris = []
    for _ in range(n_rep):
        a = RNG.choice(n, min(n_sub, n), replace=True)
        b = RNG.choice(n, min(n_sub, n), replace=True)
        la, _ = fit_predict(method, k, Z, S, a)
        lb, _ = fit_predict(method, k, Z, S, b)
        aris.append(adjusted_rand_score(la[hold], lb[hold]))
    return float(np.mean(aris)), float(np.std(aris))


def profiles(X, labels):
    df = X.copy()
    df["cluster"] = labels
    df = df.reset_index()
    g = df.groupby(["knesset", "cluster"])[C.FAMILIES].sum()
    tot = g.sum(1)
    prof = g.div(tot, axis=0)
    prof["votes"] = tot
    prof["weight"] = tot / tot.groupby(level=0).transform("sum")
    return prof.reset_index()


def main():
    vpath = os.path.join(C.OUTT, "clustering_validation.csv")
    rows = pd.read_csv(vpath).to_dict("records") if os.path.exists(vpath) else []
    done = {(r["unit"], r["method"], int(r["K"])) for r in rows}
    for unit in ["ballot", "settlement"]:
        X = family_matrix(unit)
        Z, S = hellinger(X)
        n = len(Z)
        print(f"[{unit}] {n} unit-elections")
        sil_idx = RNG.choice(n, min(6000, n), replace=False)
        fit_idx = np.arange(n) if unit == "settlement" else RNG.choice(n, min(40000, n), replace=False)
        methods = ["kmeans", "gmm", "ward", "nmf"]
        for method in methods:
            for k in KS:
                if method == "nmf" and k >= Z.shape[1]:
                    continue  # NMF cannot have more archetypes than the 9 family features
                if (unit, method, k) in done:
                    continue
                lab, model = fit_predict(method, k, Z, S, fit_idx)
                if len(np.unique(lab)) < 2:
                    continue
                r = dict(unit=unit, method=method, K=k, n_clusters_used=len(np.unique(lab)),
                         silhouette=silhouette_score(Z[sil_idx], lab[sil_idx]),
                         calinski_harabasz=calinski_harabasz_score(Z[sil_idx], lab[sil_idx]),
                         davies_bouldin=davies_bouldin_score(Z[sil_idx], lab[sil_idx]))
                if method == "gmm":
                    r["bic"] = model.bic(Z[fit_idx])
                if method == "nmf":
                    r["nmf_recon_err"] = model.reconstruction_err_
                if unit == "ballot" or method != "ward":
                    r["stability_ari"], r["stability_sd"] = stability(method, k, Z, S,
                                                                      n_sub=min(20000, n))
                rows.append(r)
                print({kk: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for kk, v in r.items()}, flush=True)
                out = pd.DataFrame({"unit_id": X.index.get_level_values(0),
                                    "knesset": X.index.get_level_values(1), "cluster": lab})
                out.to_parquet(os.path.join(CL, f"{method}_K{k}_{unit}.parquet"), index=False)
                profiles(X, lab).to_csv(os.path.join(CL, f"profiles_{method}_K{k}_{unit}.csv"), index=False)
                pd.DataFrame(rows).to_csv(vpath, index=False)  # checkpoint after every config


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
