"""
Step 04b - Second historical poll source: Hebrew Wikipedia (+ archives), final month before each election.

Input : data/raw/hewiki/polls_<election>.csv (extracted and independently verified by research agents;
        one row per poll, one column per list's ballot letters holding seats).
Output: data/processed/hewiki_polls_long.csv   (long format, same columns as hist_polls.csv)
        data/processed/hist_polls_merged.csv   (English-Wikipedia polls + polls found only in the Hebrew source)
        outputs/tables/hewiki_crosscheck.csv   (seat agreement between the two sources on matched polls)
Matching: same election, |date difference| <= 2 days (the English date against the Hebrew fieldwork OR
publication date), closest seat vector first, one-to-one; same firm lineage required unless the tables are
near-identical (<= 2 seats apart); 3-4 days apart is accepted only for near-identical tables. Matched polls are
cross-checked seat by seat; unmatched Hebrew polls are added.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

RAW = os.path.join(C.ROOT, "data", "raw", "hewiki")


def load_hewiki():
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, "polls_*.csv"))):
        e = re.search(r"polls_(\w+)\.csv", os.path.basename(f)).group(1)
        if e not in C.ELECTIONS:
            continue
        k = C.ELECTIONS[e]
        d = pd.read_csv(f, dtype=str)
        letters = [l for l in C.PARTIES[k] if l in d.columns]
        ed = pd.Timestamp(C.ELECTION_DATE[k])
        for i, r in d.iterrows():
            date = pd.to_datetime(r.get("fieldwork_end") or r.get("pub_date"), errors="coerce")
            pub = pd.to_datetime(r.get("pub_date"), errors="coerce")
            if pd.isna(date):
                date = pub
            pct = {}
            for kv in str(r.get("pct_below", "") or "").split(";"):
                if ":" in kv:
                    a, b = kv.split(":", 1)
                    try:
                        pct[a.strip()] = float(b.strip().replace("%", ""))
                    except ValueError:
                        pass
            pollster = str(r.get("pollster", ""))
            publisher = str(r.get("publisher", ""))
            for l in letters:
                v = pd.to_numeric(r[l], errors="coerce")
                if pd.isna(v):
                    continue
                rows.append(dict(election=e, knesset=k, row_idx=f"he{i}", date_end=date, pub_date=pub,
                                 days_before=(ed - date).days if not pd.isna(date) else np.nan,
                                 pollster=pollster, publisher=publisher,
                                 lineage=C.pollster_lineage(pollster, publisher, e), letters=l, seats=float(v),
                                 pct=pct.get(l), family=C.PARTIES[k][l][1], source="hewiki"))
    H = pd.DataFrame(rows)
    if len(H):
        tot = H.groupby(["election", "row_idx"]).seats.transform("sum")
        H["row_total"] = tot
        H = H[(H.days_before >= 1) & (tot >= 116) & (tot <= 124)]
    return H


def main():
    H = load_hewiki()
    if H.empty:
        print("no Hebrew-Wikipedia polls found")
        return
    H.to_csv(os.path.join(C.PROC, "hewiki_polls_long.csv"), index=False, encoding="utf-8-sig")
    E = pd.read_csv(os.path.join(C.PROC, "hist_polls.csv"))
    E["date_end"] = pd.to_datetime(E.date_end)
    E["source"] = "enwiki"
    ev = E.pivot_table(index=["election", "row_idx", "lineage", "date_end"], columns="letters", values="seats",
                       aggfunc="first")
    hv = H.pivot_table(index=["election", "row_idx", "lineage", "date_end"], columns="letters", values="seats",
                       aggfunc="first")
    # English Wikipedia dates some polls by fieldwork and others by publication: compare with both Hebrew dates
    hpub = H.groupby(["election", "row_idx"]).pub_date.first()
    cand = []
    for hk, hrow in hv.iterrows():
        e, hri, hlin, hd = hk
        sub = ev.loc[e] if e in ev.index.get_level_values(0) else None
        if sub is None:
            continue
        hp = hpub.get((e, hri), pd.NaT)
        for ek, erow in sub.iterrows():
            eri, elin, edt = ek
            dd = abs((edt - hd).days)
            if not pd.isna(hp):
                dd = min(dd, abs((edt - hp).days))
            if dd > 4:
                continue
            same_firm = (elin == hlin)
            common = hrow.dropna().index.intersection(erow.dropna().index)
            if len(common) < 5:
                continue
            dist = float(np.abs(hrow[common] - erow[common]).sum())
            cand.append((0 if same_firm else 1, dist, dd, e, hri, eri))
    cand.sort()
    used_h, used_e, matches = set(), set(), []
    for sf, dist, dd, e, hri, eri in cand:
        if (e, hri) in used_h or (e, eri) in used_e:
            continue
        if sf == 1 and (dist > 2 or dd > 2):  # different firm label: accept only near-identical tables
            continue
        if dist > (12 if dd <= 2 else 2):  # 3-4 days apart: only near-identical tables
            continue
        used_h.add((e, hri))
        used_e.add((e, eri))
        matches.append(dict(election=e, he_row=hri, en_row=eri, seat_abs_diff=dist, same_firm=sf == 0))
    M = pd.DataFrame(matches)
    M.to_csv(os.path.join(C.OUTT, "hewiki_crosscheck.csv"), index=False)
    new = H[~H.set_index(["election", "row_idx"]).index.isin(list(used_h))]
    win = new[new.days_before <= 14]
    merged = pd.concat([E, new.assign(poll_source="hewiki")], ignore_index=True)
    merged.to_csv(os.path.join(C.PROC, "hist_polls_merged.csv"), index=False, encoding="utf-8-sig")
    n_he = H.groupby(["election", "row_idx"]).ngroups
    print(f"Hebrew-source polls: {n_he}; matched to English Wikipedia: {len(M)} "
          f"(identical seat tables: {(M.seat_abs_diff == 0).sum()}, mean |diff| {M.seat_abs_diff.mean():.2f} seats)")
    print("new polls (not in English Wikipedia):", new.groupby("election").row_idx.nunique().to_dict(),
          "| of which in the final fortnight:", win.groupby("election").row_idx.nunique().to_dict())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
