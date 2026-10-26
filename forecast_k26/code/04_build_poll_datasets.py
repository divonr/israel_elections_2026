"""
Step 04 - Build clean poll datasets.

(a) hist_polls.csv  : every seat-projection poll of K18-K25 (2009-2022) from the final list
                      configuration table, mapped to ballot letters, party family and pollster lineage.
(b) actual_results.csv : actual vote share and seats for every list (K18-K25).
(c) polls_2026.csv  : 2026 polls (Wikipedia, final two list configurations: before/after the
                      Blue & White withdrawal on 19.10), cross-checked against the two AI-agent
                      collections; raw percentages from the CEC reports (agents' files) attached.
(d) polls_2026_crosscheck.csv : every seat discrepancy Wikipedia vs agents.
"""
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

BASE = os.path.dirname(C.ROOT)  # the user's folder with the agents' files

AGENT_PARTY = {"הליכוד": "LIKUD", "ישר! עם איזנקוט.": "YASHAR", "ביחד": "TOGETHER",
               "הדמוקרטים": "DEMOCRATS", "ישראל ביתנו": "YB", "עוצמה יהודית": "OTZMA", 'ש"ס': "SHAS",
               "יהדות התורה": "UTJ", "הציונות הדתית וזהות": "RZP_ZEHUT", "הרשימה המשותפת": "JOINT",
               'רע"מ': "RAAM", "מפלגת יועז הנדל וירון זליכה": "HENDEL_ZELICHA", "עמך ישראל": "AMCHA",
               "כחול לבן": "BLUE_WHITE", "הציבור החרדי": "HAREDI_PUBLIC"}
AGENT_PUB = {"חדשות 13": "Channel 13", "חדשות 12": "HaHadashot 12", "ישראל היום": "Israel Hayom",
             "חדשות 14": "Channel 14", "זמן ישראל": "Zman Yisrael", "מעריב": "Maariv", "i24NEWS": "i24 News",
             "וואלה": "Walla", "ערוץ 16": "Channel 16", "כאן 11": "Kan 11"}


def hist_polls():
    w = pd.read_csv(os.path.join(C.PROC, "wiki_polls_long.csv"))
    w["party_wiki"] = w["party_wiki"].str.strip()
    rows, drop_spans = [], []
    for e, k in C.ELECTIONS.items():
        g = w[(w.election == e)]
        t0 = g.table_idx.min()
        g = g[(g.table_idx == t0) & (~g.is_result_row)].copy()
        g["letters"] = g.party_wiki.map(C.WIKI_TO_LETTERS[k])
        # a body cell spanning columns of lists that ran SEPARATELY (e.g. a combined "Arab parties" figure in
        # 2009) cannot be split: remember it and drop it for every spanned list after the row-total check.
        # (Unmapped columns are parts of a joint list, e.g. Yisrael Beiteinu inside Likud-Beiteinu: kept.)
        jn = g[g.note.fillna("").str.startswith("joint with ")]
        for _, r in jn.iterrows():
            first = r.note[len("joint with "):].strip()
            la, lb = C.WIKI_TO_LETTERS[k].get(first), C.WIKI_TO_LETTERS[k].get(r.party_wiki)
            if la and lb and la != lb:
                drop_spans.append((e, r.row_idx, la, lb))
        g = g[g.letters.notna()]
        g["knesset"] = k
        ed = pd.Timestamp(C.ELECTION_DATE[k])
        g["date_end"] = pd.to_datetime(g.date_end, errors="coerce")
        # Wikipedia tables run newest -> oldest and often omit the year: a row dated LATER than the rows
        # above it belongs to the previous year (e.g. a "15 Mar" 2014 row inside the 2015 table)
        rd = g.groupby("row_idx").date_end.first().sort_index()
        fixed, run_min = {}, None
        for ri, d in rd.items():
            if pd.isna(d):
                continue
            while run_min is not None and d > run_min + pd.Timedelta(days=3):
                d = d - pd.DateOffset(years=1)
            fixed[ri] = d
            run_min = d if run_min is None else min(run_min, d)
        g["date_end"] = g.row_idx.map(fixed)
        g["days_before"] = (ed - g.date_end).dt.days
        g["lineage"] = [C.pollster_lineage(a, b, e) for a, b in zip(g.pollster, g.publisher)]
        g["family"] = [C.PARTIES[k].get(l, ("", "OTHER"))[1] for l in g.letters]
        # several wiki columns can map to one list (a later merger, or a joint cell): a joint cell sits in
        # one column (the other is NaN); separate pre-merger values are summed
        agg = {c: "first" for c in g.columns if c not in ("election", "row_idx", "letters", "seats", "pct")}
        agg.update(seats=lambda x: x.sum(min_count=1), pct=lambda x: x.sum(min_count=1))
        g = g.groupby(["election", "row_idx", "letters"], as_index=False).agg(agg)
        rows.append(g)
    h = pd.concat(rows, ignore_index=True)
    # drop exit polls / day-of-election rows and rows with a broken total
    h = h[(h.days_before >= 1) & ~h.pollster.str.contains("exit", case=False, na=False)]
    # and reference rows that are not polls (e.g. "Current Composition" of the outgoing Knesset)
    h = h[~h.pollster.str.contains(r"current (?:composition|seats)|election results|final results",
                                   case=False, na=False)]
    tot = h.groupby(["election", "row_idx"]).seats.transform("sum")
    h["row_total"] = tot
    h = h[(tot >= 116) & (tot <= 124)].copy()
    for e, ri, la, lb in drop_spans:
        h.loc[(h.election == e) & (h.row_idx == ri) & (h.letters.isin([la, lb])), "seats"] = np.nan
    h.to_csv(os.path.join(C.PROC, "hist_polls.csv"), index=False, encoding="utf-8-sig")
    print("historical polls:", h.groupby("election").row_idx.nunique().to_dict())
    return h


def actual_results():
    n = pd.read_csv(os.path.join(C.PROC, "national_totals.csv"))
    out = []
    for k in range(16, 26):
        g = n[n.knesset == k].copy()
        g["share"] = g.votes / g.votes.sum()
        g["name"] = [C.PARTIES[k].get(l, (l, "OTHER"))[0] for l in g.letters]
        g["family"] = [C.PARTIES[k].get(l, (l, "OTHER"))[1] for l in g.letters]
        g["seats"] = [C.ACTUAL_SEATS.get(k, {}).get(l, np.nan) for l in g.letters]
        out.append(g)
    a = pd.concat(out)
    a.to_csv(os.path.join(C.PROC, "actual_results.csv"), index=False, encoding="utf-8-sig")
    return a


def polls_2026():
    w = pd.read_csv(os.path.join(C.PROC, "wiki_polls_long.csv"))
    w = w[w.election == "2026"].copy()
    w["party_wiki"] = w["party_wiki"].str.strip()
    final_set = set(C.PARTIES_2026)
    keep_tabs = [t for t, g in w.groupby("table_idx") if set(g.party_wiki) <= final_set]
    w = w[w.table_idx.isin(keep_tabs) & ~w.is_result_row]
    w["party"] = w.party_wiki.map(lambda x: C.PARTIES_2026[x][0])
    w["date_end"] = pd.to_datetime(w.date_end)
    w["lineage"] = [C.pollster_lineage(a, b, "2026") for a, b in zip(w.pollster, w.publisher)]
    w["publisher"] = w.publisher.str.replace(r"\s*\[.*$", "", regex=True).str.strip()
    w["poll_id"] = w.date_end.dt.strftime("%Y-%m-%d") + "_" + w.publisher
    w["source"] = "wikipedia"
    w = w[["poll_id", "table_idx", "date_raw", "date_end", "pollster", "lineage", "publisher",
           "sample_size", "party", "seats", "pct", "source"]]

    # agents' collections (instinct 15.10 + 25.10; they cross-validated with perplexity on 15.10)
    ag = []
    for f in ["instinct/01 15.10 סקרי מנדטים.csv", "instinct/25.10 עדכון.csv"]:
        d = pd.read_csv(os.path.join(BASE, f), dtype=str)
        for _, r in d.iterrows():
            for he, pid in AGENT_PARTY.items():
                s = r.get(f"{he}_מנדטים")
                pr = r.get(f"{he}_אחוז_בדוח")
                pb = r.get(f"{he}_אחוז_מתחת_לחסימה_מדווח")
                ag.append(dict(agent_id=r["מזהה_סקר"], date_pub=r["תאריך_פרסום"],
                               publisher=AGENT_PUB.get(r["מפרסם"], r["מפרסם"]), firm=r["מכון_וסוקר"],
                               fieldwork_end=r["סוף_איסוף_מדווח"], n=r["משיבים_לפי_דוח"] or r["משיבים_לפי_כתבה"],
                               party=pid, seats_agent=pd.to_numeric(s, errors="coerce"),
                               pct_report=pd.to_numeric(pr, errors="coerce"),
                               pct_below=pd.to_numeric(pb, errors="coerce")))
    ag = pd.DataFrame(ag).drop_duplicates(["agent_id", "party"], keep="last")
    ag.to_csv(os.path.join(C.PROC, "agents_polls_2026_long.csv"), index=False, encoding="utf-8-sig")

    # match agents' polls to wikipedia polls by publisher and date (publication vs fieldwork +-2 days)
    wp = w.groupby("poll_id").agg(date_end=("date_end", "first"), publisher=("publisher", "first")).reset_index()
    ap = ag.groupby("agent_id").agg(date_pub=("date_pub", "first"), publisher=("publisher", "first"),
                                    fieldwork_end=("fieldwork_end", "first")).reset_index()
    ap["date_pub"] = pd.to_datetime(ap.date_pub)
    ap["date_fw"] = pd.to_datetime(ap.fieldwork_end, errors="coerce").fillna(ap.date_pub)
    # one-to-one greedy matching on (publisher, |fieldwork/publication date - wiki date|)
    cand = []
    for _, r in ap.iterrows():
        for _, c in wp[wp.publisher == r.publisher].iterrows():
            d1 = abs((c.date_end - r.date_fw).days)
            d2 = abs((c.date_end - r.date_pub).days)
            if min(d1, d2) <= 1:
                cand.append((min(d1, d2), d1, r.agent_id, c.poll_id))
    match, used = {}, set()
    for _, _, a, p in sorted(cand):
        if a not in match and p not in used:
            match[a] = p
            used.add(p)
    ag["poll_id"] = ag.agent_id.map(match)
    m = w.merge(ag[["poll_id", "party", "seats_agent", "pct_report", "pct_below", "agent_id", "n"]],
                on=["poll_id", "party"], how="left")
    cc = m[m.seats.notna() & m.seats_agent.notna() & (m.seats != m.seats_agent)]
    cc.to_csv(os.path.join(C.PROC, "polls_2026_crosscheck.csv"), index=False, encoding="utf-8-sig")
    unmatched = sorted(set(ap.agent_id) - set(match))
    print("agents' polls matched to wikipedia:", len(match), "unmatched:", unmatched)
    print("seat discrepancies wiki vs agents:", len(cc))
    if len(cc):
        print(cc[["poll_id", "party", "seats", "seats_agent"]].to_string())

    # agents' polls missing from Wikipedia (e.g. DRI) are appended from the agents' data
    extra = ag[ag.agent_id.isin(unmatched)].copy()
    if len(extra):
        extra["poll_id"] = extra.date_pub + "_" + extra.publisher.str.slice(0, 20)
        extra["date_end"] = pd.to_datetime(extra.fieldwork_end.fillna(extra.date_pub))
        extra["pollster"] = extra.firm
        extra["lineage"] = [C.pollster_lineage(f, p, "2026") for f, p in zip(extra.firm, extra.publisher)]
        extra["seats"] = extra.seats_agent
        extra["pct"] = extra.pct_report.fillna(extra.pct_below)
        extra["source"] = "agents"
        extra["sample_size"] = extra.n
        extra["table_idx"] = -1
        extra["date_raw"] = extra.date_pub
        m = pd.concat([m, extra[[c for c in m.columns if c in extra.columns]]], ignore_index=True)
    # 2026 firm per outlet (Wikipedia "Pollsters and publishers" table, valid for the whole campaign window)
    pub2firm = {"Channel 13": "Ch13_Madad2026", "HaHadashot 12": "Midgam", "Israel Hayom": "Kantar",
                "Kan 11": "Kantar", "Channel 14": "NextData_Filber", "Zman Yisrael": "Tatika",
                "Maariv": "Lazar_Panels", "Walla": "Lazar_Panels", "i24 News": "DirectPolls",
                "Channel 16": "MaagarMochot"}
    m["lineage"] = [pub2firm.get(pb, ln) for pb, ln in zip(m.publisher, m.lineage)]
    m.loc[m.publisher.astype(str).str.contains("ועדת"), "lineage"] = "DRI"
    m.loc[m.publisher.astype(str).str.contains("ועדת"), "publisher"] = "DRI (CEC report only)"
    # fieldwork end dates verified against the CEC poll reports (research workflow, 26 Oct)
    date_fixes = {"2026-10-23_Kan 11": "2026-10-22",   # CEC Survey_4260: fieldwork 21 Oct 16:40 - 22 Oct 13:40
                  "2026-10-23_i24 News": "2026-10-22"}  # i24 article: single-day fieldwork 22 Oct
    for pid, d in date_fixes.items():
        m.loc[m.poll_id == pid, "date_end"] = pd.Timestamp(d)
    # Wikipedia "N/a" cells where the agents' sources give an explicit below-threshold result
    # (e.g. Israel Hayom 22.10: Amcha 0 seats, 2.7% - confirmed by CEC Survey_4261)
    na = m.seats.isna() & (m.seats_agent == 0)
    m.loc[na, "seats"] = 0
    m.loc[na & m.pct.isna(), "pct"] = m.loc[na & m.pct.isna(), "pct_below"]
    # use the CEC-report raw percentage when available
    m["pct_raw"] = m.pct_report
    m.loc[m.pct_raw.isna(), "pct_raw"] = m.loc[m.pct_raw.isna(), "pct"]
    m.to_csv(os.path.join(C.PROC, "polls_2026.csv"), index=False, encoding="utf-8-sig")
    print("2026 polls:", m.poll_id.nunique(), "from", m.date_end.min(), "to", m.date_end.max())
    return m


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    hist_polls()
    actual_results()
    polls_2026()
