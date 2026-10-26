"""
Step 03 - Parse the Wikipedia opinion-polling pages into a long poll table.

For every "seat projection" table we expand the HTML into a full grid (rowspan/colspan aware),
identify the party columns from the header, and read each poll row:
    seats  -> integer seats as published
    "(x%)" -> party polled below the threshold: seats 0, pct x
    "–", "—", "N/a" -> not polled / not running (NaN)
A body cell that spans several party columns (joint run in that poll) is assigned to the first
party of the span and flagged in `note`.

Output: data/processed/wiki_polls_long.csv with columns
  election, table_idx, row_idx, date_raw, date_end, pollster, publisher, sample_size,
  party_wiki, seats, pct, is_result_row, note
and data/processed/wiki_poll_events.csv (banner rows such as withdrawals).
"""
import os
import re
import sys
from datetime import date

import pandas as pd
from bs4 import BeautifulSoup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIKI = os.path.join(ROOT, "data", "raw", "wiki")
OUT = os.path.join(ROOT, "data", "processed")

ELECTION_DATE = {"2009": date(2009, 2, 10), "2013": date(2013, 1, 22), "2015": date(2015, 3, 17),
                 "2019a": date(2019, 4, 9), "2019b": date(2019, 9, 17), "2020": date(2020, 3, 2),
                 "2021": date(2021, 3, 23), "2022": date(2022, 11, 1), "2026": date(2026, 10, 27)}
# headings under which the main seat-projection tables live (others are scenarios, PM polls...)
SEAT_SECTIONS = ("seat projections", "polls", "polling", "by party", "contents", "2026")
META_COLS = {"date": "date", "fieldwork date": "date", "polling firm": "pollster", "pollster": "pollster",
             "poll": "pollster", "publisher": "publisher", "media": "publisher",
             "sample size": "sample_size"}
NON_PARTY = {"others", "other", "oth.", "gov.", "opp.", "gov", "opp", "lead", "sources", "l", "r", "c", "o",
             "don't know", "undecided", "coalition", "opposition"}
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep",
                                      "oct", "nov", "dec"], 1)}


def clean(txt):
    txt = re.sub(r"\[[^\]]*\]", "", txt)          # footnotes [12], [a]
    return re.sub(r"\s+", " ", txt).strip()


def table_grid(table):
    """Expand an HTML table to a grid of (text, origin_id, tag)."""
    grid, pending = [], {}  # pending[(r, c)] = cell for rowspans
    origin = 0
    for r, tr in enumerate(table.find_all("tr")):
        row, c = [], 0
        cells = tr.find_all(["th", "td"], recursive=False)
        ci = 0
        while ci < len(cells) or (r, c) in pending:
            if (r, c) in pending:
                row.append(pending.pop((r, c)))
                c += 1
                continue
            cell = cells[ci]
            ci += 1
            origin += 1
            try:
                cs = int(re.match(r"\d+", cell.get("colspan", "1") or "1").group())
            except AttributeError:
                cs = 1
            try:
                rs = int(re.match(r"\d+", cell.get("rowspan", "1") or "1").group())
            except AttributeError:
                rs = 1
            item = (clean(cell.get_text(" ", strip=True)), origin, cell.name, cs)
            for k in range(cs):
                row.append(item)
                for rr in range(1, rs):
                    pending[(r + rr, c + k)] = item
            c += cs
        grid.append(row)
    return grid


def parse_value(v):
    v = v.replace(" ", " ").strip()
    if v in ("", "–", "—", "-", "— N/a", "N/a", "n/a", "—N/a"):
        return None, None
    m = re.match(r"^\(?\s*<?\s*(\d+(?:\.\d+)?)\s*%\s*\)?$", v)
    if m and "(" in v:
        return 0, float(m.group(1))
    m = re.match(r"^(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)", v)  # a seat range "4-5" -> midpoint
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2, None
    m = re.match(r"^(\d+(?:\.\d+)?)(?:\s*\D.*)?$", v)
    if m:
        x = float(m.group(1))
        return (int(x) if x.is_integer() else x), None
    return None, None


def parse_date(raw, election):
    """Return the END date of the fieldwork/publication range as a date (year inferred)."""
    ed = ELECTION_DATE[election]
    s = raw.replace("–", "-").replace("—", "-").replace(",", " ")
    toks = re.findall(r"(\d{1,2})\s*([A-Za-z]{3})[a-z]*\.?\s*(\d{2,4})?", s)
    us = re.findall(r"([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2})\s+(\d{4})", s)  # "Feb 6 2009"
    if us:
        mon, d, yr = us[-1]
        toks = [(d, mon, yr)]
    if not toks:
        return None
    d, mon, yr = toks[-1]
    mon = MONTHS.get(mon.lower()[:3])
    if not mon:
        return None
    if yr:
        y = int(yr) + (2000 if len(yr) == 2 else 0)
    else:
        y = ed.year
        if (mon, int(d)) > (ed.month, ed.day):
            y -= 1
    try:
        return date(y, mon, int(d))
    except ValueError:
        return None


def parse_page(election):
    html = open(os.path.join(WIKI, f"polls_{election}.html"), encoding="utf-8").read()
    soup = BeautifulSoup(html, "lxml")
    rows, events = [], []
    for ti, table in enumerate(soup.select("table.wikitable")):
        head = table.find_previous(["h2", "h3", "h4"])
        section = head.get_text(" ", strip=True).lower() if head else ""
        grid = table_grid(table)
        if not grid:
            continue
        hdr = grid[0]
        labels = [h[0].lower() for h in hdr]
        if not any(l in META_COLS for l in labels) or "polling firm" not in labels and "poll" not in labels \
                and "pollster" not in labels:
            continue
        if not any(s in section for s in SEAT_SECTIONS) and election != "2026":
            continue
        if election == "2026" and not section.startswith("2026") and "seat" not in section:
            continue
        # column roles
        meta_pos, party_groups = {}, []
        for c, (txt, oid, tag, cs) in enumerate(hdr):
            key = txt.lower()
            if key in META_COLS:
                meta_pos.setdefault(META_COLS[key], c)
            elif key.split(" [")[0] in NON_PARTY or key.startswith(("gov", "opp", "others", "lead")):
                continue
            elif txt:
                if party_groups and party_groups[-1][1] == oid:
                    party_groups[-1][2].append(c)
                else:
                    party_groups.append([txt, oid, [c]])
        if len(party_groups) < 4:
            continue
        ncol = len(hdr)
        n_head = 1
        while n_head < len(grid) and all(t == "th" for (_, _, t, _) in grid[n_head]) and grid[n_head]:
            n_head += 1
        for ri, row in enumerate(grid[n_head:], start=n_head):
            if not row:
                continue
            spans = [x[3] for x in row]
            if max(spans) >= ncol // 2:  # banner row (event)
                events.append({"election": election, "table_idx": ti, "row_idx": ri,
                               "date_raw": row[0][0], "text": row[-1][0]})
                continue
            get = lambda name: row[meta_pos[name]][0] if name in meta_pos and meta_pos[name] < len(row) else ""
            date_raw = get("date")
            pollster = get("pollster")
            publisher = get("publisher")
            is_result = "result" in pollster.lower() or "result" in publisher.lower()
            dend = parse_date(date_raw, election)
            ss = re.sub(r"[^\d]", "", get("sample_size")) if "sample_size" in meta_pos else ""
            used_origins = {}
            for name, _, cols in party_groups:
                c0 = cols[0]
                if c0 >= len(row):
                    continue
                txt, oid, tag, cs = row[c0]
                note = ""
                if oid in used_origins:  # same body cell already assigned to an earlier party
                    rows.append(dict(election=election, table_idx=ti, row_idx=ri, date_raw=date_raw,
                                     date_end=dend, pollster=pollster, publisher=publisher,
                                     sample_size=ss, party_wiki=name, seats=None, pct=None,
                                     is_result_row=is_result,
                                     note=f"joint with {used_origins[oid]}"))
                    continue
                if cs > len(cols):
                    note = "cell spans beyond party columns"
                used_origins[oid] = name
                seats, pct = parse_value(txt)
                rows.append(dict(election=election, table_idx=ti, row_idx=ri, date_raw=date_raw,
                                 date_end=dend, pollster=pollster, publisher=publisher, sample_size=ss,
                                 party_wiki=name, seats=seats, pct=pct, is_result_row=is_result,
                                 note=note, raw=txt))
    return pd.DataFrame(rows), pd.DataFrame(events)


def main():
    allr, alle = [], []
    for e in ELECTION_DATE:
        r, ev = parse_page(e)
        allr.append(r)
        alle.append(ev)
        if len(r):
            tot = r.groupby(["table_idx", "row_idx"])["seats"].sum()
            print(f"{e}: {r.groupby(['table_idx','row_idx']).ngroups:4d} poll rows in "
                  f"{r.table_idx.nunique():2d} tables; rows summing to 120: {(tot == 120).mean():.0%}")
    df = pd.concat(allr, ignore_index=True)
    df.to_csv(os.path.join(OUT, "wiki_polls_long.csv"), index=False, encoding="utf-8-sig")
    pd.concat(alle, ignore_index=True).to_csv(os.path.join(OUT, "wiki_poll_events.csv"), index=False,
                                              encoding="utf-8-sig")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
