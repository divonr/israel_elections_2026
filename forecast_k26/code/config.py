"""
Shared configuration: party mappings, party families, official seat results, pollster lineages,
and the Israeli seat-allocation rule (threshold + Bader-Ofer/D'Hondt + surplus agreements).
"""
import os
import re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
OUTT = os.path.join(ROOT, "outputs", "tables")

ELECTIONS = {  # wiki key -> knesset
    "2009": 18, "2013": 19, "2015": 20, "2019a": 21, "2019b": 22, "2020": 23, "2021": 24, "2022": 25}
ELECTION_DATE = {16: "2003-01-28", 17: "2006-03-28", 18: "2009-02-10", 19: "2013-01-22",
                 20: "2015-03-17", 21: "2019-04-09", 22: "2019-09-17", 23: "2020-03-02",
                 24: "2021-03-23", 25: "2022-11-01", 26: "2026-10-27"}
THRESHOLD = {16: 0.015, 17: 0.02, 18: 0.02, 19: 0.02, 20: 0.0325, 21: 0.0325, 22: 0.0325,
             23: 0.0325, 24: 0.0325, 25: 0.0325, 26: 0.0325}
OFFICIAL_TURNOUT = {16: 67.8, 17: 63.5, 18: 64.7, 19: 67.8, 20: 72.3, 21: 68.5, 22: 69.8,
                    23: 71.5, 24: 67.4, 25: 70.6}

FAMILIES = ["LIKUD", "SHAS", "UTJ", "NATREL", "YB", "CENTER", "LEFT", "ARAB", "OTHER"]

# ballot letters -> (english name, family) for every list above ~0.4% (rest -> OTHER)
PARTIES = {
    16: {"מחל": ("Likud", "LIKUD"), "אמת": ("Labor-Meimad", "LEFT"), "יש": ("Shinui", "CENTER"),
         "שס": ("Shas", "SHAS"), "ל": ("National Union", "NATREL"), "מרצ": ("Meretz", "LEFT"),
         "ג": ("UTJ", "UTJ"), "ב": ("NRP", "NATREL"), "ו": ("Hadash-Ta'al", "ARAB"),
         "ם": ("Am Echad", "LEFT"), "ד": ("Balad", "ARAB"), "כן": ("Yisrael BaAliyah", "YB"),
         "עם": ("UAL", "ARAB")},
    17: {"כן": ("Kadima", "CENTER"), "אמת": ("Labor-Meimad", "LEFT"), "שס": ("Shas", "SHAS"),
         "מחל": ("Likud", "LIKUD"), "ל": ("Yisrael Beiteinu", "YB"), "טב": ("NU-NRP", "NATREL"),
         "זך": ("Gil", "CENTER"), "ג": ("UTJ", "UTJ"), "מרצ": ("Meretz-Yachad", "LEFT"),
         "עם": ("UAL-Ta'al", "ARAB"), "ו": ("Hadash", "ARAB"), "ד": ("Balad", "ARAB")},
    18: {"כן": ("Kadima", "CENTER"), "מחל": ("Likud", "LIKUD"), "ל": ("Yisrael Beiteinu", "YB"),
         "אמת": ("Labor", "LEFT"), "שס": ("Shas", "SHAS"), "ג": ("UTJ", "UTJ"),
         "עם": ("UAL-Ta'al", "ARAB"), "ט": ("National Union", "NATREL"), "ו": ("Hadash", "ARAB"),
         "מרצ": ("Meretz", "LEFT"), "ב": ("Jewish Home", "NATREL"), "ד": ("Balad", "ARAB"),
         "זך": ("Gil", "CENTER")},
    19: {"מחל": ("Likud Beiteinu", "LIKUD"), "פה": ("Yesh Atid", "CENTER"), "אמת": ("Labor", "LEFT"),
         "טב": ("Jewish Home", "NATREL"), "שס": ("Shas", "SHAS"), "ג": ("UTJ", "UTJ"),
         "צפ": ("Hatnuah", "CENTER"), "מרץ": ("Meretz", "LEFT"), "עם": ("UAL-Ta'al", "ARAB"),
         "ו": ("Hadash", "ARAB"), "ד": ("Balad", "ARAB"), "כן": ("Kadima", "CENTER"),
         "נץ": ("Otzma LeYisrael", "NATREL"), "ץ": ("Am Shalem", "SHAS")},
    20: {"מחל": ("Likud", "LIKUD"), "אמת": ("Zionist Union", "LEFT"), "ודעם": ("Joint List", "ARAB"),
         "פה": ("Yesh Atid", "CENTER"), "כ": ("Kulanu", "CENTER"), "טב": ("Jewish Home", "NATREL"),
         "שס": ("Shas", "SHAS"), "ל": ("Yisrael Beiteinu", "YB"), "ג": ("UTJ", "UTJ"),
         "מרצ": ("Meretz", "LEFT"), "קץ": ("Yachad", "SHAS")},
    21: {"מחל": ("Likud", "LIKUD"), "פה": ("Blue & White", "CENTER"), "שס": ("Shas", "SHAS"),
         "ג": ("UTJ", "UTJ"), "ום": ("Hadash-Ta'al", "ARAB"), "אמת": ("Labor", "LEFT"),
         "ל": ("Yisrael Beiteinu", "YB"), "טב": ("URWP", "NATREL"), "מרצ": ("Meretz", "LEFT"),
         "כ": ("Kulanu", "CENTER"), "דעם": ("Ra'am-Balad", "ARAB"), "נ": ("New Right", "NATREL"),
         "ז": ("Zehut", "NATREL"), "נר": ("Gesher", "CENTER")},
    22: {"פה": ("Blue & White", "CENTER"), "מחל": ("Likud", "LIKUD"), "ודעם": ("Joint List", "ARAB"),
         "שס": ("Shas", "SHAS"), "ל": ("Yisrael Beiteinu", "YB"), "ג": ("UTJ", "UTJ"),
         "טב": ("Yamina", "NATREL"), "אמת": ("Labor-Gesher", "LEFT"),
         "מרצ": ("Democratic Union", "LEFT"), "כף": ("Otzma Yehudit", "NATREL")},
    23: {"מחל": ("Likud", "LIKUD"), "פה": ("Blue & White", "CENTER"), "ודעם": ("Joint List", "ARAB"),
         "שס": ("Shas", "SHAS"), "ג": ("UTJ", "UTJ"), "אמת": ("Labor-Gesher-Meretz", "LEFT"),
         "ל": ("Yisrael Beiteinu", "YB"), "טב": ("Yamina", "NATREL"), "נץ": ("Otzma Yehudit", "NATREL")},
    24: {"מחל": ("Likud", "LIKUD"), "פה": ("Yesh Atid", "CENTER"), "שס": ("Shas", "SHAS"),
         "כן": ("Blue & White", "CENTER"), "ב": ("Yamina", "NATREL"), "אמת": ("Labor", "LEFT"),
         "ג": ("UTJ", "UTJ"), "ל": ("Yisrael Beiteinu", "YB"), "ט": ("Religious Zionist", "NATREL"),
         "ודעם": ("Joint List", "ARAB"), "ת": ("New Hope", "CENTER"), "מרצ": ("Meretz", "LEFT"),
         "עם": ("Ra'am", "ARAB"), "יז": ("New Economic", "OTHER")},
    25: {"מחל": ("Likud", "LIKUD"), "פה": ("Yesh Atid", "CENTER"), "ט": ("RZP-Otzma", "NATREL"),
         "כן": ("National Unity", "CENTER"), "שס": ("Shas", "SHAS"), "ג": ("UTJ", "UTJ"),
         "ל": ("Yisrael Beiteinu", "YB"), "עם": ("Ra'am", "ARAB"), "ום": ("Hadash-Ta'al", "ARAB"),
         "אמת": ("Labor", "LEFT"), "מרצ": ("Meretz", "LEFT"), "ד": ("Balad", "ARAB"),
         "ב": ("Jewish Home", "NATREL")},
}

# official seat results (CEC final results)
ACTUAL_SEATS = {
    18: {"כן": 28, "מחל": 27, "ל": 15, "אמת": 13, "שס": 11, "ג": 5, "עם": 4, "ט": 4, "ו": 4, "מרצ": 3,
         "ב": 3, "ד": 3},
    19: {"מחל": 31, "פה": 19, "אמת": 15, "טב": 12, "שס": 11, "ג": 7, "צפ": 6, "מרץ": 6, "עם": 4, "ו": 4,
         "ד": 3, "כן": 2},
    20: {"מחל": 30, "אמת": 24, "ודעם": 13, "פה": 11, "כ": 10, "טב": 8, "שס": 7, "ל": 6, "ג": 6, "מרצ": 5},
    21: {"מחל": 35, "פה": 35, "שס": 8, "ג": 8, "ום": 6, "אמת": 6, "ל": 5, "טב": 5, "מרצ": 4, "כ": 4,
         "דעם": 4},
    22: {"פה": 33, "מחל": 32, "ודעם": 13, "שס": 9, "ל": 8, "ג": 7, "טב": 7, "אמת": 6, "מרצ": 5},
    23: {"מחל": 36, "פה": 33, "ודעם": 15, "שס": 9, "ג": 7, "אמת": 7, "ל": 7, "טב": 6},
    24: {"מחל": 30, "פה": 17, "שס": 9, "כן": 8, "ב": 7, "אמת": 7, "ג": 7, "ל": 7, "ט": 6, "ודעם": 6,
         "ת": 6, "מרצ": 6, "עם": 4},
    25: {"מחל": 32, "פה": 24, "ט": 14, "כן": 12, "שס": 11, "ג": 7, "ל": 6, "עם": 5, "ום": 5, "אמת": 4},
}

# Wikipedia column name (final seat-projection table) -> ballot letters
WIKI_TO_LETTERS = {
    18: {"Kadima": "כן", "Labor": "אמת", "Shas": "שס", "Likud": "מחל", "Yisrael Beiteinu": "ל",
         "UAL - Ta'al": "עם", "Hadash": "ו", "Balad": "ד", "Jewish Home": "ב", "National Union": "ט",
         "Gil": "זך", "UTJ": "ג", "Meretz": "מרצ"},
    19: {"Kadima": "כן", "Likud": "מחל", "Labor": "אמת", "Shas": "שס", "UTJ": "ג", "Jewish Home": "טב", "National Union": "טב",
         "UAL - Ta'al": "עם", "Hadash": "ו", "Balad": "ד", "Meretz": "מרץ", "Yesh Atid": "פה",
         "Otzma": "נץ", "Am Shalem": "ץ", "Hatnuah": "צפ"},
    20: {"Likud": "מחל", "Yisrael Beiteinu": "ל", "Yesh Atid": "פה", "Labor": "אמת", "Jewish Home": "טב",
         "Shas": "שס", "UTJ": "ג", "Meretz": "מרצ", "Hadash": "ודעם", "Otzma": "קץ", "Yachad": "קץ",
         "Kulanu": "כ"},
    21: {"Likud": "מחל", "Labor": "אמת", "Blue & White": "פה", "Kulanu": "כ", "Ra'am – Balad": "דעם",
         "Shas": "שס", "UTJ": "ג", "URWP": "טב", "Yisrael Beiteinu": "ל", "Meretz": "מרצ",
         "Hadash – Ta'al": "ום", "New Right": "נ", "Gesher": "נר", "Zehut": "ז"},
    22: {"Likud": "מחל", "Blue & White": "פה", "Joint List": "ודעם", "Shas": "שס", "UTJ": "ג",
         "Yamina": "טב", "Labor- Gesher": "אמת", "Yisrael Beiteinu": "ל", "Dem. Union": "מרצ",
         "Otzma Yehudit": "כף"},
    23: {"Blue & White": "פה", "Likud": "מחל", "Joint List": "ודעם", "Emet": "אמת", "Shas": "שס",
         "Yisrael Beiteinu": "ל", "UTJ": "ג", "Yamina": "טב", "Otzma": "נץ"},
    24: {"Likud": "מחל", "Yesh Atid": "פה", "Blue & White": "כן", "Joint List": "ודעם", "Shas": "שס",
         "UTJ": "ג", "Yisrael Beiteinu": "ל", "Meretz": "מרצ", "Ra'am": "עם", "Yamina": "ב",
         "New Hope": "ת", "Labor": "אמת", "Religious Zionist": "ט", "New Economic": "יז"},
    25: {"Likud": "מחל", "Yesh Atid": "פה", "National Unity": "כן", "Shas": "שס", "Jewish Home": "ב",
         "Labor": "אמת", "UTJ": "ג", "Yisrael Beiteinu": "ל", "RZP - OY": "ט", "Hadash –Ta'al": "ום",
         "Meretz": "מרצ", "Ra'am": "עם", "Balad": "ד"},
}

# 2026 lists (Wikipedia column names) -> canonical id, family
PARTIES_2026 = {
    "Likud": ("LIKUD", "LIKUD"), "Together": ("TOGETHER", "CENTER"), "RZP - Zehut": ("RZP_ZEHUT", "NATREL"),
    "Otzma": ("OTZMA", "NATREL"), "Blue & White": ("BLUE_WHITE", "CENTER"), "Shas": ("SHAS", "SHAS"),
    "UTJ": ("UTJ", "UTJ"), "Yisrael Beiteinu": ("YB", "YB"), "Ra'am": ("RAAM", "ARAB"),
    "Joint List": ("JOINT", "ARAB"), "Dems": ("DEMOCRATS", "LEFT"), "Yashar": ("YASHAR", "CENTER"),
    "Reserv. - NEP": ("HENDEL_ZELICHA", "CENTER"), "Amcha Yisrael": ("AMCHA", "NATREL"),
    "Haredi Public": ("HAREDI_PUBLIC", "OTHER"),
}
NAMES_2026 = {"LIKUD": "Likud", "YASHAR": "Yashar! (Eisenkot)", "TOGETHER": "Together (Bennett-Lapid)",
              "DEMOCRATS": "The Democrats", "YB": "Yisrael Beiteinu", "OTZMA": "Otzma Yehudit",
              "SHAS": "Shas", "UTJ": "United Torah Judaism", "RZP_ZEHUT": "Religious Zionism & Zehut",
              "JOINT": "Joint List (Hadash-Ta'al-Balad)", "RAAM": "Ra'am",
              "HENDEL_ZELICHA": "Hendel & Zelicha (Reservists)", "AMCHA": "Amcha Israel (Winter)",
              "BLUE_WHITE": "Blue & White (Gantz) - withdrew 19.10", "HAREDI_PUBLIC": "HaTzibur HaHaredi"}
FAMILY_2026 = {v[0]: v[1] for v in PARTIES_2026.values()}


def pollster_lineage(pollster, publisher="", election=None):
    """Map a polling-firm string to a stable firm lineage used for house effects."""
    s = f"{pollster}".lower()
    p = f"{publisher}".lower()
    if election == "2026":
        if "sf+nd" in s or "next data" in s or "filber" in s:
            return "NextData_Filber"
        if "mp+tm" in s or "madad" in s:
            return "Ch13_Madad2026"
        if "dri" in s:
            return "DRI"
    if election in ("2013", "2015"):  # these tables name the outlet, not the firm
        for key, firm in (("channel 2", "Midgam"), ("channel 10", "Dialog_Fuchs"), ("maariv", "MaagarMochot"),
                          ("walla", "Lazar_Panels"), ("knesset channel", "Lazar_Panels"),
                          ("haaretz", "Dialog_Fuchs"), ("yisrael hayom", "NewWave"),
                          ("israel hayom", "NewWave"), ("channel 1", "Other:channel_1")):
            if key in s and not any(f in s for f in ("dahaf", "smith", "geocart", "teleseker", "panels",
                                                     "midgam", "maagar", "sarid", "dialog", "fuchs")):
                return firm
    if "midgam project" in s or "midgam proj" in s or "project hamidgam" in s or "project ha-midgam" in s:
        return "MidgamProject"
    if "dahaf" in s:  # before Midgam: the Hebrew source writes "Dahaf (Mina Tzemach)"
        return "Dahaf"
    if "midgam" in s or "mina tzemach" in s or "geva" in s:  # Mano Geva (often "Meno Geva") runs Midgam
        return "Midgam"
    if "kantar" in s or "teleseker" in s or "tns" in s:
        return "Kantar"
    if "panel" in s or "lazar" in s or "lri" in s or "p4a" in s:
        return "Lazar_Panels"
    if "direct" in s:
        return "DirectPolls"
    if "maagar" in s or "ma'agar" in s:
        return "MaagarMochot"
    if "smith" in s:
        return "Smith"
    if "fuchs" in s or "dialog" in s:
        return "Dialog_Fuchs"
    if "geocart" in s:
        return "Geocartography"
    if "dahaf" in s:
        return "Dahaf"
    if "sarid" in s:
        return "Sarid"
    if "tatika" in s:
        return "Tatika"
    if "statnet" in s or "stat-net" in s:
        return "Statnet"
    if "miskar" in s:
        return "Miskar"
    # smaller firms that the two Wikipedia editions spell differently
    if "new wave" in s or "gal hadash" in s or "hagal hahadash" in s:
        return "NewWave"
    if "shvakim" in s or "panorama" in s:
        return "ShvakimPanorama"
    if "shiluv" in s or "millward" in s:
        return "Shiluv"
    if re.search(r"\btri\b", s):
        return "TRI"
    return "Other:" + re.sub(r"[^a-z0-9]+", "_", s)[:30]


def dhondt(votes, seats=120):
    """Plain D'Hondt (Bader-Ofer) over an array of votes."""
    votes = np.asarray(votes, float)
    alloc = np.zeros(len(votes), int)
    for _ in range(seats):
        alloc[np.argmax(votes / (alloc + 1))] += 1
    return alloc


def allocate(shares, threshold=0.0325, seats=120, surplus_pairs=(), total=None):
    """
    Israeli allocation (Knesset Elections Law ss. 81-83):
      1. lists with less than `threshold` of ALL valid votes are removed (`total` = all valid votes;
         if None, the sum of the given shares is used, i.e. the input must then be complete);
      2. quota Q = votes of passing lists / 120; each list gets floor(votes / Q);
      3. remaining seats by Bader-Ofer (largest votes/(seats+1)), where each surplus-agreement pair
         is treated as ONE list (combined votes and combined seats);
      4. a pair's seats are split between its two lists by Bader-Ofer.
    shares: dict name -> votes or vote share. Returns dict name -> seats.
    """
    names = list(shares)
    v = {n: max(float(shares[n]), 0.0) for n in names}
    tot = float(total) if total is not None else sum(v.values())
    passed = [n for n in names if v[n] / tot >= threshold]
    out = {n: 0 for n in names}
    if not passed:
        return out
    q = sum(v[n] for n in passed) / seats
    init = {n: int(np.floor(v[n] / q)) for n in passed}
    units, used = [], set()
    for a, b in surplus_pairs:
        if a in passed and b in passed and a not in used and b not in used:
            units.append([(a, b), v[a] + v[b], init[a] + init[b]])
            used |= {a, b}
    for n in passed:
        if n not in used:
            units.append([(n,), v[n], init[n]])
    remaining = seats - sum(u[2] for u in units)
    for _ in range(remaining):
        best = max(units, key=lambda u: u[1] / (u[2] + 1))
        best[2] += 1
    for members, _, s in units:
        if len(members) == 1:
            out[members[0]] = int(s)
        else:
            sub = dhondt([v[members[0]], v[members[1]]], int(s))
            out[members[0]], out[members[1]] = int(sub[0]), int(sub[1])
    return out


def seats_to_shares(seats, pct, others=0.015, default_below=1.8):
    """
    Convert one poll's published seats to vote shares (fractions of all valid votes), used identically
    for the 2009-2022 polls (calibration) and the 2026 polls (forecast).
    Lists at 0 seats get their reported % (default_below if none); B = their total.
    Passing lists share T = 1 - B - others. Bader-Ofer/D'Hondt gives a list with share v about
    floor(v/d) seats with divisor d ~ T/(S + J/2) (S seats, J passing lists), so a list with s seats holds
    about (s + 1/2) d: the midpoint of its feasible interval. (Plain s/120*T understates small lists:
    a 4-seat list would sit below the 3.25% threshold it passed.)
    seats, pct: pandas Series indexed by list. Returns a Series of shares.
    """
    import pandas as pd
    seats = pd.to_numeric(seats, errors="coerce")
    pct = pd.to_numeric(pct, errors="coerce").reindex(seats.index)
    zero = seats == 0
    B = (pct[zero].fillna(default_below) / 100).sum()
    T = 1 - B - others
    pas = seats > 0
    S, J = float(seats[pas].sum()), int(pas.sum())
    if S < 116:  # some lists were dropped from this poll (e.g. an unsplittable combined cell): keep their room
        T = T * S / 120.0
    d = T / (S + J / 2.0) if J else 0.0
    v = pd.Series(np.nan, index=seats.index)
    v[pas] = (seats[pas] + 0.5) * d
    v[zero] = pct[zero].fillna(default_below) / 100
    return v

