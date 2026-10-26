"""
Step 02 - Parse the official CEC results (Knesset 16-25) into one uniform long table.

Outputs (data/processed/):
  ballots_long.parquet   one row per (knesset, ballot box, party letters) with votes
  ballots_meta.parquet   one row per (knesset, ballot box): settlement, eligible, voters, valid ...
  national_totals.csv    national valid votes per party (incl. double envelopes)
  national_summary.csv   eligible / voters / valid / turnout per knesset

Notes
-----
* "Double envelopes" (soldiers, diplomats, prisoners, hospitals, disabled voters...) are not
  geographic. In K16-K18 they appear as settlement code 0 rows inside the ballot file; in
  K19-K25 they appear only in the settlement file (code 0 / 9999 / 99999). They are kept for the
  national totals but flagged (is_envelope) and excluded from the geographic clustering.
* Ballot ids are not stable across elections, settlement codes are (CBS codes).
"""
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "results")
OUT = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)

META_ALIASES = {
    "settlement_name": ["שם ישוב"],
    "settlement_code": ["סמל ישוב"],
    "ballot": ["מספר קלפי", "קלפי", "סמל קלפי"],
    "eligible": ["בזב", "בז''ב", "בוחרים"],
    "voters": ["מצביעים"],
    "invalid": ["פסולים"],
    "valid": ["כשרים"],
}
DROP_COLS = {"סמל ועדה", "ברזל", "ריכוז", "שופט", "כתובת", "ת. עדכון", ""}
ELECTION_DATES = {16: "2003-01-28", 17: "2006-03-28", 18: "2009-02-10", 19: "2013-01-22",
                  20: "2015-03-17", 21: "2019-04-09", 22: "2019-09-17", 23: "2020-03-02",
                  24: "2021-03-23", 25: "2022-11-01"}


def read_csv_any(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "cp1255"):
        try:
            txt = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    from io import StringIO
    df = pd.read_csv(StringIO(txt), dtype=str)
    return df


def normalise(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c not in DROP_COLS and not c.startswith("Unnamed")]]
    ren = {}
    for std, al in META_ALIASES.items():
        for a in al:
            if a in df.columns:
                ren[a] = std
                break
    df = df.rename(columns=ren)
    return df


def load_ballots(k):
    d = os.path.join(RAW, f"k{k}")
    if os.path.exists(os.path.join(d, "expb.xls")):
        df = pd.read_excel(os.path.join(d, "expb.xls"), sheet_name=0, dtype=str)
    else:
        df = read_csv_any(os.path.join(d, "expb.csv"))
    df = normalise(df)
    if k == 17:  # K17 file has no settlement code / eligible columns
        df["settlement_code"] = np.nan
        df["eligible"] = np.nan
        df["invalid"] = np.nan
    meta = ["settlement_name", "settlement_code", "ballot", "eligible", "voters", "invalid", "valid"]
    parties = [c for c in df.columns if c not in meta]
    for c in parties + ["eligible", "voters", "invalid", "valid"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.strip(), errors="coerce")
    df["settlement_name"] = df["settlement_name"].astype(str).str.strip()
    df["settlement_code"] = pd.to_numeric(df["settlement_code"], errors="coerce")
    df = df[df["valid"].notna()]
    df[parties] = df[parties].fillna(0)
    # envelope rows inside the ballot file (K16-K18; K17 by name only)
    env = df["settlement_name"].str.contains("מעטפות") | (df["settlement_code"] == 0)
    df["is_envelope"] = env
    df["knesset"] = k
    df["ballot_uid"] = (f"K{k}_" + df["settlement_code"].fillna(-1).astype(int).astype(str)
                        + "_" + df["settlement_name"].str.replace(",", " ")
                        + "_" + df["ballot"].astype(str))
    # a few K20 rows repeat a (settlement, ballot) pair (split boxes): make every id unique
    dup = df.groupby("ballot_uid").cumcount()
    df.loc[dup > 0, "ballot_uid"] = df.loc[dup > 0, "ballot_uid"] + "_dup" + dup[dup > 0].astype(str)
    return df, parties


def load_city_envelopes(k):
    """Return the external/double-envelope totals that appear only in the settlement file."""
    p = os.path.join(RAW, f"k{k}", "expc.csv")
    if not os.path.exists(p):
        return None
    df = normalise(read_csv_any(p))
    df["settlement_code"] = pd.to_numeric(df["settlement_code"], errors="coerce")
    name = df["settlement_name"].fillna("").astype(str)
    env = df[(df["settlement_code"].isin([0, 9999, 99999])) | name.str.contains("מעטפות")]
    if env.empty:
        return None
    meta = ["settlement_name", "settlement_code", "eligible", "voters", "invalid", "valid"]
    parties = [c for c in env.columns if c not in meta]
    out = env[parties + ["voters", "invalid", "valid"]].apply(
        lambda s: pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")).fillna(0).sum()
    return out


def main():
    long_rows, meta_rows, nat_rows, summ = [], [], [], []
    for k in range(16, 26):
        df, parties = load_ballots(k)
        geo = df[~df["is_envelope"]]
        envb = df[df["is_envelope"]]
        city_env = None if len(envb) else load_city_envelopes(k)
        # national party totals
        tot = df[parties].sum()
        if city_env is not None:
            for p in parties:
                tot[p] += city_env.get(p, 0)
        voters = df["voters"].sum() + (city_env["voters"] if city_env is not None else 0)
        valid = df["valid"].sum() + (city_env["valid"] if city_env is not None else 0)
        eligible = df["eligible"].sum()
        for p in parties:
            nat_rows.append({"knesset": k, "letters": p, "votes": int(tot[p])})
        summ.append({"knesset": k, "date": ELECTION_DATES[k], "eligible_in_ballot_file": eligible,
                     "voters": voters, "valid": valid, "sum_party_votes": tot.sum(),
                     "n_ballots_geo": len(geo), "envelope_voters": voters - geo["voters"].sum()})
        m = geo[["ballot_uid", "knesset", "settlement_code", "settlement_name", "ballot",
                 "eligible", "voters", "invalid", "valid"]].copy()
        meta_rows.append(m)
        lg = geo.melt(id_vars=["ballot_uid", "knesset"], value_vars=parties,
                      var_name="letters", value_name="votes")
        lg = lg[lg["votes"] > 0]
        long_rows.append(lg)
        print(f"K{k}: {len(geo):>6} geo ballots, {len(envb)} envelope rows in ballot file, "
              f"city-file envelopes: {'yes' if city_env is not None else 'no'}; voters={voters:,.0f} "
              f"valid={valid:,.0f} sum_parties={tot.sum():,.0f}")
    pd.concat(long_rows).to_parquet(os.path.join(OUT, "ballots_long.parquet"), index=False)
    pd.concat(meta_rows).to_parquet(os.path.join(OUT, "ballots_meta.parquet"), index=False)
    pd.DataFrame(nat_rows).to_csv(os.path.join(OUT, "national_totals.csv"), index=False,
                                  encoding="utf-8-sig")
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "national_summary.csv"), index=False)
    print(pd.DataFrame(summ).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
