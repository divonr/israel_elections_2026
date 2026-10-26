"""
Step 01 - Download all raw data used by the forecast.

Sources
-------
1. Official ballot-box ("expb") and settlement ("expc") level results published by the
   Israeli Central Elections Committee (CEC):
     * Knesset 21-25 (2019a-2022): https://mediaXX.bechirot.gov.il/files/expb.csv / expc.csv
       (linked from https://votesXX.bechirot.gov.il/ballotresults)
     * Knesset 16-20 (2003-2015): the CEC files republished on data.gov.il
       (dataset "votes-knesset", id 26f9fa06-fcd7-4173-8df5-65797b63e857)
2. Wikipedia "Opinion polling for the XXXX Israeli legislative election" pages
   (the aggregator recommended in the competition rules), raw HTML.

All files are written to ../data/raw/ . Nothing downloaded is executed.
"""
import os
import subprocess
import sys
import time
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

DATAGOV = "https://data.gov.il/dataset/26f9fa06-fcd7-4173-8df5-65797b63e857/resource"
RESULT_FILES = {
    # knesset: {local name: url}
    16: {"expb.xls": f"{DATAGOV}/498b48e9-5af6-474d-b7a4-5ac1e21d3a08/download/-16-2003.xls"},
    17: {"expb.xls": f"{DATAGOV}/70f8bc93-8d98-4c20-ad7c-768af713f1c5/download/-17-.xls"},
    18: {"expb.csv": f"{DATAGOV}/840edb33-90ac-4176-8ad9-4cdcb8e5caa5/download/-18-.csv"},
    19: {"expb.csv": f"{DATAGOV}/432d3185-545a-41d9-8c72-d10ee515919c/download/expbkalpi.csv",
         "expc.csv": f"{DATAGOV}/c20cdcef-4d42-4241-a41b-6ca7ae51002d/download/expcity.csv"},
    20: {"expb.csv": f"{DATAGOV}/c3db5581-f48d-45fc-b221-e7635e940c41/download/expbkal20.csv",
         "expc.csv": f"{DATAGOV}/929b50c6-f455-4be2-b438-ec6af01421f2/download/expc-5.csv"},
}
for k in (21, 22, 23, 24, 25):
    RESULT_FILES[k] = {"expb.csv": f"https://media{k}.bechirot.gov.il/files/expb.csv",
                       "expc.csv": f"https://media{k}.bechirot.gov.il/files/expc.csv"}

WIKI_PAGES = {
    "2006": "Opinion_polling_for_the_2006_Israeli_legislative_election",
    "2009": "Opinion_polling_for_the_2009_Israeli_legislative_election",
    "2013": "Opinion_polling_for_the_2013_Israeli_legislative_election",
    "2015": "Opinion_polling_for_the_2015_Israeli_legislative_election",
    "2019a": "Opinion_polling_for_the_April_2019_Israeli_legislative_election",
    "2019b": "Opinion_polling_for_the_September_2019_Israeli_legislative_election",
    "2020": "Opinion_polling_for_the_2020_Israeli_legislative_election",
    "2021": "Opinion_polling_for_the_2021_Israeli_legislative_election",
    "2022": "Opinion_polling_for_the_2022_Israeli_legislative_election",
    "2026": "Opinion_polling_for_the_2026_Israeli_legislative_election",
}


def fetch(url, path, retries=3):
    for i in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=180)
            r.raise_for_status()
            if r.content[:15].lower().startswith(b"<!doctype html") and path.endswith((".csv", ".xls")):
                raise RuntimeError("got HTML instead of data")
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"  ok  {len(r.content):>9,d} B  {path}")
            return True
        except requests.exceptions.SSLError:
            # media21.bechirot.gov.il only negotiates legacy TLS ciphers -> fall back to curl
            rc = subprocess.run(["curl", "-s", "-A", UA["User-Agent"], "--max-time", "180",
                                 "-o", path, url]).returncode
            if rc == 0 and os.path.getsize(path) > 1000:
                print(f"  ok (curl) {os.path.getsize(path):>9,d} B  {path}")
                return True
        except Exception as e:  # noqa: BLE001
            print(f"  retry {i + 1}: {url}: {e}")
            time.sleep(3)
    return False


def main():
    only = set(sys.argv[1:])
    if not only or "results" in only:
        for k, files in RESULT_FILES.items():
            d = os.path.join(RAW, "results", f"k{k}")
            os.makedirs(d, exist_ok=True)
            for name, url in files.items():
                fetch(url, os.path.join(d, name))
    if not only or "wiki" in only:
        d = os.path.join(RAW, "wiki")
        os.makedirs(d, exist_ok=True)
        for key, page in WIKI_PAGES.items():
            fetch(f"https://en.wikipedia.org/wiki/{page}", os.path.join(d, f"polls_{key}.html"))


if __name__ == "__main__":
    main()
