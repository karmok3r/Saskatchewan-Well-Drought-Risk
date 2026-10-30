"""Download the Water Security Agency public water-well records (driller reports).

Source: https://gis.wsask.ca/arcgiswa/rest/services/Public/Water_Wells/MapServer/4
The layer allows 10,000 records per request, so records are paged by OBJECTID.
Output: data/raw/wsa_water_wells.csv (one row per driller report, with UTM and lon/lat).
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://gis.wsask.ca/arcgiswa/rest/services/Public/Water_Wells/MapServer/4/query"
OUT = Path(__file__).parent / "data" / "raw" / "wsa_water_wells.csv"
PAGE = 10000


def query(params):
    data = urllib.parse.urlencode(params).encode()
    for attempt in range(5):
        try:
            with urllib.request.urlopen(URL, data=data, timeout=120) as r:
                return json.load(r)
        except Exception as e:  # network hiccups: retry with backoff
            print(f"  retry {attempt + 1}: {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("query failed")


def main():
    rows, last = [], 0
    while True:
        d = query({"where": f"OBJECTID > {last}", "outFields": "*", "orderByFields": "OBJECTID",
                   "returnGeometry": "true", "outSR": 4326, "resultRecordCount": PAGE, "f": "json"})
        feats = d.get("features", [])
        if not feats:
            break
        for f in feats:
            a = f["attributes"]
            g = f.get("geometry") or {}
            a["lon"], a["lat"] = g.get("x"), g.get("y")
            rows.append(a)
        last = feats[-1]["attributes"]["OBJECTID"]
        print(f"{len(rows):>7} records (last OBJECTID {last})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"saved {len(rows)} records to {OUT}")


if __name__ == "__main__":
    main()
