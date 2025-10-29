"""
Utilities to compute post-tournament transfer windows from StatsBomb matches.csv
and return 90-day windows after each tournament final date.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, date
from typing import List, Tuple

import csv


def compute_windows_from_matches(
    matches_csv: Path, days_after: int = 90
) -> List[Tuple[date, date]]:
    """Compute [start,end] windows for each (competition_id, season_id) group.

    - start = (max(match_date) + 1 day)
    - end   = (max(match_date) + days_after days)
    """
    finals: dict[tuple[str, str], date] = {}
    with open(matches_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            key = (str(row.get("competition_id")), str(row.get("season_id")))
            dstr = row.get("match_date")
            if not dstr:
                continue
            try:
                d = datetime.strptime(dstr, "%Y-%m-%d").date()
            except ValueError:
                continue
            finals[key] = max(finals.get(key, d), d)

    windows: List[Tuple[date, date]] = []
    for _, fin in finals.items():
        st = fin + timedelta(days=1)
        en = fin + timedelta(days=days_after)
        windows.append((st, en))
    return sorted(windows, key=lambda w: w[0])


def save_windows_csv(
    windows: List[Tuple[date, date]],
    out_path: Path = Path("data/processed/transfermarkt/all/windows_all.csv"),
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start_date", "end_date"])
        for st, en in windows:
            w.writerow([st.isoformat(), en.isoformat()])
