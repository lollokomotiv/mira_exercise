"""
Scarica i dati StatsBomb Open Data (GitHub).

- Funzione `run()` esistente: scarica solo EURO 2024 (retro‑compat).
- Nuove funzioni per scaricare competizioni delle nazionali (dal 2018):
  World Cup, UEFA Euro, Copa America.
"""

import os
import json
import requests
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
RAW_DIR = Path("data/raw/statsbomb")

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def download_json(url: str, out_path: Path):
    """Scarica un file JSON da GitHub solo se non esiste già localmente."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"ℹ️  Skipping {out_path} (already exists)")
        return
    print(f"⬇️  Downloading {url}")
    resp = requests.get(url)
    resp.raise_for_status()
    out_path.write_text(resp.text, encoding="utf-8")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

def download_competitions():
    """Scarica competitions.json"""
    url = f"{BASE_URL}/competitions.json"
    out = RAW_DIR / "competitions.json"
    download_json(url, out)
    return out


def get_euro2024_info(competitions_file: Path):
    """Trova competition_id e season_id per EURO 2024"""
    comps = load_json(competitions_file)
    for c in comps:
        if c["competition_name"] == "UEFA Euro" and c["season_name"] == "2024":
            return c["competition_id"], c["season_id"]
    raise ValueError("Euro 2024 not found in competitions.json!")


def _norm_competition_name(name: str | None) -> str:
    if not name:
        return ""
    name = name.lower()
    # uniform accents and common aliases
    name = name.replace("copa américa", "copa america")
    name = name.replace("uefa euro", "euro")
    name = name.replace("european championship", "euro")
    name = name.replace("fifa world cup", "world cup")
    return name


def list_national_competitions_since(competitions_file: Path, since_year: int = 2018):
    """Return list of (competition_id, season_id, competition_name, season_name)
    for national team tournaments since a given year.

    Matches competitions whose normalized name is one of:
      - world cup
      - euro
      - copa america
    """
    comps = load_json(competitions_file)
    targets = {"world cup", "euro", "copa america"}
    out = []
    for c in comps:
        cname = _norm_competition_name(c.get("competition_name"))
        sname = str(c.get("season_name") or "")
        # season_name can be "2021" or "2019/2020"; take first 4 digits
        year = None
        for tok in sname.replace("\\", "/").split("/"):
            if tok.isdigit() and len(tok) == 4:
                year = int(tok)
                break
        if cname in targets and (year is not None and year >= since_year):
            out.append((c["competition_id"], c["season_id"], c.get("competition_name"), c.get("season_name")))
    return out


def download_matches(competition_id: int, season_id: int):
    """Scarica tutti i match JSON per una competition/season"""
    url = f"{BASE_URL}/matches/{competition_id}/{season_id}.json"
    out = RAW_DIR / "matches" / f"{competition_id}_{season_id}.json"
    download_json(url, out)
    return out


def download_match_details(match_file: Path):
    """Scarica events e lineups per ogni match della lista"""
    matches = load_json(match_file)
    for m in matches:
        match_id = m["match_id"]

        # events
        url_events = f"{BASE_URL}/events/{match_id}.json"
        out_events = RAW_DIR / "events" / f"{match_id}.json"
        download_json(url_events, out_events)

        # lineups
        url_lineups = f"{BASE_URL}/lineups/{match_id}.json"
        out_lineups = RAW_DIR / "lineups" / f"{match_id}.json"
        download_json(url_lineups, out_lineups)


def run():
    comps_file = download_competitions()
    comp_id, season_id = get_euro2024_info(comps_file)
    matches_file = download_matches(comp_id, season_id)
    download_match_details(matches_file)
    print("✅ StatsBomb Euro 2024 data downloaded into data/raw/statsbomb/")


def run_nationals(since_year: int = 2018) -> None:
    """Scarica tutte le competizioni nazionali (WC, Euro, Copa America) dal `since_year` in poi."""
    comps_file = download_competitions()
    items = list_national_competitions_since(comps_file, since_year=since_year)
    if not items:
        print(f"⚠️  Nessuna competizione nazionale trovata >= {since_year}")
        return
    for comp_id, season_id, cname, sname in items:
        print(f"⬇️  Download matches {cname} {sname} ({comp_id}/{season_id})")
        matches_file = download_matches(int(comp_id), int(season_id))
        download_match_details(matches_file)
    print("✅ StatsBomb national tournaments downloaded into data/raw/statsbomb/")


if __name__ == "__main__":
    run()
