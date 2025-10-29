"""
Filtra e struttura i dati StatsBomb (EURO 2024) scaricati in data/raw/statsbomb.

Output (CSV) in data/processed/statsbomb:
- teams.csv: elenco squadre EURO 2024
- players.csv: rosa per squadra (da lineups)
- matches.csv: metadati partite (programma, risultato, arbitro, stadio)
- player_match_stats.csv: statistiche per giocatore e partita (pass, shot, goal, ecc.)
- referees.csv: anagrafica arbitri
- venues.csv: anagrafica stadi
- events_meta.csv: metadati essenziali degli eventi per tutte le partite EURO 2024

Prerequisiti: eseguire prima fetch_statsbomb.run() per scaricare i JSON.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


RAW_DIR = Path("data/raw/statsbomb")
OUT_DIR = Path("data/processed/statsbomb")


# ----------------------------------------------------------------------
# Helpers I/O
# ----------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------

def _get_euro2024_comp_season() -> tuple[int, int]:
    comps_file = RAW_DIR / "competitions.json"
    comps = load_json(comps_file)
    # StatsBomb negli anni ha usato nomi diversi per l'Europeo
    euro_names = {"UEFA Euro", "European Championship"}
    for c in comps:
        if c.get("competition_name") in euro_names and c.get("season_name") == "2024":
            return int(c["competition_id"]), int(c["season_id"])
    raise RuntimeError("EURO 2024 non trovato in competitions.json")


def _get_matches_file() -> Path:
    comp_id, season_id = _get_euro2024_comp_season()
    path = RAW_DIR / "matches" / f"{comp_id}_{season_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"File matches non trovato: {path}")
    return path


# ----------------------------------------------------------------------
# Builders: matches, teams, players
# ----------------------------------------------------------------------

def build_matches(matches_file: Path) -> list[dict[str, Any]]:
    matches = load_json(matches_file)
    rows: list[dict[str, Any]] = []
    for m in matches:
        row = {
            "match_id": m.get("match_id"),
            "match_date": m.get("match_date"),
            "kick_off": m.get("kick_off"),
            "competition_id": (m.get("competition") or {}).get("competition_id"),
            "competition_name": (m.get("competition") or {}).get("competition_name"),
            "season_id": (m.get("season") or {}).get("season_id"),
            "season_name": (m.get("season") or {}).get("season_name"),
            "stage_id": (m.get("competition_stage") or {}).get("id"),
            "stage_name": (m.get("competition_stage") or {}).get("name"),

            "home_team_id": (m.get("home_team") or {}).get("home_team_id") or (m.get("home_team") or {}).get("id"),
            "home_team_name": (m.get("home_team") or {}).get("home_team_name") or (m.get("home_team") or {}).get("name"),
            "away_team_id": (m.get("away_team") or {}).get("away_team_id") or (m.get("away_team") or {}).get("id"),
            "away_team_name": (m.get("away_team") or {}).get("away_team_name") or (m.get("away_team") or {}).get("name"),
            "home_score": m.get("home_score"),
            "away_score": m.get("away_score"),

            "referee_id": (m.get("referee") or {}).get("id"),
            "referee_name": (m.get("referee") or {}).get("name"),
            "stadium_id": (m.get("stadium") or {}).get("id"),
            "stadium_name": (m.get("stadium") or {}).get("name"),
            "country_id": ((m.get("stadium") or {}).get("country") or {}).get("id"),
            "country_name": ((m.get("stadium") or {}).get("country") or {}).get("name"),
        }
        rows.append(row)
    return rows


def build_teams(matches_rows: list[dict[str, Any]], match_ids: list[int]) -> list[dict[str, Any]]:
    # Deriva i team da lineups e events per coprire eventuali mismatch
    teams: dict[int, dict[str, Any]] = {}

    # dai matches
    for m in matches_rows:
        for side in ("home", "away"):
            tid = m.get(f"{side}_team_id")
            tname = m.get(f"{side}_team_name")
            if tid is not None:
                teams.setdefault(int(tid), {"team_id": int(tid), "team_name": tname})

    # dai lineups
    for mid in match_ids:
        path = RAW_DIR / "lineups" / f"{mid}.json"
        if not path.exists():
            continue
        lineup = load_json(path)
        for side in lineup:
            tid = side.get("team_id")
            tname = side.get("team_name")
            if tid is not None:
                teams.setdefault(int(tid), {"team_id": int(tid), "team_name": tname})

    # dagli events
    for mid in match_ids:
        path = RAW_DIR / "events" / f"{mid}.json"
        if not path.exists():
            continue
        evts = load_json(path)
        for e in evts:
            team = e.get("team") or {}
            tid = team.get("id")
            tname = team.get("name")
            if tid is not None:
                teams.setdefault(int(tid), {"team_id": int(tid), "team_name": tname})

    # ritorna ordinato per id
    return sorted(teams.values(), key=lambda x: x["team_id"])


def build_players(match_ids: list[int]) -> list[dict[str, Any]]:
    # Aggrega le rose dai lineups
    players: dict[tuple[int, int], dict[str, Any]] = {}
    for mid in match_ids:
        path = RAW_DIR / "lineups" / f"{mid}.json"
        if not path.exists():
            continue
        lineup = load_json(path)
        for side in lineup:
            tid = side.get("team_id")
            tname = side.get("team_name")
            for p in side.get("lineup", []):
                pid = p.get("player_id") or (p.get("player") or {}).get("id")
                pname = p.get("player_name") or (p.get("player") or {}).get("name")
                jersey = p.get("jersey_number")
                # Prima posizione dichiarata
                pos_name = None
                positions = p.get("positions") or []
                if positions:
                    pos_name = (positions[0].get("position") or {}).get("name") if isinstance(positions[0].get("position"), dict) else positions[0].get("position")
                    pos_name = pos_name or positions[0].get("position_name") or positions[0].get("position_id")

                if pid is None or tid is None:
                    continue
                key = (int(pid), int(tid))
                players.setdefault(
                    key,
                    {
                        "player_id": int(pid),
                        "player_name": pname,
                        "team_id": int(tid),
                        "team_name": tname,
                        "primary_position": pos_name,
                        "jersey_number": jersey,
                    },
                )

    # Ordina per team, poi per nome
    return sorted(players.values(), key=lambda x: (x["team_id"], str(x["player_name"])) )


# ----------------------------------------------------------------------
# Player-match stats e Event metadata
# ----------------------------------------------------------------------

def _inc(d: dict[str, Any], key: str, val: int = 1):
    d[key] = int(d.get(key, 0)) + val


def aggregate_player_match_stats(match_ids: list[int]) -> list[dict[str, Any]]:
    stats: dict[tuple[int, int, int], dict[str, Any]] = {}

    for mid in match_ids:
        path = RAW_DIR / "events" / f"{mid}.json"
        if not path.exists():
            continue
        events = load_json(path)
        for e in events:
            team = e.get("team") or {}
            player = e.get("player") or {}
            pid = player.get("id")
            tid = team.get("id")
            if pid is None or tid is None:
                continue
            key = (int(mid), int(pid), int(tid))
            rec = stats.setdefault(
                key,
                {
                    "match_id": int(mid),
                    "player_id": int(pid),
                    "team_id": int(tid),
                    "passes": 0,
                    "passes_completed": 0,
                    "shots": 0,
                    "goals": 0,
                    "tackles": 0,
                    "dribbles": 0,
                    "pressures": 0,
                    "interceptions": 0,
                    "clearances": 0,
                    "fouls_committed": 0,
                    "fouls_won": 0,
                    "yellow_cards": 0,
                    "red_cards": 0,
                },
            )

            tname = (e.get("type") or {}).get("name")

            if tname == "Pass":
                _inc(rec, "passes")
                # completed if no pass.outcome
                pinfo = e.get("pass") or {}
                if not pinfo.get("outcome"):
                    _inc(rec, "passes_completed")

            elif tname == "Shot":
                _inc(rec, "shots")
                sinfo = e.get("shot") or {}
                if (sinfo.get("outcome") or {}).get("name") == "Goal":
                    _inc(rec, "goals")

            elif tname == "Tackle":
                _inc(rec, "tackles")
            elif tname == "Duel":
                dinfo = e.get("duel") or {}
                if (dinfo.get("type") or {}).get("name") == "Tackle":
                    _inc(rec, "tackles")
            elif tname == "Dribble":
                _inc(rec, "dribbles")
            elif tname == "Pressure":
                _inc(rec, "pressures")
            elif tname == "Interception":
                _inc(rec, "interceptions")
            elif tname == "Clearance":
                _inc(rec, "clearances")
            elif tname == "Foul Committed":
                _inc(rec, "fouls_committed")
                finfo = e.get("foul_committed") or {}
                card = (finfo.get("card") or {}).get("name")
                if card == "Yellow Card":
                    _inc(rec, "yellow_cards")
                elif card == "Second Yellow":
                    _inc(rec, "yellow_cards")
                    _inc(rec, "red_cards")
                elif card == "Red Card":
                    _inc(rec, "red_cards")
            elif tname == "Foul Won":
                _inc(rec, "fouls_won")

    return sorted(stats.values(), key=lambda x: (x["match_id"], x["team_id"], x["player_id"]))


def dump_events_metadata(match_ids: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mid in match_ids:
        path = RAW_DIR / "events" / f"{mid}.json"
        if not path.exists():
            continue
        events = load_json(path)
        for idx, e in enumerate(events):
            team = e.get("team") or {}
            player = e.get("player") or {}
            row = {
                "match_id": int(mid),
                "idx": idx,
                "id": e.get("id"),
                "period": e.get("period"),
                "timestamp": e.get("timestamp"),
                "minute": e.get("minute"),
                "second": e.get("second"),
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "player_id": player.get("id"),
                "player_name": player.get("name"),
                "type": (e.get("type") or {}).get("name"),
                "possession": e.get("possession"),
                "possession_team_id": ((e.get("possession_team") or {}).get("id")),
                "possession_team_name": ((e.get("possession_team") or {}).get("name")),
                "duration": e.get("duration"),
                # stringify arrays (location/end_location are [x,y])
                "location": ",".join(map(str, e.get("location", []))) if e.get("location") else None,
                "end_location": ",".join(map(str, e.get("end_location", []))) if e.get("end_location") else None,
            }
            rows.append(row)
    return rows


def build_referees(matches_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[int, dict[str, Any]] = {}
    for m in matches_rows:
        rid = m.get("referee_id")
        rname = m.get("referee_name")
        if rid is not None:
            seen.setdefault(int(rid), {"referee_id": int(rid), "referee_name": rname})
    return sorted(seen.values(), key=lambda x: x["referee_id"])


def build_venues(matches_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[int, dict[str, Any]] = {}
    for m in matches_rows:
        sid = m.get("stadium_id")
        sname = m.get("stadium_name")
        cid = m.get("country_id")
        cname = m.get("country_name")
        if sid is not None:
            seen.setdefault(int(sid), {"stadium_id": int(sid), "stadium_name": sname, "country_id": cid, "country_name": cname})
    return sorted(seen.values(), key=lambda x: x["stadium_id"])


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------

def run() -> None:
    matches_file = _get_matches_file()
    matches_rows = build_matches(matches_file)
    match_ids = [int(m["match_id"]) for m in matches_rows]

    print(f"🧾 Partite EURO 2024: {len(match_ids)}")

    teams_rows = build_teams(matches_rows, match_ids)
    players_rows = build_players(match_ids)
    stats_rows = aggregate_player_match_stats(match_ids)
    events_rows = dump_events_metadata(match_ids)
    referees_rows = build_referees(matches_rows)
    venues_rows = build_venues(matches_rows)

    # Write files
    write_csv(OUT_DIR / "matches.csv", list(matches_rows[0].keys()) if matches_rows else [
        "match_id","match_date","kick_off","competition_id","competition_name","season_id","season_name","stage_id","stage_name","home_team_id","home_team_name","away_team_id","away_team_name","home_score","away_score","referee_id","referee_name","stadium_id","stadium_name","country_id","country_name"
    ], matches_rows)

    write_csv(OUT_DIR / "teams.csv", ["team_id", "team_name"], teams_rows)

    write_csv(OUT_DIR / "players.csv", [
        "player_id","player_name","team_id","team_name","primary_position","jersey_number"
    ], players_rows)

    write_csv(OUT_DIR / "player_match_stats.csv", [
        "match_id","team_id","player_id","passes","passes_completed","shots","goals","tackles","dribbles","pressures","interceptions","clearances","fouls_committed","fouls_won","yellow_cards","red_cards"
    ], stats_rows)

    write_csv(OUT_DIR / "events_meta.csv", [
        "match_id","idx","id","period","timestamp","minute","second","team_id","team_name","player_id","player_name","type","possession","possession_team_id","possession_team_name","duration","location","end_location"
    ], events_rows)

    write_csv(OUT_DIR / "referees.csv", ["referee_id", "referee_name"], referees_rows)

    write_csv(OUT_DIR / "venues.csv", ["stadium_id", "stadium_name", "country_id", "country_name"], venues_rows)

    print("✅ Data filtrati e salvati in data/processed/statsbomb/")


if __name__ == "__main__":
    run()
