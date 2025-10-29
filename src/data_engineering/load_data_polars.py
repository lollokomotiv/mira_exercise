"""
Polars-based ETL for StatsBomb EURO 2024 into tidy CSVs.

Outputs (to data/processed/statsbomb):
- matches.csv, teams.csv, players.csv
- player_appearances.csv (started, minutes, position, shirt)
- player_match_stats.csv (incl. xg, xa)
- referees.csv, venues.csv, events_meta.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import polars as pl


RAW_DIR = Path("data/raw/statsbomb")
OUT_DIR = Path("data/processed/statsbomb/euro24")


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_out():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _get_euro2024_comp_season() -> Tuple[int, int]:
    comps = _load_json(RAW_DIR / "competitions.json")
    euro_names = {"UEFA Euro", "European Championship"}
    for c in comps:
        if c.get("competition_name") in euro_names and c.get("season_name") == "2024":
            return int(c["competition_id"]), int(c["season_id"])
    raise RuntimeError("EURO 2024 not found in competitions.json")


def build_matches_df() -> pl.DataFrame:
    comp_id, season_id = _get_euro2024_comp_season()
    matches_path = RAW_DIR / "matches" / f"{comp_id}_{season_id}.json"
    matches = _load_json(matches_path)
    rows: List[Dict[str, Any]] = []
    for m in matches:
        rows.append(
            {
                "match_id": m.get("match_id"),
                "match_date": m.get("match_date"),
                "kick_off": m.get("kick_off"),
                "competition_id": (m.get("competition") or {}).get("competition_id"),
                "competition_name": (m.get("competition") or {}).get(
                    "competition_name"
                ),
                "season_id": (m.get("season") or {}).get("season_id"),
                "season_name": (m.get("season") or {}).get("season_name"),
                "stage_id": (m.get("competition_stage") or {}).get("id"),
                "stage_name": (m.get("competition_stage") or {}).get("name"),
                "home_team_id": (m.get("home_team") or {}).get("home_team_id")
                or (m.get("home_team") or {}).get("id"),
                "home_team_name": (m.get("home_team") or {}).get("home_team_name")
                or (m.get("home_team") or {}).get("name"),
                "away_team_id": (m.get("away_team") or {}).get("away_team_id")
                or (m.get("away_team") or {}).get("id"),
                "away_team_name": (m.get("away_team") or {}).get("away_team_name")
                or (m.get("away_team") or {}).get("name"),
                "home_score": m.get("home_score"),
                "away_score": m.get("away_score"),
                "referee_id": (m.get("referee") or {}).get("id"),
                "referee_name": (m.get("referee") or {}).get("name"),
                "stadium_id": (m.get("stadium") or {}).get("id"),
                "stadium_name": (m.get("stadium") or {}).get("name"),
                "country_id": ((m.get("stadium") or {}).get("country") or {}).get("id"),
                "country_name": ((m.get("stadium") or {}).get("country") or {}).get(
                    "name"
                ),
            }
        )
    return pl.DataFrame(rows)


def build_teams_df(matches_df: pl.DataFrame, match_ids: List[int]) -> pl.DataFrame:
    """Collect unique teams across matches, lineups, and events using Polars ops."""

    # Home/away team identifiers straight from matches_df
    home = matches_df.select(
        pl.col("home_team_id").cast(pl.Int64).alias("team_id"),
        pl.col("home_team_name").alias("team_name"),
    )
    away = matches_df.select(
        pl.col("away_team_id").cast(pl.Int64).alias("team_id"),
        pl.col("away_team_name").alias("team_name"),
    )
    team_frames = [home, away]

    # Complimentary teams coming from lineups JSON payloads
    lineup_rows: list[dict[str, Any]] = []
    for mid in match_ids:
        p = RAW_DIR / "lineups" / f"{mid}.json"
        if not p.exists():
            continue
        for side in _load_json(p):
            tid = side.get("team_id")
            tname = side.get("team_name")
            if tid is None:
                continue
            lineup_rows.append({"team_id": int(tid), "team_name": tname})
    if lineup_rows:
        team_frames.append(pl.DataFrame(lineup_rows))

    # Teams observed in events JSON payloads (e.g. for missing lineups)
    event_rows: list[dict[str, Any]] = []
    for mid in match_ids:
        p = RAW_DIR / "events" / f"{mid}.json"
        if not p.exists():
            continue
        for e in _load_json(p):
            team = e.get("team") or {}
            tid = team.get("id")
            tname = team.get("name")
            if tid is None:
                continue
            event_rows.append({"team_id": int(tid), "team_name": tname})
    if event_rows:
        team_frames.append(pl.DataFrame(event_rows))

    teams_df = (
        pl.concat(team_frames, how="vertical_relaxed")
        .drop_nulls("team_id")
        .with_columns(pl.col("team_id").cast(pl.Int64))
        .unique(subset=["team_id"], keep="last")
        .sort("team_id")
    )
    return teams_df


def _parse_timecode(tc: str | None) -> float:
    if not tc:
        return 0.0
    # format "MM:SS" or "HH:MM:SS.xxx"; StatsBomb uses "MM:SS" within halves, but
    # lineups positions often are "00:00:00.000" style. Parse generically.
    parts = tc.split(":")
    try:
        if len(parts) == 2:
            m, s = parts
            return int(m) + int(s) / 60.0
        elif len(parts) == 3:
            h, m, s = parts
            sec = float(s)
            return int(h) * 60 + int(m) + sec / 60.0
    except Exception:
        return 0.0
    return 0.0


def build_players_and_appearances_df(
    match_ids: List[int],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    players: Dict[Tuple[int, int], Dict[str, Any]] = {}
    appearances_rows: List[Dict[str, Any]] = []

    for mid in match_ids:
        p = RAW_DIR / "lineups" / f"{mid}.json"
        if not p.exists():
            continue
        data = _load_json(p)
        for side in data:
            tid = side.get("team_id")
            tname = side.get("team_name")
            for plr in side.get("lineup", []):
                pid = plr.get("player_id") or (plr.get("player") or {}).get("id")
                pname = plr.get("player_name") or (plr.get("player") or {}).get("name")
                jersey = plr.get("jersey_number")
                positions = plr.get("positions") or []
                primary_pos = None
                started = 0
                minutes = 0.0
                if positions:
                    # first position as primary
                    pos0 = positions[0]
                    if isinstance(pos0.get("position"), dict):
                        primary_pos = (pos0.get("position") or {}).get("name")
                    else:
                        primary_pos = pos0.get("position") or pos0.get("position_name")
                    # accumulate minutes across stints if from/to available
                    for stint in positions:
                        frm = _parse_timecode(stint.get("from"))
                        to = _parse_timecode(stint.get("to"))
                        if to <= 0:
                            # default halves: to 45 or 90
                            to = 90.0 if stint.get("to_period", 2) == 2 else 45.0
                        minutes += max(0.0, to - frm)
                        if (stint.get("from")) and (
                            stint.get("start_reason") == "Starting XI"
                            or stint.get("from") == "00:00:00.000"
                        ):
                            started = 1

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
                        "primary_position": primary_pos,
                        "jersey_number": jersey,
                    },
                )
                appearances_rows.append(
                    {
                        "appearance_id": f"{mid}_{pid}",
                        "match_id": int(mid),
                        "team_id": int(tid),
                        "player_id": int(pid),
                        "started": started,
                        "minutes": round(minutes, 2),
                        "position_played": primary_pos,
                        "jersey_number": jersey,
                    }
                )

    players_df = pl.DataFrame(
        sorted(players.values(), key=lambda x: (x["team_id"], str(x["player_name"])))
    )
    apps_df = pl.DataFrame(appearances_rows).unique(
        subset=["appearance_id"], keep="last"
    )
    return players_df, apps_df


def build_player_match_stats_df(match_ids: List[int]) -> pl.DataFrame:
    # aggregate counts and xG/xA using event ids
    rows: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for mid in match_ids:
        p = RAW_DIR / "events" / f"{mid}.json"
        if not p.exists():
            continue
        events = _load_json(p)
        # map event id -> player id for key passes
        pass_by_id: Dict[str, int] = {}
        for e in events:
            if (e.get("type") or {}).get("name") == "Pass" and e.get("id") is not None:
                pid = (e.get("player") or {}).get("id")
                if pid is not None:
                    pass_by_id[str(e["id"])] = int(pid)

        for e in events:
            team = e.get("team") or {}
            player = e.get("player") or {}
            pid = player.get("id")
            tid = team.get("id")
            if pid is None or tid is None:
                continue
            key = (int(mid), int(pid), int(tid))
            rec = rows.setdefault(
                key,
                {
                    "match_id": int(mid),
                    "player_id": int(pid),
                    "team_id": int(tid),
                    "minutes": None,
                    "shots": 0,
                    "goals": 0,
                    "xg": 0.0,
                    "xa": 0.0,
                    "passes": 0,
                    "passes_completed": 0,
                    "dribbles": 0,
                    "pressures": 0,
                    "tackles": 0,
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
                rec["passes"] += 1
                pinfo = e.get("pass") or {}
                if not pinfo.get("outcome"):
                    rec["passes_completed"] += 1
            elif tname == "Shot":
                rec["shots"] += 1
                sinfo = e.get("shot") or {}
                xg = sinfo.get("statsbomb_xg") or sinfo.get("xg") or 0.0
                try:
                    rec["xg"] += float(xg)
                except Exception:
                    pass
                if (sinfo.get("outcome") or {}).get("name") == "Goal":
                    rec["goals"] += 1
                # xA credit to the assister via key_pass_id
                kpid = sinfo.get("key_pass_id")
                if kpid is not None:
                    apid = pass_by_id.get(str(kpid))
                    if apid is not None:
                        akey = (int(mid), int(apid), int(tid))
                        arec = rows.setdefault(
                            akey,
                            {
                                "match_id": int(mid),
                                "player_id": int(apid),
                                "team_id": int(tid),
                                "minutes": None,
                                "shots": 0,
                                "goals": 0,
                                "xg": 0.0,
                                "xa": 0.0,
                                "passes": 0,
                                "passes_completed": 0,
                                "dribbles": 0,
                                "pressures": 0,
                                "tackles": 0,
                                "interceptions": 0,
                                "clearances": 0,
                                "fouls_committed": 0,
                                "fouls_won": 0,
                                "yellow_cards": 0,
                                "red_cards": 0,
                            },
                        )
                        try:
                            arec["xa"] += float(xg or 0.0)
                        except Exception:
                            pass
            elif tname == "Tackle":
                rec["tackles"] += 1
            elif tname == "Duel":
                dinfo = e.get("duel") or {}
                if (dinfo.get("type") or {}).get("name") == "Tackle":
                    rec["tackles"] += 1
            elif tname == "Dribble":
                rec["dribbles"] += 1
            elif tname == "Pressure":
                rec["pressures"] += 1
            elif tname == "Interception":
                rec["interceptions"] += 1
            elif tname == "Clearance":
                rec["clearances"] += 1
            elif tname == "Foul Committed":
                rec["fouls_committed"] += 1
                finfo = e.get("foul_committed") or {}
                card = (finfo.get("card") or {}).get("name")
                if card == "Yellow Card":
                    rec["yellow_cards"] += 1
                elif card == "Second Yellow":
                    rec["yellow_cards"] += 1
                    rec["red_cards"] += 1
                elif card == "Red Card":
                    rec["red_cards"] += 1
            elif tname == "Foul Won":
                rec["fouls_won"] += 1

    df = pl.DataFrame(list(rows.values()))
    return df.sort(["match_id", "team_id", "player_id"]) if df.height else df


def build_events_meta_df(match_ids: List[int]) -> pl.DataFrame:
    out_rows: List[Dict[str, Any]] = []
    for mid in match_ids:
        p = RAW_DIR / "events" / f"{mid}.json"
        if not p.exists():
            continue
        events = _load_json(p)
        for idx, e in enumerate(events):
            team = e.get("team") or {}
            player = e.get("player") or {}
            out_rows.append(
                {
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
                    "possession_team_name": (
                        (e.get("possession_team") or {}).get("name")
                    ),
                    "duration": e.get("duration"),
                    "location": (
                        ",".join(map(str, e.get("location", [])))
                        if e.get("location")
                        else None
                    ),
                    "end_location": (
                        ",".join(map(str, e.get("end_location", [])))
                        if e.get("end_location")
                        else None
                    ),
                }
            )
    return pl.DataFrame(out_rows)


def build_referees_df(matches_df: pl.DataFrame) -> pl.DataFrame:
    return (
        matches_df.select([pl.col("referee_id").cast(pl.Int64), pl.col("referee_name")])
        .unique()
        .drop_nulls("referee_id")
        .sort("referee_id")
        .rename({"referee_id": "referee_id", "referee_name": "referee_name"})
    )


def build_venues_df(matches_df: pl.DataFrame) -> pl.DataFrame:
    return (
        matches_df.select(
            [
                pl.col("stadium_id").alias("stadium_id"),
                pl.col("stadium_name").alias("stadium_name"),
                pl.col("country_id").alias("country_id"),
                pl.col("country_name").alias("country_name"),
            ]
        )
        .unique()
        .drop_nulls("stadium_id")
        .sort("stadium_id")
    )


def run() -> None:
    _ensure_out()
    matches_df = build_matches_df()
    match_ids = [int(x) for x in matches_df.select("match_id").to_series().to_list()]
    teams_df = build_teams_df(matches_df, match_ids)
    players_df, apps_df = build_players_and_appearances_df(match_ids)
    stats_df = build_player_match_stats_df(match_ids)
    # attach minutes from appearances to stats, ensuring single 'minutes' column
    if stats_df.height and apps_df.height:
        if "minutes" in stats_df.columns:
            stats_df = stats_df.drop("minutes")
        stats_df = stats_df.join(
            apps_df.select(["match_id", "player_id", "minutes"]),
            on=["match_id", "player_id"],
            how="left",
        )
    events_meta_df = build_events_meta_df(match_ids)
    referees_df = build_referees_df(matches_df)
    venues_df = build_venues_df(matches_df)

    # Write CSVs to match existing loader expectations and new tables
    matches_df.write_csv(OUT_DIR / "matches.csv")
    teams_df.write_csv(OUT_DIR / "teams.csv")
    players_df.write_csv(OUT_DIR / "players.csv")
    apps_df.write_csv(OUT_DIR / "player_appearances.csv")
    stats_df.write_csv(OUT_DIR / "player_match_stats.csv")
    events_meta_df.write_csv(OUT_DIR / "events_meta.csv")
    referees_df.write_csv(OUT_DIR / "referees.csv")
    venues_df.write_csv(OUT_DIR / "venues.csv")

    print("✅ Polars ETL complete: data/processed/statsbomb/euro24")


if __name__ == "__main__":
    run()
