"""
Costruisce uno schema SQLite e carica i CSV strutturati/filtrati.

Origini attese:
- data/processed/statsbomb/ (matches.csv, teams.csv, players.csv, player_match_stats.csv, referees.csv, venues.csv, events_meta.csv)
- data/processed/transfermarkt/transfers_24.csv

Uso:
  python src/data_engineering/build_schema.py --db data/db/euro24.sqlite
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path
from typing import Iterable, Dict, Any


PROC_SB = Path("data/processed/statsbomb/euro24")
PROC_TM = Path("data/processed/transfermarkt/euro24")
RAW_TM = Path("data/raw/transfermarkt")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def execute_script(conn: sqlite3.Connection, sql: str) -> None:
    conn.executescript(sql)
    conn.commit()


def create_schema(conn: sqlite3.Connection) -> None:
    ddl = """
    PRAGMA foreign_keys = ON;

    -- Drop/refresh views that depend on base tables
    DROP VIEW IF EXISTS v_players_transfers_euro24_window_all;
    DROP VIEW IF EXISTS v_players_transfers_euro24_window;
    DROP VIEW IF EXISTS transfers_euro24_window;

    DROP TABLE IF EXISTS events_meta;
    DROP TABLE IF EXISTS player_match_stats;
    DROP TABLE IF EXISTS player_appearances;
    DROP TABLE IF EXISTS matches;
    DROP TABLE IF EXISTS players;
    DROP TABLE IF EXISTS teams;
    DROP TABLE IF EXISTS referees;
    DROP TABLE IF EXISTS venues;
    DROP TABLE IF EXISTS transfers;
    DROP TABLE IF EXISTS player_xref_all;
    DROP TABLE IF EXISTS player_xref;
    DROP TABLE IF EXISTS club_xref;
    DROP TABLE IF EXISTS tm_players;

    CREATE TABLE teams (
        team_id INTEGER PRIMARY KEY,
        team_name TEXT
    );

    CREATE TABLE players (
        player_id INTEGER,
        team_id INTEGER,
        player_name TEXT,
        team_name TEXT,
        primary_position TEXT,
        jersey_number INTEGER,
        PRIMARY KEY (player_id, team_id),
        FOREIGN KEY (team_id) REFERENCES teams(team_id)
    );

    CREATE TABLE matches (
        match_id INTEGER PRIMARY KEY,
        match_date TEXT,
        kick_off TEXT,
        competition_id INTEGER,
        competition_name TEXT,
        season_id INTEGER,
        season_name TEXT,
        stage_id INTEGER,
        stage_name TEXT,
        home_team_id INTEGER,
        home_team_name TEXT,
        away_team_id INTEGER,
        away_team_name TEXT,
        home_score INTEGER,
        away_score INTEGER,
        referee_id INTEGER,
        referee_name TEXT,
        stadium_id INTEGER,
        stadium_name TEXT,
        country_id INTEGER,
        country_name TEXT
    );

    CREATE TABLE player_match_stats (
        match_id INTEGER,
        team_id INTEGER,
        player_id INTEGER,
        minutes REAL,
        passes INTEGER,
        passes_completed INTEGER,
        shots INTEGER,
        goals INTEGER,
        xg REAL,
        xa REAL,
        tackles INTEGER,
        dribbles INTEGER,
        pressures INTEGER,
        interceptions INTEGER,
        clearances INTEGER,
        fouls_committed INTEGER,
        fouls_won INTEGER,
        yellow_cards INTEGER,
        red_cards INTEGER,
        PRIMARY KEY (match_id, player_id),
        FOREIGN KEY (match_id) REFERENCES matches(match_id),
        FOREIGN KEY (team_id) REFERENCES teams(team_id)
    );

    CREATE TABLE player_appearances (
        appearance_id TEXT PRIMARY KEY,
        match_id INTEGER,
        team_id INTEGER,
        player_id INTEGER,
        started INTEGER,
        minutes REAL,
        position_played TEXT,
        jersey_number INTEGER,
        FOREIGN KEY (match_id) REFERENCES matches(match_id),
        FOREIGN KEY (team_id) REFERENCES teams(team_id)
    );

    CREATE TABLE events_meta (
        match_id INTEGER,
        idx INTEGER,
        id TEXT,
        period INTEGER,
        timestamp TEXT,
        minute INTEGER,
        second INTEGER,
        team_id INTEGER,
        team_name TEXT,
        player_id INTEGER,
        player_name TEXT,
        type TEXT,
        possession INTEGER,
        possession_team_id INTEGER,
        possession_team_name TEXT,
        duration REAL,
        location TEXT,
        end_location TEXT
    );

    CREATE TABLE referees (
        referee_id INTEGER PRIMARY KEY,
        referee_name TEXT
    );

    CREATE TABLE venues (
        stadium_id INTEGER PRIMARY KEY,
        stadium_name TEXT,
        country_id INTEGER,
        country_name TEXT
    );

    CREATE TABLE transfers (
        player_id INTEGER,
        transfer_date TEXT,
        transfer_season TEXT,
        from_club_id INTEGER,
        to_club_id INTEGER,
        from_club_name TEXT,
        to_club_name TEXT,
        transfer_fee REAL,
        market_value_in_eur REAL,
        player_name TEXT
    );

    -- tm_players
    CREATE TABLE tm_players (
        player_id INTEGER PRIMARY KEY,
        first_name TEXT,
        last_name TEXT,
        name TEXT,
        last_season TEXT,
        current_club_id INTEGER,
        player_code TEXT,
        country_of_birth TEXT,
        city_of_birth TEXT,
        country_of_citizenship TEXT,
        date_of_birth TEXT,
        sub_position TEXT,
        position TEXT,
        foot TEXT,
        height_in_cm INTEGER,
        contract_expiration_date TEXT,
        agent_name TEXT,
        image_url TEXT,
        url TEXT,
        current_club_domestic_competition_id TEXT,
        current_club_name TEXT,
        market_value_in_eur REAL,
        highest_market_value_in_eur REAL
    );

    CREATE TABLE player_xref (
        statsbomb_player_id INTEGER,
        tm_player_id INTEGER,
        statsbomb_player_name TEXT,
        tm_player_name TEXT,
        confidence REAL,
        method TEXT,
        PRIMARY KEY (statsbomb_player_id, tm_player_id),
        FOREIGN KEY (tm_player_id) REFERENCES tm_players(player_id)
    );

    -- Optional wider xref (lower confidence), same core columns; extra columns ignored in load
    CREATE TABLE player_xref_all (
        statsbomb_player_id INTEGER,
        tm_player_id INTEGER,
        statsbomb_player_name TEXT,
        tm_player_name TEXT,
        confidence REAL,
        method TEXT,
        PRIMARY KEY (statsbomb_player_id, tm_player_id)
        FOREIGN KEY (tm_player_id) REFERENCES tm_players(player_id)
    );

    CREATE TABLE club_xref (
        tm_club_id INTEGER PRIMARY KEY,
        tm_club_name TEXT,
        alias TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_stats_match ON player_match_stats(match_id);
    CREATE INDEX IF NOT EXISTS idx_stats_player ON player_match_stats(player_id);
    CREATE INDEX IF NOT EXISTS idx_events_match ON events_meta(match_id);
    CREATE INDEX IF NOT EXISTS idx_transfers_date ON transfers(transfer_date);
    CREATE INDEX IF NOT EXISTS idx_transfers_player_date ON transfers(player_id, transfer_date);
    CREATE INDEX IF NOT EXISTS idx_transfers_to_club ON transfers(to_club_id);
    CREATE INDEX IF NOT EXISTS idx_transfers_from_club ON transfers(from_club_id);
    CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_id);
    CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);
    CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team_id, away_team_id);
    CREATE INDEX IF NOT EXISTS idx_appearances_match ON player_appearances(match_id);
    CREATE INDEX IF NOT EXISTS idx_appearances_player ON player_appearances(player_id);
    CREATE INDEX IF NOT EXISTS idx_xref_tm_player ON player_xref(tm_player_id);
    CREATE INDEX IF NOT EXISTS idx_xref_all_tm_player ON player_xref_all(tm_player_id);

    -- Transfers in EURO 2024 window: after final and before 2024-10-01
    CREATE VIEW IF NOT EXISTS transfers_euro24_window AS
    SELECT *
    FROM transfers
    WHERE transfer_date >= '2024-07-15' AND transfer_date < '2024-10-01';

    -- Players joined with all transfers in the window (via xref)
    CREATE VIEW IF NOT EXISTS v_players_transfers_euro24_window AS
    SELECT
      sp.player_id AS statsbomb_player_id,
      sp.player_name,
      sp.team_id AS national_team_id,
      sp.team_name AS national_team,
      sp.primary_position,
      sp.jersey_number,
      px.tm_player_id,
      tm.date_of_birth AS tm_date_of_birth,
      tm.position AS tm_position,
      tm.foot AS tm_foot,
      tm.height_in_cm AS tm_height_cm,
      t.transfer_date,
      t.from_club_id,
      t.from_club_name,
      t.to_club_id,
      t.to_club_name,
      t.transfer_fee,
      t.market_value_in_eur
    FROM players sp
    LEFT JOIN player_xref px ON px.statsbomb_player_id = sp.player_id
    LEFT JOIN tm_players tm ON tm.player_id = px.tm_player_id
    LEFT JOIN transfers_euro24_window t ON t.player_id = px.tm_player_id;

    -- Same join but using the wider xref (player_xref_all)
    CREATE VIEW IF NOT EXISTS v_players_transfers_euro24_window_all AS
    SELECT
      sp.player_id AS statsbomb_player_id,
      sp.player_name,
      sp.team_id AS national_team_id,
      sp.team_name AS national_team,
      sp.primary_position,
      sp.jersey_number,
      px.tm_player_id,
      tm.date_of_birth AS tm_date_of_birth,
      tm.position AS tm_position,
      tm.foot AS tm_foot,
      tm.height_in_cm AS tm_height_cm,
      t.transfer_date,
      t.from_club_id,
      t.from_club_name,
      t.to_club_id,
      t.to_club_name,
      t.transfer_fee,
      t.market_value_in_eur
    FROM players sp
    LEFT JOIN player_xref_all px ON px.statsbomb_player_id = sp.player_id
    LEFT JOIN tm_players tm ON tm.player_id = px.tm_player_id
    LEFT JOIN transfers_euro24_window t ON t.player_id = px.tm_player_id;
    """
    execute_script(conn, ddl)


def load_csv(conn: sqlite3.Connection, table: str, path: Path) -> None:
    if not path.exists():
        print(f"⚠️  Skip {table}: file non trovato {path}")
        return
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Usa solo le colonne effettivamente presenti nella tabella
        cur = conn.execute(f"PRAGMA table_info({table})")
        table_cols = [r[1] for r in cur.fetchall()]
        cols = [c for c in (reader.fieldnames or []) if c in table_cols]
        placeholders = ",".join([":" + c for c in cols])
        sql = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
        conn.executemany(sql, reader)
        conn.commit()
        print(f"✅ Loaded {table}: {path}")


def run(db_path: str) -> None:
    db = Path(db_path)
    ensure_parent(db)
    conn = sqlite3.connect(str(db))
    try:
        create_schema(conn)

        # Load StatsBomb processed CSV
        load_csv(conn, "teams", PROC_SB / "teams.csv")
        load_csv(conn, "players", PROC_SB / "players.csv")
        load_csv(conn, "matches", PROC_SB / "matches.csv")
        load_csv(conn, "player_match_stats", PROC_SB / "player_match_stats.csv")
        load_csv(conn, "player_appearances", PROC_SB / "player_appearances.csv")
        load_csv(conn, "events_meta", PROC_SB / "events_meta.csv")
        load_csv(conn, "referees", PROC_SB / "referees.csv")
        load_csv(conn, "venues", PROC_SB / "venues.csv")

        # Load transfers filtered
        load_csv(conn, "transfers", PROC_TM / "transfers_euro24.csv")

        # Load Transfermarkt players (for date_of_birth, position, market value, etc)
        # load_csv(conn, "tm_players", PROC_TM / "players_euro24.csv")
        load_csv(conn, "tm_players", RAW_TM / "players.csv")

        # Optional: load xref files if present
        XREF = Path("data/processed/xref/euro24")
        load_csv(conn, "player_xref", XREF / "player_xref_euro24_strict.csv")
        load_csv(conn, "player_xref_all", XREF / "player_xref_euro24_full.csv")
        load_csv(conn, "club_xref", XREF / "club_xref_euro24.csv")

        print(f"🎯 Database pronto: {db}")
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Costruisce DB SQLite con dati EURO24 + transfers"
    )
    p.add_argument(
        "--db", default="data/db/euro24.sqlite", help="Percorso DB SQLite di output"
    )
    args = p.parse_args(argv)
    run(args.db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
