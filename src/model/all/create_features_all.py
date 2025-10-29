"""
Feature builder for EURO 2024 transfer value modeling (pandas-first).

Outputs:
- In-memory pandas DataFrame with player-level features + labels
  (optionally saved as CSV if requested).

Notes:
- Reads from SQLite DB created by build_schema.py
- Aggregates StatsBomb player_match_stats + appearances into totals and per90 rates
- Joins transfer labels from v_players_transfers_euro24_window or _all
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd


DEFAULT_DB = Path("data/db/euro24.sqlite")
DEFAULT_VALUATIONS = Path("data/raw/transfermarkt/player_valuations.csv")
MV_CUTOFF = "2024-06-13"  # ultimo valore PRIMA dell'inizio EURO24 (14/06)
DEFAULT_TM_PLAYERS = Path("data/raw/transfermarkt/players.csv")
AGE_REF_DATE = "2024-07-01"  # età calcolata a inizio EURO24
MIN_MINUTES_PER90 = 180.0  # capping: per-90 calcolati su almeno 180' per stabilità


def _safe_div(n: pd.Series, d: pd.Series) -> pd.Series:
    out = n.astype(float) / d.replace({0: np.nan})
    return out.fillna(0.0)


def _position_bucket(pos: pd.Series) -> pd.Series:
    s = pos.fillna("").str.lower()
    conds = [
        s.str.contains("keep"),
        s.str.contains("back")
        | s.str.contains("defen")
        | s.str.contains("centre")
        | s.str.contains("center")
        | s.str.contains("cb"),
        s.str.contains("mid"),
        s.str.contains("wing")
        | s.str.contains("forward")
        | s.str.contains("striker")
        | s.str.contains("attack"),
    ]
    choices = ["GK", "DEF", "MID", "ATT"]
    return np.select(conds, choices, default="UNK")


def load_player_aggregates(conn: sqlite3.Connection) -> pd.DataFrame:
    sql = """
    WITH stats AS (
      SELECT
        s.player_id,
        SUM(COALESCE(s.minutes,0)) AS minutes,
        COUNT(DISTINCT s.match_id) AS matches,
        SUM(COALESCE(s.goals,0)) AS goals,
        SUM(COALESCE(s.shots,0)) AS shots,
        SUM(COALESCE(s.xg,0)) AS xg,
        SUM(COALESCE(s.xa,0)) AS xa,
        SUM(COALESCE(s.passes,0)) AS passes,
        SUM(COALESCE(s.passes_completed,0)) AS passes_completed,
        SUM(COALESCE(s.tackles,0)) AS tackles,
        SUM(COALESCE(s.dribbles,0)) AS dribbles,
        SUM(COALESCE(s.pressures,0)) AS pressures,
        SUM(COALESCE(s.interceptions,0)) AS interceptions,
        SUM(COALESCE(s.clearances,0)) AS clearances,
        SUM(COALESCE(s.fouls_committed,0)) AS fouls_committed,
        SUM(COALESCE(s.fouls_won,0)) AS fouls_won,
        SUM(COALESCE(s.yellow_cards,0)) AS yellow_cards,
        SUM(COALESCE(s.red_cards,0)) AS red_cards
      FROM player_match_stats s
      GROUP BY s.player_id
    ), apps AS (
      SELECT
        a.player_id,
        SUM(CASE WHEN COALESCE(a.started,0)=1 THEN 1 ELSE 0 END) AS starts
      FROM player_appearances a
      GROUP BY a.player_id
    )
    SELECT
      p.player_id,
      p.player_name,
      p.team_id,
      p.team_name,
      p.primary_position,
      COALESCE(stats.minutes,0) AS minutes,
      COALESCE(stats.matches,0) AS matches,
      COALESCE(apps.starts,0) AS starts,
      COALESCE(stats.goals,0) AS goals,
      COALESCE(stats.shots,0) AS shots,
      COALESCE(stats.xg,0) AS xg,
      COALESCE(stats.xa,0) AS xa,
      COALESCE(stats.passes,0) AS passes,
      COALESCE(stats.passes_completed,0) AS passes_completed,
      COALESCE(stats.tackles,0) AS tackles,
      COALESCE(stats.dribbles,0) AS dribbles,
      COALESCE(stats.pressures,0) AS pressures,
      COALESCE(stats.interceptions,0) AS interceptions,
      COALESCE(stats.clearances,0) AS clearances,
      COALESCE(stats.fouls_committed,0) AS fouls_committed,
      COALESCE(stats.fouls_won,0) AS fouls_won,
      COALESCE(stats.yellow_cards,0) AS yellow_cards,
      COALESCE(stats.red_cards,0) AS red_cards
    FROM players p
    LEFT JOIN stats ON stats.player_id = p.player_id
    LEFT JOIN apps ON apps.player_id = p.player_id
    """
    df = pd.read_sql_query(sql, conn)

    # Derived features
    df["pos_bucket"] = _position_bucket(df["primary_position"])
    # Capping: per-90 su almeno 180' per evitare esplosioni con pochi minuti
    m_raw = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)
    m = m_raw.clip(lower=MIN_MINUTES_PER90)
    for col in [
        "goals",
        "shots",
        "xg",
        "xa",
        "tackles",
        "dribbles",
        "pressures",
        "interceptions",
        "clearances",
        "fouls_committed",
        "fouls_won",
    ]:
        df[f"{col}_per90"] = (df[col].astype(float) * 90.0 / m).fillna(0.0)
    df["pass_pct"] = _safe_div(df["passes_completed"], df["passes"]) * 100.0
    df["minutes_per_match"] = _safe_div(df["minutes"], df["matches"]).replace(
        np.inf, 0.0
    )

    return df


def load_labels(conn: sqlite3.Connection, use_all_xref: bool = False) -> pd.DataFrame:
    view = (
        "v_players_transfers_euro24_window_all"
        if use_all_xref
        else "v_players_transfers_euro24_window"
    )
    lbl = pd.read_sql_query(
        f"""
        SELECT statsbomb_player_id AS player_id, tm_player_id AS tm_player_id_label, transfer_date, transfer_fee, market_value_in_eur
        FROM {view}
        WHERE transfer_date IS NOT NULL
        """,
        conn,
        parse_dates=["transfer_date"],
    )
    # Keep the last transfer per player_id (if duplicates)
    lbl = lbl.sort_values(["player_id", "transfer_date"]).drop_duplicates(
        "player_id", keep="last"
    )
    return lbl


def load_tm_mapping(conn: sqlite3.Connection) -> pd.DataFrame:
    """Restituisce una mappatura StatsBomb→TM per tutti i giocatori disponibili.

    Priorità ai match "strict" (player_xref). Dove manca, usa player_xref_all.
    """
    try:
        strict = pd.read_sql_query(
            "SELECT statsbomb_player_id AS player_id, tm_player_id FROM player_xref",
            conn,
        )
    except Exception:
        strict = pd.DataFrame(columns=["player_id", "tm_player_id"])  # empty

    try:
        wide = pd.read_sql_query(
            "SELECT statsbomb_player_id AS player_id, tm_player_id, confidence FROM player_xref_all",
            conn,
        )
    except Exception:
        wide = pd.DataFrame(
            columns=["player_id", "tm_player_id", "confidence"]
        )  # empty

    # Dedup per player_id (se più righe, tieni la più alta confidence per xref_all)
    if not wide.empty and "confidence" in wide.columns:
        wide = wide.sort_values(
            ["player_id", "confidence"], ascending=[True, False]
        ).drop_duplicates("player_id")
    else:
        wide = wide.drop_duplicates("player_id")
    strict = strict.drop_duplicates("player_id")

    m = strict.merge(wide, on="player_id", how="outer", suffixes=("_strict", "_all"))
    # Scegli tm_player_id strict se presente, altrimenti all
    tm = m["tm_player_id_strict"].combine_first(m.get("tm_player_id_all"))
    out = pd.DataFrame(
        {
            "player_id": m["player_id"],
            "tm_player_id": tm,
        }
    )
    return out


def load_tm_pre_euro_mv(
    csv_path: Path | str = DEFAULT_VALUATIONS, cutoff_date: str = MV_CUTOFF
) -> pd.DataFrame:
    """Carica l'ultimo market value Transfermarkt PRIMA di EURO24 per ogni tm_player_id.

    Ritorna colonne: tm_player_id, mv_pre_euro, mv_pre_euro_date
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return pd.DataFrame(
            columns=["tm_player_id", "mv_pre_euro", "mv_pre_euro_date"]
        )  # empty

    df = pd.read_csv(csv_path)
    # ensure expected columns exist
    required = {"player_id", "date", "market_value_in_eur"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame(
            columns=["tm_player_id", "mv_pre_euro", "mv_pre_euro_date"]
        )  # empty

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()
    df = df[df["market_value_in_eur"].notna()].copy()

    cutoff = pd.to_datetime(cutoff_date)
    df = df[df["date"] <= cutoff]
    if df.empty:
        return pd.DataFrame(
            columns=["tm_player_id", "mv_pre_euro", "mv_pre_euro_date"]
        )  # empty

    df = df.sort_values(["player_id", "date"])
    last = df.groupby("player_id").tail(1).copy()
    out = last.rename(
        columns={
            "player_id": "tm_player_id",
            "market_value_in_eur": "mv_pre_euro",
            "date": "mv_pre_euro_date",
        }
    )[["tm_player_id", "mv_pre_euro", "mv_pre_euro_date"]]
    return out


def load_tm_age(
    csv_path: Path | str = DEFAULT_TM_PLAYERS, ref_date: str = AGE_REF_DATE
) -> pd.DataFrame:
    """Carica l'età dei giocatori (in anni) calcolata a una data di riferimento.

    Ritorna: tm_player_id, age_years
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return pd.DataFrame(columns=["tm_player_id", "age_years"])  # empty

    df = pd.read_csv(csv_path)
    if "player_id" not in df.columns or "date_of_birth" not in df.columns:
        return pd.DataFrame(columns=["tm_player_id", "age_years"])  # empty

    dob = pd.to_datetime(df["date_of_birth"], errors="coerce")
    ref = pd.to_datetime(ref_date)
    # Use days/365.25 to avoid unsupported 'Y' timedelta unit
    age_days = (ref - dob).dt.days
    age_years = age_days / 365.25
    out = pd.DataFrame(
        {
            "tm_player_id": df["player_id"],
            "age_years": age_years.astype(float),
        }
    )
    return out.dropna(subset=["age_years"])  # keep valid ages


def build_dataset(
    db_path: Path | str = DEFAULT_DB,
    use_all_xref: bool = False,
    include_market_value: bool = False,
    include_mv_pre: bool = True,
    valuations_csv: Path | str = DEFAULT_VALUATIONS,
    mv_cutoff_date: str = MV_CUTOFF,
    tm_players_csv: Path | str = DEFAULT_TM_PLAYERS,
    age_ref_date: str = AGE_REF_DATE,
    supervised_only: bool = True,
) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        X = load_player_aggregates(conn)

        # Aggiungi mapping TM per tutti i giocatori (strict → fallback all)
        map_tm = load_tm_mapping(conn)
        df = X.merge(map_tm, on="player_id", how="left")

        # Etichette di trasferimento (potrebbero avere tm_player_id_label)
        y = load_labels(conn, use_all_xref=use_all_xref)
        df = df.merge(y, on="player_id", how="left")
        # Se manca tm_player_id dalla mappatura, usa quello dell'etichetta
        if "tm_player_id_label" in df.columns:
            df["tm_player_id"] = df["tm_player_id"].combine_first(
                df["tm_player_id_label"]
            )  # fill missing
            df = df.drop(columns=["tm_player_id_label"])  # pulizia

        # Ensure numeric transfer_fee column (NaN if not parseable)
        if "transfer_fee" in df.columns:
            df["transfer_fee"] = pd.to_numeric(df["transfer_fee"], errors="coerce")

        # Join pre-EURO market value from Transfermarkt (optional)
        if include_mv_pre:
            mv = load_tm_pre_euro_mv(valuations_csv, cutoff_date=mv_cutoff_date)
            if not mv.empty and "tm_player_id" in df.columns:
                df = df.merge(mv, on="tm_player_id", how="left")
                # Derived comparatives
                if "mv_pre_euro" in df.columns:
                    tfee = pd.to_numeric(df["transfer_fee"], errors="coerce")
                    mvpre = pd.to_numeric(df["mv_pre_euro"], errors="coerce")
                    df["fee_over_mv_pre"] = _safe_div(tfee, mvpre)
                if "transfer_date" in df.columns and "mv_pre_euro_date" in df.columns:
                    try:
                        df["days_since_mv_pre"] = (
                            pd.to_datetime(df["transfer_date"])
                            - pd.to_datetime(df["mv_pre_euro_date"])
                        ) / np.timedelta64(1, "D")
                    except Exception:
                        df["days_since_mv_pre"] = np.nan

        # Join age from TM players
        age = load_tm_age(tm_players_csv, ref_date=age_ref_date)
        if not age.empty and "tm_player_id" in df.columns:
            df = df.merge(age, on="tm_player_id", how="left")

        # Supervised rows only (optional)
        if supervised_only and "transfer_fee" in df.columns:
            df = df[df["transfer_fee"].notna() & (df["transfer_fee"] > 0)].copy()

        # Optional: drop market_value if not requested
        if not include_market_value and "market_value_in_eur" in df.columns:
            df = df.drop(columns=["market_value_in_eur"])

        return df
    finally:
        conn.close()


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Build features + labels DataFrame from SQLite DB"
    )
    p.add_argument("--db", default=str(DEFAULT_DB), help="Path to SQLite DB")
    p.add_argument(
        "--use-all-xref", action="store_true", help="Use wider xref view for labels"
    )
    p.add_argument(
        "--include-market-value",
        action="store_true",
        help="Include Transfermarkt MV as feature (beware leakage)",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--use-mv-pre",
        dest="use_mv_pre",
        action="store_true",
        help="Include pre-EURO market value feature (default)",
    )
    group.add_argument(
        "--no-mv-pre",
        dest="use_mv_pre",
        action="store_false",
        help="Exclude pre-EURO market value feature",
    )
    p.set_defaults(use_mv_pre=True)
    p.add_argument(
        "--valuations-csv",
        default=str(DEFAULT_VALUATIONS),
        help="Path to player_valuations.csv from Transfermarkt dataset",
    )
    p.add_argument(
        "--mv-cutoff-date",
        default=MV_CUTOFF,
        help="Cutoff date for pre-EURO MV (YYYY-MM-DD)",
    )
    p.add_argument(
        "--tm-players-csv",
        default=str(DEFAULT_TM_PLAYERS),
        help="Path to Transfermarkt players.csv",
    )
    p.add_argument(
        "--age-ref-date",
        default=AGE_REF_DATE,
        help="Reference date for age (YYYY-MM-DD)",
    )
    p.add_argument(
        "--out",
        default="data/processed/model/features.csv",
        help="Output CSV path (primary dataset)",
    )
    p.add_argument(
        "--all-players",
        action="store_true",
        help="Produce dataset including players without valid transfer fee (no label filter)",
    )
    p.add_argument(
        "--out-all",
        default="",
        help="Optional secondary CSV path for the complementary dataset (if set, saves both supervised and all)",
    )
    args = p.parse_args(argv)

    # Primary dataset
    df = build_dataset(
        args.db,
        use_all_xref=args.use_all_xref,
        include_market_value=args.include_market_value,
        include_mv_pre=args.use_mv_pre,
        valuations_csv=args.valuations_csv,
        mv_cutoff_date=args.mv_cutoff_date,
        tm_players_csv=args.tm_players_csv,
        age_ref_date=args.age_ref_date,
        supervised_only=(not args.all_players),
    )

    # Save the primary dataset as features_all.csv
    features_all_path = Path("data/processed/model/features_all.csv")
    features_all_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(features_all_path, index=False)
    print(f"✅ Saved features_all.csv: {features_all_path} (rows={len(df)})")

    # Optional complementary dataset
    if args.out_all:
        df2 = build_dataset(
            args.db,
            use_all_xref=args.use_all_xref,
            include_market_value=args.include_market_value,
            include_mv_pre=args.use_mv_pre,
            valuations_csv=args.valuations_csv,
            mv_cutoff_date=args.mv_cutoff_date,
            tm_players_csv=args.tm_players_csv,
            age_ref_date=args.age_ref_date,
            supervised_only=not (not args.all_players),
        )
        out2 = Path(args.out_all)
        out2.parent.mkdir(parents=True, exist_ok=True)
        df2.to_csv(out2, index=False)
        print(f"✅ Also saved complementary dataset: {out2} (rows={len(df2)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
