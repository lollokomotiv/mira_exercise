from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np


def build_dataset(
    db_path: str | Path,
    use_all_xref: bool = True,
    include_market_value: bool = True,
    include_mv_pre: bool = False,
    dataset: str = "nt",
    **_: Any,
) -> pd.DataFrame:
    """Minimal dataset builder for training.

    Returns a DataFrame with (at least) columns used by the trainer:
    - transfer_fee, transfer_date, market_value_in_eur
    - team_id, team_name, player_id, player_name
    - minutes, xg_per90, xa_per90, goals_per90, tackles_per90, interceptions_per90
    - age, tm_position, tm_foot, tm_height_cm

    Notes: ignores mv_pre and external CSVs; computes per-90 with cap 180'.
    """
    db_path = str(db_path)
    conn = sqlite3.connect(db_path)
    try:
        # Labels view: euro24 (default) or national tournaments aggregate (nt)
        if dataset == "nt":
            view = (
                "v_players_transfers_nt_window_all"
                if use_all_xref
                else "v_players_transfers_nt_window"
            )
        else:
            view = "v_players_transfers_euro24_window"

        lbl = pd.read_sql_query(
            f"""
            SELECT
              statsbomb_player_id AS player_id,
              player_name,
              national_team_id    AS team_id,
              national_team       AS team_name,
              transfer_date,
              transfer_fee,
              market_value_in_eur,
              tm_date_of_birth,
              tm_position,
              tm_foot,
              tm_height_cm
            FROM {view}
            WHERE transfer_date IS NOT NULL
            """,
            conn,
            parse_dates=["transfer_date", "tm_date_of_birth"],
        )

        # Calculate age at the time of transfer
        lbl["age"] = lbl.apply(
            lambda row: (
                (row["transfer_date"] - row["tm_date_of_birth"]).days // 365
                if pd.notnull(row["transfer_date"])
                and pd.notnull(row["tm_date_of_birth"])
                else np.nan
            ),
            axis=1,
        )

        # Aggregates per player from player_match_stats
        agg = pd.read_sql_query(
            """
            SELECT
              s.player_id,
              SUM(COALESCE(s.minutes,0)) AS minutes,
              SUM(COALESCE(s.xg,0))      AS xg,
              SUM(COALESCE(s.xa,0))      AS xa,
              SUM(COALESCE(s.goals,0))   AS goals,
              SUM(COALESCE(s.tackles,0)) AS tackles,
              SUM(COALESCE(s.interceptions,0)) AS interceptions
            FROM player_match_stats s
            GROUP BY s.player_id
            """,
            conn,
        )

        df = lbl.merge(agg, on="player_id", how="left")

        # Per-90 with cap 180'
        m = (
            pd.Series(pd.to_numeric(df["minutes"], errors="coerce"))
            .fillna(0)
            .astype(float)
        )
        denom = np.where(m < 180.0, 180.0, m)
        for c in ["xg", "xa", "goals", "tackles", "interceptions"]:
            if c in df.columns:
                df[f"{c}_per90"] = (
                    pd.Series(pd.to_numeric(df[c], errors="coerce"))
                    .fillna(0)
                    .astype(float)
                    * 90.0
                    / denom
                )

        # Check for NULL/empty strings and coerce numerics before imputation
        for col in ["tm_position", "tm_foot", "tm_height_cm", "market_value_in_eur"]:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                empty_mask = (
                    df[col].eq("")
                    if df[col].dtype == "object"
                    else pd.Series(False, index=df.index)
                )
                empty_count = int(empty_mask.sum())
                print(
                    f"Feature '{col}' has {null_count} NULL values and {empty_count} empty strings."
                )
                if null_count > 0 or empty_count > 0:
                    if col == "tm_position":
                        df[col] = df[col].fillna("Unknown").mask(empty_mask, "Unknown")
                    elif col == "tm_foot":
                        df[col] = df[col].fillna("right").mask(empty_mask, "right")
                    elif col == "tm_height_cm":
                        # Coerce to numeric (treat empty as NaN), then impute median
                        s = df[col]
                        if s.dtype == "object":
                            s = s.mask(empty_mask, np.nan)
                        s = pd.to_numeric(s, errors="coerce")
                        med = s.median(skipna=True)
                        df[col] = s.fillna(med)
                    elif col == "market_value_in_eur":
                        # Coerce to numeric (treat empty as NaN), then DROP rows with MV missing
                        s = df[col]
                        if s.dtype == "object":
                            s = s.mask(empty_mask, np.nan)
                        s = pd.to_numeric(s, errors="coerce")
                        missing_before = int(s.isna().sum())
                        df[col] = s
                        if missing_before > 0:
                            before_n = len(df)
                            df = df[~df[col].isna()].copy()
                            after_n = len(df)
                            print(
                                f"Dropped {before_n - after_n} rows due to missing market_value_in_eur"
                            )

        if include_market_value and "market_value_in_eur" not in df.columns:
            df["market_value_in_eur"] = np.nan

        return df
    finally:
        conn.close()
