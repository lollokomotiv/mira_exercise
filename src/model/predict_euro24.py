"""
Predict transfer fees for EURO24 window players using a trained residual model.

Usage
  python -m src.model.predict_euro24 \
    --db data/db/euro24.sqlite \
    --model artifacts/model_final_euro24/model.joblib \
    --out artifacts/predictions_euro24.csv \
    [--scope transfers|all] [--age-ref-date 2024-07-01]

Notes
- The trained pipelines in this repo predict the residual on log scale:
    y_resid = log1p(fee) - log1p(market_value_in_eur)
  We reconstruct the fee as: fee_hat = expm1(log1p(MV) + resid_hat).
- If the pipeline is a baseline fallback (zero residual), predictions coincide
  with MV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from src.model.euro24.features import build_dataset
import sqlite3
from datetime import datetime


def _used_input_columns(pipe) -> list[str]:
    used: list[str] = []
    prep = pipe.named_steps.get("prep") if hasattr(pipe, "named_steps") else None
    if prep is None or not hasattr(prep, "transformers_"):
        return used
    for name, trans, cols in prep.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        used.extend(list(cols))
    # Deduplicate preserving order
    seen = set()
    out: list[str] = []
    for c in used:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _get_prep_column_groups(pipe) -> tuple[list[str], list[str]]:
    """Return (num_cols, cat_cols) as learned by the fitted preprocessor."""
    num_cols: list[str] = []
    cat_cols: list[str] = []
    prep = pipe.named_steps.get("prep") if hasattr(pipe, "named_steps") else None
    if prep is not None and hasattr(prep, "transformers_"):
        for name, trans, cols in prep.transformers_:
            if name == "num":
                num_cols = list(cols)
            elif name == "cat":
                cat_cols = list(cols)
    return num_cols, cat_cols

def run(db: str, model_path: str, out_csv: str) -> None:
    # Load model
    pipe = load(model_path)

    # Build dataset depending on scope
    if ARGS.scope == "all":
        df = _build_all_players_dataset(db, age_ref_date=ARGS.age_ref_date)
    else:
        # EURO24 window transfers only
        df = build_dataset(
            db_path=db,
            use_all_xref=True,
            include_market_value=True,
            include_mv_pre=False,
            dataset="euro24",
        )

    # Sanitize numeric/categorical columns to match training expectations
    num_cols, cat_cols = _get_prep_column_groups(pipe)
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].replace("", np.nan), errors="coerce")
    # Basic cleanup for categorical defaults if present
    if "tm_position" in df.columns and (not cat_cols or "tm_position" in cat_cols):
        df["tm_position"] = df["tm_position"].fillna("Unknown").replace("", "Unknown")
    if "tm_foot" in df.columns and (not cat_cols or "tm_foot" in cat_cols):
        df["tm_foot"] = df["tm_foot"].fillna("right").replace("", "right")

    # Columns used by the pipeline
    used_cols = _used_input_columns(pipe)
    # If pipeline has no prep (baseline), used_cols can be empty
    X = df[used_cols] if used_cols else df.iloc[:, :0]

    # Predict residual (or zero if baseline)
    resid_hat = pipe.predict(X)
    mv = pd.to_numeric(df["market_value_in_eur"], errors="coerce").astype(float).values
    pred_log_fee = np.log1p(mv) + resid_hat
    pred_fee = np.expm1(pred_log_fee)

    # Output predictions (ensure StatsBomb player_id is present even if agg missing)
    if "statsbomb_player_id" in df.columns:
        out = pd.DataFrame({"player_id": df["statsbomb_player_id"]})
    elif "player_id" in df.columns:
        out = pd.DataFrame({"player_id": df["player_id"]})
    else:
        out = pd.DataFrame()
    for c in [
        "tm_player_id",
        "player_name",
        "team_name",
        "transfer_date",
        "market_value_in_eur",
        "transfer_fee",
    ]:
        if c in df.columns and c not in out.columns:
            out[c] = df[c]
    out["pred_fee"] = pred_fee
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"✅ Saved predictions → {out_csv}  (rows={len(out)})")


def _build_all_players_dataset(
    db_path: str,
    age_ref_date: str = "2024-07-01",
) -> pd.DataFrame:
    """Build a dataset of all EURO24 players (not only those with transfers).

    Joins StatsBomb players with Transfermarkt tm_players via xref_all to obtain
    demographics and current market value. Aggregates per-90 from player_match_stats.
    Age is computed at a reference date (default 2024-07-01).
    """
    conn = sqlite3.connect(str(db_path))
    try:
        sql_players = """
            SELECT
              sp.player_id      AS statsbomb_player_id,
              sp.player_name,
              sp.team_id        AS team_id,
              sp.team_name      AS team_name
            FROM players sp
        """
        players = pd.read_sql_query(sql_players, conn)

        # Use STRICT xref to avoid false positives when predicting all players
        sql_xref = """
            SELECT statsbomb_player_id, tm_player_id
            FROM player_xref
        """
        xref = pd.read_sql_query(sql_xref, conn)

        sql_tm = """
            SELECT
              player_id           AS tm_player_id,
              date_of_birth       AS tm_date_of_birth,
              position            AS tm_position,
              foot                AS tm_foot,
              height_in_cm        AS tm_height_cm,
              market_value_in_eur AS tm_market_value
            FROM tm_players
        """
        tm = pd.read_sql_query(sql_tm, conn, parse_dates=["tm_date_of_birth"])

        sql_agg = """
            SELECT
              s.player_id,
              SUM(COALESCE(s.minutes,0)) AS minutes,
              SUM(COALESCE(s.xg,0))      AS xg,
              SUM(COALESCE(s.xa,0))      AS xa,
              SUM(COALESCE(s.goals,0))   AS goals,
              SUM(COALESCE(s.tackles,0)) AS tackles,
              SUM(COALESCE(s.interceptions,0)) AS interceptions,
              SUM(COALESCE(s.pressures,0)) AS pressures
            FROM player_match_stats s
            GROUP BY s.player_id
        """
        agg = pd.read_sql_query(sql_agg, conn)

        # Keep ONLY players present in STRICT XREF (inner join), as requested
        df = players.merge(xref, on="statsbomb_player_id", how="inner").merge(
            tm, on="tm_player_id", how="left"
        )
        df = df.merge(agg, left_on="statsbomb_player_id", right_on="player_id", how="left")

        # Age at reference date
        ref = pd.to_datetime(age_ref_date)
        dob = pd.to_datetime(df.get("tm_date_of_birth"), errors="coerce")
        df["age"] = ((ref - dob).dt.days / 365.25).astype(float)

        # Per-90 with cap 180'
        m = pd.to_numeric(df.get("minutes"), errors="coerce").fillna(0).astype(float)
        denom = np.where(m < 180.0, 180.0, m)
        for c in ["xg", "xa", "goals", "tackles", "interceptions", "pressures"]:
            if c in df.columns:
                df[f"{c}_per90"] = pd.to_numeric(df.get(c), errors="coerce").fillna(0).astype(float) * 90.0 / denom

        # Align MV column name to expected one
        df["market_value_in_eur"] = pd.to_numeric(df.get("tm_market_value"), errors="coerce")

        return df
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Predict EURO24 transfer fees with trained model")
    p.add_argument("--db", default="data/db/euro24.sqlite")
    p.add_argument(
        "--model", default="artifacts/model_final_euro24/model.joblib", help="Path to trained model.joblib"
    )
    p.add_argument("--out", default="artifacts/predictions_euro24.csv")
    p.add_argument("--scope", choices=["transfers", "all"], default="transfers", help="Predict on transfers only or all EURO24 players")
    p.add_argument("--age-ref-date", default="2024-07-01", help="Reference date to compute age for 'all' scope")
    args = p.parse_args(argv)
    global ARGS
    ARGS = args
    run(args.db, args.model, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
