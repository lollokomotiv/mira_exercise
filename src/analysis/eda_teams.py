"""
EDA by team (EURO24) using Altair.

This version focuses ONLY on aggregated team totals, not per-player
distributions or per-player means. Kept teams must have at least 4 matches.

Inputs:
- features CSV (prefer features_all.csv for full roster coverage)
- SQLite DB path to compute matches per team

Outputs: artifacts/eda_teams/
- bar_team_sum_<metric>.html (xg, xa, shots, goals, tackles, interceptions, dribbles, pressures, clearances, passes, passes_completed)
- scatter_team_sum_xg_vs_xa.html
 - scatter_team_avg_fee_vs_avg_mv.html (avg transfer fee vs avg market value by national team)

Usage:
  python -m src.analysis.eda_teams \
    --features data/processed/model/features_all.csv \
    --db data/db/euro24.sqlite \
    --out artifacts/eda_teams \
    --min-minutes 180
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd


DEFAULT_DB = "data/db/euro24.sqlite"
DEFAULT_FEATURES = "data/processed/model/features_all.csv"


def teams_with_min_matches(db_path: str | Path, min_matches: int = 4) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        sql = """
        WITH t AS (
          SELECT home_team_id AS team_id, home_team_name AS team_name FROM matches
          UNION ALL
          SELECT away_team_id AS team_id, away_team_name AS team_name FROM matches
        )
        SELECT team_id, team_name, COUNT(*) AS matches_played
        FROM t
        GROUP BY team_id, team_name
        HAVING COUNT(*) >= ?
        ORDER BY matches_played DESC
        """
        df = pd.read_sql_query(sql, conn, params=(min_matches,))
        return df
    finally:
        conn.close()


# Removed per-player distribution/means and per90 aggregations per request.


def alt_bar_team_sum(df: pd.DataFrame, metric_total: str, out_dir: Path) -> None:
    """Bar chart with team sums for a total metric (e.g., xg, xa, shots)."""
    if metric_total not in df.columns:
        return
    g = (
        df.groupby(['team_id','team_name'], as_index=False)[metric_total]
          .sum()
          .sort_values(metric_total, ascending=False)
    )
    bars = alt.Chart(g).mark_bar().encode(
        x=alt.X('team_name:N', sort='-y', title='Team'),
        y=alt.Y(f'{metric_total}:Q', title=f'Sum {metric_total}'),
        tooltip=[alt.Tooltip(f'{metric_total}:Q', format=',.2f')]
    )
    labels = alt.Chart(g).mark_text(align='left', dx=3, dy=0, baseline='middle', fontSize=11).encode(
        x=alt.X('team_name:N', sort='-y'),
        y=alt.Y(f'{metric_total}:Q'),
        text=alt.Text(f'{metric_total}:Q', format=',.2f')
    )
    (bars + labels).properties(width=650, height=380, title=f'Team sum {metric_total}').save(out_dir / f'bar_team_sum_{metric_total}.html')


def alt_scatter_team_per90_xg_vs_xa(df: pd.DataFrame, keep_df: pd.DataFrame, out_dir: Path) -> None:
    """Scatter of team xG/90 vs xA/90 computed from totals divided by matches played."""
    if not {'xg','xa'}.issubset(df.columns):
        return
    totals = df.groupby(['team_id','team_name'], as_index=False)[['xg','xa']].sum()
    g = totals.merge(keep_df[['team_id','team_name','matches_played']], on=['team_id','team_name'], how='inner')
    g = g[g['matches_played'] > 0].copy()
    g['xg_per90_team'] = g['xg'] / g['matches_played']
    g['xa_per90_team'] = g['xa'] / g['matches_played']
    points = alt.Chart(g).mark_circle(size=110, color='#4682B4', opacity=0.85).encode(
        x=alt.X('xg_per90_team:Q', title='Team xG per 90 (matches-based)'),
        y=alt.Y('xa_per90_team:Q', title='Team xA per 90 (matches-based)'),
        tooltip=['team_name:N', alt.Tooltip('xg_per90_team:Q', format=',.2f'), alt.Tooltip('xa_per90_team:Q', format=',.2f'), 'matches_played:Q']
    )
    labels = alt.Chart(g).mark_text(align='left', dx=7, dy=0, fontSize=11).encode(
        x='xg_per90_team:Q', y='xa_per90_team:Q', text='team_name:N'
    )
    (points + labels).properties(width=650, height=380, title='Team xG/90 vs xA/90 (matches-based)').save(out_dir / 'scatter_team_xg_per90_vs_xa_per90.html')


def alt_scatter_team_per90_pair(df: pd.DataFrame, keep_df: pd.DataFrame, metric_x: str, metric_y: str, out_dir: Path) -> None:
    """Generic team-level per-90 scatter using matches-based denominator.

    per90_team(metric) = sum(metric_total) / matches_played
    """
    if not {metric_x, metric_y}.issubset(df.columns):
        return
    totals = df.groupby(['team_id','team_name'], as_index=False)[[metric_x, metric_y]].sum()
    g = totals.merge(keep_df[['team_id','team_name','matches_played']], on=['team_id','team_name'], how='inner')
    g = g[g['matches_played'] > 0].copy()
    g[f'{metric_x}_per90_team'] = g[metric_x] / g['matches_played']
    g[f'{metric_y}_per90_team'] = g[metric_y] / g['matches_played']
    title = f"Team {metric_x}/90 vs {metric_y}/90 (matches-based)"
    fname = f"scatter_team_{metric_x}_per90_vs_{metric_y}_per90.html"
    points = alt.Chart(g).mark_circle(size=110, color='#6C8EBF', opacity=0.85).encode(
        x=alt.X(f'{metric_x}_per90_team:Q', title=f'{metric_x} per 90 (team)'),
        y=alt.Y(f'{metric_y}_per90_team:Q', title=f'{metric_y} per 90 (team)'),
        tooltip=['team_name:N', alt.Tooltip(f'{metric_x}_per90_team:Q', format=',.2f'), alt.Tooltip(f'{metric_y}_per90_team:Q', format=',.2f'), 'matches_played:Q']
    )
    labels = alt.Chart(g).mark_text(align='left', dx=7, dy=0, fontSize=11).encode(
        x=alt.X(f'{metric_x}_per90_team:Q'), y=alt.Y(f'{metric_y}_per90_team:Q'), text='team_name:N'
    )
    (points + labels).properties(width=650, height=380, title=title).save(out_dir / fname)


def alt_bar_team_per90_from_matches(df: pd.DataFrame, keep_df: pd.DataFrame, metric_total: str, out_dir: Path) -> None:
    """Aggregated team metric per 90 mins using matches played (team-level minutes ≈ matches*90).

    per90_team = sum(metric_total) / matches_played
    """
    if metric_total not in df.columns:
        return
    totals = df.groupby(['team_id','team_name'], as_index=False)[metric_total].sum()
    merged = totals.merge(keep_df[['team_id','matches_played','team_name']], on=['team_id','team_name'], how='inner')
    merged = merged[merged['matches_played'] > 0].copy()
    merged[f'{metric_total}_per90_team'] = merged[metric_total] / merged['matches_played']
    merged = merged.sort_values(f'{metric_total}_per90_team', ascending=False)
    bars = alt.Chart(merged).mark_bar().encode(
        x=alt.X('team_name:N', sort='-y', title='Team'),
        y=alt.Y(f'{metric_total}_per90_team:Q', title=f'{metric_total} per 90 (team)'),
        tooltip=['team_name:N', alt.Tooltip(f'{metric_total}_per90_team:Q', format=',.2f'), 'matches_played:Q']
    )
    labels = alt.Chart(merged).mark_text(align='left', dx=3, dy=0, baseline='middle', fontSize=11).encode(
        x=alt.X('team_name:N', sort='-y'),
        y=alt.Y(f'{metric_total}_per90_team:Q'),
        text=alt.Text(f'{metric_total}_per90_team:Q', format=',.2f')
    )
    title = f'Team {metric_total}/90 (using matches count)'
    (bars + labels).properties(width=650, height=380, title=title).save(out_dir / f'bar_team_{metric_total}_per90.html')


def alt_scatter_team_avg_fee_vs_avg_mv(db_path: str | Path, out_dir: Path) -> None:
    """Scatter plot: for players with a transfer in the window, group by national team
    and plot average transfer fee (y) vs average market value (x).

    Data source: view v_players_transfers_euro24_window_all created in the schema.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        sql = """
        SELECT
          national_team_id AS team_id,
          national_team    AS team_name,
          transfer_fee,
          market_value_in_eur
        FROM v_players_transfers_euro24_window_all
        WHERE transfer_fee IS NOT NULL
        """
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()

    if df.empty:
        return
    df["transfer_fee"] = pd.to_numeric(df["transfer_fee"], errors="coerce")
    df["market_value_in_eur"] = pd.to_numeric(df["market_value_in_eur"], errors="coerce")
    m = df[(df["transfer_fee"] > 0) & (df["market_value_in_eur"].fillna(0) > 0)].copy()
    if m.empty:
        return
    agg = (
        m.groupby(["team_id", "team_name"], as_index=False)
         .agg(
            avg_transfer_fee=("transfer_fee", "mean"),
            avg_mv=("market_value_in_eur", "mean"),
            n_transfers=("transfer_fee", "count"),
         )
    )
    # Scatter with labels and a y=x reference line
    scatter = alt.Chart(agg).mark_circle(size=120, opacity=0.85, color="#3749A6").encode(
        x=alt.X("avg_mv:Q", title="Average market value (€)"),
        y=alt.Y("avg_transfer_fee:Q", title="Average transfer fee (€)"),
        size=alt.Size("n_transfers:Q", title="# transfers", scale=alt.Scale(range=[60, 400])),
        tooltip=[
            "team_name:N",
            alt.Tooltip("avg_mv:Q", format=",.0f"),
            alt.Tooltip("avg_transfer_fee:Q", format=",.0f"),
            alt.Tooltip("n_transfers:Q", format=",d"),
        ],
    )
    labels = alt.Chart(agg).mark_text(align="left", dx=7, dy=0, fontSize=11).encode(
        x="avg_mv:Q", y="avg_transfer_fee:Q", text="team_name:N"
    )
    # y=x line (reference)
    max_val = float(max(agg["avg_mv"].max(), agg["avg_transfer_fee"].max()))
    ref = pd.DataFrame({"x": [0, max_val], "y": [0, max_val]})
    line = alt.Chart(ref).mark_rule(color="#999", strokeDash=[6, 6]).encode(x="x:Q", y="y:Q")
    (scatter + labels + line).properties(
        width=720, height=460, title="Avg transfer fee vs Avg market value by national team"
    ).save(out_dir / "scatter_team_avg_fee_vs_avg_mv.html")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description='EDA by team (EURO24) with Altair')
    p.add_argument('--features', default=DEFAULT_FEATURES, help='Path to features_all.csv (or features.csv)')
    p.add_argument('--db', default=DEFAULT_DB, help='SQLite DB path to read matches')
    p.add_argument('--out', default='artifacts/eda_teams', help='Output directory')
    p.add_argument('--min-minutes', type=float, default=180.0, help='Min minutes to include in per90 metrics')
    p.add_argument('--min-matches', type=int, default=4, help='Min matches played to keep a team')
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Eligible teams (>= min matches)
    keep = teams_with_min_matches(args.db, min_matches=args.min_matches)
    keep_ids = set(keep['team_id'].tolist())

    df = pd.read_csv(args.features)
    # Filter by eligible teams (no minutes filter; sums require all players)
    df = df[df['team_id'].isin(keep_ids)].copy()

    # Team per-90 scatter (matches-based)
    alt_scatter_team_per90_xg_vs_xa(df, keep, out_dir)
    # Additional per-90 scatters as requested
    alt_scatter_team_per90_pair(df, keep, 'tackles', 'interceptions', out_dir)
    alt_scatter_team_per90_pair(df, keep, 'passes', 'interceptions', out_dir)
    # dribbles vs assists (fallback to xa if assists is missing)
    metric_assist = 'assists' if 'assists' in df.columns else 'xa'
    alt_scatter_team_per90_pair(df, keep, 'dribbles', metric_assist, out_dir)

    # Team sum bars for key totals
    for tot in ['xg','xa','shots','goals','tackles','interceptions','dribbles','pressures','clearances','passes','passes_completed']:
        alt_bar_team_sum(df[['team_id','team_name',tot]].dropna(), tot, out_dir)

    # Per 90 using matches count for xg/xa (team-level per90)
    alt_bar_team_per90_from_matches(df[['team_id','team_name','xg']].dropna(), keep, 'xg', out_dir)
    alt_bar_team_per90_from_matches(df[['team_id','team_name','xa']].dropna(), keep, 'xa', out_dir)

    # Avg transfer fee vs avg MV by national team (only players with a transfer)
    alt_scatter_team_avg_fee_vs_avg_mv(args.db, out_dir)

    print(f'✅ Team EDA artifacts saved to {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
