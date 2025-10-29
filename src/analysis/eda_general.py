"""
Focused EDA for EURO24 transfers (post-EURO window).

Outputs (default): artifacts/eda/
- missing_summary.csv (dataset-level, per column)
- table_counts.csv (DB tables overview)
- scatter_log_fee_vs_log_mv_post.(png|html)
- scatter_mv_pre_vs_mv_post.(png|html)
 - bar_height_hist.html (if height available through features CSV)

Usage:
  python -m src.analysis.eda \
    --db data/db/euro24.sqlite \
    --valuations-csv data/raw/transfermarkt/player_valuations.csv \
    --mv-cutoff-date 2024-06-13 \
    --out artifacts/eda_general \
    [--features data/processed/model/features_all.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from pyecharts.charts import Scatter, Bar, HeatMap
from pyecharts import options as opts
from pyecharts.commons.utils import JsCode
import json


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _table_counts(db_path: str, out_dir: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        tbls = [
            "teams",
            "players",
            "matches",
            "player_match_stats",
            "player_appearances",
            "events_meta",
            "transfers",
            "player_xref",
            "player_xref_all",
            "club_xref",
        ]
        rows = []
        for t in tbls:
            try:
                n = pd.read_sql_query(f"SELECT COUNT(*) AS n FROM {t}", conn).iloc[0, 0]
            except Exception:
                n = None
            rows.append({"table": t, "rows": n})
        pd.DataFrame(rows).to_csv(out_dir / "table_counts.csv", index=False)
    finally:
        conn.close()


def _missing_summary(df: pd.DataFrame, out_dir: Path) -> None:
    miss = df.isna().sum().to_frame("missing")
    miss["pct"] = (miss["missing"] / len(df) * 100).round(2)
    miss.reset_index().rename(columns={"index": "column"}).to_csv(
        out_dir / "missing_summary.csv", index=False
    )


def _load_transfers_from_db(db_path: str, euro24_only: bool = False) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        if euro24_only:
            sql = """
            SELECT 
              t.player_id AS tm_player_id,
              v.statsbomb_player_id,
              t.player_name,
              t.transfer_date,
              t.transfer_fee,
              t.market_value_in_eur
            FROM transfers_euro24_window t
            INNER JOIN v_players_transfers_euro24_window_all v
              ON v.tm_player_id = t.player_id
            """
        else:
            sql = """
            SELECT 
              t.player_id AS tm_player_id,
              t.player_name,
              t.transfer_date,
              t.transfer_fee,
              t.market_value_in_eur
            FROM transfers_euro24_window t
            """
        df = pd.read_sql_query(sql, conn, parse_dates=["transfer_date"])
        df["transfer_fee"] = _to_num(df["transfer_fee"])
        df["market_value_in_eur"] = _to_num(df["market_value_in_eur"])
        return df
    finally:
        conn.close()


def _load_player_aggregates_from_db(db_path: str) -> pd.DataFrame:
    """Aggregate StatsBomb player stats directly from SQLite for EDA.

    Returns per-StatsBomb player_id totals and per90 metrics.
    """
    conn = sqlite3.connect(db_path)
    try:
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
            SUM(COALESCE(s.clearances,0)) AS clearances
          FROM player_match_stats s
          GROUP BY s.player_id
        )
        SELECT * FROM stats
        """
        df = pd.read_sql_query(sql, conn)
        # per90 with min 180'
        mins = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)
        denom = mins.clip(lower=180.0)
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
        ]:
            df[f"{col}_per90"] = (
                pd.to_numeric(df[col], errors="coerce").fillna(0.0) * 90.0 / denom
            ).astype(float)
        return df.rename(columns={"player_id": "statsbomb_player_id"})
    finally:
        conn.close()


def _load_mv_pre(csv_path: str, cutoff: str = "2024-06-13") -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        return pd.DataFrame(columns=["tm_player_id", "mv_pre_euro", "mv_pre_euro_date"])
    df = pd.read_csv(p)
    if not {"player_id", "date", "market_value_in_eur"}.issubset(df.columns):
        return pd.DataFrame(columns=["tm_player_id", "mv_pre_euro", "mv_pre_euro_date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()
    df = df[df["date"] <= pd.to_datetime(cutoff)]
    df = df.sort_values(["player_id", "date"]).groupby("player_id").tail(1)
    out = df.rename(
        columns={
            "player_id": "tm_player_id",
            "market_value_in_eur": "mv_pre_euro",
            "date": "mv_pre_euro_date",
        }
    )[["tm_player_id", "mv_pre_euro", "mv_pre_euro_date"]]
    out["mv_pre_euro"] = _to_num(out["mv_pre_euro"])
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="EDA for EURO24 transfers (post‑EURO window)"
    )
    p.add_argument(
        "--db",
        default="data/db/euro24.sqlite",
        help="SQLite DB path (reads transfers_euro24_window)",
    )
    p.add_argument(
        "--valuations-csv",
        default="data/raw/transfermarkt/player_valuations.csv",
        help="Transfermarkt valuations CSV (for mv_pre_euro)",
    )
    p.add_argument(
        "--mv-cutoff-date",
        default="2024-06-13",
        help="Cutoff date for pre‑EURO MV (YYYY-MM-DD)",
    )
    p.add_argument(
        "--out",
        default="artifacts/eda_general",
        help="Output directory for figures and summaries",
    )
    p.add_argument(
        "--euro24-only",
        action="store_true",
        help="Filter to transfers of EURO24 players only",
    )
    p.add_argument(
        "--tm-players-csv",
        default="data/raw/transfermarkt/players.csv",
        help="Transfermarkt players.csv (for age)",
    )
    p.add_argument(
        "--age-ref-date",
        default="2024-07-01",
        help="Reference date for age in YYYY-MM-DD",
    )
    p.add_argument(
        "--features",
        default="",
        help="Optional features CSV (e.g., features_all.csv) to enrich with per-90 metrics and demographics",
    )
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset: transfers in window; optionally restrict to EURO24 players only
    base = _load_transfers_from_db(args.db, euro24_only=args.euro24_only)
    mv = _load_mv_pre(args.valuations_csv, cutoff=args.mv_cutoff_date)
    df = base.merge(mv, on="tm_player_id", how="left")

    # Add age and TM demographics from players.csv (position/foot/height)
    try:
        tm_players_path = Path(args.tm_players_csv)
        if tm_players_path.exists():
            tm_df = pd.read_csv(tm_players_path)
            if {"player_id"}.issubset(tm_df.columns):
                out_cols = {"player_id": "tm_player_id"}
                if "position" in tm_df.columns:
                    out_cols["position"] = "tm_position"
                if "foot" in tm_df.columns:
                    out_cols["foot"] = "tm_foot"
                if "height_in_cm" in tm_df.columns:
                    out_cols["height_in_cm"] = "tm_height_cm"
                tmp = tm_df.rename(columns=out_cols)[list(out_cols.values())].copy()
                df = df.merge(tmp, on="tm_player_id", how="left")
                if "date_of_birth" in tm_df.columns:
                    dob = pd.to_datetime(tm_df["date_of_birth"], errors="coerce")
                    ref = pd.to_datetime(args.age_ref_date)
                    age_years = ((ref - dob).dt.days / 365.25).astype(float)
                    df = df.merge(
                        pd.DataFrame(
                            {"tm_player_id": tm_df["player_id"], "age_years": age_years}
                        ),
                        on="tm_player_id",
                        how="left",
                    )
    except Exception as e:
        print(f"ℹ️  Skipping age merge: {e}")

    # Optionally merge player features (per-90, minutes, age)
    if args.features:
        feat_path = Path(args.features)
        if feat_path.exists():
            feats = pd.read_csv(feat_path)
            # Expect tm_player_id to merge on; if missing, skip
            if "tm_player_id" in feats.columns:
                cols_keep = [
                    c
                    for c in [
                        "tm_player_id",
                        "minutes",
                        "age_years",
                        "age",
                        "xg_per90",
                        "xa_per90",
                        "goals_per90",
                        "shots_per90",
                        "tackles_per90",
                        "dribbles_per90",
                        "pressures_per90",
                        "interceptions_per90",
                    ]
                    if c in feats.columns
                ]
                feats = feats[cols_keep].copy()
                if "age" in feats.columns and "age_years" not in feats.columns:
                    feats["age_years"] = pd.to_numeric(feats["age"], errors="coerce")
                df = df.merge(feats, on="tm_player_id", how="left")
            else:
                print(
                    "ℹ️  features CSV provided but 'tm_player_id' not found; skipping feature merge."
                )
    else:
        # No features CSV provided: compute basic per-90 aggregates directly from DB
        try:
            agg = _load_player_aggregates_from_db(args.db)
            # Prefer direct join if we have statsbomb_player_id; otherwise map via xref_all
            if "statsbomb_player_id" in df.columns:
                df = df.merge(agg, on="statsbomb_player_id", how="left")
            else:
                # map tm->statsbomb via xref_all
                conn = sqlite3.connect(args.db)
                try:
                    m = pd.read_sql_query(
                        "SELECT statsbomb_player_id, tm_player_id FROM player_xref_all",
                        conn,
                    ).dropna()
                finally:
                    conn.close()
                df = df.merge(
                    m, left_on="tm_player_id", right_on="tm_player_id", how="left"
                )
                df = df.merge(agg, on="statsbomb_player_id", how="left")
        except Exception as e:
            print(f"ℹ️  Skipping on-the-fly aggregates: {e}")

    # Coalesce duplicated columns that may arise from merges (e.g., age_years_x/_y)
    def _coalesce_column(df_in: pd.DataFrame, base: str) -> pd.DataFrame:
        if base in df_in.columns:
            return df_in
        candidates = [c for c in df_in.columns if c == base or c.startswith(base + "_")]
        if not candidates:
            return df_in
        s = None
        for c in candidates:
            col = (
                pd.to_numeric(df_in[c], errors="coerce")
                if df_in[c].dtype != object
                else df_in[c]
            )
            s = col if s is None else s.combine_first(col)
        if s is not None:
            df_in[base] = s
        return df_in

    df = _coalesce_column(df, "age_years")

    # Missing summary and table counts
    _missing_summary(df, out_dir)
    _table_counts(args.db, out_dir)

    # Optional basic distributions for new model features
    def _hist(col: str, title: str, out_name: str) -> None:
        if col not in df.columns:
            return
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            return
        counts, edges = np.histogram(s.values.astype(float), bins=20)
        labels = [
            f"{round(edges[i])}–{round(edges[i+1])}" for i in range(len(edges) - 1)
        ]
        bar = (
            Bar(init_opts=opts.InitOpts(width="800px", height="520px"))
            .add_xaxis(labels)
            .add_yaxis("count", counts.tolist())
            .set_global_opts(
                title_opts=opts.TitleOpts(title=title),
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=15)),
                yaxis_opts=opts.AxisOpts(name="count"),
                datazoom_opts=[opts.DataZoomOpts(type_="inside"), opts.DataZoomOpts()],
            )
        )
        bar.render(str(out_dir / out_name))

    # Age: from features CSV (age_years) or from TM merge above
    if "age_years" in df.columns:
        _hist("age_years", "Age distribution", "bar_age_hist.html")
    # Height: preferably from features CSV
    if "tm_height_cm" in df.columns:
        _hist("tm_height_cm", "Height (cm) distribution", "bar_height_hist.html")

    # 1) fee vs MV post (piecewise axis to reduce crowding at low values)
    # Define buckets: equal visual width per range
    bounds = [0, 2_000_000, 5_000_000, 20_000_000, 50_000_000, 100_000_000]

    def _piecewise_transform(values: pd.Series) -> list[float]:
        v = _to_num(values).astype(float).clip(bounds[0], bounds[-1])
        res: list[float] = []
        for val in v:
            for i in range(len(bounds) - 1):
                lo, hi = bounds[i], bounds[i + 1]
                if lo <= val <= hi:
                    frac = (val - lo) / (hi - lo) if hi > lo else 0.0
                    res.append(round(i + frac, 6))
                    break
        return res

    # Axis label: show bucket labels on integer ticks
    label_js = JsCode(
        """
        function (val) {
            var idx = Math.round(val);
            if (Math.abs(val - idx) < 0.05) {
                var labs = ['0M','2M','5M','20M','50M','100M'];
                if (idx >= 0 && idx < labs.length) return labs[idx];
            }
            return '';
        }
    """
    )
    m1 = df[(df["transfer_fee"] > 0) & (df["market_value_in_eur"].fillna(0) > 0)].copy()
    if len(m1) >= 3:
        x_bucket = _piecewise_transform(m1["market_value_in_eur"])  # 0..5
        y_bucket = _piecewise_transform(m1["transfer_fee"])  # 0..5
        names = m1["player_name"].astype(str).tolist()
        mv_raw = _to_num(m1["market_value_in_eur"]).astype(float).round(0).tolist()
        fee_raw = _to_num(m1["transfer_fee"]).astype(float).round(0).tolist()
        chart = (
            Scatter(init_opts=opts.InitOpts(width="900px", height="560px"))
            .add_xaxis(x_bucket)
            .add_yaxis(
                "players",
                y_bucket,
                symbol_size=6,
                label_opts=opts.LabelOpts(
                    is_show=True,
                    position="right",
                    formatter=JsCode("function(p){return names[p.dataIndex];}"),
                ),
            )
            .add_js_funcs(
                f"var names = {json.dumps(names)};"
                + f"var mv_raw = {json.dumps(mv_raw)};"
                + f"var fee_raw = {json.dumps(fee_raw)};"
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="fee vs mv_post (bucketed axis)"),
                xaxis_opts=opts.AxisOpts(
                    name="mv_post",
                    type_="value",
                    min_=0,
                    max_=5,
                    axislabel_opts=opts.LabelOpts(formatter=label_js),
                ),
                yaxis_opts=opts.AxisOpts(
                    name="fee",
                    type_="value",
                    min_=0,
                    max_=5,
                    axislabel_opts=opts.LabelOpts(formatter=label_js),
                ),
                tooltip_opts=opts.TooltipOpts(
                    trigger="item",
                    formatter=JsCode(
                        """
                        function (p) {
                            var i = p.dataIndex;
                            function fmt(n){ try { return '€' + Math.round(n).toLocaleString(); } catch(e){ return '€' + Math.round(n); } }
                            return names[i] + '<br/>' + 'mv_post: ' + fmt(mv_raw[i]) + '<br/>' + 'fee: ' + fmt(fee_raw[i]);
                        }
                        """
                    ),
                ),
            )
        )
        chart.render(str(out_dir / "scatter_fee_vs_mv_post.html"))

    # 2) MV pre vs MV post (log axes)
    m2 = df[
        (df["mv_pre_euro"].fillna(0) > 0) & (df["market_value_in_eur"].fillna(0) > 0)
    ].copy()
    if len(m2) >= 3:
        x2 = _to_num(m2["mv_pre_euro"]).astype(float).round(6).tolist()
        y2 = _to_num(m2["market_value_in_eur"]).astype(float).round(6).tolist()
        chart2 = (
            Scatter(init_opts=opts.InitOpts(width="800px", height="520px"))
            .add_xaxis(x2)
            .add_yaxis(
                "players",
                y2,
                symbol_size=6,
                label_opts=opts.LabelOpts(is_show=False),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="MV pre‑EURO vs MV post (log axes)"),
                xaxis_opts=opts.AxisOpts(name="MV pre‑EURO", type_="log"),
                yaxis_opts=opts.AxisOpts(name="MV post", type_="log"),
                tooltip_opts=opts.TooltipOpts(trigger="item"),
            )
        )
        chart2.render(str(out_dir / "scatter_mv_pre_vs_mv_post.html"))

    # 3) Correlation of features with future label (fee)
    corr_rows = []
    supervised = df[(df["transfer_fee"].notna()) & (df["transfer_fee"] > 0)].copy()
    if len(supervised) >= 5:
        y = _to_num(supervised["transfer_fee"]).astype(float)
        candidate_cols = [
            c
            for c in [
                "market_value_in_eur",
                "mv_pre_euro",
                # demographics/ctx
                "age_years",
                "age",
                "team_id",
                "tm_height_cm",
                "tm_position",
                "tm_foot",
                # minutes + per90
                "minutes",
                "xg_per90",
                "xa_per90",
                "goals_per90",
                "shots_per90",
                "tackles_per90",
                "dribbles_per90",
                "pressures_per90",
                "interceptions_per90",
            ]
            if c in supervised.columns
        ]
        # prefer age_years if both age and age_years exist
        if "age_years" in candidate_cols and "age" in candidate_cols:
            candidate_cols.remove("age")
        for c in candidate_cols:
            if supervised[c].dtype == "object":
                # Encode categoricals to numeric codes for correlation
                x = (
                    supervised[c]
                    .astype("category")
                    .cat.codes.replace({-1: np.nan})
                    .astype(float)
                )
            else:
                x = _to_num(supervised[c]).astype(float)
            valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
            if valid.sum() >= 3:
                r = float(np.corrcoef(x[valid], y[valid])[0, 1])
                corr_rows.append({"feature": c, "pearson_with_fee": r})
        if corr_rows:
            corr_df = pd.DataFrame(corr_rows).sort_values(
                "pearson_with_fee", ascending=False
            )
            corr_df.to_csv(out_dir / "correlations_with_fee.csv", index=False)
            # Bar chart (descending)
            bar = (
                Bar(init_opts=opts.InitOpts(width="860px", height="520px"))
                .add_xaxis(corr_df["feature"].tolist())
                .add_yaxis(
                    "pearson r",
                    corr_df["pearson_with_fee"].round(4).tolist(),
                    category_gap="35%",
                )
                .set_global_opts(
                    title_opts=opts.TitleOpts(
                        title="Correlation with fee (descending)"
                    ),
                    xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=15)),
                    yaxis_opts=opts.AxisOpts(name="r"),
                    datazoom_opts=[
                        opts.DataZoomOpts(type_="inside"),
                        opts.DataZoomOpts(),
                    ],
                )
            )
            bar.render(str(out_dir / "bar_correlations_with_fee.html"))

            # Build a correlation matrix for the most important parameters
            try:
                # Select top-N features by absolute correlation with fee
                top_n = 10
                top_feats = (
                    corr_df.reindex(
                        corr_df["pearson_with_fee"].abs().sort_values(ascending=False).index
                    )["feature"].head(top_n).tolist()
                )
                matrix_cols = ["transfer_fee"] + top_feats

                # Prepare a numeric dataframe for correlation
                mat = supervised[matrix_cols].copy()
                for c in mat.columns:
                    if mat[c].dtype == "object":
                        mat[c] = (
                            mat[c].astype("category").cat.codes.replace({-1: np.nan}).astype(float)
                        )
                    else:
                        mat[c] = _to_num(mat[c]).astype(float)
                corr_m = mat.corr(method="pearson")
                # Persist CSV
                corr_csv = out_dir / "correlation_matrix_top_features.csv"
                corr_m.to_csv(corr_csv)

                # Heatmap visualization
                xlabs = corr_m.columns.tolist()
                ylabs = corr_m.index.tolist()
                data = [
                    [i, j, float(round(corr_m.iloc[i, j], 3))]
                    for i in range(len(ylabs))
                    for j in range(len(xlabs))
                ]
                hm = (
                    HeatMap(init_opts=opts.InitOpts(width="900px", height="720px"))
                    .add_xaxis(xlabs)
                    .add_yaxis("corr", ylabs, data, label_opts=opts.LabelOpts(is_show=False))
                    .set_global_opts(
                        title_opts=opts.TitleOpts(title="Correlation Matrix — Top Features + Fee"),
                        visualmap_opts=opts.VisualMapOpts(min_=-1, max_=1),
                        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=20)),
                    )
                )
                hm.render(str(out_dir / "heatmap_correlation_top_features.html"))
            except Exception as e:
                print(f"ℹ️  Skipping correlation matrix: {e}")

            # Also produce a variant excluding MV features to highlight performance-driven signals
            corr_df_nomv = corr_df[
                ~corr_df["feature"].isin(["market_value_in_eur", "mv_pre_euro"])
            ].copy()
            if not corr_df_nomv.empty:
                corr_df_nomv.to_csv(
                    out_dir / "correlations_with_fee_no_mv.csv", index=False
                )
                bar2 = (
                    Bar(init_opts=opts.InitOpts(width="860px", height="520px"))
                    .add_xaxis(corr_df_nomv["feature"].tolist())
                    .add_yaxis(
                        "pearson r",
                        corr_df_nomv["pearson_with_fee"].round(4).tolist(),
                        category_gap="35%",
                    )
                    .set_global_opts(
                        title_opts=opts.TitleOpts(
                            title="Correlation with fee (no MV features)"
                        ),
                        xaxis_opts=opts.AxisOpts(
                            axislabel_opts=opts.LabelOpts(rotate=15)
                        ),
                        yaxis_opts=opts.AxisOpts(name="r"),
                        datazoom_opts=[
                            opts.DataZoomOpts(type_="inside"),
                            opts.DataZoomOpts(),
                        ],
                    )
                )
                bar2.render(str(out_dir / "bar_correlations_with_fee_no_mv.html"))

    # Calculate the average difference between transfer_fee and market_value_in_eur
    def calculate_fee_mv_difference(df: pd.DataFrame, out_dir: Path) -> float:
        filtered_df = df[
            (df["transfer_fee"] > 0) & (df["market_value_in_eur"].fillna(0) > 0)
        ]
        if filtered_df.empty:
            print("ℹ️ No valid data to calculate fee vs market value difference.")
            return 0.0
        # Fix SettingWithCopyWarning by using .loc
        filtered_df = filtered_df.copy()
        filtered_df.loc[:, "fee_mv_diff"] = (
            filtered_df["transfer_fee"] - filtered_df["market_value_in_eur"]
        )
        avg_difference = filtered_df["fee_mv_diff"].mean()
        print(f"✅ Average fee vs market value difference: {avg_difference}")

        # Save the result to a CSV file
        result_file = out_dir / "average_fee_mv_difference.csv"
        pd.DataFrame({"average_fee_mv_difference": [avg_difference]}).to_csv(
            result_file, index=False
        )
        print(f"✅ Result saved to {result_file}")

        return avg_difference

    # Call the function (message printed inside)
    _ = calculate_fee_mv_difference(df, out_dir)

    print(f"✅ EDA artifacts saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
