"""
EDA focused on EURO24 player performance metrics (independent of transfers).

Reads the features CSV (ideally features_all.csv) and creates Pyecharts plots
focusing on minutes and per-90 stats without grouping by position, plus top-10
leaderboards per metric.

Outputs (default: artifacts/eda_players/):
 - hist_minutes.png
 - violin_minutes_by_pos.png
 - box_xg_per90_by_pos.png
 - box_xa_per90_by_pos.png
 - box_shots_per90_by_pos.png
 - box_tackles_per90_by_pos.png
 - box_dribbles_per90_by_pos.png
 - corr_heatmap_per90.png
 - scatter_xg_per90_vs_goals_per90.png
 - scatter_pass_pct_vs_passes_completed_per90.png
- bar_top_xg_per90.png (and others for each metric)
 - bar_age_hist.html (if age/age_years present)
 - bar_height_hist.html (if tm_height_cm present)

Usage
  python -m src.analysis.eda_players \
    --features data/processed/model/features_all.csv \
    --out artifacts/eda_players \
    --min-minutes 180
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pyecharts.charts import Bar, Scatter, HeatMap
from pyecharts import options as opts
from pyecharts.commons.utils import JsCode
import json


DEFAULT_FEATURES = "data/processed/model/features_all.csv"


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _hist_minutes_bar(df: pd.DataFrame, out_dir: Path, min_minutes: float) -> None:
    m = df[df["minutes"].notna()]["minutes"].astype(float)
    if m.empty:
        return
    counts, edges = np.histogram(m, bins=20)
    labels = [f"{int(edges[i])}–{int(edges[i+1])}" for i in range(len(edges)-1)]
    bar = (
        Bar(init_opts=opts.InitOpts(width="800px", height="520px"))
        .add_xaxis(labels)
        .add_yaxis("players", counts.tolist())
        .set_global_opts(title_opts=opts.TitleOpts(title="Minutes distribution (all players)"),
                         xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=15)),
                         yaxis_opts=opts.AxisOpts(name="count"),
                         datazoom_opts=[opts.DataZoomOpts(type_="inside"), opts.DataZoomOpts()])
    )
    bar.render(str(out_dir / "bar_minutes_hist.html"))


def _simple_hist_numeric(df: pd.DataFrame, col: str, out_dir: Path, bins: int = 20, title: str | None = None, outfile: str | None = None) -> None:
    if col not in df.columns:
        return
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return
    counts, edges = np.histogram(s.values.astype(float), bins=bins)
    labels = [f"{round(edges[i])}–{round(edges[i+1])}" for i in range(len(edges)-1)]
    bar = (
        Bar(init_opts=opts.InitOpts(width="800px", height="520px"))
        .add_xaxis(labels)
        .add_yaxis("count", counts.tolist())
        .set_global_opts(title_opts=opts.TitleOpts(title=title or f"Distribution of {col}"),
                         xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=15)),
                         yaxis_opts=opts.AxisOpts(name="count"),
                         datazoom_opts=[opts.DataZoomOpts(type_="inside"), opts.DataZoomOpts()])
    )
    outname = outfile or f"bar_{col}_hist.html"
    bar.render(str(out_dir / outname))


def _leaders_plot(df: pd.DataFrame, metric: str, out_dir: Path, min_minutes: float, n: int = 10) -> None:
    if metric not in df.columns:
        return
    m = df[df["minutes"].fillna(0) >= min_minutes].copy()
    if len(m) == 0:
        return
    cols = ["player_name", "team_name", metric]
    top = (
        m[cols]
        .sort_values(metric, ascending=False)
        .head(n)
        .iloc[::-1]  # reverse for horizontal bar ascending
    )
    labels = (top["player_name"].astype(str) + " (" + top["team_name"].astype(str) + ")").tolist()
    values = top[metric].astype(float).round(4).tolist()
    bar = (
        Bar(init_opts=opts.InitOpts(width="1100px", height="640px"))
        .add_xaxis(labels)
        .add_yaxis(metric, values)
        .reversal_axis()  # horizontal bars to fit long labels
        .set_global_opts(
            title_opts=opts.TitleOpts(title=f"Top {n} {metric} (Player – Team)"),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts()),
            # Big margin pushes label text far to the right to avoid browser clipping
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(margin=140)),
            datazoom_opts=[opts.DataZoomOpts(type_="inside"), opts.DataZoomOpts()],
        )
        # Show the player's name and the numeric value directly on the bar
        .set_series_opts(
            label_opts=opts.LabelOpts(
                is_show=True,
                position="insideRight",
                formatter=JsCode(
                    "function (p) { return p.name + ' — ' + Number(p.value).toFixed(2); }"
                )
            )
        )
    )
    bar.render(str(out_dir / f"bar_top_{metric}.html"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="EDA on EURO24 player performance metrics")
    p.add_argument("--features", default=DEFAULT_FEATURES, help="Path to features_all.csv (or features.csv)")
    p.add_argument("--out", default="artifacts/eda_players", help="Output directory for plots")
    p.add_argument("--min-minutes", type=float, default=180.0, help="Min minutes to include in per90/box/leaderboards")
    p.add_argument("--leaders-n", type=int, default=15, help="Top N for leaderboards")
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    # Ensure expected columns exist
    needed = [
        "player_id", "player_name", "team_name", "pos_bucket", "minutes",
        "goals_per90", "shots_per90", "xg_per90", "xa_per90",
        "tackles_per90", "dribbles_per90", "pressures_per90",
        "interceptions_per90", "clearances_per90", "pass_pct", "passes_completed",
        "passes", "passes_completed_per90" if "passes_completed_per90" in df.columns else "passes_completed",
    ]
    # Convert basics to numeric
    df["minutes"] = _to_num(df.get("minutes", np.nan))
    for c in df.columns:
        if c.endswith("_per90") or c in ["pass_pct", "passes", "passes_completed"]:
            df[c] = _to_num(df[c])

    # 1) Minutes distributions (Pyecharts bar histogram)
    _hist_minutes_bar(df, out_dir, args.min_minutes)
    # Age/height distributions if available
    if "age" in df.columns or "age_years" in df.columns:
        _simple_hist_numeric(df, "age" if "age" in df.columns else "age_years", out_dir, title="Age distribution", outfile="bar_age_hist.html")
    if "tm_height_cm" in df.columns:
        _simple_hist_numeric(df, "tm_height_cm", out_dir, title="Height (cm) distribution", outfile="bar_height_hist.html")

    # (Removed) position-based boxplots

    # 3) Correlation heatmap among per90 features (filtered by minutes)
    per90_cols = [
        c for c in [
            "goals_per90", "shots_per90", "xg_per90", "xa_per90",
            "tackles_per90", "pressures_per90", "interceptions_per90", "clearances_per90",
            "dribbles_per90",
        ] if c in df.columns
    ]
    mm = df[df["minutes"].fillna(0) >= args.min_minutes][per90_cols].dropna(how="all")
    if len(mm) > 2 and len(per90_cols) >= 2:
        corr = mm.corr(numeric_only=True)
        x_labels = per90_cols
        y_labels = per90_cols
        data = []
        for i, xi in enumerate(x_labels):
            for j, yj in enumerate(y_labels):
                data.append([i, j, round(float(corr.loc[xi, yj]), 3)])
        heat = (
            HeatMap(init_opts=opts.InitOpts(width="700px", height="580px"))
            .add_xaxis(x_labels)
            .add_yaxis("corr", y_labels, data, label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(title_opts=opts.TitleOpts(title="Correlation (per90 metrics)"),
                             visualmap_opts=opts.VisualMapOpts(min_=-1, max_=1))
        )
        heat.render(str(out_dir / "corr_heatmap_per90.html"))

    # 4) Scatters
    if "xg_per90" in df.columns and "goals_per90" in df.columns:
        m = df[df["minutes"].fillna(0) >= args.min_minutes]
        names1 = (m["player_name"].astype(str) + " (" + m["team_name"].astype(str) + ")").tolist()
        x1 = m["xg_per90"].astype(float).round(4).tolist()
        y1 = m["goals_per90"].astype(float).round(4).tolist()
        s1 = (
            Scatter(init_opts=opts.InitOpts(width="800px", height="520px"))
            .add_xaxis(x1)
            .add_yaxis("players", y1, symbol_size=6,
                       label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(title_opts=opts.TitleOpts(title="xG/90 vs Goals/90 (min minutes filter)"),
                             xaxis_opts=opts.AxisOpts(name="xg_per90", type_="value"),
                             yaxis_opts=opts.AxisOpts(name="goals_per90", type_="value"),
                             tooltip_opts=opts.TooltipOpts(formatter=JsCode(
                                 """
                                 function (p) {
                                     var i = p.dataIndex;
                                     return names1[i] + '<br/>' + 'xg/90: ' + x1[i] + '<br/>' + 'goals/90: ' + y1[i];
                                 }
                                 """
                             )))
            .add_js_funcs(f"var names1 = {json.dumps(names1)}; var x1 = {json.dumps(x1)}; var y1 = {json.dumps(y1)};")
        )
        s1.render(str(out_dir / "scatter_xg_per90_vs_goals_per90.html"))

    # New: xG/90 vs xA/90
    if {"xg_per90","xa_per90"}.issubset(df.columns):
        m = df[df["minutes"].fillna(0) >= args.min_minutes]
        names2 = names1  # same rows/ordering
        x2 = x1
        y2 = m["xa_per90"].astype(float).round(4).tolist()
        # Piecewise transform for xg_per90 to equalize visual spacing across ranges
        bounds = [0, 0.1, 0.2, 0.4, 0.7, 1.2, 1.8]
        def _piecewise(vals: list[float]) -> list[float]:
            out: list[float] = []
            for v in vals:
                vv = max(bounds[0], min(bounds[-1], float(v)))
                placed = False
                for i in range(len(bounds)-1):
                    lo, hi = bounds[i], bounds[i+1]
                    if lo <= vv <= hi:
                        frac = 0.0 if hi == lo else (vv - lo)/(hi - lo)
                        out.append(round(i + frac, 6))
                        placed = True
                        break
                if not placed:
                    out.append(float(len(bounds)-1))
            return out
        x2_bucket = _piecewise(x2)
        s2 = (
            Scatter(init_opts=opts.InitOpts(width="800px", height="520px"))
            .add_xaxis(x2_bucket)
            .add_yaxis("players", y2, symbol_size=6,
                       label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(title_opts=opts.TitleOpts(title="xG/90 vs xA/90 (min minutes filter)"),
                             xaxis_opts=opts.AxisOpts(
                                 name="xg_per90",
                                 type_="value",
                                 min_=0,
                                 max_=len(bounds)-1,
                                 axislabel_opts=opts.LabelOpts(
                                     formatter=JsCode(
                                         """
                                         function (val) {
                                             var idx = Math.round(val);
                                             var labs = [0,0.1,0.2,0.4,0.7,1.2,1.8];
                                             if (Math.abs(val-idx)<0.05 && idx>=0 && idx<labs.length) {
                                                 return labs[idx];
                                             }
                                             return '';
                                         }
                                         """
                                     )
                                 )
                             ),
                             yaxis_opts=opts.AxisOpts(name="xa_per90", type_="value"),
                             tooltip_opts=opts.TooltipOpts(formatter=JsCode(
                                 """
                                 function (p) {
                                     var i = p.dataIndex;
                                     return names2[i] + '<br/>' + 'xg/90: ' + x2[i] + '<br/>' + 'xa/90: ' + y2[i];
                                 }
                                 """
                             )))
            .add_js_funcs(f"var names2 = {json.dumps(names2)}; var x2 = {json.dumps(x2)}; var y2 = {json.dumps(y2)};")
        )
        s2.render(str(out_dir / "scatter_xg_per90_vs_xa_per90.html"))

    if "pass_pct" in df.columns and "passes_completed" in df.columns:
        m = df[df["minutes"].fillna(0) >= args.min_minutes]
        names3 = names1
        x3 = m["pass_pct"].astype(float).round(2).tolist()
        y3 = m["passes_completed"].astype(float).round(2).tolist()
        s3 = (
            Scatter(init_opts=opts.InitOpts(width="800px", height="520px"))
            .add_xaxis(x3)
            .add_yaxis("players", y3, symbol_size=6,
                       label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(title_opts=opts.TitleOpts(title="Pass% vs Completed passes (min minutes filter)"),
                             xaxis_opts=opts.AxisOpts(name="pass_pct", type_="value"),
                             yaxis_opts=opts.AxisOpts(name="passes_completed", type_="value"),
                             tooltip_opts=opts.TooltipOpts(formatter=JsCode(
                                 """
                                 function (p) {
                                     var i = p.dataIndex;
                                     return names3[i] + '<br/>' + 'pass%: ' + x3[i] + '<br/>' + 'completed: ' + y3[i];
                                 }
                                 """
                             )))
            .add_js_funcs(f"var names3 = {json.dumps(names3)}; var x3 = {json.dumps(x3)}; var y3 = {json.dumps(y3)};")
        )
        s3.render(str(out_dir / "scatter_pass_pct_vs_passes_completed.html"))

    # New scatter: interceptions/90 vs tackles/90
    if {"interceptions_per90","tackles_per90"}.issubset(df.columns):
        m = df[df["minutes"].fillna(0) >= args.min_minutes]
        names4 = names1
        x4 = m["interceptions_per90"].astype(float).round(4).tolist()
        y4 = m["tackles_per90"].astype(float).round(4).tolist()
        s4 = (
            Scatter(init_opts=opts.InitOpts(width="800px", height="520px"))
            .add_xaxis(x4)
            .add_yaxis("players", y4, symbol_size=6,
                       label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(title_opts=opts.TitleOpts(title="Interceptions/90 vs Tackles/90 (min minutes filter)"),
                             xaxis_opts=opts.AxisOpts(name="interceptions_per90", type_="value"),
                             yaxis_opts=opts.AxisOpts(name="tackles_per90", type_="value"),
                             tooltip_opts=opts.TooltipOpts(formatter=JsCode(
                                 """
                                 function (p) {
                                     var i = p.dataIndex;
                                     return names4[i] + '<br/>' + 'interceptions/90: ' + x4[i] + '<br/>' + 'tackles/90: ' + y4[i];
                                 }
                                 """
                             )))
            .add_js_funcs(f"var names4 = {json.dumps(names4)}; var x4 = {json.dumps(x4)}; var y4 = {json.dumps(y4)};")
        )
        s4.render(str(out_dir / "scatter_interceptions_per90_vs_tackles_per90.html"))

    # 5) Leaderboards (bar charts for top 10)
    for metric in [
        "xg_per90", "xa_per90", "goals_per90", "shots_per90",
        "tackles_per90", "interceptions_per90", "pressures_per90", "dribbles_per90",
    ]:
        _leaders_plot(df, metric, out_dir, args.min_minutes, n=10)

    print(f"✅ Players EDA artifacts saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
