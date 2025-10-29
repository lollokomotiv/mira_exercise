"""
XGBoost variants for all-tournaments model (basic/complex + KFold).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
from pyecharts.charts import Bar
from pyecharts import options as opts

from xgboost import XGBRegressor

from src.model.all.features import build_dataset

RANDOM_STATE = 42


def _train_val_split_time(df: pd.DataFrame, test_size: float = 0.2):
    if "transfer_date" in df.columns and df["transfer_date"].notna().any():
        df = df.sort_values("transfer_date").reset_index(drop=True)
        n = len(df)
        split = max(1, int(n * (1 - test_size)))
        tr, va = df.iloc[:split].copy(), df.iloc[split:].copy()
        if len(va) == 0:
            va = df.tail(max(1, int(n * test_size))).copy()
        return tr, va
    n = len(df)
    split = max(1, int(n * (1 - test_size)))
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def _get_feature_names(pipe) -> list[str]:
    prep = pipe.named_steps.get("prep")
    names: list[str] = []
    if prep is None:
        return names
    for name, trans, cols in prep.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if hasattr(trans, "get_feature_names_out"):
            got = list(trans.get_feature_names_out(cols))
        else:
            got = list(cols)
        names.extend(got)
    return names


def _plot_importance(names: list[str], scores: np.ndarray, out_path: Path, title: str):
    order = np.argsort(scores)[::-1]
    names_s = [names[i] for i in order]
    vals_s = [float(scores[i]) for i in order]
    bar = (
        Bar(init_opts=opts.InitOpts(width="900px", height="560px"))
        .add_xaxis(names_s)
        .add_yaxis("perm_importance", vals_s)
        .reversal_axis()
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=0)),
            datazoom_opts=[opts.DataZoomOpts(type_="inside"), opts.DataZoomOpts()],
        )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=True, position="right", formatter="{c}"))
    )
    bar.render(str(out_path))


def _coerce_numerics(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace("", np.nan), errors="coerce")
    return df


def train_basic(db: str, out_dir: str) -> dict:
    df = build_dataset(db_path=db, use_all_xref=True, include_market_value=True, include_mv_pre=False)
    df = _coerce_numerics(
        df,
        [
            "market_value_in_eur",
            "minutes",
            "xg_per90",
            "xa_per90",
            "goals_per90",
            "tackles_per90",
            "interceptions_per90",
            "tm_height_cm",
            "age",
        ],
    )
    # Drop rows with MV missing
    if "market_value_in_eur" in df.columns:
        df = df[~df["market_value_in_eur"].isna()].copy()

    df["transfer_fee_num"] = pd.to_numeric(df.get("transfer_fee"), errors="coerce").astype(float)
    df = df[df["transfer_fee_num"].notna() & (df["transfer_fee_num"] > 0)].copy()
    df["y_log"] = np.log1p(df["transfer_fee_num"]).astype(float)

    num_cols = [
        "market_value_in_eur",
        "age",
        "minutes",
        "xg_per90",
        "xa_per90",
        "goals_per90",
        "tackles_per90",
        "interceptions_per90",
        "tm_height_cm",
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in ["team_id", "tm_position", "tm_foot"] if c in df.columns]
    used_cols = num_cols + cat_cols

    train_df, val_df = _train_val_split_time(df, test_size=0.2)

    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    transformers = [("num", num_pipe, num_cols)]
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), cat_cols))
    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)

    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipe = Pipeline([("prep", pre), ("model", model)])
    pipe.fit(train_df[used_cols], train_df["y_log"])

    pred_fee = np.expm1(pipe.predict(val_df[used_cols]))
    true_fee = val_df["transfer_fee_num"].astype(float).values
    rmse = float(np.sqrt(mean_squared_error(true_fee, pred_fee)))
    mae = float(mean_absolute_error(true_fee, pred_fee))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump(pipe, out / "model.joblib")
    metrics = {"variant": "basic_xgb", "rmse_eur": rmse, "mae_eur": mae, "n_train": int(len(train_df)), "n_val": int(len(val_df))}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    names = _get_feature_names(pipe)
    result = permutation_importance(pipe, val_df[used_cols], val_df["y_log"], n_repeats=10, random_state=RANDOM_STATE)
    _plot_importance(names, result.importances_mean, out / "feature_importance_basic_xgb.html", title="Permutation importance (basic XGB)")
    return metrics


def train_complex(db: str, out_dir: str) -> dict:
    df = build_dataset(db_path=db, use_all_xref=True, include_market_value=True, include_mv_pre=False)
    df = _coerce_numerics(
        df,
        [
            "minutes",
            "xg_per90",
            "xa_per90",
            "goals_per90",
            "tackles_per90",
            "interceptions_per90",
            "pressures_per90",
            "tm_height_cm",
            "age",
            "market_value_in_eur",
        ],
    )
    mv = pd.to_numeric(df.get("market_value_in_eur"), errors="coerce").astype(float)
    fee = pd.to_numeric(df.get("transfer_fee"), errors="coerce").astype(float)
    df = df[(mv > 0) & (fee > 0)].copy()
    df["y_resid"] = np.log1p(fee) - np.log1p(mv)

    per90_cols = ["xg_per90", "xa_per90", "goals_per90", "tackles_per90", "interceptions_per90", "pressures_per90"]
    base_num = ["age", "minutes", "tm_height_cm"]
    num_cols = [c for c in base_num + per90_cols if c in df.columns]
    cat_cols = [c for c in ["team_id", "tm_position", "tm_foot"] if c in df.columns]
    used_cols = num_cols + cat_cols

    train_df, val_df = _train_val_split_time(df, test_size=0.2)

    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    transformers = [("num", num_pipe, num_cols)]
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), cat_cols))
    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)

    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipe = Pipeline([("prep", pre), ("model", model)])
    pipe.fit(train_df[used_cols], train_df["y_resid"])

    resid_pred = pipe.predict(val_df[used_cols])
    pred_log_fee = np.log1p(val_df["market_value_in_eur"].astype(float).values) + resid_pred
    pred_fee = np.expm1(pred_log_fee)
    true_fee = pd.to_numeric(val_df["transfer_fee"], errors="coerce").astype(float).values
    rmse = float(np.sqrt(mean_squared_error(true_fee, pred_fee)))
    mae = float(mean_absolute_error(true_fee, pred_fee))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump(pipe, out / "model.joblib")
    metrics = {"variant": "complex_xgb", "rmse_eur": rmse, "mae_eur": mae, "n_train": int(len(train_df)), "n_val": int(len(val_df))}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    names = _get_feature_names(pipe)
    result = permutation_importance(pipe, val_df[used_cols], val_df["y_resid"], n_repeats=10, random_state=RANDOM_STATE)
    _plot_importance(names, result.importances_mean, out / "feature_importance_complex_xgb.html", title="Permutation importance (complex XGB)")
    return metrics


def kfold_basic(db: str, out_dir: str, k: int = 5) -> dict:
    df = build_dataset(db_path=db, use_all_xref=True, include_market_value=True, include_mv_pre=False)
    df = _coerce_numerics(
        df,
        [
            "market_value_in_eur",
            "minutes",
            "xg_per90",
            "xa_per90",
            "goals_per90",
            "tackles_per90",
            "interceptions_per90",
            "tm_height_cm",
            "age",
        ],
    )
    if "market_value_in_eur" in df.columns:
        df = df[~df["market_value_in_eur"].isna()].copy()

    df["transfer_fee_num"] = pd.to_numeric(df.get("transfer_fee"), errors="coerce").astype(float)
    df = df[df["transfer_fee_num"].notna() & (df["transfer_fee_num"] > 0)].copy()
    df["y_log"] = np.log1p(df["transfer_fee_num"]).astype(float)

    num_cols = [
        "market_value_in_eur",
        "age",
        "minutes",
        "xg_per90",
        "xa_per90",
        "goals_per90",
        "tackles_per90",
        "interceptions_per90",
        "tm_height_cm",
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in ["team_id", "tm_position", "tm_foot"] if c in df.columns]
    used_cols = num_cols + cat_cols

    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    transformers = [("num", num_pipe, num_cols)]
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), cat_cols))
    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)

    pipe = Pipeline([("prep", pre), ("model", XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ))])

    kf = KFold(n_splits=max(2, k), shuffle=True, random_state=RANDOM_STATE)
    oof_rows = []
    maes, rmses = [], []
    fold = 0
    for tr_idx, va_idx in kf.split(df):
        fold += 1
        tr, va = df.iloc[tr_idx], df.iloc[va_idx]
        pipe.fit(tr[used_cols], tr["y_log"])
        pred_log = pipe.predict(va[used_cols])
        pred_fee = np.expm1(pred_log)
        true_fee = pd.to_numeric(va["transfer_fee"], errors="coerce").astype(float).values
        maes.append(mean_absolute_error(true_fee, pred_fee))
        rmses.append(np.sqrt(mean_squared_error(true_fee, pred_fee)))
        tmp = va[["player_id", "player_name", "team_name", "transfer_fee", "transfer_date"]].copy()
        tmp["pred_fee"] = pred_fee
        tmp["fold"] = fold
        oof_rows.append(tmp)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    oof = pd.concat(oof_rows, ignore_index=True)
    oof.to_csv(out / "oof_predictions_basic_xgb.csv", index=False)

    metrics = {
        "variant": "basic_kfold_xgb",
        "folds": int(fold),
        "mae_mean_eur": float(np.mean(maes)),
        "mae_std_eur": float(np.std(maes)),
        "rmse_mean_eur": float(np.mean(rmses)),
        "rmse_std_eur": float(np.std(rmses)),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Fit on full data for a single importance plot
    pipe.fit(df[used_cols], df["y_log"])
    names = _get_feature_names(pipe)
    result = permutation_importance(pipe, df[used_cols], df["y_log"], n_repeats=10, random_state=RANDOM_STATE)
    _plot_importance(names, result.importances_mean, out / "feature_importance_basic_kfold_xgb.html", title="Permutation importance (basic XGB KFold, fit on all)")
    return metrics


def kfold_complex(db: str, out_dir: str, k: int = 5) -> dict:
    df = build_dataset(db_path=db, use_all_xref=True, include_market_value=True, include_mv_pre=False)
    df = _coerce_numerics(
        df,
        [
            "minutes",
            "xg_per90",
            "xa_per90",
            "goals_per90",
            "tackles_per90",
            "interceptions_per90",
            "pressures_per90",
            "tm_height_cm",
            "age",
            "market_value_in_eur",
        ],
    )
    mv = pd.to_numeric(df.get("market_value_in_eur"), errors="coerce").astype(float)
    fee = pd.to_numeric(df.get("transfer_fee"), errors="coerce").astype(float)
    df = df[(mv > 0) & (fee > 0)].copy()
    df["y_resid"] = np.log1p(fee) - np.log1p(mv)

    per90_cols = ["xg_per90", "xa_per90", "goals_per90", "tackles_per90", "interceptions_per90", "pressures_per90"]
    base_num = ["age", "minutes", "tm_height_cm"]
    num_cols = [c for c in base_num + per90_cols if c in df.columns]
    cat_cols = [c for c in ["team_id", "tm_position", "tm_foot"] if c in df.columns]
    used_cols = num_cols + cat_cols

    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    transformers = [("num", num_pipe, num_cols)]
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), cat_cols))
    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)
    pipe = Pipeline([("prep", pre), ("model", XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ))])

    kf = KFold(n_splits=max(2, k), shuffle=True, random_state=RANDOM_STATE)
    maes, rmses = [], []
    oof_rows = []
    fold = 0
    for tr_idx, va_idx in kf.split(df):
        fold += 1
        tr, va = df.iloc[tr_idx], df.iloc[va_idx]
        pipe.fit(tr[used_cols], tr["y_resid"])
        resid_pred = pipe.predict(va[used_cols])
        pred_log_fee = np.log1p(pd.to_numeric(va["market_value_in_eur"], errors="coerce").astype(float)) + resid_pred
        pred_fee = np.expm1(pred_log_fee)
        true_fee = pd.to_numeric(va["transfer_fee"], errors="coerce").astype(float).values
        maes.append(mean_absolute_error(true_fee, pred_fee))
        rmses.append(np.sqrt(mean_squared_error(true_fee, pred_fee)))
        tmp = va[["player_id", "player_name", "team_name", "transfer_fee", "transfer_date", "market_value_in_eur"]].copy()
        tmp["pred_fee"] = pred_fee
        tmp["fold"] = fold
        oof_rows.append(tmp)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    oof = pd.concat(oof_rows, ignore_index=True)
    oof.to_csv(out / "oof_predictions_complex_xgb.csv", index=False)

    metrics = {
        "variant": "complex_kfold_xgb",
        "folds": int(fold),
        "mae_mean_eur": float(np.mean(maes)),
        "mae_std_eur": float(np.std(maes)),
        "rmse_mean_eur": float(np.mean(rmses)),
        "rmse_std_eur": float(np.std(rmses)),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Fit best on full data and plot importance
    pipe.fit(df[used_cols], df["y_resid"])
    names = _get_feature_names(pipe)
    result = permutation_importance(pipe, df[used_cols], df["y_resid"], n_repeats=10, random_state=RANDOM_STATE)
    _plot_importance(names, result.importances_mean, out / "feature_importance_complex_kfold_xgb.html", title="Permutation importance (complex XGB KFold, fit on all)")
    return metrics


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train all-tournaments model with XGBoost")
    p.add_argument("--db", default="data/db/all_tournaments.sqlite")
    p.add_argument("--out", default="artifacts/model_all_basic_xgb")
    p.add_argument("--variant", choices=["basic", "complex"], default="basic")
    p.add_argument("--cv", choices=["none", "kfold"], default="none")
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args(argv)

    if args.variant == "basic":
        if args.cv == "kfold":
            out = args.out if args.out != "artifacts/model_all_basic_xgb" else "artifacts/model_all_basic_kfold_xgb"
            m = kfold_basic(args.db, out, k=args.k)
            print(f"✅ All-tournaments XGB (basic KFold {m['folds']}x). MAE={m['mae_mean_eur']:,.0f}€ ±{m['mae_std_eur']:,.0f}€  RMSE={m['rmse_mean_eur']:,.0f}€ ±{m['rmse_std_eur']:,.0f}€")
            print(f"Artifacts → {out}")
        else:
            m = train_basic(args.db, args.out)
            print(f"✅ All-tournaments XGB (basic) trained. RMSE={m['rmse_eur']:,.0f}€ MAE={m['mae_eur']:,.0f}€ (n_train={m['n_train']}, n_val={m['n_val']})")
            print(f"Artifacts → {args.out}")
    else:
        out_default = "artifacts/model_all_complex_xgb"
        if args.cv == "kfold":
            out = args.out if args.out != "artifacts/model_all_basic_xgb" else f"{out_default}_kfold"
            m = kfold_complex(args.db, out, k=args.k)
            print(f"✅ All-tournaments XGB (complex KFold {m['folds']}x). MAE={m['mae_mean_eur']:,.0f}€ ±{m['mae_std_eur']:,.0f}€  RMSE={m['rmse_mean_eur']:,.0f}€ ±{m['rmse_std_eur']:,.0f}€")
            print(f"Artifacts → {out}")
        else:
            out = args.out if args.out != "artifacts/model_all_basic_xgb" else out_default
            m = train_complex(args.db, out)
            print(f"✅ All-tournaments XGB (complex residual) trained. RMSE={m['rmse_eur']:,.0f}€ MAE={m['mae_eur']:,.0f}€ (n_train={m['n_train']}, n_val={m['n_val']})")
            print(f"Artifacts → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

