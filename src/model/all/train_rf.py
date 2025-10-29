"""
RandomForest variants for all-tournaments model (basic RF).

Aligns with euro24 RF pipeline but targets the all_tournaments DB.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from pyecharts.charts import Bar
from pyecharts import options as opts
import shap

from src.model.all.features import build_dataset

RANDOM_STATE = 42


def _train_val_split_time(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def _plot_importance(
    names: list[str], scores: np.ndarray, out_path: Path, title: str
) -> None:
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
        .set_series_opts(
            label_opts=opts.LabelOpts(is_show=True, position="right", formatter="{c}")
        )
    )
    bar.render(str(out_path))


def train_basic(db: str, out_dir: str, shap_eval: bool = False) -> dict:
    df = build_dataset(
        db_path=db, use_all_xref=True, include_market_value=True, include_mv_pre=False
    )
    df = df.copy()
    df["transfer_fee_num"] = pd.to_numeric(
        df.get("transfer_fee"), errors="coerce"
    ).astype(float)
    df = df[df["transfer_fee_num"].notna() & (df["transfer_fee_num"] > 0)].copy()
    df["y_log"] = np.log1p(df["transfer_fee_num"])

    # Handle NULL/empty strings; coerce numerics before median
    for col in ["tm_position", "tm_foot", "tm_height_cm", "market_value_in_eur"]:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            empty_count = (df[col] == "").sum() if df[col].dtype == "object" else 0
            print(
                f"Feature '{col}' has {null_count} NULL values and {empty_count} empty strings."
            )
            if null_count > 0 or empty_count > 0:
                if col == "tm_position":
                    df[col] = df[col].fillna("Unknown").replace("", "Unknown")
                elif col == "tm_foot":
                    df[col] = df[col].fillna("right").replace("", "right")
                elif col in ("tm_height_cm", "market_value_in_eur"):
                    df[col] = pd.to_numeric(df[col].replace("", np.nan), errors="coerce")
                    med = df[col].median(skipna=True)
                    df[col] = df[col].fillna(med)

    num_cols = [
        "market_value_in_eur",
        "age",
        "minutes",
        "xg_per90",
        "xa_per90",
        "goals_per90",
        "tackles_per90",
        "interceptions_per90",
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in ["team_id"] if c in df.columns]
    # Include new features added in features.py
    num_cols.extend(["tm_height_cm"])
    cat_cols.extend(["tm_position", "tm_foot"])
    # Ensure numeric columns are numeric for imputer
    for col in [
        "market_value_in_eur",
        "minutes",
        "xg_per90",
        "xa_per90",
        "goals_per90",
        "tackles_per90",
        "interceptions_per90",
        "tm_height_cm",
        "age",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace("", np.nan), errors="coerce")

    # Ensure numeric columns are numeric for imputer (complex)
    for col in [
        "minutes",
        "xg_per90",
        "xa_per90",
        "goals_per90",
        "tackles_per90",
        "interceptions_per90",
        "pressures_per90",
        "tm_height_cm",
        "age",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace("", np.nan), errors="coerce")

    # Drop rows with missing market_value_in_eur for basic
    if "market_value_in_eur" in df.columns:
        before = len(df)
        df = df[~pd.to_numeric(df["market_value_in_eur"], errors="coerce").isna()].copy()
        after = len(df)
        if before - after > 0:
            print(f"Dropped {before - after} rows due to missing market_value_in_eur (basic RF)")

    train_df, val_df = _train_val_split_time(df, test_size=0.2)

    num_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    transformers = [("num", num_pipe, num_cols)]
    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), cat_cols)
        )
    pre = ColumnTransformer(
        transformers=transformers, remainder="drop", sparse_threshold=0.3
    )

    model = RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipe = Pipeline([("prep", pre), ("model", model)])

    used_cols = num_cols + cat_cols
    pipe.fit(train_df[used_cols], train_df["y_log"])
    pred_fee = np.expm1(pipe.predict(val_df[used_cols]))
    true_fee = val_df["transfer_fee_num"].astype(float).values
    rmse = float(np.sqrt(mean_squared_error(true_fee, pred_fee)))
    mae = float(mean_absolute_error(true_fee, pred_fee))

    # Print the features used for training and prediction
    print("Features used for training and prediction:")
    print("Numeric features:", num_cols)
    print("Categorical features:", cat_cols)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump(pipe, out / "model.joblib")
    metrics = {
        "variant": "basic_rf",
        "rmse_eur": rmse,
        "mae_eur": mae,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    names = _get_feature_names(pipe)
    result = permutation_importance(
        pipe,
        val_df[used_cols],
        val_df["y_log"],
        n_repeats=10,
        random_state=RANDOM_STATE,
    )
    _plot_importance(
        names,
        result.importances_mean,
        out / "feature_importance_basic_rf.html",
        title="Permutation importance (basic RF)",
    )
    if shap_eval:
        X_val_t = pipe.named_steps["prep"].transform(val_df[used_cols])
        model = pipe.named_steps["model"]
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_val_t)
        shap_imp = np.abs(sv).mean(axis=0)
        shap_df = pd.DataFrame({"feature": names, "shap_importance": shap_imp})\
            .sort_values("shap_importance", ascending=False)
        shap_df.to_csv(out / "feature_importance_basic_rf_shap.csv", index=False)
    return metrics


def train_complex(db: str, out_dir: str, shap_eval: bool = False) -> dict:
    df = build_dataset(
        db_path=db, use_all_xref=True, include_market_value=True, include_mv_pre=False
    )
    df = df.copy()
    mv = pd.to_numeric(df.get("market_value_in_eur"), errors="coerce").astype(float)
    fee = pd.to_numeric(df.get("transfer_fee"), errors="coerce").astype(float)
    df = df[(mv > 0) & (fee > 0)].copy()
    df["y_resid"] = np.log1p(fee) - np.log1p(mv)

    per90_cols = [
        "xg_per90",
        "xa_per90",
        "goals_per90",
        "tackles_per90",
        "interceptions_per90",
        "pressures_per90",
    ]
    base_num = ["age", "minutes", "tm_height_cm"]
    num_cols = [c for c in base_num + per90_cols if c in df.columns]
    cat_cols = [c for c in ["team_id", "tm_position", "tm_foot"] if c in df.columns]

    print("Features used for training and prediction:")
    print("Numeric features:", num_cols)
    print("Categorical features:", cat_cols)

    train_df, val_df = _train_val_split_time(df, test_size=0.2)
    used_cols = num_cols + cat_cols

    num_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    transformers = [("num", num_pipe, num_cols)]
    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), cat_cols)
        )
    pre = ColumnTransformer(
        transformers=transformers, remainder="drop", sparse_threshold=0.3
    )

    model = RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipe = Pipeline([("prep", pre), ("model", model)])

    pipe.fit(train_df[used_cols], train_df["y_resid"])
    resid_pred = pipe.predict(val_df[used_cols])
    pred_log_fee = (
        np.log1p(
            pd.to_numeric(val_df["market_value_in_eur"], errors="coerce").astype(float)
        )
        + resid_pred
    )
    pred_fee = np.expm1(pred_log_fee)
    true_fee = pd.to_numeric(val_df["transfer_fee"], errors="coerce").astype(float)
    rmse = float(np.sqrt(mean_squared_error(true_fee, pred_fee)))
    mae = float(mean_absolute_error(true_fee, pred_fee))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump(pipe, out / "model.joblib")
    metrics = {
        "variant": "complex_rf",
        "rmse_eur": rmse,
        "mae_eur": mae,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "num_features": num_cols,
        "cat_features": cat_cols,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    names = _get_feature_names(pipe)
    result = permutation_importance(
        pipe,
        val_df[used_cols],
        val_df["y_resid"],
        n_repeats=10,
        random_state=RANDOM_STATE,
    )
    _plot_importance(
        names,
        result.importances_mean,
        out / "feature_importance_complex_rf.html",
        title="Permutation importance (complex RF)",
    )
    if shap_eval:
        X_val_t = pipe.named_steps["prep"].transform(val_df[used_cols])
        model = pipe.named_steps["model"]
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_val_t)
        shap_imp = np.abs(sv).mean(axis=0)
        shap_df = pd.DataFrame({"feature": names, "shap_importance": shap_imp})\
            .sort_values("shap_importance", ascending=False)
        shap_df.to_csv(out / "feature_importance_complex_rf_shap.csv", index=False)
    return metrics


def _calculate_baseline_vs_model_kfold(oof: pd.DataFrame, out_path: Path) -> None:
    mv_baseline = pd.to_numeric(oof.get("market_value_in_eur"), errors="coerce").astype(
        float
    )
    true_fee = pd.to_numeric(oof.get("transfer_fee"), errors="coerce").astype(float)
    mask = ~np.isnan(mv_baseline)
    if mask.any():
        mae_b = float(mean_absolute_error(true_fee[mask], mv_baseline[mask]))
        rmse_b = float(np.sqrt(mean_squared_error(true_fee[mask], mv_baseline[mask])))
        mae_m = float(mean_absolute_error(true_fee, oof["pred_fee"].values))
        rmse_m = float(np.sqrt(mean_squared_error(true_fee, oof["pred_fee"].values)))
        metrics = {
            "n_val": int(len(oof)),
            "baseline": {
                "mae_eur": mae_b,
                "rmse_eur": rmse_b,
                "formula": "fee_hat = market_value_in_eur",
            },
            "model": {"mae_eur": mae_m, "rmse_eur": rmse_m},
        }
        out_path.write_text(json.dumps(metrics, indent=2))


def kfold_basic(db: str, out_dir: str, k: int = 5, shap_eval: bool = False) -> dict:
    df = build_dataset(
        db_path=db, use_all_xref=True, include_market_value=True, include_mv_pre=False
    )
    df = df.copy()
    df["transfer_fee_num"] = pd.to_numeric(
        df.get("transfer_fee"), errors="coerce"
    ).astype(float)
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

    num_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    transformers = [("num", num_pipe, num_cols)]
    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), cat_cols)
        )
    pre = ColumnTransformer(
        transformers=transformers, remainder="drop", sparse_threshold=0.3
    )
    pipe = Pipeline([("prep", pre), ("model", RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ))])

    from sklearn.model_selection import KFold

    # Ensure numerics in DF to avoid imputer errors
    for col in [
        "market_value_in_eur",
        "minutes",
        "xg_per90",
        "xa_per90",
        "goals_per90",
        "tackles_per90",
        "interceptions_per90",
        "tm_height_cm",
        "age",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace("", np.nan), errors="coerce")

    # Drop rows with missing market_value_in_eur for KFold basic
    if "market_value_in_eur" in df.columns:
        before = len(df)
        df = df[~pd.to_numeric(df["market_value_in_eur"], errors="coerce").isna()].copy()
        after = len(df)
        if before - after > 0:
            print(f"Dropped {before - after} rows due to missing market_value_in_eur (kfold basic RF)")

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
        true_fee = (
            pd.to_numeric(va["transfer_fee"], errors="coerce").astype(float).values
        )
        maes.append(mean_absolute_error(true_fee, pred_fee))
        rmses.append(np.sqrt(mean_squared_error(true_fee, pred_fee)))
        tmp = va[[
            "player_id","player_name","team_name","transfer_fee","transfer_date"]].copy()
        tmp["pred_fee"] = pred_fee
        tmp["market_value_in_eur"] = va["market_value_in_eur"].values
        tmp["fold"] = fold
        oof_rows.append(tmp)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    oof = pd.concat(oof_rows, ignore_index=True)
    oof.to_csv(out / "oof_predictions_basic.csv", index=False)
    _calculate_baseline_vs_model_kfold(oof, out / "baseline_vs_model.json")

    metrics = {
        "variant": "basic_kfold_rf",
        "folds": int(fold),
        "mae_mean_eur": float(np.mean(maes)),
        "mae_std_eur": float(np.std(maes)),
        "rmse_mean_eur": float(np.mean(rmses)),
        "rmse_std_eur": float(np.std(rmses)),
        "num_features": num_cols,
        "cat_features": cat_cols,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Fit on full data for a single importance plot
    pipe.fit(df[used_cols], df["y_log"])
    names = _get_feature_names(pipe)
    result = permutation_importance(
        pipe, df[used_cols], df["y_log"], n_repeats=10, random_state=RANDOM_STATE
    )
    _plot_importance(
        names,
        result.importances_mean,
        out / "feature_importance_basic_kfold_rf.html",
        title="Permutation importance (basic RF KFold, fit on all)",
    )
    if shap_eval:
        X_all_t = pipe.named_steps["prep"].transform(df[used_cols])
        model = pipe.named_steps["model"]
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_all_t)
        shap_imp = np.abs(sv).mean(axis=0)
        shap_df = pd.DataFrame({"feature": names, "shap_importance": shap_imp})\
            .sort_values("shap_importance", ascending=False)
        shap_df.to_csv(out / "feature_importance_basic_kfold_rf_shap.csv", index=False)
    return metrics


def kfold_complex(db: str, out_dir: str, k: int = 5, shap_eval: bool = False) -> dict:
    df = build_dataset(
        db_path=db, use_all_xref=True, include_market_value=True, include_mv_pre=False
    )
    df = df.copy()
    mv = pd.to_numeric(df.get("market_value_in_eur"), errors="coerce").astype(float)
    fee = pd.to_numeric(df.get("transfer_fee"), errors="coerce").astype(float)
    df = df[(mv > 0) & (fee > 0)].copy()
    df["y_resid"] = np.log1p(fee) - np.log1p(mv)

    per90_cols = [
        "xg_per90",
        "xa_per90",
        "goals_per90",
        "tackles_per90",
        "interceptions_per90",
        "pressures_per90",
    ]
    base_num = ["age", "minutes", "tm_height_cm"]
    num_cols = [c for c in base_num + per90_cols if c in df.columns]
    cat_cols = [c for c in ["team_id", "tm_position", "tm_foot"] if c in df.columns]
    used_cols = num_cols + cat_cols

    num_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    transformers = [("num", num_pipe, num_cols)]
    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), cat_cols)
        )
    pre = ColumnTransformer(
        transformers=transformers, remainder="drop", sparse_threshold=0.3
    )
    pipe = Pipeline([("prep", pre), ("model", RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ))])

    from sklearn.model_selection import KFold

    # Ensure numerics in DF to avoid imputer errors
    for col in [
        "minutes",
        "xg_per90",
        "xa_per90",
        "goals_per90",
        "tackles_per90",
        "interceptions_per90",
        "pressures_per90",
        "tm_height_cm",
        "age",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace("", np.nan), errors="coerce")

    kf = KFold(n_splits=max(2, k), shuffle=True, random_state=RANDOM_STATE)
    maes, rmses = [], []
    oof_rows = []
    fold = 0
    for tr_idx, va_idx in kf.split(df):
        fold += 1
        tr, va = df.iloc[tr_idx], df.iloc[va_idx]
        pipe.fit(tr[used_cols], tr["y_resid"])
        resid_pred = pipe.predict(va[used_cols])
        pred_log_fee = (
            np.log1p(
                pd.to_numeric(va["market_value_in_eur"], errors="coerce").astype(float)
            )
            + resid_pred
        )
        pred_fee = np.expm1(pred_log_fee)
        true_fee = (
            pd.to_numeric(va["transfer_fee"], errors="coerce").astype(float).values
        )
        maes.append(mean_absolute_error(true_fee, pred_fee))
        rmses.append(np.sqrt(mean_squared_error(true_fee, pred_fee)))
        tmp = va[[
            "player_id","player_name","team_name","transfer_fee","transfer_date","market_value_in_eur"
        ]].copy()
        tmp["pred_fee"] = pred_fee
        tmp["fold"] = fold
        oof_rows.append(tmp)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    oof = pd.concat(oof_rows, ignore_index=True)
    oof.to_csv(out / "oof_predictions_complex.csv", index=False)
    _calculate_baseline_vs_model_kfold(oof, out / "baseline_vs_model.json")

    metrics = {
        "variant": "complex_kfold_rf",
        "folds": int(fold),
        "mae_mean_eur": float(np.mean(maes)),
        "mae_std_eur": float(np.std(maes)),
        "rmse_mean_eur": float(np.mean(rmses)),
        "rmse_std_eur": float(np.std(rmses)),
        "num_features": num_cols,
        "cat_features": cat_cols,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Fit best on full data and plot importance
    pipe.fit(df[used_cols], df["y_resid"])
    names = _get_feature_names(pipe)
    result = permutation_importance(
        pipe, df[used_cols], df["y_resid"], n_repeats=10, random_state=RANDOM_STATE
    )
    _plot_importance(
        names,
        result.importances_mean,
        out / "feature_importance_complex_kfold_rf.html",
        title="Permutation importance (complex RF KFold, fit on all)",
    )
    if shap_eval:
        X_all_t = pipe.named_steps["prep"].transform(df[used_cols])
        model = pipe.named_steps["model"]
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_all_t)
        shap_imp = np.abs(sv).mean(axis=0)
        shap_df = pd.DataFrame({"feature": names, "shap_importance": shap_imp})\
            .sort_values("shap_importance", ascending=False)
        shap_df.to_csv(out / "feature_importance_complex_kfold_rf_shap.csv", index=False)
    return metrics


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train all-tournaments model with RandomForest")
    p.add_argument("--db", default="data/db/all_tournaments.sqlite")
    p.add_argument("--out", default="artifacts/model_all_basic_rf")
    p.add_argument("--variant", choices=["basic", "complex"], default="basic")
    p.add_argument("--cv", choices=["none", "kfold"], default="none")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--shap", action="store_true", help="Calcola SHAP importance")
    args = p.parse_args(argv)

    if args.variant == "basic":
        if args.cv == "kfold":
            out = (
                args.out
                if args.out != "artifacts/model_all_basic_rf"
                else "artifacts/model_all_basic_kfold_rf"
            )
            m = kfold_basic(args.db, out, k=args.k, shap_eval=args.shap)
            print(
                f"✅ All-tournaments RF (basic KFold {m['folds']}x). MAE={m['mae_mean_eur']:,.0f}€ ±{m['mae_std_eur']:,.0f}€  RMSE={m['rmse_mean_eur']:,.0f}€ ±{m['rmse_std_eur']:,.0f}€"
            )
            print(f"Artifacts → {out}")
        else:
            m = train_basic(args.db, args.out, shap_eval=args.shap)
            print(
                f"✅ All-tournaments RF (basic) trained. RMSE={m['rmse_eur']:,.0f}€ MAE={m['mae_eur']:,.0f}€ (n_train={m['n_train']}, n_val={m['n_val']})"
            )
            print(f"Artifacts → {args.out}")
    else:
        out_default = "artifacts/model_all_complex_rf"
        if args.cv == "kfold":
            out = (
                args.out
                if args.out != "artifacts/model_all_basic_rf"
                else f"{out_default}_kfold"
            )
            m = kfold_complex(args.db, out, k=args.k, shap_eval=args.shap)
            print(
                f"✅ All-tournaments RF (complex KFold {m['folds']}x). MAE={m['mae_mean_eur']:,.0f}€ ±{m['mae_std_eur']:,.0f}€  RMSE={m['rmse_mean_eur']:,.0f}€ ±{m['rmse_std_eur']:,.0f}€"
            )
            print(f"Artifacts → {out}")
        else:
            out = args.out if args.out != "artifacts/model_all_basic_rf" else out_default
            m = train_complex(args.db, out, shap_eval=args.shap)
            print(
                f"✅ All-tournaments RF (complex residual) trained. RMSE={m['rmse_eur']:,.0f}€ MAE={m['mae_eur']:,.0f}€ (n_train={m['n_train']}, n_val={m['n_val']})"
            )
            print(f"Artifacts → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
