"""
2024-only transfer fee model (minimal, opinionated).

- Loads supervised 2024 dataset from SQLite via model.features.build_dataset
- Uses MV post-transfer (market_value_in_eur) + few stable features
- Trains a Ridge regression on log1p(fee) with a simple time-aware holdout
- Saves artifacts under artifacts/model_2024_only/

Usage
  python -m model.2024_only.train \
    --db data/db/euro24.sqlite \
    --out artifacts/model_2024_only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.inspection import permutation_importance
from pyecharts.charts import Bar
from pyecharts import options as opts
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer

from src.model.euro24.features import build_dataset


RANDOM_STATE = 42


def _train_val_split_time(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Sort by transfer_date if present, otherwise a random split
    if "transfer_date" in df.columns and df["transfer_date"].notna().any():
        df = df.sort_values("transfer_date").reset_index(drop=True)
        n = len(df)
        split = max(1, int(n * (1 - test_size)))
        tr, va = df.iloc[:split].copy(), df.iloc[split:].copy()
        if len(va) == 0:
            va = df.tail(max(1, int(n * test_size))).copy()
        return tr, va
    # Fallback: last N% rows as validation
    n = len(df)
    split = max(1, int(n * (1 - test_size)))
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def _feature_sets_basic(df: pd.DataFrame) -> list[dict]:
    """Iterative feature sets for basic variant (EURO24)."""
    base_num = ["market_value_in_eur", "age", "xg_per90", "xa_per90"]
    base_num = [c for c in base_num if c in df.columns]
    steps: list[dict] = []
    steps.append({"name": "base", "num": base_num.copy(), "cat": []})
    if "minutes" in df.columns:
        steps.append({"name": "base+minutes", "num": base_num + ["minutes"], "cat": []})
    if "team_id" in df.columns:
        last = steps[-1]
        steps.append({"name": last["name"] + "+team", "num": last["num"], "cat": ["team_id"]})
    if "tm_height_cm" in df.columns:
        last = steps[-1]
        steps.append({"name": last["name"] + "+height", "num": last["num"] + ["tm_height_cm"], "cat": last["cat"]})
    return steps


def train_2024(
    db: str,
    out_dir: str,
    alpha: float = 10.0,
    num_cols_override: list[str] | None = None,
    cat_cols_override: list[str] | None = None,
) -> dict:
    # Build dataset (2024 window is already enforced by build_dataset labels)
    # Choices: use_all_xref=True (greater coverage), include_market_value=True (mv_post), exclude mv_pre
    df = build_dataset(
        db_path=db,
        use_all_xref=True,
        include_market_value=True,
        include_mv_pre=False,
    )

    # Target (log space) with robust numeric cast
    df = df.copy()
    df["transfer_fee_num"] = pd.to_numeric(
        df.get("transfer_fee"), errors="coerce"
    ).astype(float)
    df = df[df["transfer_fee_num"].notna() & (df["transfer_fee_num"] > 0)].copy()
    df["y_log"] = np.log1p(df["transfer_fee_num"])

    # Minimal, stable feature set (numeric)
    num_cols = [
        "market_value_in_eur",  # MV post
        "age",
        "minutes",
        "xg_per90",
        "xa_per90",
        "goals_per90",
        "tackles_per90",
        "interceptions_per90",
    ]
    num_cols = [c for c in num_cols if c in df.columns]

    # Categorical: national team id (squadra). One‑hot encoded with small frequency threshold
    cat_cols = [c for c in ["team_id"] if c in df.columns]

    # Include new features added in features.py
    num_cols.extend(["tm_height_cm"])
    cat_cols.extend(["tm_position", "tm_foot"])

    # Allow overrides (iterative sweeps)
    if num_cols_override is not None:
        num_cols = [c for c in num_cols_override if c in df.columns]
    if cat_cols_override is not None:
        cat_cols = [c for c in cat_cols_override if c in df.columns]

    # Train/Val split (time-aware)
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
    model = Ridge(alpha=alpha)
    pipe = Pipeline(
        [
            ("prep", pre),
            ("model", model),
        ]
    )

    used_cols = num_cols + cat_cols
    X_tr, y_tr = train_df[used_cols], train_df["y_log"]
    X_va, y_va = val_df[used_cols], val_df["y_log"]
    pipe.fit(X_tr, y_tr)

    # Evaluate back in EUR
    pred_log = pipe.predict(X_va)
    pred_fee = np.expm1(pred_log)
    true_fee = val_df["transfer_fee_num"].astype(float).values
    rmse = float(np.sqrt(mean_squared_error(true_fee, pred_fee)))
    mae = float(mean_absolute_error(true_fee, pred_fee))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump(pipe, out / "model.joblib")
    metrics = {
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "rmse_eur": rmse,
        "mae_eur": mae,
        "alpha": float(alpha),
        "num_features": num_cols,
        "cat_features": cat_cols,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    val_out = val_df[
        ["player_id", "player_name", "team_name", "transfer_fee", "transfer_date"]
    ].copy()
    val_out["pred_fee"] = pred_fee
    val_out.to_csv(out / "val_predictions_2024.csv", index=False)

    # Baseline vs model (same validation split): fee_hat = market_value_in_eur
    mv_baseline = (
        pd.to_numeric(val_df.get("market_value_in_eur"), errors="coerce")
        .astype(float)
        .values
    )
    mask = ~np.isnan(mv_baseline)
    if mask.any():
        mae_b = float(mean_absolute_error(true_fee[mask], mv_baseline[mask]))
        rmse_b = float(np.sqrt(mean_squared_error(true_fee[mask], mv_baseline[mask])))
        (out / "baseline_vs_model.json").write_text(
            json.dumps(
                {
                    "n_val": int(len(val_df)),
                    "baseline": {
                        "mae_eur": mae_b,
                        "rmse_eur": rmse_b,
                        "formula": "fee_hat = market_value_in_eur",
                    },
                    "model": {"mae_eur": mae, "rmse_eur": rmse},
                },
                indent=2,
            )
        )
    # Feature importance on validation (permutation on log-target)
    result = permutation_importance(
        pipe, X_va, y_va, n_repeats=10, random_state=RANDOM_STATE
    )
    names = _get_feature_names(pipe)
    _plot_importance(
        names,
        result.importances_mean,
        out / "feature_importance_basic.html",
        title="Permutation importance (basic)",
    )

    # Print the features used for prediction
    print("Features used for prediction:")
    print("Numeric features:", num_cols)
    print("Categorical features:", cat_cols)

    return metrics


def _winsorize(df: pd.DataFrame, cols: list[str], p: float = 0.01) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        lo, hi = s.quantile(p), s.quantile(1 - p)
        out[c] = s.clip(lo, hi)
    return out


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


def train_2024_complex(db: str, out_dir: str) -> dict:
    # Build dataset with mv_post and enriched features; exclude mv_pre
    df = build_dataset(
        db_path=db,
        use_all_xref=True,
        include_market_value=True,
        include_mv_pre=False,
    )

    # Residual target: y_resid = log(fee) − log(mv_post)
    df = df.copy()
    mv = pd.to_numeric(df.get("market_value_in_eur"), errors="coerce").astype(float)
    fee = pd.to_numeric(df.get("transfer_fee"), errors="coerce").astype(float)
    df = df[(mv > 0) & (fee > 0)].copy()
    df["y_resid"] = np.log1p(fee) - np.log1p(mv)

    # Feature set (exclude market_value_in_eur which is used in baseline term)
    per90_cols = [
        "xg_per90",
        "xa_per90",
        "goals_per90",
        "tackles_per90",
        "interceptions_per90",
        "pressures_per90",
    ]
    base_num = ["age", "minutes"]
    num_cols = [c for c in base_num + per90_cols if c in df.columns]
    df = _winsorize(df, per90_cols, p=0.01)

    cat_cols = [c for c in ["team_id"] if c in df.columns]

    # Include new features added in features.py for the complex variant
    num_cols.extend(["tm_height_cm"])
    cat_cols.extend(["tm_position", "tm_foot"])

    # Print the features used for prediction
    print("Features used for prediction:")
    print("Numeric features:", num_cols)
    print("Categorical features:", cat_cols)

    # Split train/val time-aware on transfer_date
    train_df, val_df = _train_val_split_time(df, test_size=0.2)
    used_cols = num_cols + cat_cols

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

    pipe = Pipeline([("prep", pre), ("model", Ridge())])

    # Simple hyperparameter search on alpha with 5-fold CV (on training set)
    param_grid = {"model__alpha": [3.0, 10.0, 30.0, 100.0]}
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipe, param_grid=param_grid, cv=cv, scoring="neg_mean_absolute_error"
    )
    search.fit(train_df[used_cols], train_df["y_resid"])
    best_pipe = search.best_estimator_

    # Evaluate on holdout (reconstruct fee)
    resid_pred = best_pipe.predict(val_df[used_cols])
    pred_log_fee = (
        np.log1p(
            pd.to_numeric(val_df["market_value_in_eur"], errors="coerce").astype(float)
        )
        + resid_pred
    )
    pred_fee = np.expm1(pred_log_fee)
    true_fee = (
        pd.to_numeric(val_df["transfer_fee"], errors="coerce").astype(float).values
    )
    rmse = float(np.sqrt(mean_squared_error(true_fee, pred_fee)))
    mae = float(mean_absolute_error(true_fee, pred_fee))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump(best_pipe, out / "model.joblib")
    metrics = {
        "variant": "complex_residual",
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "rmse_eur": rmse,
        "mae_eur": mae,
        "best_alpha": float(search.best_params_["model__alpha"]),
        "num_features": num_cols,
        "cat_features": cat_cols,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    val_out = val_df[
        [
            "player_id",
            "player_name",
            "team_name",
            "transfer_fee",
            "transfer_date",
            "market_value_in_eur",
        ]
    ].copy()
    val_out["pred_fee"] = pred_fee
    val_out["pred_log_fee"] = pred_log_fee
    val_out["pred_residual"] = resid_pred
    val_out.to_csv(out / "val_predictions_2024_complex.csv", index=False)

    # Baseline vs model (same validation split): fee_hat = market_value_in_eur
    mv_baseline = (
        pd.to_numeric(val_df.get("market_value_in_eur"), errors="coerce")
        .astype(float)
        .values
    )
    true_fee_arr = (
        pd.to_numeric(val_df["transfer_fee"], errors="coerce").astype(float).values
    )
    mask = ~np.isnan(mv_baseline)
    if mask.any():
        mae_b = float(mean_absolute_error(true_fee_arr[mask], mv_baseline[mask]))
        rmse_b = float(
            np.sqrt(mean_squared_error(true_fee_arr[mask], mv_baseline[mask]))
        )
        (out / "baseline_vs_model.json").write_text(
            json.dumps(
                {
                    "n_val": int(len(val_df)),
                    "baseline": {
                        "mae_eur": mae_b,
                        "rmse_eur": rmse_b,
                        "formula": "fee_hat = market_value_in_eur",
                    },
                    "model": {"mae_eur": mae, "rmse_eur": rmse},
                },
                indent=2,
            )
        )
    # Permutation importance on residual target (validation)
    names = _get_feature_names(best_pipe)
    result = permutation_importance(
        best_pipe,
        val_df[used_cols],
        val_df["y_resid"],
        n_repeats=10,
        random_state=RANDOM_STATE,
    )
    _plot_importance(
        names,
        result.importances_mean,
        out / "feature_importance_complex.html",
        title="Permutation importance (complex residual)",
    )

    return metrics


def kfold_basic(
    db: str,
    out_dir: str,
    alpha: float = 10.0,
    k: int = 5,
    num_cols_override: list[str] | None = None,
    cat_cols_override: list[str] | None = None,
) -> dict:
    df = build_dataset(
        db_path=db,
        use_all_xref=True,
        include_market_value=True,
        include_mv_pre=False,
    )

    # Prepare robust target in log space, dropping invalid rows
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
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in ["team_id"] if c in df.columns]

    # Include new features added in features.py for the complex variant
    num_cols.extend(["tm_height_cm"])
    cat_cols.extend(["tm_position", "tm_foot"])

    # Allow overrides (iterative sweeps)
    if num_cols_override is not None:
        num_cols = [c for c in num_cols_override if c in df.columns]
    if cat_cols_override is not None:
        cat_cols = [c for c in cat_cols_override if c in df.columns]

    # Allow overrides via function arguments by wrapping below (we'll expose from main)

    used_cols = num_cols + cat_cols

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
    pipe = Pipeline([("prep", pre), ("model", Ridge(alpha=alpha))])

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
        tmp = va[
            ["player_id", "player_name", "team_name", "transfer_fee", "transfer_date"]
        ].copy()
        tmp["pred_fee"] = pred_fee
        tmp["fold"] = fold
        oof_rows.append(tmp)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    oof = pd.concat(oof_rows, ignore_index=True)
    oof.to_csv(out / "oof_predictions_basic.csv", index=False)

    # Calculate baseline vs model metrics
    _calculate_baseline_vs_model_kfold(oof, out / "baseline_vs_model.json")

    metrics = {
        "variant": "basic_kfold",
        "folds": int(fold),
        "mae_mean_eur": float(np.mean(maes)),
        "mae_std_eur": float(np.std(maes)),
        "rmse_mean_eur": float(np.mean(rmses)),
        "rmse_std_eur": float(np.std(rmses)),
        "alpha": float(alpha),
        "num_features": num_cols,
        "cat_features": cat_cols,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # Fit on full data to produce a single importance chart
    pipe.fit(df[used_cols], df["y_log"])
    names = _get_feature_names(pipe)
    result = permutation_importance(
        pipe, df[used_cols], df["y_log"], n_repeats=10, random_state=RANDOM_STATE
    )
    _plot_importance(
        names,
        result.importances_mean,
        out / "feature_importance_basic_kfold.html",
        title="Permutation importance (basic KFold, fit on all)",
    )
    return metrics


def kfold_complex(db: str, out_dir: str, k: int = 5) -> dict:
    df = build_dataset(
        db_path=db,
        use_all_xref=True,
        include_market_value=True,
        include_mv_pre=False,
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
    df = _winsorize(df, per90_cols, p=0.01)
    cat_cols = [c for c in ["team_id", "tm_position", "tm_foot"] if c in df.columns]

    # Print the features used for prediction
    print("Features used for prediction:")
    print("Numeric features:", num_cols)
    print("Categorical features:", cat_cols)

    used_cols = num_cols + cat_cols

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

    # Use inner CV per fold to pick alpha from small grid
    param_grid = {"model__alpha": [3.0, 10.0, 30.0, 100.0]}
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    kf = KFold(n_splits=max(2, k), shuffle=True, random_state=RANDOM_STATE)
    maes, rmses = [], []
    oof_rows = []
    fold = 0
    for tr_idx, va_idx in kf.split(df):
        fold += 1
        tr, va = df.iloc[tr_idx], df.iloc[va_idx]
        pipe = Pipeline([("prep", pre), ("model", Ridge())])
        search = GridSearchCV(
            pipe, param_grid=param_grid, cv=inner_cv, scoring="neg_mean_absolute_error"
        )
        search.fit(tr[used_cols], tr["y_resid"])
        best = search.best_estimator_
        resid_pred = best.predict(va[used_cols])
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
        tmp = va[
            [
                "player_id",
                "player_name",
                "team_name",
                "transfer_fee",
                "transfer_date",
                "market_value_in_eur",
            ]
        ].copy()
        tmp["pred_fee"] = pred_fee
        tmp["fold"] = fold
        oof_rows.append(tmp)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    oof = pd.concat(oof_rows, ignore_index=True)
    oof.to_csv(out / "oof_predictions_complex.csv", index=False)

    # Calculate baseline vs model metrics
    _calculate_baseline_vs_model_kfold(oof, out / "baseline_vs_model.json")

    metrics = {
        "variant": "complex_kfold",
        "folds": int(fold),
        "mae_mean_eur": float(np.mean(maes)),
        "mae_std_eur": float(np.std(maes)),
        "rmse_mean_eur": float(np.mean(rmses)),
        "rmse_std_eur": float(np.std(rmses)),
        "num_features": num_cols,
        "cat_features": cat_cols,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # Fit best alpha on full data and plot importance
    pipe_full = Pipeline([("prep", pre), ("model", Ridge())])
    search_full = GridSearchCV(
        pipe_full, param_grid=param_grid, cv=inner_cv, scoring="neg_mean_absolute_error"
    )
    search_full.fit(df[used_cols], df["y_resid"])
    best_full = search_full.best_estimator_
    names = _get_feature_names(best_full)
    result = permutation_importance(
        best_full, df[used_cols], df["y_resid"], n_repeats=10, random_state=RANDOM_STATE
    )
    _plot_importance(
        names,
        result.importances_mean,
        out / "feature_importance_complex_kfold.html",
        title="Permutation importance (complex KFold, fit on all)",
    )
    return metrics


def _calculate_baseline_vs_model_kfold(oof: pd.DataFrame, out_path: Path) -> None:
    """Calculate baseline vs model metrics for k-fold and save to JSON."""
    # Baseline: fee_hat = market_value_in_eur
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
            "model": {
                "mae_eur": mae_m,
                "rmse_eur": rmse_m,
            },
        }

        out_path.write_text(json.dumps(metrics, indent=2))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Train 2024-only transfer fee model (basic or complex)"
    )
    p.add_argument("--db", default="data/db/euro24.sqlite")
    p.add_argument("--out", default="artifacts/model_2024_only")
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--variant", choices=["basic", "complex"], default="basic")
    p.add_argument("--cv", choices=["none", "kfold"], default="none")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--iter-features", action="store_true", help="Esegue iterazioni di feature set (basic/kfold)")
    args = p.parse_args(argv)

    if args.variant == "basic":
        if args.cv == "kfold":
            out_root = (
                args.out
                if args.out != "artifacts/model_2024_only"
                else "artifacts/model_2024_only_kfold"
            )
            if args.iter_features:
                df_tmp = build_dataset(
                    db_path=args.db,
                    use_all_xref=True,
                    include_market_value=True,
                    include_mv_pre=False,
                )
                steps = _feature_sets_basic(df_tmp)
                summary = []
                for i, st in enumerate(steps, start=1):
                    out = str(Path(out_root) / f"step_{i:02d}_{st['name']}")
                    m = kfold_basic(
                        args.db,
                        out,
                        alpha=args.alpha,
                        k=args.k,
                        num_cols_override=st["num"],
                        cat_cols_override=st["cat"],
                    )
                    summary.append({"step": st["name"], **m})
                Path(out_root).mkdir(parents=True, exist_ok=True)
                (Path(out_root) / "feature_sweep_summary.json").write_text(
                    json.dumps(summary, indent=2)
                )
                print(f"✅ 2024 KFold feature sweep completato → {out_root}
⚠️ Nota: in EURO24 kfold sweep usa set di default; per sweep preciso servirebbe estendere kfold_basic con override come in model/all.")
            else:
                m = kfold_basic(args.db, out_root, alpha=args.alpha, k=args.k)
                print(
                    f"✅ 2024-only (basic KFold {m['folds']}x). MAE={m['mae_mean_eur']:,.0f}€ ±{m['mae_std_eur']:,.0f}€  RMSE={m['rmse_mean_eur']:,.0f}€ ±{m['rmse_std_eur']:,.0f}€"
                )
                print(f"Artifacts → {out_root}")
        else:
            if args.iter_features:
                df_tmp = build_dataset(
                    db_path=args.db,
                    use_all_xref=True,
                    include_market_value=True,
                    include_mv_pre=False,
                )
                steps = _feature_sets_basic(df_tmp)
                summary = []
                for i, st in enumerate(steps, start=1):
                    out = str(Path(args.out) / f"step_{i:02d}_{st['name']}")
                    m = train_2024(
                        args.db,
                        out,
                        alpha=args.alpha,
                        num_cols_override=st["num"],
                        cat_cols_override=st["cat"],
                    )
                    summary.append({"step": st["name"], **m})
                Path(args.out).mkdir(parents=True, exist_ok=True)
                (Path(args.out) / "feature_sweep_summary.json").write_text(
                    json.dumps(summary, indent=2)
                )
                print(f"✅ 2024 Holdout feature sweep completato → {args.out}")
            else:
                m = train_2024(args.db, args.out, alpha=args.alpha)
                print(
                    f"✅ 2024-only (basic) trained. RMSE={m['rmse_eur']:,.0f}€ MAE={m['mae_eur']:,.0f}€  (n_train={m['n_train']}, n_val={m['n_val']})"
                )
                print(f"Artifacts → {args.out}")
    else:
        out_default = "artifacts/model_2024_only_complex"
        if args.cv == "kfold":
            out = (
                args.out
                if args.out != "artifacts/model_2024_only"
                else f"{out_default}_kfold"
            )
            m = kfold_complex(args.db, out, k=args.k)
            print(
                f"✅ 2024-only (complex KFold {m['folds']}x). MAE={m['mae_mean_eur']:,.0f}€ ±{m['mae_std_eur']:,.0f}€  RMSE={m['rmse_mean_eur']:,.0f}€ ±{m['rmse_std_eur']:,.0f}€"
            )
            print(f"Artifacts → {out}")
        else:
            out = args.out if args.out != "artifacts/model_2024_only" else out_default
            m = train_2024_complex(args.db, out)
            print(
                f"✅ 2024-only (complex residual) trained. RMSE={m['rmse_eur']:,.0f}€ MAE={m['mae_eur']:,.0f}€  (n_train={m['n_train']}, n_val={m['n_val']})"
            )
            print(f"Artifacts → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
