"""
RandomForest variants for 2024-only model (basic and complex).
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

from .features import build_dataset

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

    # Handle NULL and empty string values in features
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
                elif col == "tm_height_cm":
                    df[col] = df[col].fillna(df[col].median())
                elif col == "market_value_in_eur":
                    df[col] = df[col].fillna(df[col].median())

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
    train_df, val_df = _train_val_split_time(df, test_size=0.2)

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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train 2024-only model with RandomForest")
    p.add_argument("--db", default="data/db/euro24.sqlite")
    p.add_argument("--out", default="artifacts/model_2024_only_basic_rf")
    p.add_argument("--shap", action="store_true", help="Calcola SHAP importance")
    args = p.parse_args(argv)
    m = train_basic(args.db, args.out, shap_eval=args.shap)
    print(
        f"✅ 2024-only RF trained. RMSE={m['rmse_eur']:,.0f}€ MAE={m['mae_eur']:,.0f}€ (n_train={m['n_train']}, n_val={m['n_val']})"
    )
    print(f"Artifacts → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
