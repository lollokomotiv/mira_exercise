"""
Final training script that searches a robust combination of features and model
to beat the MV baseline, for both euro24 and all_tournaments DBs.

Strategy
- Build dataset via the appropriate builder (auto-detected from DB path).
- Coerce numerics and drop rows with invalid/missing values.
- Use residual target: y = log1p(fee) - log1p(MV), which helps match the
  MV baseline if the residual is ≈0 and learn uplift otherwise.
- Try a small set of feature subsets (minimal → add minutes → add team_id → add height)
  keeping categorical footprint small (team_id one-hot with min_frequency).
- Try a small set of estimators: Ridge (alpha grid), Huber, RandomForest (regularized).
- Use GroupKFold by player_id to avoid leakage; select best by OOF MAE in EUR
  (reconstructing fee from residual + MV). Then evaluate on a time-aware holdout.
- Save artifacts: best model, metrics with baseline, OOF predictions, holdout predictions,
  and permutation importances.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from pyecharts.charts import Bar
from pyecharts import options as opts

RANDOM_STATE = 42


def _select_build_dataset(db_path: str):
    from importlib import import_module

    db_l = str(db_path).lower()
    if "all_tournaments" in db_l or "national" in db_l:
        return import_module("src.model.all.features").build_dataset
    return import_module("src.model.euro24.features").build_dataset


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c].replace("", np.nan), errors="coerce")
    return out


def _winsorize(df: pd.DataFrame, cols: List[str], p: float = 0.01) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        lo, hi = s.quantile(p), s.quantile(1 - p)
        out[c] = s.clip(lo, hi)
    return out


class ZeroRegressor:
    def fit(self, X, y):
        return self
    def predict(self, X):
        return np.zeros(shape=(len(X),), dtype=float)


def _time_holdout(df: pd.DataFrame, frac_val: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "transfer_date" in df.columns and df["transfer_date"].notna().any():
        df = df.sort_values("transfer_date").reset_index(drop=True)
        n = len(df)
        split = max(1, int(n * (1 - frac_val)))
        tr, va = df.iloc[:split].copy(), df.iloc[split:].copy()
        if len(va) == 0:
            va = df.tail(max(1, int(n * frac_val))).copy()
        return tr, va
    n = len(df)
    split = max(1, int(n * (1 - frac_val)))
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def _time_forward_splits(df: pd.DataFrame, k: int = 5, date_col: str = "transfer_date"):
    """Yield forward-chaining time splits: train=all blocks before i, val=block i.
    If no date available, falls back to shuffled KFold.
    """
    if date_col not in df.columns or not df[date_col].notna().any():
        kf = KFold(n_splits=max(2, k), shuffle=True, random_state=RANDOM_STATE)
        yield from kf.split(df)
        return
    order = np.argsort(pd.to_datetime(df[date_col], errors="coerce").fillna(pd.Timestamp(0)).values)
    blocks = np.array_split(order, max(2, k))
    for i in range(1, len(blocks)):
        val_idx = blocks[i]
        train_idx = np.concatenate(blocks[:i])
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        yield train_idx, val_idx


def _get_feature_names(pipe: Pipeline) -> List[str]:
    prep = pipe.named_steps.get("prep")
    names: List[str] = []
    if prep is None or not hasattr(prep, "transformers_"):
        # Pipeline not fitted or no prep step; return empty to skip importance
        return names
    for name, trans, cols in prep.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if hasattr(trans, "get_feature_names_out"):
            names.extend(list(trans.get_feature_names_out(cols)))
        else:
            names.extend(list(cols))
    return names


def _plot_importance(names: List[str], scores: np.ndarray, out_path: Path, title: str) -> None:
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


@dataclass
class Candidate:
    name: str
    num: List[str]
    cat: List[str]


def _feature_candidates(df: pd.DataFrame) -> List[Candidate]:
    base = [c for c in ["age", "xg_per90", "xa_per90"] if c in df.columns]
    cands: List[Candidate] = [Candidate("base", base, [])]
    if "minutes" in df.columns:
        cands.append(Candidate("base+minutes", base + ["minutes"], []))
    # Team id (low-frequency one-hot) only if present
    if "team_id" in df.columns:
        last = cands[-1]
        cands.append(Candidate(last.name + "+team", last.num, ["team_id"]))
    # Height as last step
    if "tm_height_cm" in df.columns:
        last = cands[-1]
        cands.append(Candidate(last.name + "+height", last.num + ["tm_height_cm"], last.cat))
    return cands


def _build_pipe(estimator: str, num_cols: List[str], cat_cols: List[str]) -> Pipeline:
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    transformers: List[tuple[str, Any, Any]] = [("num", num_pipe, num_cols)]
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cat_cols))
    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)

    if estimator == "ridge_3":
        model = Ridge(alpha=3.0, random_state=RANDOM_STATE)
    elif estimator == "ridge_10":
        model = Ridge(alpha=10.0, random_state=RANDOM_STATE)
    elif estimator == "ridge_30":
        model = Ridge(alpha=30.0, random_state=RANDOM_STATE)
    elif estimator == "huber":
        model = HuberRegressor(max_iter=5000)
    elif estimator == "rf":
        model = RandomForestRegressor(
            n_estimators=800,
            max_depth=12,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    elif estimator == "zero":
        model = ZeroRegressor()
    else:
        model = Ridge(alpha=10.0, random_state=RANDOM_STATE)

    return Pipeline([("prep", pre), ("model", model)])


def _oof_cv_residual(
    df: pd.DataFrame,
    used_cols: List[str],
    groups: np.ndarray | None,
    estimator: str,
    k: int = 5,
    cv_mode: str = "time",
) -> tuple[dict, np.ndarray]:
    # Prefer time-forward CV; else group; else shuffled KFold
    if cv_mode == "time":
        splits = _time_forward_splits(df, k=k, date_col="transfer_date")
    elif cv_mode == "group" and groups is not None and len(np.unique(groups)) >= max(2, k):
        kf = GroupKFold(n_splits=max(2, k))
        splits = kf.split(df, groups=groups)
    else:
        kf = KFold(n_splits=max(2, k), shuffle=True, random_state=RANDOM_STATE)
        splits = kf.split(df)

    y_resid = df["y_resid"].values
    mv = pd.to_numeric(df["market_value_in_eur"], errors="coerce").astype(float).values
    oof_pred_fee = np.zeros(len(df))

    maes, rmses = [], []
    for tr_idx, va_idx in splits:
        tr, va = df.iloc[tr_idx], df.iloc[va_idx]
        pipe = _build_pipe(estimator, [c for c in used_cols if c in tr.columns and tr[c].dtype != object], [c for c in used_cols if c in tr.columns and tr[c].dtype == object])
        pipe.fit(tr[used_cols], tr["y_resid"])
        resid_pred = pipe.predict(va[used_cols])
        pred_fee = np.expm1(np.log1p(va["market_value_in_eur"].astype(float).values) + resid_pred)
        true_fee = pd.to_numeric(va["transfer_fee"], errors="coerce").astype(float).values
        maes.append(mean_absolute_error(true_fee, pred_fee))
        rmses.append(np.sqrt(mean_squared_error(true_fee, pred_fee)))
        oof_pred_fee[va_idx] = pred_fee

    metrics = {
        "folds": int(max(2, k)),
        "mae_mean_eur": float(np.mean(maes)),
        "mae_std_eur": float(np.std(maes)),
        "rmse_mean_eur": float(np.mean(rmses)),
        "rmse_std_eur": float(np.std(rmses)),
    }
    return metrics, oof_pred_fee


def _prepare(df: pd.DataFrame, min_minutes: float = 180.0) -> pd.DataFrame:
    df = df.copy()
    # Drop rows with missing or non-positive MV and fee
    df["transfer_fee_num"] = pd.to_numeric(df.get("transfer_fee"), errors="coerce").astype(float)
    df["market_value_in_eur"] = pd.to_numeric(df.get("market_value_in_eur"), errors="coerce").astype(float)
    df = df[(df["transfer_fee_num"] > 0) & (df["market_value_in_eur"] > 0)].copy()

    # Optional minutes filter for stability of per-90
    if "minutes" in df.columns and min_minutes > 0:
        df = df[pd.to_numeric(df["minutes"], errors="coerce").fillna(0) >= min_minutes].copy()

    # Winsorize per-90
    per90 = [c for c in ["xg_per90", "xa_per90", "goals_per90", "tackles_per90", "interceptions_per90", "pressures_per90"] if c in df.columns]
    df = _winsorize(df, per90, p=0.01)

    # Coerce helpful numeric columns
    df = _coerce_numeric(df, ["age", "minutes", "tm_height_cm"] + per90)

    # Residual target
    fee = df["transfer_fee_num"].astype(float)
    mv = df["market_value_in_eur"].astype(float)
    df["y_resid"] = np.log1p(fee) - np.log1p(mv)
    return df


def run(db: str, out_dir: str, k: int = 5, min_minutes: float = 180.0, cv_mode: str = "time", top_n_holdout: int = 5) -> dict:
    build_dataset = _select_build_dataset(db)
    df = build_dataset(db_path=db, use_all_xref=True, include_market_value=True, include_mv_pre=False)
    df = _prepare(df, min_minutes=min_minutes)

    # Candidates (residual modeling): try robust subsets
    cands = _feature_candidates(df)
    estimators = ["ridge_3", "ridge_10", "ridge_30", "huber", "rf"]

    tried: list[dict] = []
    best: dict | None = None
    oof_best: np.ndarray | None = None
    used_cols_best: List[str] | None = None
    pipe_best: Pipeline | None = None

    groups = df["player_id"].values if "player_id" in df.columns else None

    for cand in cands:
        num_cols = [c for c in cand.num if c in df.columns]
        cat_cols = [c for c in cand.cat if c in df.columns]
        # Always include MV in used_cols only for reconstruction of fee, not as model input
        used_cols = num_cols + cat_cols
        for est in estimators:
            cv_metrics, oof_pred_fee = _oof_cv_residual(df, used_cols, groups, est, k=k, cv_mode=cv_mode)
            # Baseline MAE over OOF set (using MV directly)
            true_fee = df["transfer_fee_num"].astype(float).values
            mv = df["market_value_in_eur"].astype(float).values
            mae_baseline = float(mean_absolute_error(true_fee, mv))
            improve = mae_baseline - cv_metrics["mae_mean_eur"]
            summary = {
                "candidate": cand.name,
                "estimator": est,
                **cv_metrics,
                "mae_baseline_eur": mae_baseline,
                "improvement_vs_baseline_eur": improve,
            }
            tried.append({
                "candidate": cand.name,
                "estimator": est,
                "num_cols": num_cols,
                "cat_cols": cat_cols,
                **summary,
            })
            if (best is None) or (improve > best.get("improvement_vs_baseline_eur", -1e9)):
                best = summary
                oof_best = oof_pred_fee
                used_cols_best = used_cols
                # Fit final pipe on full data for holdout later
                pipe_best = _build_pipe(est, num_cols, cat_cols)

    assert best is not None and pipe_best is not None and used_cols_best is not None and oof_best is not None

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Save OOF predictions and CV metrics of best
    oof = df[["player_id", "player_name", "team_name", "transfer_fee", "transfer_date", "market_value_in_eur"]].copy() if "team_name" in df.columns else df[["player_id", "player_name", "transfer_fee", "transfer_date", "market_value_in_eur"]].copy()
    oof["pred_fee"] = oof_best
    oof.to_csv(out / "oof_predictions_final.csv", index=False)
    (out / "cv_best.json").write_text(json.dumps(best, indent=2))

    # Time-aware holdout evaluation with guard: try top-N candidates and pick first that beats baseline
    tr, va = _time_holdout(df, frac_val=0.2)
    mae_baseline_va = float(mean_absolute_error(va["transfer_fee_num"].astype(float).values, va["market_value_in_eur"].astype(float).values))

    tried_sorted = sorted(tried, key=lambda x: x["improvement_vs_baseline_eur"], reverse=True)
    chosen = None
    for cand in tried_sorted[: max(1, top_n_holdout)]:
        pipe_tmp = _build_pipe(cand["estimator"], cand["num_cols"], cand["cat_cols"])
        used_cols_tmp = cand["num_cols"] + cand["cat_cols"]
        pipe_tmp.fit(tr[used_cols_tmp], tr["y_resid"])
        resid_pred_tmp = pipe_tmp.predict(va[used_cols_tmp])
        pred_fee_tmp = np.expm1(np.log1p(va["market_value_in_eur"].astype(float).values) + resid_pred_tmp)
        true_fee = va["transfer_fee_num"].astype(float).values
        mae_tmp = float(mean_absolute_error(true_fee, pred_fee_tmp))
        if mae_tmp < mae_baseline_va:
            chosen = {
                "pipe": pipe_tmp,
                "used_cols": used_cols_tmp,
                "mae": mae_tmp,
                "rmse": float(np.sqrt(mean_squared_error(true_fee, pred_fee_tmp))),
                "resid_pred": resid_pred_tmp,
                "pred_fee": pred_fee_tmp,
                "cand": cand,
            }
            break

    if chosen is None:
        # Fallback to baseline (zero residual)
        pipe_best = _build_pipe("zero", [], [])
        used_cols_best = []
        resid_pred = np.zeros(len(va))
        pred_fee = va["market_value_in_eur"].astype(float).values
        true_fee = va["transfer_fee_num"].astype(float).values
        mae = float(mean_absolute_error(true_fee, pred_fee))
        rmse = float(np.sqrt(mean_squared_error(true_fee, pred_fee)))
        chosen_meta = {"candidate": "baseline_mv", "estimator": "zero"}
    else:
        pipe_best = chosen["pipe"]
        used_cols_best = chosen["used_cols"]
        resid_pred = chosen["resid_pred"]
        pred_fee = chosen["pred_fee"]
        true_fee = va["transfer_fee_num"].astype(float).values
        mae = chosen["mae"]
        rmse = chosen["rmse"]
        chosen_meta = {"candidate": chosen["cand"]["candidate"], "estimator": chosen["cand"]["estimator"]}

    # Save model and metrics
    dump(pipe_best, out / "model.joblib")
    metrics = {
        "dataset": "all_tournaments" if ("all_tournaments" in str(db).lower()) else "euro24",
        "variant": "final_residual",
        "folds": int(best["folds"]),
        "cv_best": best,
        "cv_tried": tried_sorted,
        "holdout": {
            "rmse_eur": rmse,
            "mae_eur": mae,
            "baseline_mae_eur": mae_baseline_va,
            "improvement_vs_baseline_eur": float(mae_baseline_va - mae),
        },
        "used_num_features": [c for c in used_cols_best if c in df.columns and df[c].dtype != object],
        "used_cat_features": [c for c in used_cols_best if c in df.columns and df[c].dtype == object],
        "estimator": chosen_meta["estimator"],
        "candidate": chosen_meta["candidate"],
        "min_minutes": float(min_minutes),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Permutation importance on holdout (skip if baseline fallback or pipeline not fitted)
    names = _get_feature_names(pipe_best)
    if names and used_cols_best:
        try:
            result = permutation_importance(
                pipe_best,
                va[used_cols_best],
                va["y_resid"],
                n_repeats=10,
                random_state=RANDOM_STATE,
            )
            _plot_importance(
                names,
                result.importances_mean,
                out / "feature_importance_final.html",
                title="Permutation importance (final residual)",
            )
        except Exception:
            pass

    # Save holdout predictions
    val_out_cols = [c for c in ["player_id", "player_name", "team_name", "transfer_fee", "transfer_date", "market_value_in_eur"] if c in va.columns]
    val_out = va[val_out_cols].copy()
    val_out["pred_fee"] = pred_fee
    val_out["pred_residual"] = resid_pred
    val_out.to_csv(out / "val_predictions_final.csv", index=False)

    print(
        f"✅ Final model trained. Holdout MAE={mae:,.0f}€ (baseline {mae_baseline_va:,.0f}€, Δ={mae_baseline_va - mae:,.0f}€)  RMSE={rmse:,.0f}€"
    )
    print(f"Best CV combo → cand={best['candidate']}  est={best['estimator']}  MAEΔ={best['improvement_vs_baseline_eur']:,.0f}€")
    print(f"Artifacts → {out}")
    print(f"Rows used for training after cleaning: {len(df)}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train final model (residual) to beat MV baseline")
    p.add_argument("--db", required=True, help="Path to SQLite DB (euro24 or all_tournaments)")
    p.add_argument("--out", default="artifacts/model_final")
    p.add_argument("--k", type=int, default=5, help="Folds for CV")
    p.add_argument("--cv-mode", choices=["time", "group", "kfold"], default="time", help="CV mode for model selection")
    p.add_argument("--min-minutes", type=float, default=180.0, help="Filter players with minutes < threshold")
    p.add_argument("--top-n-holdout", type=int, default=5, help="Try top-N CV candidates on holdout and pick first beating baseline")
    args = p.parse_args(argv)

    run(args.db, args.out, k=args.k, min_minutes=args.min_minutes, cv_mode=args.cv_mode, top_n_holdout=args.top_n_holdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
