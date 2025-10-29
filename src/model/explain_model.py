"""
Model explanation utilities (feature contributions and importance).

Two paths:
- Linear model (Ridge): show standardized coefficients (coef * std of feature)
- If SHAP is available and model is tree-based, compute SHAP values (optional)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import load
from sklearn.inspection import permutation_importance

def _select_build_dataset(db_path: str):
    """Return the appropriate build_dataset function based on DB path.

    - If the path hints at all tournaments (e.g., contains 'all_tournaments'),
      use src.model.all.features.build_dataset
    - Otherwise default to euro24 builder
    """
    from importlib import import_module

    db_l = str(db_path).lower()
    if "all_tournaments" in db_l or "national" in db_l:
        return import_module("src.model.all.features").build_dataset
    return import_module("src.model.euro24.features").build_dataset


def _get_feature_names(pipe) -> list[str]:
    # Retrieve feature names from ColumnTransformer + OneHot
    prep = pipe.named_steps.get("prep")
    feat_names: list[str] = []
    if prep is None:
        return feat_names
    for name, trans, cols in prep.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if hasattr(trans, "get_feature_names_out"):
            # For OneHotEncoder
            got = list(trans.get_feature_names_out(cols))
        else:
            got = list(cols)
        feat_names.extend(got)
    return feat_names


def explain(
    model_path: Path | str = "artifacts/model/model.joblib",
    db: Path | str = "data/db/euro24.sqlite",
    use_all_xref: bool = False,
    include_market_value: bool = False,
    out_dir: Path | str = "artifacts/model",
) -> dict:
    pipe = load(model_path)
    build_dataset = _select_build_dataset(str(db))
    df = build_dataset(
        db,
        use_all_xref=use_all_xref,
        include_market_value=include_market_value,
    )
    df = df.copy()
    df["y_log"] = np.log1p(df["transfer_fee"].astype(float))

    # Use last 20% as validation (same logic as train)
    if "transfer_date" in df.columns and df["transfer_date"].notna().any():
        df = df.sort_values("transfer_date")
        split = max(1, int(len(df) * 0.8))
        val_df = df.iloc[split:].copy()
    else:
        val_df = df.sample(frac=0.2, random_state=42)

    # Identify columns used by the pipeline
    prep = pipe.named_steps["prep"]
    num_cols = prep.transformers_[0][2]
    cat_cols = prep.transformers_[1][2] if len(prep.transformers_) > 1 else []
    X_val = val_df[num_cols + cat_cols]
    y_val = val_df["y_log"]

    # Permutation importance (on log-target)
    result = permutation_importance(pipe, X_val, y_val, n_repeats=10, random_state=42, n_jobs=1)
    feat_names = _get_feature_names(pipe)
    imp = (
        pd.DataFrame({
            "feature": feat_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    imp.to_csv(out / "feature_importance_permutation.csv", index=False)

    # Try SHAP if available and model supports it
    shap_summary_path = out / "feature_importance_shap.csv"
    shap_done = False
    try:
        import shap  # type: ignore

        # KernelExplainer can be used as a fallback for any model; sample for speed
        X_sample = X_val.sample(min(200, len(X_val)), random_state=42)
        explainer = shap.Explainer(pipe.predict, X_sample)
        shap_values = explainer(X_sample)
        # Aggregate absolute mean per feature
        vals = np.abs(shap_values.values).mean(axis=0)
        names = _get_feature_names(pipe)
        shap_imp = pd.DataFrame({"feature": names, "shap_importance": vals}).sort_values("shap_importance", ascending=False)
        shap_imp.to_csv(shap_summary_path, index=False)
        shap_done = True
    except Exception:
        shap_done = False

    report = {
        "permutation_importance_csv": str((out / "feature_importance_permutation.csv").as_posix()),
        "shap_importance_csv": str(shap_summary_path.as_posix()) if shap_done else None,
    }
    (out / "explain_report.json").write_text(json.dumps(report, indent=2))
    print(f"✅ Saved permutation importance → {out/'feature_importance_permutation.csv'}")
    if shap_done:
        print(f"✅ Saved SHAP importance → {shap_summary_path}")
    else:
        print("ℹ️ SHAP skipped (not available or failed)")
    return report


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Explain trained model with permutation importance and SHAP (optional)")
    p.add_argument("--model", default="artifacts/model/model.joblib")
    p.add_argument("--db", default="data/db/euro24.sqlite")
    p.add_argument("--use-all-xref", action="store_true")
    p.add_argument("--include-market-value", action="store_true")
    p.add_argument("--out", default="artifacts/model")
    args = p.parse_args(argv)

    explain(args.model, db=args.db, use_all_xref=args.use_all_xref, include_market_value=args.include_market_value, out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
