#!/usr/bin/env sh
set -euo pipefail

# Configurable settings
PYTHON_BIN="${PYTHON_BIN:-python3.12}"   # Require: python3.12 (no fallback)
REQ_FILE="requirements.txt"
VENV=".venv"

# ----------------------------
# Python selection & checks
# ----------------------------

# Require python3.12 explicitly (no fallback)
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "❌ $PYTHON_BIN non trovato. Installare Python 3.12 e riprovare."
  exit 1
fi

# Strict version check: must be exactly 3.12
VERSION=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
MAJOR=$(echo "$VERSION" | cut -d. -f1)
MINOR=$(echo "$VERSION" | cut -d. -f2)

if [ "$MAJOR" -ne 3 ] || [ "$MINOR" -ne 12 ]; then
  echo "❌ Python 3.12 richiesto, trovato $VERSION"
  exit 1
fi

echo "✅ Using Python $VERSION ($PYTHON_BIN)"

# Create a Virtual environment
if [ ! -d "$VENV" ]; then
  echo "📦 Creating virtual environment..."
  "$PYTHON_BIN" -m venv "$VENV"
fi

. "$VENV/bin/activate"

# Install Dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip >/dev/null
pip install -r "$REQ_FILE"

# Setup Plotly Chrome (for Kaleido PNG export)
SETUP_PLOTLY_CHROME="${SETUP_PLOTLY_CHROME:-1}"
if [ "$SETUP_PLOTLY_CHROME" = "1" ]; then
  if [ -x "$VENV/bin/plotly_get_chrome" ]; then
    echo "🧩 Ensuring Plotly Chrome (for kaleido)..."
    # This is idempotent; if Chrome is already present it will no-op.
    "$VENV/bin/plotly_get_chrome" || echo "⚠️  plotly_get_chrome failed; HTML exports will still work."
  else
    echo "ℹ️  plotly_get_chrome not found. Ensure 'plotly' is installed or set SETUP_PLOTLY_CHROME=0 to skip."
  fi
fi

printf "\n🚀 Running full project orchestration...\n"

# 1) Build databases (all tournaments + euro24)
printf "\n[1/4] Build databases (all_tournaments, euro24)\n"
"$VENV/bin/python" -m src.pipeline.run_all_tournaments --all --db data/db/all_tournaments.sqlite
"$VENV/bin/python" -m src.pipeline.run_euro24 --all --db data/db/euro24.sqlite

# Generate features_all.csv for all players
"$VENV/bin/python" -m src.model.all.create_features_all --all-players

# 2) EDA (general, players, teams)
printf "\n[2/4] Run EDA (general, players, teams)\n"
"$VENV/bin/python" -m src.analysis.eda_general \
  --db data/db/euro24.sqlite \
  --valuations-csv data/raw/transfermarkt/player_valuations.csv \
  --mv-cutoff-date 2024-06-13 \
  --out artifacts/eda_general \
  --euro24-only \
  --features data/processed/model/features_all.csv || true

"$VENV/bin/python" -m src.analysis.eda_players \
  --features data/processed/model/features_all.csv \
  --out artifacts/eda_players \
  --min-minutes 180 || true

"$VENV/bin/python" -m src.analysis.eda_teams \
  --features data/processed/model/features_all.csv \
  --db data/db/euro24.sqlite \
  --out artifacts/eda_teams \
  --min-minutes 180 \
  --min-matches 4 || true

# 3) Train final models (residual mixed estimators)
printf "\n[3/4] Train final models (residual mixed estimators)\n"
"$VENV/bin/python" -m src.model.train_final \
  --db data/db/all_tournaments.sqlite \
  --out artifacts/model_final_all \
  --k 5 \
  --cv-mode time \
  --min-minutes 180 \
  --top-n-holdout 5

"$VENV/bin/python" -m src.model.train_final \
  --db data/db/euro24.sqlite \
  --out artifacts/model_final_euro24 \
  --k 5 \
  --cv-mode time \
  --min-minutes 180 \
  --top-n-holdout 5

# 4) Train final RF models (+ SHAP)
printf "\n[4/4] Train final RF models (+ SHAP)\n"
"$VENV/bin/python" -m src.model.train_final_rf \
  --db data/db/all_tournaments.sqlite \
  --out artifacts/model_final_all_rf \
  --k 5 \
  --min-minutes 260 \
  --top-n-holdout 5

"$VENV/bin/python" -m src.model.train_final_rf \
  --db data/db/euro24.sqlite \
  --out artifacts/model_final_euro24_rf \
  --k 5 \
  --min-minutes 260 \
  --top-n-holdout 5

printf "\n[5/5] Export predictions for ALL EURO24 players (ALL models)\n"
# model_final_all (residual mixed estimators)
"$VENV/bin/python" -m src.model.predict_euro24 \
  --db data/db/euro24.sqlite \
  --model artifacts/model_final_all/model.joblib \
  --out artifacts/predictions_euro24_all_players.csv \
  --scope all \
  --age-ref-date 2024-07-01

# model_final_all_rf (RandomForest residual)
"$VENV/bin/python" -m src.model.predict_euro24 \
  --db data/db/euro24.sqlite \
  --model artifacts/model_final_all_rf/model.joblib \
  --out artifacts/predictions_euro24_rf_all_players.csv \
  --scope all \
  --age-ref-date 2024-07-01

printf "\n✅ Done. Artifacts are under artifacts/\n"
