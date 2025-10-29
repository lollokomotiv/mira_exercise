# DATA SCIENCE & ENGINEERING EXERCISE — README

Questo file spiega la struttura del progetto, come allestire l’ambiente, come eseguire lo script di orchestrazione, e quali assunzioni minime servono per riprodurre tutto. I report tecnici e narrativi sono in `docs/`:
- `docs/README_DATAENG.md` (data model e pipeline dati)
- `docs/README_EDA.md` (analisi esplorativa)
- `docs/README_MODEL.md` (modelli, selezione, predizioni)

## Struttura delle cartelle
- `data/raw/…`: sorgenti (StatsBomb JSON/CSV, Transfermarkt CSV)
- `data/processed/…`: CSV normalizzati/filtrati e xref (strict e all)
- `data/db/…`: database SQLite costruiti (`euro24.sqlite`, `all_tournaments.sqlite`)
- `src/data_engineering/`: script per schema e caricamento CSV
  - `build_schema_euro24.py`, `build_schema_all.py`
  - `fetch_transfers.py`, `build_xref_euro24.py`, `build_xref_all.py`, `windows.py`
- `src/pipeline/`: pipeline end‑to‑end per costruire i DB
  - `run_euro24.py`, `run_all_tournaments.py`
- `src/analysis/`: EDA generale/giocatori/squadre
  - `eda_general.py` (richiamato anche come `src/analysis/eda.py`), `eda_players.py`, `eda_teams.py`
- `src/model/`: training e predizione
  - `create_features_all.py`: creazione del file con tutte le features usato in fase di EDA
  - `train_final.py`, `train_final_rf.py` (selezione time‑aware + guardrail)
  - `predict_euro24.py` (predizioni per EURO24, anche “all players” con XRef STRICT)
- `artifacts/`: output di EDA e modelli (metrics, grafici, modelli .joblib, predizioni)
- `run.sh`: script di orchestrazione (setup venv → build DB → EDA → training → predizioni)

## Setup dell’ambiente
- Requisito: Python 3.12, bash/zsh su UNIX‑like; spazio su disco per CSV/SQLite.
- Lo script `run.sh` crea e attiva automaticamente un virtualenv `.venv` e installa le dipendenze da `requirements.txt`.

Esecuzione manuale (opzionale):
```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Esecuzione completa (script di orchestrazione)
Lo script `run.sh` esegue in sequenza:
1) Creazione del venv Python, attivazione dello stesso e installazione e dei requirements
2) Fetch dei dati dalle repository di StatsBomb e Transfermarkt
3) Build dei DB `all_tournaments.sqlite` e `euro24.sqlite`
4) EDA (generale, giocatori, squadre) con output in `artifacts/`
5) Training dei modelli finali (`model_final`, `model_final_rf`) per ALL ed EURO24
6) Predizioni finali per tutti i giocatori EURO24 usando il modello allenato su ALL

Comando:
```bash
./run.sh
```

**Prerequisiti dati**
- Lo script va a fare la fetch dei dati dalle repository di StatsBomb e Transfermarkt. Il download via Kaggle (Transfermarkt) è opzionale in quanto richiede un account personale e l'inserimento di variabili d’ambiente (`KAGGLE_USERNAME`, `KAGGLE_KEY`) se si decide di automatizzarlo; per questo motivo, i CSV di Transfermarkt sono già presenti nella folder `data/raw/transfermarkt`.

## Esecuzioni mirate (facoltative)
- Solo EDA generale (come in run.sh):
```bash
. .venv/bin/activate
python -m src.analysis.eda_general \
  --db data/db/euro24.sqlite \
  --valuations-csv data/raw/transfermarkt/player_valuations.csv \
  --mv-cutoff-date 2024-06-13 \
  --out artifacts/eda_general \
  --euro24-only \
  --features data/processed/model/features_all.csv
```
- Predizioni EURO24 “all players” (XRef STRICT) con modello ALL (regressione):
```bash
. .venv/bin/activate
python -m src.model.predict_euro24 \
  --db data/db/euro24.sqlite \
  --model artifacts/model_final_all/model.joblib \
  --out artifacts/predictions_euro24_all_players.csv \
  --scope all --age-ref-date 2024-07-01
```
- Predizioni EURO24 “all players” con modello ALL Random Forest:
```bash
. .venv/bin/activate
python -m src.model.predict_euro24 \
  --db data/db/euro24.sqlite \
  --model artifacts/model_final_all_rf/model.joblib \
  --out artifacts/predictions_euro24_rf_all_players.csv \
  --scope all --age-ref-date 2024-07-01
```

## Assunzioni e note
- Python 3.12; non sono richieste altre dipendenze di sistema oltre a quelle installate dal `requirements.txt`.
- Alcuni grafici EDA sono HTML interattivi e vanno caricati sul browser per una migliore analisi.
