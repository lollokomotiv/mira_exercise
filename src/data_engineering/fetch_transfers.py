"""
fetch_transfers.py

Scarica i dati dal dataset Kaggle "davidcariboo/player-scores" e
salva i file selezionati in `data/raw/transfermarkt/`.

Inoltre, filtra `transfers.csv` per mantenere solo i trasferimenti
dal 14/07/2024 (incluso) al 30/09/2024 (incluso) e salva il risultato
come `data/processed/transfermarkt/transfers_24.csv`.

Scarica solo i seguenti file utili al task:
- `transfers.csv`
- `players.csv`
- `clubs.csv`
- `player_valuations.csv`

Strategia:
- Se tutti i file richiesti esistono già, skippa il download.
- Altrimenti scarica solo i file mancanti (se possibile) o esegue fallback al download completo.
"""

from __future__ import annotations

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional
from datetime import datetime, date
import csv

RAW_DIR = Path("data/raw/transfermarkt")
PROC_DIR_EURO24 = Path("data/processed/transfermarkt/euro24")
PROC_DIR_ALL = Path("data/processed/transfermarkt/all")
DATASET = "davidcariboo/player-scores"
DEFAULT_FILES = [
    "transfers.csv",
    "players.csv",
    "clubs.csv",
    "player_valuations.csv",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def has_kaggle_cli() -> bool:
    return shutil.which("kaggle") is not None


def ensure_kaggle_credentials() -> None:
    """Controlla credenziali; se mancano, chiede inline."""
    creds_ok = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if creds_ok or kaggle_json.exists():
        return

    # Se non ci sono, chiedi all'utente
    print("⚠️  Credenziali Kaggle mancanti.")
    username = input("Inserisci KAGGLE_USERNAME: ").strip()
    key = input("Inserisci KAGGLE_KEY: ").strip()
    if not username or not key:
        raise RuntimeError("❌ Credenziali Kaggle non fornite.")
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key


def _download_whole_dataset(dataset: str, out_dir: Path) -> None:
    """Scarica e unzip dell'intero dataset Kaggle (fallback)."""
    ensure_dir(out_dir)
    ensure_kaggle_credentials()

    if not has_kaggle_cli():
        raise RuntimeError(
            "❌ La CLI 'kaggle' non è installata o non è nel PATH. "
            "Installa con: pip install kaggle"
        )

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(out_dir),
        "--unzip",
        "--force",
    ]
    print(f"⬇️  Downloading Kaggle dataset (full): {dataset}")
    subprocess.run(cmd, check=True)


def download_specific_files(
    files: Iterable[str], dataset: str = DATASET, out_dir: Path = RAW_DIR
) -> None:
    """Scarica solo alcuni file dal dataset Kaggle usando l'opzione -f.

    Se il download selettivo fallisce, esegue il fallback sull'intero dataset.
    """
    ensure_dir(out_dir)
    ensure_kaggle_credentials()

    if not has_kaggle_cli():
        raise RuntimeError(
            "❌ La CLI 'kaggle' non è installata o non è nel PATH. "
            "Installa con: pip install kaggle"
        )

    files = list(files)
    try:
        for fname in files:
            cmd = [
                "kaggle",
                "datasets",
                "download",
                "-d",
                dataset,
                "-f",
                fname,
                "-p",
                str(out_dir),
                "--unzip",
                "--force",
            ]
            print(f"⬇️  Downloading {dataset}:{fname} -> {out_dir}")
            subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("⚠️  Download selettivo fallito. Eseguo fallback al dataset completo…")
        _download_whole_dataset(dataset, out_dir)


def filter_transfers_2024(
    src_dir: Path = RAW_DIR,
    out_dir: Path = PROC_DIR_EURO24,
    out_name: str = "transfers_euro24.csv",
    start: date | None = None,
    end: date | None = None,
) -> Path:
    """Filtra transfers.csv per il periodo post-EURO 2024.

    Default: dal 2024-07-14 (incluso) al 2024-09-30 (incluso).
    Output di default: data/processed/transfermarkt/transfers_euro24.csv
    """
    start = start or date(2024, 7, 14)
    # "prima del 1 Ottobre" < 2024-10-01; inclusivo al 30/09
    end = end or date(2024, 9, 30)

    src = src_dir / "transfers.csv"
    if not src.exists():
        print(f"ℹ️  Impossibile filtrare: file non trovato {src}")
        return src_dir / out_name

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name

    with open(src, "r", encoding="utf-8") as fin, open(
        out_path, "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames or []
        # garantisce presenza della colonna transfer_date
        if "transfer_date" not in fieldnames:
            raise RuntimeError("Colonna 'transfer_date' non presente in transfers.csv")
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            dstr = row.get("transfer_date")
            if not dstr:
                continue
            try:
                d = datetime.strptime(dstr, "%Y-%m-%d").date()
            except ValueError:
                # formatto inatteso: skippa
                continue
            if start <= d <= end:
                writer.writerow(row)

    print(f"✅ Filtrati trasferimenti {start}..{end} -> {out_path}")
    return out_path


def _read_windows_csv(windows_csv: Path) -> list[tuple[date, date]]:
    """Legge un CSV con colonne start_date,end_date e ritorna lista di tuple date."""
    if not windows_csv.exists():
        print(f"⚠️  Windows file non trovato: {windows_csv}")
        return []
    wins: list[tuple[date, date]] = []
    with open(windows_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            s = row.get("start_date")
            e = row.get("end_date")
            if not s or not e:
                continue
            try:
                st = datetime.strptime(s.strip(), "%Y-%m-%d").date()
                en = datetime.strptime(e.strip(), "%Y-%m-%d").date()
            except ValueError:
                continue
            wins.append((st, en))
    return wins


def create_transfers_all_from_windows(
    windows_csv: Path = PROC_DIR_ALL / "windows_all.csv",
    src_dir: Path = RAW_DIR,
    out_dir: Path = PROC_DIR_ALL,
    out_name: str = "transfers_all.csv",
) -> Path:
    """Crea transfers_all.csv includendo tutte le righe di transfers.csv la cui transfer_date
    cade in QUALSIASI finestra definita in windows_csv.
    """
    wins = _read_windows_csv(windows_csv)
    if not wins:
        print("⚠️  Nessuna finestra trovata — skip transfers_all")
        return out_dir / out_name
    return filter_transfers_by_windows(
        wins, src_dir=src_dir, out_dir=out_dir, out_name=out_name
    )


def filter_players_for_transfers(
    src_players: Path = RAW_DIR / "players.csv",
    transfers_filtered: Path = PROC_DIR_EURO24 / "transfers_24.csv",
    out_path: Path = PROC_DIR_EURO24 / "players_24.csv",
) -> Path:
    """Filtra players.csv mantenendo solo i player_id presenti in transfers_24.csv o transfers_all.csv.

    Usa la colonna "player_id" in entrambi i file (come presente nei CSV nel workspace).
    Restituisce il Path dell'output (out_path).
    """
    if not src_players.exists():
        print(f"⚠️  Skip filter players: source not found {src_players}")
        return out_path
    if not transfers_filtered.exists():
        print(f"⚠️  Skip filter players: transfers file not found {transfers_filtered}")
        return out_path

    # leggi player_id dai transfers filtrati
    ids: set[str] = set()
    with open(transfers_filtered, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if "player_id" not in (r.fieldnames or []):
            print(f"⚠️  Colonna 'player_id' non trovata in {transfers_filtered}; skip")
            return out_path
        for row in r:
            pid = row.get("player_id")
            if pid is None:
                continue
            pid = pid.strip()
            if not pid:
                continue
            ids.add(pid)

    # filtra players.csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(src_players, "r", encoding="utf-8") as fin, open(
        out_path, "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames or []
        if "player_id" not in fieldnames and "id" in fieldnames:
            # alcuni CSV potrebbero usare 'id' come nome della colonna
            id_col = "id"
        else:
            id_col = "player_id"
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            pid = row.get(id_col)
            if pid is None:
                continue
            if str(pid).strip() in ids:
                writer.writerow(row)
                written += 1

    print(f"✅ Wrote filtered players -> {out_path} ({written} rows)")
    return out_path


def filter_transfers_by_windows(
    windows: list[tuple[date, date]],
    src_dir: Path = RAW_DIR,
    out_dir: Path = PROC_DIR_ALL,
    out_name: str = "transfers_all.csv",
) -> Path:
    """Filtra transfers.csv mantenendo righe con transfer_date in QUALSIASI finestra.

    - `windows`: lista di tuple (start_date_incluso, end_date_incluso)
    - Scrive l'output in `out_dir/out_name`
    """
    src = src_dir / "transfers.csv"
    if not src.exists():
        print(f"ℹ️  Impossibile filtrare: file non trovato {src}")
        return out_dir / out_name

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name

    # Normalizza finestre (ordina e unisce sovrapposizioni)
    ws = sorted(windows, key=lambda w: w[0])
    merged: list[tuple[date, date]] = []
    for st, en in ws:
        if not merged or st > merged[-1][1]:
            merged.append((st, en))
        else:
            prev_st, prev_en = merged[-1]
            merged[-1] = (prev_st, max(prev_en, en))

    with open(src, "r", encoding="utf-8") as fin, open(
        out_path, "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames or []
        if "transfer_date" not in fieldnames:
            raise RuntimeError("Colonna 'transfer_date' non presente in transfers.csv")
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            dstr = row.get("transfer_date")
            if not dstr:
                continue
            try:
                d = datetime.strptime(dstr, "%Y-%m-%d").date()
            except ValueError:
                continue
            for st, en in merged:
                if st <= d <= en:
                    writer.writerow(row)
                    break

    print(f"✅ Filtrati trasferimenti per {len(merged)} finestre -> {out_path}")
    return out_path


def run(files: Optional[List[str]] = None) -> None:
    files = files or DEFAULT_FILES

    # Skip se tutti i file richiesti sono già presenti -> NON uscire: vogliamo comunque creare i processed
    ensure_dir(RAW_DIR)
    missing = [f for f in files if not (RAW_DIR / f).exists()]
    if not missing:
        print("ℹ️  Tutti i file Kaggle già presenti, skip download.")
    else:
        try:
            download_specific_files(missing)
        except subprocess.CalledProcessError as e:
            print("❌ Errore eseguendo la CLI Kaggle:", e, file=sys.stderr)
            raise

    # Filtra la finestra temporale richiesta e salva transfers_euro24.csv (processed)
    euro_path = filter_transfers_2024(
        RAW_DIR, PROC_DIR_EURO24, out_name="transfers_euro24.csv"
    )

    # Crea transfers_all.csv basato sulle finestre in data/processed/transfermarkt/windows_all.csv
    all_path = create_transfers_all_from_windows(
        PROC_DIR_ALL / "windows_all.csv",
        RAW_DIR,
        PROC_DIR_ALL,
        out_name="transfers_all.csv",
    )

    # Filtra players corrispondenti ai trasferimenti (due file: euro24 e all)
    filter_players_for_transfers(
        src_players=RAW_DIR / "players.csv",
        transfers_filtered=euro_path,
        out_path=PROC_DIR_EURO24 / "players_euro24.csv",
    )
    filter_players_for_transfers(
        src_players=RAW_DIR / "players.csv",
        transfers_filtered=all_path,
        out_path=PROC_DIR_ALL / "players_all.csv",
    )

    print("✅ File Kaggle aggiornati in data/processed/transfermarkt/")


if __name__ == "__main__":
    # Supporto CLI minimale: "--files transfers.csv players.csv"
    import argparse

    parser = argparse.ArgumentParser(
        description="Scarica file selezionati dal dataset player-scores"
    )
    parser.add_argument(
        "--files", nargs="*", default=DEFAULT_FILES, help="Lista di file da scaricare"
    )
    args = parser.parse_args()
    run(files=args.files)
