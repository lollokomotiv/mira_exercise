"""
Pipeline per costruire il DB `all_tournaments.sqlite` includendo
competizioni con nazionali (World Cup, UEFA Euro, Copa America) dal 2018.

Step:
- fetch_statsbomb.run_nationals: scarica competitions/matches/events/lineups
- load_data_polars_multi.run: JSON → CSV consolidati
- windows.compute_windows_from_matches: calcola finestre (90gg dopo le finali)
- fetch_transfers.filter_transfers_by_windows: filtra transfers.csv → transfers_all.csv
- build_xref.run: genera player_xref / club_xref
- build_schema_national.run: crea schema e carica CSV nel nuovo DB
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

from src.data_engineering.fetch_statsbomb import run_nationals as sb_run_nationals
from src.data_engineering.load_data_polars_multi import run as load_multi_run
from src.data_engineering.windows import compute_windows_from_matches, save_windows_csv
from src.data_engineering.fetch_transfers import run as tm_fetch_run
from src.data_engineering.build_xref_all import run as xref_run
from src.data_engineering.build_schema_all import run as build_db_run


def run_pipeline(
    since_year: int = 2018,
    db_path: str = "data/db/all_tournaments.sqlite",
    skip_statsbomb: bool = False,
    skip_transfers: bool = False,
    skip_load: bool = False,
    build_xref: bool = True,
) -> None:
    # 1) StatsBomb open data (multi-competition)
    if not skip_statsbomb:
        print(f"🚀 Estrazione: StatsBomb nazionali (>= {since_year})")
        sb_run_nationals(since_year)
    else:
        print("⏭️  Skip StatsBomb download")

    # 2) Trasformazione JSON → CSV
    if not skip_load:
        print("🛠️  Trasformazione: load_data_polars_multi")
        load_multi_run()
    else:
        print("⏭️  Skip load_data_polars_multi")

    # 3) Calcolo finestre 90gg e filtro transfers
    matches_csv = Path("data/processed/statsbomb/all/matches.csv")
    if not skip_transfers:
        print("📅 Calcolo finestre 90gg post-finali")
        windows = compute_windows_from_matches(matches_csv, days_after=90)
        save_windows_csv(
            windows, Path("data/processed/transfermarkt/all/windows_all.csv")
        )
        print("🚀 Estrazione/Kaggle: player-scores (se necessario) e filtro transfers")
        # Scarica file raw se mancanti; poi filtra
        tm_fetch_run(files=None)
    else:
        print("⏭️  Skip filtro transfers")

    # 4) XREF
    if build_xref:
        print("🔗 Costruzione XREF")
        xref_run()
    else:
        print("⏭️  Skip XREF")

    # 5) Build DB
    print(f"🏗️  Build schema + load CSV → {db_path}")
    build_db_run(db_path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pipeline DB nazionali (>=2018)")
    p.add_argument("--since-year", type=int, default=2018)
    p.add_argument("--db", default="data/db/all_tournaments.sqlite")
    p.add_argument("--skip-statsbomb", action="store_true")
    p.add_argument("--skip-transfers", action="store_true")
    p.add_argument("--skip-load", action="store_true")
    p.add_argument("--no-xref", action="store_true")
    p.add_argument("--all", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.all:
        run_pipeline(
            since_year=args.since_year,
            db_path=args.db,
            skip_statsbomb=False,
            skip_transfers=False,
            skip_load=False,
            build_xref=True,
        )
        return 0
    run_pipeline(
        since_year=args.since_year,
        db_path=args.db,
        skip_statsbomb=args.skip_statsbomb,
        skip_transfers=args.skip_transfers,
        skip_load=args.skip_load,
        build_xref=not args.no_xref,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
