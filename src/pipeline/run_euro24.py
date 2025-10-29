"""
Pipeline per costruire il DB `euro24.sqlite` includendo
competizioni EURO 2024.

Step:
- fetch_statsbomb.run: scarica competitions/matches/events/lineups
- load_data_polars.run: JSON → CSV consolidati
- fetch_transfers.run: filtra transfers.csv → transfers_euro24.csv
- build_xref_euro24.run: genera player_xref / club_xref
- build_schema_euro24.run: crea schema e carica CSV nel nuovo DB
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_engineering.fetch_statsbomb import run as fetch_statsbomb_run
from src.data_engineering.load_data_polars import run as load_data_polars_run
from src.data_engineering.fetch_transfers import run as fetch_transfers_run
from src.data_engineering.build_xref_euro24 import run as xref_run
from src.data_engineering.build_schema_euro24 import run as build_db_run


def run_pipeline(
    db_path: str = "data/db/euro24.sqlite",
    skip_statsbomb: bool = False,
    skip_transfers: bool = False,
    skip_load: bool = False,
    build_xref: bool = True,
) -> None:
    # 1) StatsBomb open data (EURO 2024)
    if not skip_statsbomb:
        print("🚀 Estrazione: StatsBomb EURO 2024")
        fetch_statsbomb_run()
    else:
        print("⏭️  Skip StatsBomb download")

    # 2) Trasformazione JSON → CSV
    if not skip_load:
        print("🛠️  Trasformazione: load_data_polars")
        load_data_polars_run()
    else:
        print("⏭️  Skip load_data_polars")

    # 3) Filtro transfers
    if not skip_transfers:
        print("🚀 Estrazione/Kaggle: player-scores (se necessario) e filtro transfers")
        fetch_transfers_run()
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
    p = argparse.ArgumentParser(description="Pipeline DB EURO 2024")
    p.add_argument("--db", default="data/db/euro24.sqlite")
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
            db_path=args.db,
            skip_statsbomb=False,
            skip_transfers=False,
            skip_load=False,
            build_xref=True,
        )
        return 0
    run_pipeline(
        db_path=args.db,
        skip_statsbomb=args.skip_statsbomb,
        skip_transfers=args.skip_transfers,
        skip_load=args.skip_load,
        build_xref=not args.no_xref,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
