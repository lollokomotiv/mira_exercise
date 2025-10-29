"""
Generate a Mermaid ER diagram from a SQLite DB by introspecting tables and FKs.

Usage
  python -m src.data_engineering.export_schema_mermaid --db data/db/euro24.sqlite --out docs/SCHEMA_EURO24.mmd

Then embed the contents as a Mermaid code block in your README:

```mermaid
<paste file contents here>
```
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple


def _get_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    return [r[0] for r in cur.fetchall()]


def _get_columns(conn: sqlite3.Connection, table: str) -> List[Tuple[str, str, bool]]:
    """Return list of (name, type, is_pk)."""
    cur = conn.execute(f"PRAGMA table_info({table})")
    cols: List[Tuple[str, str, bool]] = []
    for cid, name, coltype, notnull, dflt, pk in cur.fetchall():
        cols.append((name, (coltype or '').upper(), bool(pk)))
    return cols


def _get_foreign_keys(conn: sqlite3.Connection, table: str) -> List[Tuple[str, str, str, str]]:
    """Return list of (from_col, ref_table, ref_col, fk_name)."""
    cur = conn.execute(f"PRAGMA foreign_key_list({table})")
    fks: List[Tuple[str, str, str, str]] = []
    for row in cur.fetchall():
        # columns: id, seq, table, from, to, on_update, on_delete, match
        _, _, ref_table, from_col, to_col, *_ = row
        fk_name = f"{table}.{from_col}→{ref_table}.{to_col}"
        fks.append((from_col, ref_table, to_col, fk_name))
    return fks


def _type_label(sql_type: str) -> str:
    t = sql_type.strip().upper()
    if not t:
        return "TEXT"
    if any(k in t for k in ("INT",)):
        return "INTEGER"
    if any(k in t for k in ("CHAR", "CLOB", "TEXT")):
        return "TEXT"
    if any(k in t for k in ("BLOB",)):
        return "BLOB"
    if any(k in t for k in ("REAL", "FLOA", "DOUB")):
        return "REAL"
    if any(k in t for k in ("NUM", "DEC")):
        return "NUMERIC"
    return t


def export_mermaid(conn: sqlite3.Connection) -> str:
    tables = _get_tables(conn)
    cols: Dict[str, List[Tuple[str, str, bool]]] = {t: _get_columns(conn, t) for t in tables}
    fks_by_child: Dict[str, List[Tuple[str, str, str, str]]] = {
        t: _get_foreign_keys(conn, t) for t in tables
    }

    lines: List[str] = ["erDiagram"]
    # Entities
    for t in tables:
        lines.append(f"  {t} {{")
        for name, coltype, is_pk in cols[t]:
            label = _type_label(coltype)
            suffix = " PK" if is_pk else ""
            lines.append(f"    {label} {name}{suffix}")
        lines.append("  }")

    # Relationships (parent ||--o{ child)
    for child, fks in fks_by_child.items():
        for from_col, parent, to_col, fk_name in fks:
            rel_label = f'"{from_col}→{to_col}"'
            lines.append(f"  {parent} ||--o{{ {child} : {rel_label}")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export Mermaid ER diagram from a SQLite DB")
    p.add_argument("--db", required=True, help="Path to SQLite DB")
    p.add_argument("--out", required=True, help="Path to output .mmd file")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.db)
    try:
        mermaid = export_mermaid(conn)
    finally:
        conn.close()
    out.write_text(mermaid, encoding="utf-8")
    print(f"✅ Mermaid ER exported → {out}")
    print("Paste this into a Markdown code block with ```mermaid to render it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

