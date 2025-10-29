"""
Create derived xref outputs for national 'all' dataset from StatsBomb + Transfermarkt using fuzzy matching.

Inputs (expected):
- data/processed/statsbomb/all/players.csv
- data/processed/transfermarkt/players_all.csv
- data/raw/transfermarkt/clubs.csv

Outputs (written to data/processed/xref/all/):
- club_xref_all.csv
- player_xref_all_full.csv (diagnostics)
- player_xref_all_review_needed.csv
- player_xref_all_strict.csv
"""

from __future__ import annotations
import csv
import unicodedata
import difflib
from pathlib import Path
from typing import Dict, Any, List, Tuple

import polars as pl

try:
    from rapidfuzz.fuzz import token_set_ratio as rf_token_set_ratio
    from rapidfuzz.fuzz import partial_ratio as rf_partial_ratio

    HAVE_RAPIDFUZZ = True
except Exception:
    HAVE_RAPIDFUZZ = False


XREF_DIR = Path("data/processed/xref/all")
PROC_SB = Path("data/processed/statsbomb/all")
RAW_TM = Path("data/raw/transfermarkt")

IN_PLAYERS_TM = RAW_TM / "players.csv"
IN_SB_PLAYERS = PROC_SB / "players.csv"

OUT_CLUB = XREF_DIR / "club_xref_all.csv"
OUT_PLAYER_FULL = XREF_DIR / "player_xref_all_full.csv"
OUT_PLAYER_REVIEW = XREF_DIR / "player_xref_all_review_needed.csv"
OUT_PLAYER_STRICT = XREF_DIR / "player_xref_all_strict.csv"


def normalize_name(name: str) -> str:
    """Normalize a name by lowercasing, removing accents, and stripping punctuation."""
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = name.lower()
    for ch in ",.;:'\"()[]{}-/&|":
        name = name.replace(ch, " ")
    return " ".join(name.split())


def _tokens(s: str) -> List[str]:
    return [t for t in s.split() if t]


def _surname_tokens(name_norm: str) -> List[str]:
    toks = _tokens(name_norm)
    if not toks:
        return []
    if len(toks) >= 2:
        return [toks[-1], toks[-2]]
    return [toks[-1]]


def _score(a: str, b: str) -> float:
    if HAVE_RAPIDFUZZ:
        ts = rf_token_set_ratio(a, b) / 100.0
        pr = rf_partial_ratio(a, b) / 100.0
        return max(ts, pr)
    return difflib.SequenceMatcher(None, a, b).ratio()


NICK_MAP: Dict[str, List[str]] = {
    "dani": ["daniel"],
    "rodri": ["rodrigo"],
    "joselu": ["jose luis"],
    "koke": ["jorge"],
    "isco": ["francisco"],
    "nico": ["nicolas", "nicholas"],
}


def _variants(name_norm: str) -> List[str]:
    toks = _tokens(name_norm)
    if not toks:
        return [name_norm]
    first = toks[0]
    vars_: List[str] = [name_norm]
    if first in NICK_MAP:
        for repl in NICK_MAP[first]:
            v = " ".join([repl] + toks[1:])
            vars_.append(v)
    return list(dict.fromkeys(vars_))


def compare_names(sb_name: str, tm_name: str, nicknames: Dict[str, List[str]]) -> float:
    """Compare two names using multiple strategies."""
    sb_norm = normalize_name(sb_name)
    tm_norm = normalize_name(tm_name)

    # Exact match
    if sb_norm == tm_norm:
        return 1.0

    # First + Last name match
    sb_parts = sb_norm.split()
    tm_parts = tm_norm.split()
    if len(sb_parts) > 1 and len(tm_parts) > 1:
        if sb_parts[0] == tm_parts[0] and sb_parts[-1] == tm_parts[-1]:
            return 0.95

    # Nickname matching
    if sb_parts[0] in nicknames:
        for nick in nicknames[sb_parts[0]]:
            if nick == tm_parts[0]:
                return 0.9

    # Fuzzy matching
    if HAVE_RAPIDFUZZ:
        return rf_token_set_ratio(sb_norm, tm_norm) / 100.0
    return difflib.SequenceMatcher(None, sb_norm, tm_norm).ratio()


def build_player_xref(
    threshold_exact: float = 0.95,
    threshold_high: float = 0.90,
    threshold_fuzzy: float = 0.81,
) -> pl.DataFrame:
    # Validate inputs
    if not IN_SB_PLAYERS.exists():
        print(f"⚠️  Missing StatsBomb players CSV: {IN_SB_PLAYERS}")
        return pl.DataFrame()
    if not IN_PLAYERS_TM.exists():
        print(f"⚠️  Missing TM players CSV: {IN_PLAYERS_TM}")
        return pl.DataFrame()

    sb_players = pl.read_csv(IN_SB_PLAYERS)
    tm_players = pl.read_csv(IN_PLAYERS_TM)

    # Prepare normalized fields and keep last_name from TM when available
    sb = sb_players.with_columns(
        [
            pl.col("player_name").map_elements(normalize_name).alias("name_norm"),
            pl.col("team_name").cast(pl.Utf8).alias("nation"),
        ]
    ).select(
        [
            "player_id",
            "player_name",
            "team_id",
            "team_name",
            "name_norm",
            "nation",
            "primary_position",
        ]
    )

    # Include last_name if present in TM
    tm = tm_players.with_columns(
        [
            pl.col("name").map_elements(normalize_name).alias("name_norm"),
            pl.col("country_of_citizenship").cast(pl.Utf8).alias("nation"),
            pl.col("last_name")
            .map_elements(normalize_name)
            .alias("last_name_norm")
            .fill_null(""),
        ]
    ).select(
        [
            "player_id",
            "name",
            "name_norm",
            "last_name_norm",
            "nation",
            "current_club_name",
            "date_of_birth",
            "sub_position",
            "position",
        ]
    )

    def _cat_from_pos(p: str | None) -> str | None:
        if not p:
            return None
        s = p.lower()
        if "keeper" in s:
            return "GK"
        if "back" in s or "defen" in s or "centre-back" in s or "center back" in s:
            return "DEF"
        if "mid" in s:
            return "MID"
        if "wing" in s or "forward" in s or "striker" in s or "attack" in s:
            return "ATT"
        return None

    # Index TM by nation to narrow pool; lowercase keys
    tm_by_nation: Dict[str, List[dict]] = {}
    tm_dicts = tm.to_dicts()
    for row in tm_dicts:
        key = (row.get("nation") or "").lower()
        tm_by_nation.setdefault(key, []).append(row)

    rows: List[Dict[str, Any]] = []

    # helper to compute score between two normalized strings
    def compute_match_score(p_name_norm: str, q_name_norm: str) -> float:
        # fast exact
        if p_name_norm == q_name_norm and p_name_norm:
            return 1.0
        base = _score(p_name_norm, q_name_norm)
        return float(base)

    nicknames = {
        "danny": ["daniel"],
        "dan": ["daniel"],
        "alex": ["alexander", "aleksander"],
        # Add more nicknames as needed
    }

    for p in sb.to_dicts():
        nation = (p.get("nation") or "").lower()
        # prefer TM players with matching nation, fallback to all
        cands = tm_by_nation.get(nation, []) or tm_dicts

        # prepare surname tokens
        surs = set(_surname_tokens(p["name_norm"]))

        best = None
        second_best_score = 0.0
        second_best_name = None

        # Evaluate candidates
        for q in cands:
            # quick exact checks on normalized name or last name
            q_name_norm = q.get("name_norm") or ""
            q_last_norm = q.get("last_name_norm") or ""
            p_name_norm = p.get("name_norm") or ""

            # compute base similarity
            base_score = compute_match_score(p_name_norm, q_name_norm)

            # start from base
            s = base_score

            # strong surname match boost
            if q_last_norm and surs and q_last_norm in surs:
                s += 0.08

            # nation match boost (we already filtered by nation but still helpful)
            tm_nation = (q.get("nation") or "").lower()
            if tm_nation and nation and tm_nation == nation:
                s += 0.06

            # token_set/token overlap proportional bonus
            p_tokens = set(_tokens(p_name_norm))
            q_tokens = set(_tokens(q_name_norm))
            if p_tokens:
                overlap = len(p_tokens & q_tokens)
                overlap_ratio = overlap / max(1, len(p_tokens))
                s += 0.05 * overlap_ratio

            # club/team name hint
            cur_club = (q.get("current_club_name") or "").lower()
            sb_team = (p.get("team_name") or "").lower()
            if cur_club and sb_team and (sb_team in cur_club or cur_club in sb_team):
                s += 0.03

            # penalize very short names producing spurious high matches
            if len(p_name_norm) < 5 or len(q_name_norm) < 5:
                s -= 0.03

            # cap
            s = min(1.0, max(0.0, s))

            diag = {
                "score_raw": round(float(base_score), 4),
                "surname_bonus": (
                    1 if (q_last_norm and surs and q_last_norm in surs) else 0
                ),
                "nation_bonus": (
                    1 if (tm_nation and nation and tm_nation == nation) else 0
                ),
                "overlap": int(len(p_tokens & q_tokens)),
                "pair_a": p_name_norm,
                "pair_b": q_name_norm,
            }

            if best is None or s > best[0]:
                if best is not None:
                    second_best_score = max(second_best_score, best[0])
                    second_best_name = best[1].get("name")
                best = (s, q, diag)
            else:
                if s > second_best_score:
                    second_best_score = s
                    second_best_name = q.get("name")

        if best is None:
            continue

        score, q, diag = best
        # determine method with margin consideration
        margin = round(float(max(0.0, score - second_best_score)), 4)
        score_capped = min(1.0, round(float(score), 4))

        if score_capped >= 0.98:
            method = "exact"
        elif score_capped >= 0.92 and margin >= 0.05:
            method = "exact"
        elif score_capped >= 0.90:
            method = "high"
        elif score_capped >= 0.82 and margin >= 0.03:
            method = "fuzzy"
        else:
            method = "low"

        sb_sur = p["name_norm"].split(" ")[-1] if p["name_norm"] else ""
        tm_sur = (
            (q.get("name_norm") or "").split(" ")[-1]
            if (q.get("name_norm") or "")
            else ""
        )
        surname_exact = 1 if sb_sur and tm_sur and sb_sur == tm_sur else 0

        sb_cat = _cat_from_pos(p.get("primary_position"))
        tm_cat = _cat_from_pos(q.get("sub_position") or q.get("position"))

        rows.append(
            {
                "statsbomb_player_id": p["player_id"],
                "statsbomb_player_name": p["player_name"],
                "tm_player_id": q["player_id"],
                "tm_player_name": q["name"],
                "confidence": score_capped,
                "method": method,
                **diag,
                "surname_exact": surname_exact,
                "position_ok": (
                    1 if (sb_cat is None or tm_cat is None or sb_cat == tm_cat) else 0
                ),
                "margin": margin,
                "second_best_name": second_best_name or "",
            }
        )

    df = pl.DataFrame(rows)
    return df


def build_club_xref() -> pl.DataFrame:
    clubs = pl.read_csv(RAW_TM / "clubs.csv")
    df = clubs.select(
        [
            pl.col("club_id").alias("tm_club_id"),
            pl.col("name").alias("tm_club_name"),
        ]
    ).with_columns([pl.col("tm_club_name").map_elements(normalize_name).alias("alias")])
    return df


def _read_rows(path: Path):
    if not path.exists():
        print(f"⚠️  Missing input file: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def _write_rows(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Wrote {len(rows)} rows -> {path}")


def _read_tm_player_ids(path: Path) -> set[str]:
    if not path.exists():
        print(f"⚠️  Missing TM players file: {path}")
        return set()
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pid = row.get("player_id")
            if pid:
                ids.add(pid.strip())
    return ids


def run() -> None:
    XREF_DIR.mkdir(parents=True, exist_ok=True)

    # Build xref from scratch using fuzzy matching on StatsBomb vs TM players
    px_all = build_player_xref()
    if px_all.is_empty():
        print("No xref generated; aborting.")
        return

    # Write full diagnostics
    px_all.write_csv(OUT_PLAYER_FULL)

    # strict: exact + high confidence (>=0.95)
    strict = px_all.filter(
        (pl.col("method") == "exact") & (pl.col("confidence") >= 0.95)
    )
    strict.write_csv(OUT_PLAYER_STRICT)

    # review: complement of strict
    review = px_all.filter(
        ~((pl.col("method") == "exact") & (pl.col("confidence") >= 0.95))
    )
    review.write_csv(OUT_PLAYER_REVIEW)

    # clubs xref
    cx = build_club_xref()
    cx.write_csv(OUT_CLUB)

    print("✅ Wrote ALL xref outputs")


if __name__ == "__main__":
    run()
