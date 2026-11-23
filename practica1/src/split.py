"""Split temporal agrupado por equipo.

Reglas:
- La fecha del equipo es la fecha máxima de sus documentos.
- Los equipos más recientes van a test; luego validación; el resto a train.
- Ningún equipo aparece en dos splits.
- Los equipos sin fecha se quedan en train y se declaran como limitación.

Uso:
    .venv/bin/python -m practica1.src.split
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .build_dataset import (
    _doc_key,
    _iter_heading_segments,
    _iter_page_fallback,
    _known_names,
)
from .config import (
    ANALYSIS_JSON,
    AUG_MAX_CHARS,
    AUG_TARGET_CHARS,
    FRAGMENTS_CSV,
    FRAGMENTS_SPLIT_CSV,
    LABELS_REVIEW_CSV,
    MIN_TEST_FRAGMENTS,
    MIN_TEST_NEGATIVES,
    MIN_TEST_POSITIVES,
    SPLIT_JSON,
    TEST_FRACTION,
    TRAIN_MIN_FRAGMENTS,
    VAL_FRACTION,
)
from .text import dedup_key, sha256_text


def _load_json(path) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    match = re.match(r"D:(\d{4})(\d{2})(\d{2})", str(value))
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def _team_key(team_id: str) -> str:
    return sha256_text("team:" + str(team_id), 12)


def _team_dates(analysis: Dict) -> Dict[str, Optional[date]]:
    dates: Dict[str, List[date]] = defaultdict(list)
    for doc in (analysis.get("documents") or {}).values():
        team = str(doc.get("team_id") or "")
        if not team:
            continue
        metadata = doc.get("metadata") or {}
        dt = _parse_date(metadata.get("CreationDate")) or _parse_date(metadata.get("ModDate"))
        if dt:
            dates[team].append(dt)
    out: Dict[str, Optional[date]] = {}
    for team in {str(d.get("team_id")) for d in (analysis.get("documents") or {}).values()}:
        out[team] = max(dates[team]) if dates.get(team) else None
    return out


def _counts(frame: pd.DataFrame) -> Dict:
    labels = Counter(frame["label"].astype(int).tolist())
    return {
        "fragments": int(len(frame)),
        "positive": int(labels.get(1, 0)),
        "negative": int(labels.get(0, 0)),
    }


def _accumulate(teams: List[Dict], target_fragments: int, min_pos: int, min_neg: int) -> List[Dict]:
    selected: List[Dict] = []
    total = pos = neg = 0
    for team in teams:
        selected.append(team)
        total += team["fragments"]
        pos += team["positive"]
        neg += team["negative"]
        if total >= target_fragments and pos >= min_pos and neg >= min_neg:
            break
    return selected


def _augment_train(
    df: pd.DataFrame,
    analysis: Dict,
    train_keys: set,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """Añade fragmentos finos solo a train si faltan ejemplos para el mínimo.

    Los fragmentos extra salen de documentos cuyos equipos ya están en train.
    Nunca toca val/test. Se deduplican contra todo el dataset principal.
    """
    frame = df.copy()
    frame["augmented"] = 0
    current = int((frame["split"] == "train").sum())
    need = max(0, TRAIN_MIN_FRAGMENTS - current)
    meta = {
        "target_train_fragments": TRAIN_MIN_FRAGMENTS,
        "base_train_fragments": current,
        "added": 0,
        "method": "extra_small_chunks_from_train_documents",
        "chunk_target_chars": AUG_TARGET_CHARS,
        "chunk_max_chars": AUG_MAX_CHARS,
        "test_untouched": True,
    }
    if need == 0:
        return frame, meta

    known_names = _known_names(analysis)
    seen = {dedup_key(str(text)) for text in frame["text"].astype(str)}
    extras: List[Dict] = []
    for doc in (analysis.get("documents") or {}).values():
        team_key = _team_key(str(doc.get("team_id") or ""))
        if team_key not in train_keys:
            continue
        doc_key = _doc_key(str(doc.get("id") or ""))
        segments = list(_iter_heading_segments(doc, known_names, AUG_TARGET_CHARS, AUG_MAX_CHARS))
        if not segments:
            segments = list(_iter_page_fallback(doc, known_names, AUG_TARGET_CHARS, AUG_MAX_CHARS))
        for seg in segments:
            key = dedup_key(seg["text"])
            if key in seen:
                continue
            seen.add(key)
            extras.append(
                {
                    "fragment_id": "aug_" + sha256_text(f"{doc_key}|{seg['page']}|{seg['section']}|{key}", 16),
                    "doc_key": doc_key,
                    "team_key": team_key,
                    "section": seg["section"],
                    "label": int(seg["label"]),
                    "label_name": seg["label_name"],
                    "text": seg["text"],
                    "char_len": len(seg["text"]),
                    "page": int(seg["page"]),
                    "split": "train",
                    "augmented": 1,
                }
            )

    if len(extras) < need:
        raise SystemExit(
            f"Faltan {need - len(extras)} fragmentos para llegar a {TRAIN_MIN_FRAGMENTS} de train. "
            "Agrega más reportes PDF y vuelve a construir el dataset."
        )
    if len(extras) > need:
        extras = pd.DataFrame(extras).sample(n=need, random_state=seed).to_dict("records")
    frame = pd.concat([frame, pd.DataFrame(extras)], ignore_index=True)
    # Orden estable para que el CSV sea reproducible.
    frame = frame.sort_values(["split", "doc_key", "page", "fragment_id"]).reset_index(drop=True)
    meta["added"] = len(extras)
    meta["train_fragments_after"] = int((frame["split"] == "train").sum())
    return frame, meta


def build_split() -> pd.DataFrame:
    if not FRAGMENTS_CSV.exists():
        raise SystemExit(f"Falta {FRAGMENTS_CSV}. Ejecuta build_dataset primero.")

    df = pd.read_csv(FRAGMENTS_CSV)
    analysis = _load_json(ANALYSIS_JSON)
    team_dates = _team_dates(analysis)

    # Conteos por team_key.
    grouped = {}
    for team_key, frame in df.groupby("team_key", sort=False):
        labels = Counter(frame["label"].astype(int).tolist())
        grouped[str(team_key)] = {
            "team_key": str(team_key),
            "fragments": int(len(frame)),
            "positive": int(labels.get(1, 0)),
            "negative": int(labels.get(0, 0)),
        }

    # Mapear team_key -> fecha usando el team_id original.
    key_to_date: Dict[str, Optional[date]] = {}
    for team_id, dt in team_dates.items():
        key_to_date[_team_key(team_id)] = dt
    for key, row in grouped.items():
        row["date"] = key_to_date.get(key)

    dated = [row for row in grouped.values() if row.get("date") is not None]
    undated = [row for row in grouped.values() if row.get("date") is None]
    # Más recientes primero.
    dated.sort(key=lambda r: (r["date"], r["team_key"]), reverse=True)

    total_fragments = len(df)
    target_test = max(MIN_TEST_FRAGMENTS, int(math.ceil(total_fragments * TEST_FRACTION)))
    test_teams = _accumulate(
        dated,
        target_fragments=target_test,
        min_pos=MIN_TEST_POSITIVES,
        min_neg=MIN_TEST_NEGATIVES,
    )

    if not test_teams:
        raise SystemExit("No hay equipos con fecha para construir un test temporal.")

    test_keys = {t["team_key"] for t in test_teams}
    remaining = [t for t in dated if t["team_key"] not in test_keys]
    target_val = max(60, int(math.ceil(total_fragments * VAL_FRACTION)))
    val_teams = _accumulate(remaining, target_fragments=target_val, min_pos=20, min_neg=20)

    val_keys = {t["team_key"] for t in val_teams}
    train_teams = [t for t in remaining if t["team_key"] not in val_keys] + undated

    method = "temporal_grouped_by_team"
    fallback_reason = None
    test_counts = _counts(df[df["team_key"].isin({t["team_key"] for t in test_teams})])
    if test_counts["positive"] < MIN_TEST_POSITIVES:
        raise SystemExit(
            "El test temporal no alcanzó el mínimo de positivos. "
            "Amplía el test con más equipos recientes o documenta un fallback."
        )

    team_to_split = {}
    for team in train_teams:
        team_to_split[team["team_key"]] = "train"
    for team in val_teams:
        team_to_split[team["team_key"]] = "val"
    for team in test_teams:
        team_to_split[team["team_key"]] = "test"

    df = df.copy()
    df["split"] = df["team_key"].map(team_to_split)
    if df["split"].isna().any():
        missing = sorted(df[df["split"].isna()]["team_key"].unique().tolist())
        raise SystemExit(f"Equipos sin split: {missing}")
    leakage = df.groupby("team_key")["split"].nunique()
    if (leakage > 1).any():
        bad = sorted(leakage[leakage > 1].index.tolist())
        raise SystemExit(f"Fuga de equipo entre splits: {bad}")
    for split_name in ("train", "val", "test"):
        if not (df["split"] == split_name).any():
            raise SystemExit(f"El split {split_name} quedó vacío.")

    train_keys = {t["team_key"] for t in train_teams}
    df, augmentation = _augment_train(df, analysis, train_keys)
    if df["split"].isna().any():
        raise SystemExit("El refuerzo de train produjo filas sin split.")

    FRAGMENTS_SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FRAGMENTS_SPLIT_CSV, index=False, encoding="utf-8")

    splits = {}
    for name in ("train", "val", "test"):
        frame = df[df["split"] == name]
        counts = _counts(frame)
        counts["teams"] = int(frame["team_key"].nunique())
        counts["documents"] = int(frame["doc_key"].nunique())
        counts["team_dates"] = sorted(
            {
                str(team["date"])
                for team in (train_teams if name == "train" else val_teams if name == "val" else test_teams)
                if team.get("date")
            }
        )
        splits[name] = counts

    payload = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "method": method,
        "fallback_reason": fallback_reason,
        "dataset_sha256": hashlib.sha256(FRAGMENTS_CSV.read_bytes()).hexdigest(),
        "counts": splits,
        "teams": {
            "train": sorted(team_to_split.keys() - {t["team_key"] for t in val_teams} - {t["team_key"] for t in test_teams}),
            "val": sorted({t["team_key"] for t in val_teams}),
            "test": sorted({t["team_key"] for t in test_teams}),
        },
        "undated_teams_in_train": sorted({t["team_key"] for t in undated}),
        "augmentation": augmentation,
        "notes": [
            "Ningún equipo aparece en más de un split.",
            "El test se etiqueta manualmente al 100% antes de la corrida final.",
            "Los equipos sin fecha se asignan a train.",
            "El refuerzo de train usa fragmentos más finos de documentos de train; val/test no cambian.",
        ],
    }
    SPLIT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not LABELS_REVIEW_CSV.exists():
        review = df[df["split"] == "test"][
            ["fragment_id", "doc_key", "team_key", "section", "page", "label", "text"]
        ].copy()
        review = review.rename(columns={"label": "weak_label"})
        review["manual_label"] = ""
        review["notes"] = ""
        review.to_csv(LABELS_REVIEW_CSV, index=False, encoding="utf-8")
        print(f"Creado {LABELS_REVIEW_CSV} ({len(review)} fragmentos de test).")
    else:
        print(f"Se conserva {LABELS_REVIEW_CSV} existente.")

    return df


def main() -> None:
    df = build_split()
    print(f"Split: {SPLIT_JSON}")
    for name in ("train", "val", "test"):
        sub = df[df["split"] == name]
        print(
            f"{name:5s}: {len(sub):4d} fragmentos | "
            f"pos={int((sub['label'] == 1).sum()):3d} neg={int((sub['label'] == 0).sum()):3d} | "
            f"equipos={sub['team_key'].nunique():2d} docs={sub['doc_key'].nunique():2d}"
        )


if __name__ == "__main__":
    main()
