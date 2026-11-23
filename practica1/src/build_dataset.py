"""Construye el dataset de fragmentos etiquetados a partir de analysis.json.

Estrategia principal: segmentar por headings. Cada heading con sección canónica
abre un segmento que termina en el siguiente heading (aunque ese siguiente no
tenga sección). Esto evita mezclar, por ejemplo, "Inversión" con "Operaciones"
en una misma página.

Si un documento no produce segmentos, se usa un fallback por página.

Uso:
    .venv/bin/python -m practica1.src.build_dataset

El resultado es local (practica1/data/fragments.csv). No se publica en la PWA.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from .config import (
    AMBIGUOUS_SECTIONS,
    ANALYSIS_JSON,
    DATASET_MANIFEST,
    EXCLUDED_SECTIONS,
    FRAGMENTS_CSV,
    MAX_CHARS,
    MIN_CHARS,
    TARGET_CHARS,
    NEGATIVE_SECTIONS,
    POSITIVE_SECTIONS,
    chosen_section,
    label_for_sections,
)
from .text import chunk_text, dedup_key, is_quality_text, normalize_for_match, normalize_whitespace, scrub_pii, sha256_text


def _load_json(path) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _team_key(team_id: str) -> str:
    return sha256_text("team:" + str(team_id), 12)


def _doc_key(doc_id: str) -> str:
    return sha256_text("doc:" + str(doc_id), 16)


def _known_names(analysis: Dict) -> Set[str]:
    names: Set[str] = set()
    for team in analysis.get("teams", []):
        for key in ("name", "university"):
            value = (team.get(key) or "").strip()
            if value and len(value) >= 4:
                names.add(value)
        context = team.get("context") or {}
        for key in ("name", "university"):
            value = (context.get(key) or "").strip()
            if value and len(value) >= 4:
                names.add(value)
    for doc in (analysis.get("documents") or {}).values():
        author = ((doc.get("metadata") or {}).get("Author") or "").strip()
        if author:
            for part in re.split(r"[,;&]| y | and ", author):
                part = part.strip()
                if len(part) >= 4:
                    names.add(part)
    names.update(
        {
            "Tecnológico de Monterrey",
            "Tecnologico de Monterrey",
            "ITESM",
            "UNAM",
            "IPN",
            "UPAEP",
            "UAM",
            "UPQ",
            "UPIIT",
            "UPIIG",
            "UPIIH",
            "ESIME",
        }
    )
    return {n for n in names if len(n) >= 4}


def _heading_lines(doc: Dict) -> Set[Tuple[int, str]]:
    out: Set[Tuple[int, str]] = set()
    for heading in (doc.get("structure") or {}).get("headings") or []:
        text = heading.get("text") or ""
        if text:
            out.add((int(heading.get("page") or 0), normalize_for_match(text)))
    return out


def _repeated_margin_lines(doc: Dict) -> Set[str]:
    """Líneas cortas que se repiten en la mayoría de las páginas (headers/footers)."""
    counts: Counter = Counter()
    pages = doc.get("pages") or []
    for page in pages:
        seen = set()
        for line in page.get("lines") or []:
            text = normalize_for_match(line.get("text") or "")
            if not text or len(text) > 120:
                continue
            seen.add(text)
        counts.update(seen)
    threshold = max(3, int(len(pages) * 0.5) + 1)
    return {text for text, count in counts.items() if count >= threshold}


def _page_text(
    doc: Dict,
    page: Dict,
    headings: Set[Tuple[int, str]],
    margins: Set[str],
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> str:
    page_no = int(page.get("number") or 0)
    lines: List[str] = []
    for line in page.get("lines") or []:
        text = (line.get("text") or "").strip()
        if not text:
            continue
        top = float(line.get("top") or 0)
        if y_min is not None and top <= y_min + 1:
            continue
        if y_max is not None and top >= y_max - 1:
            continue
        norm = normalize_for_match(text)
        if not norm:
            continue
        if (page_no, norm) in headings:
            continue
        if norm in margins:
            continue
        if re.fullmatch(r"\d{1,3}", norm):
            continue
        letters = sum(1 for ch in norm if ch.isalpha())
        if letters < 3:
            continue
        lines.append(text)
    return " ".join(lines)


def _page_section_map(doc: Dict) -> Dict[int, Set[str]]:
    """Fallback: mapea páginas a secciones usando las ventanas de section_hits."""
    hits = (doc.get("structure") or {}).get("section_hits") or {}
    windows: List[Tuple[int, str, Set[int]]] = []
    n_pages = max(1, len(doc.get("pages") or []))
    for sid, hit in hits.items():
        start = int(hit.get("page") or 0)
        pages = {int(p) for p in (hit.get("pages") or []) if str(p).isdigit()}
        if not pages and start:
            pages = {start}
        pages = {p for p in pages if 1 <= p <= n_pages}
        if start and pages:
            windows.append((start, sid, pages))

    result: Dict[int, Set[str]] = {}
    for page in range(1, n_pages + 1):
        candidates = [(start, sid) for start, sid, pages in windows if page in pages]
        if not candidates:
            continue
        latest = max(start for start, _ in candidates)
        result[page] = {sid for start, sid in candidates if start == latest}
    return result


def _iter_heading_segments(
    doc: Dict,
    known_names: Set[str],
    target_chars: int = TARGET_CHARS,
    max_chars: int = MAX_CHARS,
) -> Iterable[Dict]:
    headings_all = [
        h
        for h in ((doc.get("structure") or {}).get("headings") or [])
        if (h.get("text") or "").strip()
    ]
    headings_all.sort(key=lambda h: (int(h.get("page") or 0), float(h.get("top") or 0)))
    if not headings_all:
        return
    heading_set = _heading_lines(doc)
    margins = _repeated_margin_lines(doc)
    pages = {int(p.get("number") or 0): p for p in (doc.get("pages") or [])}
    n_pages = max(pages) if pages else 0

    for idx, heading in enumerate(headings_all):
        sections = set(heading.get("sections") or [])
        label_name = label_for_sections(sections)
        if label_name not in {"pos", "neg"}:
            continue
        section = chosen_section(sections)
        if not section:
            continue
        p0 = int(heading.get("page") or 0)
        top0 = float(heading.get("top") or 0)
        if p0 <= 0 or p0 > n_pages:
            continue
        nxt = headings_all[idx + 1] if idx + 1 < len(headings_all) else None
        p1 = int(nxt.get("page") or 0) if nxt else n_pages + 1
        top1 = float(nxt.get("top") or 0) if nxt else None

        pieces: List[str] = []
        for page_no in range(p0, min(p1, n_pages) + 1):
            page = pages.get(page_no)
            if not page:
                continue
            pieces.append(
                _page_text(
                    doc,
                    page,
                    heading_set,
                    margins,
                    y_min=top0 if page_no == p0 else None,
                    y_max=top1 if (nxt and page_no == p1) else None,
                )
            )
        text = normalize_whitespace(" ".join(pieces))
        text = scrub_pii(text, known_names)
        for chunk in chunk_text(text, target=target_chars, max_chars=max_chars, min_chars=MIN_CHARS):
            if not is_quality_text(chunk):
                continue
            yield {
                "section": section,
                "label": 1 if label_name == "pos" else 0,
                "label_name": label_name,
                "text": chunk,
                "page": p0,
            }


def _iter_page_fallback(
    doc: Dict,
    known_names: Set[str],
    target_chars: int = TARGET_CHARS,
    max_chars: int = MAX_CHARS,
) -> Iterable[Dict]:
    page_sections = _page_section_map(doc)
    headings = _heading_lines(doc)
    margins = _repeated_margin_lines(doc)
    toc_pages = set((doc.get("structure") or {}).get("toc", {}).get("pages") or [])
    for page in doc.get("pages") or []:
        page_no = int(page.get("number") or 0)
        if page_no in toc_pages:
            continue
        sections = page_sections.get(page_no)
        if not sections:
            continue
        label_name = label_for_sections(sections)
        if label_name not in {"pos", "neg"}:
            continue
        section = chosen_section(sections)
        if not section:
            continue
        raw_text = _page_text(doc, page, headings, margins)
        text = scrub_pii(raw_text, known_names)
        for chunk in chunk_text(text, target=target_chars, max_chars=max_chars, min_chars=MIN_CHARS):
            if not is_quality_text(chunk):
                continue
            yield {
                "section": section,
                "label": 1 if label_name == "pos" else 0,
                "label_name": label_name,
                "text": chunk,
                "page": page_no,
            }


def build_dataset() -> pd.DataFrame:
    analysis = _load_json(ANALYSIS_JSON)
    known_names = _known_names(analysis)
    rows: List[Dict] = []
    seen = set()

    for doc in (analysis.get("documents") or {}).values():
        doc_id = str(doc.get("id") or "")
        team_id = str(doc.get("team_id") or "")
        if not doc_id or not team_id:
            continue
        dkey = _doc_key(doc_id)
        tkey = _team_key(team_id)

        segments = list(_iter_heading_segments(doc, known_names))
        if not segments:
            segments = list(_iter_page_fallback(doc, known_names))

        for seg in segments:
            key = dedup_key(seg["text"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "fragment_id": sha256_text(f"{dkey}|{seg['page']}|{seg['section']}|{key}", 16),
                    "doc_key": dkey,
                    "team_key": tkey,
                    "section": seg["section"],
                    "label": int(seg["label"]),
                    "label_name": seg["label_name"],
                    "text": seg["text"],
                    "char_len": len(seg["text"]),
                    "page": int(seg["page"]),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No se generaron fragmentos. Revisa analysis.json y la asignación de secciones.")
    df = df.sort_values(["doc_key", "page", "fragment_id"]).reset_index(drop=True)
    FRAGMENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FRAGMENTS_CSV, index=False, encoding="utf-8")

    by_label = Counter(df["label_name"])
    by_section = Counter(df["section"])
    manifest = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_analysis": str(ANALYSIS_JSON),
        "source_analysis_sha256": _sha256_file(ANALYSIS_JSON),
        "fragments_sha256": _sha256_file(FRAGMENTS_CSV),
        "fragment_count": int(len(df)),
        "label_counts": dict(by_label),
        "positive_rate": float(by_label.get("pos", 0) / len(df)),
        "section_counts": dict(sorted(by_section.items())),
        "document_count": int(df["doc_key"].nunique()),
        "team_count": int(df["team_key"].nunique()),
        "char_stats": {
            "min": int(df["char_len"].min()),
            "median": float(df["char_len"].median()),
            "mean": float(df["char_len"].mean()),
            "max": int(df["char_len"].max()),
        },
        "params": {
            "strategy": "heading_segments_with_page_fallback",
            "target_chars": TARGET_CHARS,
            "max_chars": MAX_CHARS,
            "min_chars": MIN_CHARS,
            "deduplicated": True,
            "overlap": False,
            "positive_sections": sorted(POSITIVE_SECTIONS),
            "negative_sections": sorted(NEGATIVE_SECTIONS),
            "excluded_sections": sorted(EXCLUDED_SECTIONS),
            "ambiguous_sections": sorted(AMBIGUOUS_SECTIONS),
        },
        "pii": {
            "scrubbed": ["emails", "urls", "handles", "phones", "team_names", "university_names", "authors"],
            "known_name_count": len(known_names),
        },
        "notes": [
            "Segmentación principal por headings; fallback por página si el documento no tiene headings útiles.",
            "Etiqueta débil a nivel de sección/heading; el test se audita manualmente.",
            "Los textos se limpian antes de escribir el CSV local.",
            "fragments.csv no se publica en GitHub Pages.",
        ],
    }
    DATASET_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return df


def main() -> None:
    df = build_dataset()
    print(f"Fragmentos: {len(df)}")
    print(df["label_name"].value_counts().to_string())
    print("\nPor sección:")
    print(df["section"].value_counts().to_string())
    print(f"\nDocumentos: {df['doc_key'].nunique()}  Equipos: {df['team_key'].nunique()}")
    print(f"CSV: {FRAGMENTS_CSV}")
    print(f"Manifiesto: {DATASET_MANIFEST}")


if __name__ == "__main__":
    main()
