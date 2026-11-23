"""Limpieza, anonimización, tokenización y fragmentación de texto."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable, List, Optional, Sequence

from .config import MAX_CHARS, MIN_CHARS, MIN_LETTER_RATIO, MIN_LETTERS, TARGET_CHARS

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HANDLE_RE = re.compile(r"(?<![\w.])@[\w.]{2,}", re.IGNORECASE)
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?52[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{3,4}[\s.-]\d{3,4}(?:[\s.-]?\d{2,4})?(?!\d)"
)
SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"\b\w\w+\b", re.UNICODE)


def strip_accents(text: str) -> str:
    """Quita acentos preservando letras base (equivalente a strip_accents='unicode')."""
    text = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def normalize_whitespace(text: str) -> str:
    return SPACE_RE.sub(" ", (text or "").replace("\x00", " ")).strip()


def normalize_for_match(text: str) -> str:
    return normalize_whitespace(strip_accents(text).lower())


def sha256_text(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def scrub_pii(text: str, known_names: Optional[Iterable[str]] = None) -> str:
    """Elimina identificadores evidentes.

    No intenta detectar todos los nombres personales; para eso se revisa el
    test manualmente. Sí elimina correos, URLs, handles, teléfonos y los nombres
    de equipos/universidades/autores que se le pasen.
    """
    text = text or ""
    text = EMAIL_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = HANDLE_RE.sub(" ", text)
    text = PHONE_RE.sub(" ", text)
    if known_names:
        # Primero las frases largas para evitar reemplazos parciales.
        for name in sorted({n.strip() for n in known_names if n and len(n.strip()) >= 4}, key=len, reverse=True):
            # Escape y límites suaves; no usamos \b porque hay acentos y signos.
            pattern = re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)", re.IGNORECASE)
            text = pattern.sub(" ", text)
    return normalize_whitespace(text)


def tokenize(text: str) -> List[str]:
    """Tokenizador compatible con `TfidfVectorizer(token_pattern=r'\\b\\w\\w+\\b')`."""
    return TOKEN_RE.findall(strip_accents(text or "").lower())


def is_quality_text(text: str) -> bool:
    """Filtra fragmentos de tablas, ruido OCR y títulos."""
    text = normalize_whitespace(text)
    if len(text) < MIN_CHARS:
        return False
    letters = sum(1 for ch in text if ch.isalpha())
    digits = sum(1 for ch in text if ch.isdigit())
    visible = max(1, len(text))
    if letters < MIN_LETTERS:
        return False
    if letters / visible < MIN_LETTER_RATIO:
        return False
    if digits / visible > 0.55:
        return False
    # Texto casi todo en mayúsculas y corto: probable título.
    upper = sum(1 for ch in text if ch.isupper())
    if letters and upper / letters > 0.85 and len(text) < 140:
        return False
    # Evita cadenas de un solo carácter o repeticiones.
    if len(set(text.lower())) < 8:
        return False
    return True


def _hard_split(sentence: str, max_chars: int) -> List[str]:
    words = sentence.split()
    out: List[str] = []
    buf = ""
    for word in words:
        candidate = (buf + " " + word).strip()
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                out.append(buf)
            buf = word
    if buf:
        out.append(buf)
    return out


def chunk_text(text: str, target: int = TARGET_CHARS, max_chars: int = MAX_CHARS, min_chars: int = MIN_CHARS) -> List[str]:
    """Fragmenta por oraciones en bloques de tamaño objetivo.

    No hay solapamiento: evita que un mismo contenido aparezca en train y test.
    """
    text = normalize_whitespace(text)
    if len(text) < min_chars:
        return []
    sentences = re.split(r"(?<=[.!?;:])\s+", text)
    chunks: List[str] = []
    buf = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            for part in _hard_split(sentence, max_chars):
                if len(part) >= min_chars and is_quality_text(part):
                    chunks.append(part)
            continue
        candidate = (buf + " " + sentence).strip() if buf else sentence
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if len(buf) >= min_chars and is_quality_text(buf):
                chunks.append(buf)
            buf = sentence
    if len(buf) >= min_chars and is_quality_text(buf):
        chunks.append(buf)
    return chunks


def dedup_key(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", normalize_for_match(text)).strip()
    return sha256_text(normalized, length=24)
