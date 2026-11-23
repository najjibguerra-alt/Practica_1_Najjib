"""Configuración de la Práctica 1.

Todas las rutas son relativas al repositorio. El código no modifica el dashboard
existente; solo lee `data/analysis.json`.
"""

from __future__ import annotations

from pathlib import Path

PRACTICA_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PRACTICA_DIR.parent

ANALYSIS_JSON = REPO_DIR / "data" / "analysis.json"
TEXT_DIR = REPO_DIR / "data" / "text"

DATA_DIR = PRACTICA_DIR / "data"
ARTIFACTS_DIR = PRACTICA_DIR / "artifacts"
PWA_DIR = PRACTICA_DIR / "pwa"

FRAGMENTS_CSV = DATA_DIR / "fragments.csv"
FRAGMENTS_SPLIT_CSV = DATA_DIR / "fragments_split.csv"
DATASET_MANIFEST = DATA_DIR / "dataset_manifest.json"
SPLIT_JSON = DATA_DIR / "split.json"
LABELS_REVIEW_CSV = DATA_DIR / "labels_review.csv"
TEST_LABELS_CSV = DATA_DIR / "test_labels.csv"

MODEL_JSON = ARTIFACTS_DIR / "model.json"
METRICS_JSON = ARTIFACTS_DIR / "metrics.json"
PWA_MODEL_JSON = PWA_DIR / "model.json"

# Etiqueta binaria: 1 = finanzas/riesgo; 0 = mercado/propuesta/operación.
POSITIVE_SECTIONS = frozenset({"inversion", "ingresos", "financiera", "riesgos"})
NEGATIVE_SECTIONS = frozenset(
    {
        "cliente",
        "mercado",
        "competencia",
        "segmentacion",
        "solucion",
        "vehiculo",
        "ventas",
        "marketing",
        "operaciones",
        "suministro",
        "organizacion",
    }
)
EXCLUDED_SECTIONS = frozenset({"portada", "indice", "resumen", "conclusion", "anexos"})
AMBIGUOUS_SECTIONS = frozenset(
    {"empresa", "mision", "sustentabilidad", "implementacion", "posventa"}
)

SECTION_PRIORITY = {
    **{s: 0 for s in POSITIVE_SECTIONS},
    **{s: 1 for s in NEGATIVE_SECTIONS},
    **{s: 2 for s in AMBIGUOUS_SECTIONS},
    **{s: 3 for s in EXCLUDED_SECTIONS},
}

# Fragmentación principal.
TARGET_CHARS = 450
MAX_CHARS = 650
MIN_CHARS = 80

# Refuerzo solo de train: fragmentos más finos de documentos que ya están
# en train. Nunca se aplica a val/test.
TRAIN_MIN_FRAGMENTS = 500
AUG_TARGET_CHARS = 260
AUG_MAX_CHARS = 350
MIN_LETTER_RATIO = 0.40
MIN_LETTERS = 40

# Split.
TEST_FRACTION = 0.20
VAL_FRACTION = 0.15
MIN_TEST_FRAGMENTS = 120
MIN_TEST_POSITIVES = 30
MIN_TEST_NEGATIVES = 40

# Modelo.
RANDOM_STATE = 42
MAX_FEATURES = 12000
MIN_DF = 2
HIDDEN_UNITS = 12
MLP_ALPHA = 1e-3

# Costos.
COST_FP = 1.0
COST_FN = 5.0
COST_SENSITIVITY = (3.0, 10.0)

# OOD / inferencia.
OOD_MIN_CHARS = 80
OOD_MIN_TOKENS = 8
OOD_MAX_OOV_RATE = 0.60
OOD_SPANISH_MIN_HITS = 2
OOD_SPANISH_MIN_TOKENS = 20
OOD_REVIEW_MARGIN = 0.08  # |p - umbral| <= margen => revisar manualmente


def label_for_sections(sections) -> str | None:
    """Devuelve 'pos', 'neg' o None para un conjunto de secciones.

    Si una página toca varias secciones, gana la de mayor prioridad:
    positivo > negativo > ambiguo > excluido.
    """
    sections = set(sections)
    if not sections:
        return None
    if sections & POSITIVE_SECTIONS:
        return "pos"
    if sections & NEGATIVE_SECTIONS:
        return "neg"
    if sections & AMBIGUOUS_SECTIONS:
        return "amb"
    if sections & EXCLUDED_SECTIONS:
        return "exc"
    return None


def chosen_section(sections) -> str | None:
    """Sección representativa para auditoría, con la misma prioridad."""
    sections = set(sections)
    if not sections:
        return None
    for group in (POSITIVE_SECTIONS, NEGATIVE_SECTIONS, AMBIGUOUS_SECTIONS, EXCLUDED_SECTIONS):
        matches = sorted(sections & group)
        if matches:
            return matches[0]
    return sorted(sections)[0]
