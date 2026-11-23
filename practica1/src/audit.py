"""Valida la auditoría manual del test y crea test_labels.csv.

Uso:
    .venv/bin/python -m practica1.src.audit --check
    .venv/bin/python -m practica1.src.audit
"""

from __future__ import annotations

import argparse

import pandas as pd

from .config import LABELS_REVIEW_CSV, TEST_LABELS_CSV
from .train_eval import _manual_label_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Solo valida; no escribe test_labels.csv.")
    args = parser.parse_args()

    if not LABELS_REVIEW_CSV.exists():
        raise SystemExit(f"No existe {LABELS_REVIEW_CSV}. Ejecuta split primero.")
    review = pd.read_csv(LABELS_REVIEW_CSV)
    if "manual_label" not in review.columns:
        raise SystemExit("labels_review.csv no tiene columna manual_label.")
    parsed = review["manual_label"].map(_manual_label_value)
    missing = parsed.isna()
    if missing.any():
        print(f"Faltan {int(missing.sum())} de {len(review)} etiquetas manuales.")
        print("Abre labels_review.csv y llena manual_label con 0 o 1.")
        raise SystemExit(2)

    review = review.copy()
    review["manual_label"] = parsed.astype(int)
    agreement = (review["manual_label"].astype(int) == review["weak_label"].astype(int)).mean()
    print(f"Test auditado: {len(review)} fragmentos")
    print(f"Positivos manuales: {int(review['manual_label'].sum())}")
    print(f"Acuerdo con etiqueta débil: {agreement:.3f}")
    if args.check:
        return
    out = review[["fragment_id", "manual_label", "notes"]].copy()
    out.to_csv(TEST_LABELS_CSV, index=False, encoding="utf-8")
    print(f"Escrito {TEST_LABELS_CSV}")


if __name__ == "__main__":
    main()
