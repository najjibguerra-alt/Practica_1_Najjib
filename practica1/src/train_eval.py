"""Entrena TF-IDF + MLP, calibra, barre el umbral y exporta el modelo a la PWA.

Uso provisional (sin etiquetas manuales de test):
    .venv/bin/python -m practica1.src.train_eval --provisional

Uso final (después de llenar practica1/data/test_labels.csv):
    .venv/bin/python -m practica1.src.train_eval
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier

from .config import (
    ARTIFACTS_DIR,
    COST_FP,
    COST_FN,
    DATASET_MANIFEST,
    FRAGMENTS_SPLIT_CSV,
    HIDDEN_UNITS,
    MAX_FEATURES,
    METRICS_JSON,
    MIN_DF,
    MLP_ALPHA,
    MODEL_JSON,
    OOD_MAX_OOV_RATE,
    OOD_MIN_CHARS,
    OOD_MIN_TOKENS,
    OOD_REVIEW_MARGIN,
    OOD_SPANISH_MIN_HITS,
    OOD_SPANISH_MIN_TOKENS,
    PRACTICA_DIR,
    PWA_DIR,
    PWA_MODEL_JSON,
    RANDOM_STATE,
    SPLIT_JSON,
    TEST_LABELS_CSV,
)
from .text import normalize_for_match, normalize_whitespace, sha256_text, strip_accents, tokenize

# Algunos builds de scikit-learn/BLAS reportan matmuls con ceros como
# divide/overflow/invalid durante lbfgs y predict_proba. Los pesos finales se
# verifican con np.isfinite; el warning no aporta información útil aquí.
warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)

SPANISH_STOPWORDS = {
    "el", "la", "de", "que", "y", "en", "un", "una", "los", "las", "del", "al",
    "se", "con", "por", "para", "es", "su", "sus", "como", "mas", "pero", "sin",
    "sobre", "entre", "este", "esta", "estos", "estas", "o", "no", "lo", "le",
    "les", "ya", "muy", "tambien", "puede", "desde", "hasta", "cuando", "donde",
    "porque", "cual", "quien", "ser", "son", "fue", "han", "hay", "nos", "les",
}

FINANCIAL_KEYWORDS = {
    "inversion", "inversiones", "costo", "costos", "ingreso", "ingresos", "precio",
    "precios", "margen", "margenes", "utilidad", "utilidades", "rentabilidad",
    "flujo", "flujos", "presupuesto", "capex", "van", "tir", "roi", "payback",
    "financiero", "financiera", "riesgo", "riesgos", "revenue", "cost", "costs",
    "pricing", "margin", "profit", "investment", "budget", "cash", "npv", "irr",
    "forecast", "break", "even", "retorno", "capital", "financing", "financiamiento",
}


def _load_json(path) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _sigmoid(z):
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def _ece(y_true, p, bins: int = 10) -> Tuple[float, List[Dict]]:
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    ece = 0.0
    curve = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        n = int(mask.sum())
        if n == 0:
            curve.append({"bin": i, "lo": float(lo), "hi": float(hi), "n": 0, "confidence": None, "accuracy": None})
            continue
        conf = float(p[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (n / total) * abs(acc - conf)
        curve.append({"bin": i, "lo": float(lo), "hi": float(hi), "n": n, "confidence": conf, "accuracy": acc})
    return float(ece), curve


def _fit_temperature(logits, y_true) -> float:
    logits = np.asarray(logits, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    def nll(log_t):
        t = float(np.exp(log_t))
        p = _sigmoid(logits / t)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

    result = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(result.x))


def _cost(y_true, y_pred, cost_fp: float = COST_FP, cost_fn: float = COST_FN) -> Dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    total = max(1, len(y_true))
    value = cost_fp * fp + cost_fn * fn
    return {
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tn": tn,
        "cost": float(value),
        "cost_per_1000": float(value / total * 1000.0),
        "cost_fp": float(cost_fp),
        "cost_fn": float(cost_fn),
    }


def _threshold_sweep(y_true, probs, cost_fp: float = COST_FP, cost_fn: float = COST_FN) -> Dict:
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    rows = []
    for t in np.linspace(0.01, 0.99, 99):
        pred = (probs >= t).astype(int)
        c = _cost(y_true, pred, cost_fp, cost_fn)
        rows.append({"threshold": round(float(t), 2), "cost": c["cost"], "cost_per_1000": c["cost_per_1000"]})
    best = min(rows, key=lambda r: (r["cost"], abs(r["threshold"] - 0.5)))
    return {"best": best, "rows": rows}


def _basic_metrics(y_true, probs, threshold: float) -> Dict:
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    pred = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    payload = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "cost": _cost(y_true, pred),
        "brier": float(brier_score_loss(y_true, probs)),
        "log_loss": float(log_loss(y_true, probs, labels=[0, 1])),
    }
    try:
        payload["roc_auc"] = float(roc_auc_score(y_true, probs))
    except ValueError:
        payload["roc_auc"] = None
    try:
        payload["pr_auc"] = float(average_precision_score(y_true, probs))
    except ValueError:
        payload["pr_auc"] = None
    return payload


def _keyword_predict(text: str) -> int:
    normalized = normalize_for_match(text)
    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    return int(bool(tokens & FINANCIAL_KEYWORDS))


def _ood_decision(text: str, p: float, vocab: Dict[str, int], threshold: float) -> Optional[str]:
    stripped = (text or "").strip()
    if len(stripped) < OOD_MIN_CHARS:
        return "texto_vacio_o_muy_corto"
    tokens = tokenize(text)
    if len(tokens) < OOD_MIN_TOKENS:
        return "muy_pocas_palabras"
    if len(tokens) >= OOD_SPANISH_MIN_TOKENS:
        spanish_hits = sum(1 for t in tokens if t in SPANISH_STOPWORDS)
        if spanish_hits < OOD_SPANISH_MIN_HITS:
            return "idioma_no_espanol"
    known = sum(1 for t in tokens if t in vocab)
    oov_rate = 1.0 - (known / max(1, len(tokens)))
    if oov_rate > OOD_MAX_OOV_RATE:
        return "fuera_de_vocabulario"
    if abs(p - threshold) <= OOD_REVIEW_MARGIN:
        return "cerca_del_umbral"
    return None


def _error_category(row: pd.Series, p: float, threshold: float) -> Tuple[str, str]:
    text = str(row.get("text") or "")
    letters = sum(1 for ch in text if ch.isalpha())
    digits = sum(1 for ch in text if ch.isdigit())
    ratio_digits = digits / max(1, len(text))
    if len(text) < 120:
        return "texto_corto", "El fragmento tiene poco contexto para decidir."
    if ratio_digits > 0.25:
        return "tabla_numerica", "El fragmento es predominantemente numérico o tabular."
    if abs(p - threshold) < 0.12:
        return "cerca_del_umbral", "La probabilidad está cerca del umbral; el caso es ambiguo."
    if re.search(r"\b(no|sin|excepto|salvo|nunca|pero)\b", normalize_for_match(text)):
        return "negacion_o_contraste", "Una negación o contraste cambia el sentido financiero."
    if row.get("section") in {"solucion", "vehiculo", "operaciones", "suministro"}:
        return "seccion_mixta", "La sección mezcla propuesta técnica con costos o riesgos."
    return "confusion_de_clase", "El vocabulario del fragmento se parece a la otra clase."


def _manual_label_value(value) -> Optional[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().lower()
    if text in {"1", "pos", "positivo", "positive", "finanzas", "financiero", "riesgo"}:
        return 1
    if text in {"0", "neg", "negativo", "negative", "mercado", "propuesta"}:
        return 0
    return None


def load_labels(df: pd.DataFrame, provisional: bool) -> Tuple[pd.DataFrame, str]:
    test = df[df["split"] == "test"].copy()
    if TEST_LABELS_CSV.exists():
        manual = pd.read_csv(TEST_LABELS_CSV)
        if "fragment_id" not in manual.columns or "manual_label" not in manual.columns:
            raise SystemExit("test_labels.csv debe tener columnas fragment_id y manual_label.")
        manual["_manual"] = manual["manual_label"].map(_manual_label_value)
        missing = set(test["fragment_id"]) - set(manual["fragment_id"])
        if missing:
            raise SystemExit(f"Faltan {len(missing)} fragmentos de test en test_labels.csv.")
        test = test.merge(manual[["fragment_id", "_manual"]], on="fragment_id", how="left")
        if test["_manual"].isna().any():
            raise SystemExit("Hay etiquetas manuales vacías o inválidas en test_labels.csv.")
        test["label"] = test["_manual"].astype(int)
        return test, "manual"
    if not provisional:
        raise SystemExit(
            "No existe test_labels.csv. Llena el 100% del test o usa --provisional "
            "para una corrida de desarrollo."
        )
    return test, "weak_provisional"


def _round_list(values, digits: int = 6):
    return [round(float(v), digits) for v in np.asarray(values).ravel().tolist()]


def train(provisional: bool = False) -> Dict:
    if not FRAGMENTS_SPLIT_CSV.exists():
        raise SystemExit(f"Falta {FRAGMENTS_SPLIT_CSV}. Ejecuta build_dataset y split primero.")
    df = pd.read_csv(FRAGMENTS_SPLIT_CSV)
    if "split" not in df.columns:
        raise SystemExit("fragments_split.csv no tiene columna split.")

    test, label_source = load_labels(df, provisional=provisional)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        token_pattern=r"\b\w\w+\b",
        ngram_range=(1, 2),
        min_df=MIN_DF,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float64,
    )
    X_train_sparse = vectorizer.fit_transform(train_df["text"].astype(str))
    X_val_sparse = vectorizer.transform(val_df["text"].astype(str))
    X_test_sparse = vectorizer.transform(test["text"].astype(str))
    # La MLP es pequeña; densificar evita matmuls sparse inestables en algunos
    # builds de scikit-learn/BLAS y no cambia el modelo.
    X_train = X_train_sparse.toarray()
    X_val = X_val_sparse.toarray()
    X_test = X_test_sparse.toarray()
    y_train = train_df["label"].astype(int).to_numpy()
    y_val = val_df["label"].astype(int).to_numpy()
    y_test = test["label"].astype(int).to_numpy()

    mlp = MLPClassifier(
        hidden_layer_sizes=(HIDDEN_UNITS,),
        activation="relu",
        solver="lbfgs",
        alpha=MLP_ALPHA,
        max_iter=1200,
        random_state=RANDOM_STATE,
    )
    # Algunos builds de scikit-learn/BLAS emiten RuntimeWarnings de matmul
    # durante la línea de búsqueda de lbfgs, aunque los pesos finales son
    # finitos. Se silencian solo en el fit y se verifica el resultado.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)
        mlp.fit(X_train, y_train)
    for coef in mlp.coefs_:
        if not np.isfinite(coef).all():
            raise SystemExit("La MLP produjo pesos no finitos.")

    p_val_raw = mlp.predict_proba(X_val)[:, 1]
    p_test_raw = mlp.predict_proba(X_test)[:, 1]
    logits_val = _logit(p_val_raw)
    logits_test = _logit(p_test_raw)
    temperature = _fit_temperature(logits_val, y_val)
    p_val_cal = _sigmoid(logits_val / temperature)
    p_test_cal = _sigmoid(logits_test / temperature)

    sweep_raw = _threshold_sweep(y_val, p_val_raw)
    sweep_cal = _threshold_sweep(y_val, p_val_cal)
    threshold_raw = float(sweep_raw["best"]["threshold"])
    threshold_cal = float(sweep_cal["best"]["threshold"])

    ece_before, curve_before = _ece(y_val, p_val_raw)
    ece_after, curve_after = _ece(y_val, p_val_cal)

    test_raw = _basic_metrics(y_test, p_test_raw, threshold_raw)
    test_cal = _basic_metrics(y_test, p_test_cal, threshold_cal)
    test_cal_at_raw = _basic_metrics(y_test, p_test_cal, threshold_raw)

    # Baseline mayoría.
    baseline_majority_pred = np.zeros_like(y_test)
    baseline_majority = {
        "accuracy": float(accuracy_score(y_test, baseline_majority_pred)),
        "cost": _cost(y_test, baseline_majority_pred),
        "confusion": {
            "tn": int(((baseline_majority_pred == 0) & (y_test == 0)).sum()),
            "fp": int(((baseline_majority_pred == 1) & (y_test == 0)).sum()),
            "fn": int(((baseline_majority_pred == 0) & (y_test == 1)).sum()),
            "tp": int(((baseline_majority_pred == 1) & (y_test == 1)).sum()),
        },
    }
    # Baseline keyword.
    keyword_pred = np.array([_keyword_predict(t) for t in test["text"].astype(str)], dtype=int)
    baseline_keyword = {
        "accuracy": float(accuracy_score(y_test, keyword_pred)),
        "precision": float(precision_recall_fscore_support(y_test, keyword_pred, average="binary", zero_division=0)[0]),
        "recall": float(precision_recall_fscore_support(y_test, keyword_pred, average="binary", zero_division=0)[1]),
        "f1": float(f1_score(y_test, keyword_pred, zero_division=0)),
        "cost": _cost(y_test, keyword_pred),
        "confusion": {
            "tn": int(((keyword_pred == 0) & (y_test == 0)).sum()),
            "fp": int(((keyword_pred == 1) & (y_test == 0)).sum()),
            "fn": int(((keyword_pred == 0) & (y_test == 1)).sum()),
            "tp": int(((keyword_pred == 1) & (y_test == 1)).sum()),
        },
    }

    # Baseline entrenado: logistic regression. No es el modelo de la PWA, pero
    # ayuda a saber si la MLP aporta algo frente a un modelo lineal simple.
    logistic = LogisticRegression(
        max_iter=3000,
        C=0.3,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
    )
    logistic.fit(X_train, y_train)
    p_log_val = logistic.predict_proba(X_val)[:, 1]
    p_log_test = logistic.predict_proba(X_test)[:, 1]
    log_sweep = _threshold_sweep(y_val, p_log_val)
    log_threshold = float(log_sweep["best"]["threshold"])
    baseline_logistic = _basic_metrics(y_test, p_log_test, log_threshold)
    baseline_logistic["threshold_source"] = "validation_cost_sweep"

    # Errores más informativos del test.
    test_eval = test.copy()
    test_eval["p_raw"] = p_test_raw
    test_eval["p_cal"] = p_test_cal
    test_eval["pred"] = (p_test_cal >= threshold_cal).astype(int)
    errors = test_eval[test_eval["pred"] != test_eval["label"]].copy()
    errors["distance"] = (errors["p_cal"] - threshold_cal).abs()
    errors = errors.sort_values("distance", ascending=False).head(20)
    error_rows = []
    for _, row in errors.iterrows():
        category, hypothesis = _error_category(row, float(row["p_cal"]), threshold_cal)
        error_rows.append(
            {
                "fragment_id": row["fragment_id"],
                "doc_key": row["doc_key"],
                "team_key": row["team_key"],
                "section": row["section"],
                "true_label": int(row["label"]),
                "predicted_label": int(row["pred"]),
                "probability": float(row["p_cal"]),
                "threshold": threshold_cal,
                "category": category,
                "hypothesis": hypothesis,
                "text": str(row["text"])[:700],
            }
        )
    grouped_errors = defaultdict(list)
    for row in error_rows:
        grouped_errors[row["category"]].append(row)

    # OOD: casos sintéticos definidos en practica1/ood_cases.json.
    ood_path = PRACTICA_DIR / "ood_cases.json"
    ood_results = []
    if ood_path.exists():
        for case in _load_json(ood_path):
            text = case.get("text") or ""
            if text.strip():
                p = float(mlp.predict_proba(vectorizer.transform([text]).toarray())[:, 1][0])
                p_cal = float(_sigmoid(_logit(p) / temperature))
            else:
                p = p_cal = 0.0
            reason = _ood_decision(text, p_cal, vectorizer.vocabulary_, threshold_cal)
            ood_results.append(
                {
                    "id": case.get("id"),
                    "expected": case.get("expected", "rechazar"),
                    "reason_expected": case.get("reason"),
                    "actual_reject_reason": reason,
                    "rejected": bool(reason and reason != "cerca_del_umbral"),
                    "probability": p_cal,
                }
            )

    metrics = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "provisional": bool(provisional and label_source == "weak_provisional"),
        "test_label_source": label_source,
        "dataset_manifest": _load_json(DATASET_MANIFEST) if DATASET_MANIFEST.exists() else None,
        "split": _load_json(SPLIT_JSON) if SPLIT_JSON.exists() else None,
        "model": {
            "type": "tfidf_word_1_2 + mlp_1_hidden",
            "hidden_units": HIDDEN_UNITS,
            "max_features": int(X_train.shape[1]),
            "train_size": int(X_train.shape[0]),
            "train_augmented": int(train_df["augmented"].sum()) if "augmented" in train_df.columns else 0,
            "val_size": int(X_val.shape[0]),
            "test_size": int(X_test.shape[0]),
            "train_positive": int(y_train.sum()),
            "val_positive": int(y_val.sum()),
            "test_positive": int(y_test.sum()),
        },
        "calibration": {
            "temperature": float(temperature),
            "ece_before": float(ece_before),
            "ece_after": float(ece_after),
            "curve_before": curve_before,
            "curve_after": curve_after,
        },
        "threshold": {
            "raw": threshold_raw,
            "calibrated": threshold_cal,
            "sweep_raw_best": sweep_raw["best"],
            "sweep_calibrated_best": sweep_cal["best"],
            "sweep_calibrated": sweep_cal["rows"],
        },
        "validation": {
            "raw_at_raw_threshold": _basic_metrics(y_val, p_val_raw, threshold_raw),
            "calibrated_at_calibrated_threshold": _basic_metrics(y_val, p_val_cal, threshold_cal),
        },
        "test": {
            "raw_at_raw_threshold": test_raw,
            "calibrated_at_calibrated_threshold": test_cal,
            "calibrated_at_raw_threshold": test_cal_at_raw,
        },
        "baselines": {
            "majority": baseline_majority,
            "keyword": baseline_keyword,
            "logistic": baseline_logistic,
        },
        "errors": error_rows,
        "errors_grouped": {k: len(v) for k, v in grouped_errors.items()},
        "ood": ood_results,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Exportar modelo para la PWA.
    coefs = mlp.coefs_
    intercepts = mlp.intercepts_
    vocab = {str(k): int(v) for k, v in vectorizer.vocabulary_.items()}
    model_payload = {
        "version": "1.0.0",
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model_type": "tfidf_mlp",
        "provisional": bool(metrics["provisional"]),
        "preprocess": {
            "lowercase": True,
            "strip_accents": "unicode",
            "token_pattern": r"\b\w\w+\b",
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "norm": "l2",
        },
        "vocab": vocab,
        "idf": _round_list(vectorizer.idf_),
        "hidden_activation": "relu",
        "w1": [[round(float(v), 6) for v in row] for row in coefs[0].tolist()],
        "b1": _round_list(intercepts[0]),
        "w2": _round_list(coefs[1][:, 0]),
        "b2": float(intercepts[1][0]),
        "temperature": float(temperature),
        "threshold": float(threshold_cal),
        "ood": {
            "min_chars": OOD_MIN_CHARS,
            "min_tokens": OOD_MIN_TOKENS,
            "max_oov_rate": OOD_MAX_OOV_RATE,
            "spanish_min_hits": OOD_SPANISH_MIN_HITS,
            "spanish_min_tokens": OOD_SPANISH_MIN_TOKENS,
            "review_margin": OOD_REVIEW_MARGIN,
            "spanish_stopwords": sorted(SPANISH_STOPWORDS),
        },
        "metrics_summary": {
            "test_label_source": label_source,
            "fragment_count": int(
                len(y_train) - int(metrics["model"].get("train_augmented") or 0) + len(y_val) + len(y_test)
            ),
            "total_with_augmentation": int(len(y_train) + len(y_val) + len(y_test)),
            "train_size": int(len(y_train)),
            "train_augmented": int(train_df["augmented"].sum()) if "augmented" in train_df.columns else 0,
            "train_positive": int(y_train.sum()),
            "train_negative": int((y_train == 0).sum()),
            "val_size": int(len(y_val)),
            "val_positive": int(y_val.sum()),
            "test_size": int(len(y_test)),
            "test_positive": int(y_test.sum()),
            "accuracy": test_cal["accuracy"],
            "precision": test_cal["precision"],
            "recall": test_cal["recall"],
            "f1": test_cal["f1"],
            "cost_per_1000": test_cal["cost"]["cost_per_1000"],
            "ece_after": float(ece_after),
        },
    }
    MODEL_JSON.write_text(json.dumps(model_payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    PWA_MODEL_JSON.write_text(json.dumps(model_payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    _write_pwa_evaluation(metrics)
    return metrics


def _snippet(text: str, limit: int = 280) -> str:
    clean = normalize_whitespace(text or "")
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


def _write_pwa_evaluation(metrics: dict) -> None:
    """Exporta el análisis de la rúbrica §4 sin identificadores de equipo o documento."""
    ood_text = {}
    ood_path = PRACTICA_DIR / "ood_cases.json"
    if ood_path.exists():
        for case in _load_json(ood_path):
            ood_text[case.get("id")] = case.get("text") or ""

    test = metrics["test"]["calibrated_at_calibrated_threshold"]
    raw = metrics["test"]["raw_at_raw_threshold"]
    errors = []
    for row in metrics.get("errors") or []:
        errors.append(
            {
                "section": row.get("section"),
                "true_label": int(row.get("true_label")),
                "predicted_label": int(row.get("predicted_label")),
                "probability": round(float(row.get("probability")), 4),
                "threshold": round(float(row.get("threshold")), 4),
                "category": row.get("category"),
                "hypothesis": row.get("hypothesis"),
                "text": _snippet(str(row.get("text") or ""), 640),
            }
        )
    ood = []
    for case in metrics.get("ood") or []:
        ood.append(
            {
                "id": case.get("id"),
                "expected": case.get("expected"),
                "reason_expected": case.get("reason_expected"),
                "actual_reject_reason": case.get("actual_reject_reason"),
                "rejected": bool(case.get("rejected")),
                "probability": round(float(case.get("probability") or 0), 4),
                "text": _snippet(ood_text.get(case.get("id"), ""), 220),
            }
        )
    payload = {
        "provisional": bool(metrics.get("provisional")),
        "test_label_source": metrics.get("test_label_source"),
        "dataset": {
            "fragment_count": int(
                metrics["model"]["train_size"]
                - int(metrics["model"].get("train_augmented") or 0)
                + metrics["model"]["val_size"]
                + metrics["model"]["test_size"]
            ),
            "total_with_augmentation": int(
                metrics["model"]["train_size"] + metrics["model"]["val_size"] + metrics["model"]["test_size"]
            ),
            "train_size": int(metrics["model"]["train_size"]),
            "train_augmented": int(metrics["model"].get("train_augmented") or 0),
            "train_positive": int(metrics["model"]["train_positive"]),
            "train_negative": int(metrics["model"]["train_size"] - metrics["model"]["train_positive"]),
            "val_size": int(metrics["model"]["val_size"]),
            "val_positive": int(metrics["model"]["val_positive"]),
            "test_size": int(metrics["model"]["test_size"]),
            "test_positive": int(metrics["model"]["test_positive"]),
            "document_count": int((metrics.get("dataset_manifest") or {}).get("document_count") or 0),
            "team_count": int((metrics.get("dataset_manifest") or {}).get("team_count") or 0),
            "fit_on": "train",
        },
        "threshold": metrics["threshold"]["calibrated"],
        "costs": {
            "fp": COST_FP,
            "fn": COST_FN,
            "model_per_1000": test["cost"]["cost_per_1000"],
            "raw_per_1000": raw["cost"]["cost_per_1000"],
            "majority_per_1000": metrics["baselines"]["majority"]["cost"]["cost_per_1000"],
            "keyword_per_1000": metrics["baselines"]["keyword"]["cost"]["cost_per_1000"],
            "logistic_per_1000": metrics["baselines"]["logistic"]["cost"]["cost_per_1000"],
        },
        "confusion": test["confusion"],
        "metrics": {
            "accuracy": test["accuracy"],
            "precision": test["precision"],
            "recall": test["recall"],
            "f1": test["f1"],
        },
        "calibration": {
            "temperature": metrics["calibration"]["temperature"],
            "ece_before": metrics["calibration"]["ece_before"],
            "ece_after": metrics["calibration"]["ece_after"],
        },
        "errors_grouped": metrics.get("errors_grouped") or {},
        "errors": errors,
        "ood": ood,
    }
    PWA_DIR.mkdir(parents=True, exist_ok=True)
    (PWA_DIR / "evaluation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provisional", action="store_true", help="Permite usar etiquetas débiles en test.")
    args = parser.parse_args()
    metrics = train(provisional=args.provisional)
    test = metrics["test"]["calibrated_at_calibrated_threshold"]
    print(f"Etiquetas de test: {metrics['test_label_source']}  (provisional={metrics['provisional']})")
    print(f"Umbral calibrado: {metrics['threshold']['calibrated']:.2f}")
    print(
        f"Test: acc={test['accuracy']:.3f} precision={test['precision']:.3f} "
        f"recall={test['recall']:.3f} f1={test['f1']:.3f} "
        f"costo/1000={test['cost']['cost_per_1000']:.2f}"
    )
    print(f"ECE antes={metrics['calibration']['ece_before']:.3f} después={metrics['calibration']['ece_after']:.3f}")
    print(f"Métricas: {METRICS_JSON}")
    print(f"Modelo: {MODEL_JSON}")
    print(f"Modelo PWA: {PWA_MODEL_JSON}")


if __name__ == "__main__":
    main()
