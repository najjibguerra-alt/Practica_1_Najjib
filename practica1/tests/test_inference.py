import math
import json
import shutil
import subprocess
import unittest
from pathlib import Path

from practica1.src.config import label_for_sections
from practica1.src.text import chunk_text, scrub_pii, tokenize

ROOT = Path(__file__).resolve().parents[2]
PRACTICA = ROOT / "practica1"
MODEL_JSON = PRACTICA / "pwa" / "model.json"
PARITY_CASES = PRACTICA / "tests" / "parity_cases.json"


def _count_features(tokens):
    counts = {}
    for i, token in enumerate(tokens):
        counts[token] = counts.get(token, 0) + 1
        if i + 1 < len(tokens):
            bigram = token + " " + tokens[i + 1]
            counts[bigram] = counts.get(bigram, 0) + 1
    return counts


def model_json_predict(model, text):
    tokens = tokenize(text)
    counts = _count_features(tokens)
    sparse = []
    norm = 0.0
    for token, count in counts.items():
        if token not in model["vocab"]:
            continue
        idx = model["vocab"][token]
        tf = 1.0 + math.log(count) if model["preprocess"].get("sublinear_tf") else float(count)
        value = tf * model["idf"][idx]
        sparse.append((idx, value))
        norm += value * value
    denom = math.sqrt(norm) or 1.0
    hidden = [0.0] * len(model["w1"][0])
    for idx, value in sparse:
        row = model["w1"][idx]
        scaled = value / denom
        for h in range(len(hidden)):
            hidden[h] += scaled * row[h]
    for h in range(len(hidden)):
        hidden[h] = max(0.0, hidden[h] + model["b1"][h])
    logit = model["b2"] + sum(hidden[h] * model["w2"][h] for h in range(len(hidden)))
    z = max(-60.0, min(60.0, logit / model["temperature"]))
    probability = 1.0 / (1.0 + math.exp(-z))
    threshold = model["threshold"]
    return {
        "probability": probability,
        "label": int(probability >= threshold),
        "tokenCount": len(tokens),
        "featureCount": len(sparse),
    }


class TextTests(unittest.TestCase):
    def test_tokenize_strips_accents(self):
        self.assertEqual(tokenize("Inversión y evaluación financiera"), ["inversion", "evaluacion", "financiera"])

    def test_scrub_pii(self):
        text = "Contacto: Ana Pérez ana@example.com, https://example.com, +52 55 1234 5678, equipo KURUMA."
        cleaned = scrub_pii(text, ["KURUMA", "Ana Pérez"])
        self.assertNotIn("ana@example.com", cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("KURUMA", cleaned)
        self.assertNotIn("Ana Pérez", cleaned)

    def test_chunk_limits(self):
        text = " ".join(["La inversión y los riesgos del proyecto requieren revisión."] * 30)
        chunks = chunk_text(text)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertGreaterEqual(len(chunk), 80)
            self.assertLessEqual(len(chunk), 650)

    def test_label_priority(self):
        self.assertEqual(label_for_sections({"inversion", "operaciones"}), "pos")
        self.assertEqual(label_for_sections({"operaciones"}), "neg")
        self.assertEqual(label_for_sections({"empresa"}), "amb")
        self.assertEqual(label_for_sections({"portada"}), "exc")


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODEL_JSON.exists():
            raise unittest.SkipTest("Falta practica1/pwa/model.json")
        cls.model = json.loads(MODEL_JSON.read_text(encoding="utf-8"))

    def test_shape_and_ranges(self):
        model = self.model
        n_features = len(model["vocab"])
        self.assertEqual(n_features, len(model["idf"]))
        self.assertEqual(n_features, len(model["w1"]))
        hidden = len(model["w1"][0])
        self.assertEqual(hidden, len(model["b1"]))
        self.assertEqual(hidden, len(model["w2"]))
        self.assertGreater(model["temperature"], 0)
        self.assertTrue(0 < model["threshold"] < 1)

    def test_predicts_finance_and_market(self):
        cases = {item["id"]: item["text"] for item in json.loads(PARITY_CASES.read_text(encoding="utf-8"))}
        finance = model_json_predict(self.model, cases["finance"])
        market = model_json_predict(self.model, cases["market"])
        self.assertEqual(finance["label"], 1)
        self.assertEqual(market["label"], 0)
        self.assertGreater(finance["probability"], market["probability"])

    @unittest.skipUnless(shutil.which("node"), "Node no está instalado")
    def test_js_python_parity(self):
        script = PRACTICA / "tests" / "parity.mjs"
        result = subprocess.run(
            ["node", str(script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        js = json.loads(result.stdout)
        cases = json.loads(PARITY_CASES.read_text(encoding="utf-8"))
        for item in cases:
            py = model_json_predict(self.model, item["text"])
            self.assertAlmostEqual(py["probability"], js[item["id"]]["probability"], places=9)
            self.assertEqual(py["label"], js[item["id"]]["label"])
            self.assertEqual(py["tokenCount"], js[item["id"]]["tokenCount"])
            self.assertEqual(py["featureCount"], js[item["id"]]["featureCount"])


if __name__ == "__main__":
    unittest.main()
