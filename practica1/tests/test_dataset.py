import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRACTICA = ROOT / "practica1"
SPLIT_CSV = PRACTICA / "data" / "fragments_split.csv"
REVIEW_CSV = PRACTICA / "data" / "labels_review.csv"


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SPLIT_CSV.exists():
            raise unittest.SkipTest("Falta practica1/data/fragments_split.csv")
        cls.df = pd.read_csv(SPLIT_CSV)

    def test_train_has_at_least_500_examples(self):
        train = self.df[self.df["split"] == "train"]
        self.assertGreaterEqual(len(train), 500)
        self.assertGreater(int(train["label"].sum()), 0)
        self.assertGreater(int((train["label"] == 0).sum()), 0)

    def test_no_team_leakage(self):
        leakage = self.df.groupby("team_key")["split"].nunique()
        self.assertFalse((leakage > 1).any(), leakage[leakage > 1].to_dict())

    def test_augmented_rows_are_train_only(self):
        self.assertIn("augmented", self.df.columns)
        self.assertGreater(int(self.df["augmented"].sum()), 0)
        self.assertEqual(int(self.df.loc[self.df["augmented"] == 1, "split"].ne("train").sum()), 0)

    def test_test_review_has_one_row_per_test_fragment(self):
        if not REVIEW_CSV.exists():
            self.skipTest("Falta labels_review.csv")
        review = pd.read_csv(REVIEW_CSV)
        test = self.df[self.df["split"] == "test"]
        self.assertEqual(len(review), len(test))
        self.assertEqual(set(review["fragment_id"]), set(test["fragment_id"]))


if __name__ == "__main__":
    unittest.main()
