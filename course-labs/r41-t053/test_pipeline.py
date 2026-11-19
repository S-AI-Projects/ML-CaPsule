import math
import pathlib
import sys
import unittest


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline


class TrainingMeanBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = pipeline.load_fixture(HERE / "fixture.json")

    def test_literal_formula_fixture(self):
        self.assertEqual(
            [record["x1"] for record in self.records],
            [0, 1, 2, 3, 4, 5],
        )
        self.assertEqual([record["x2"] for record in self.records], [1] * 6)
        self.assertEqual(
            [record["target"] for record in self.records],
            [1, 3, 5, 7, 9, 11],
        )
        self.assertTrue(
            all(record["target"] == 2 * record["x1"] + 1 for record in self.records)
        )

    def test_fixed_split_is_exact_and_disjoint(self):
        splits = pipeline.fixed_split_ids(seed=42)
        self.assertEqual(
            splits,
            {
                "train": ["r1", "r2", "r3", "r4"],
                "validation": ["r5"],
                "test": ["r6"],
            },
        )
        train = set(splits["train"])
        validation = set(splits["validation"])
        test = set(splits["test"])
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)

    def test_training_mean_uses_only_training_rows_and_evaluates_held_out_mae(self):
        result = pipeline.run_baseline(self.records)
        self.assertEqual(result["baseline"]["training_ids"], ["r1", "r2", "r3", "r4"])
        self.assertEqual(result["baseline"]["training_mean"], 4.0)
        self.assertEqual(result["validation_predictions"], {"r5": 4.0})
        self.assertEqual(result["test_predictions"], {"r6": 4.0})
        self.assertEqual(result["validation_mae"], 5.0)
        self.assertEqual(result["test_mae"], 7.0)
        self.assertEqual(result["held_out_mae"], 6.0)
        self.assertTrue(math.isfinite(result["held_out_mae"]))

    def test_held_out_contamination_is_rejected(self):
        import json

        with (HERE / "contaminated_split.json").open("r", encoding="utf-8") as handle:
            contaminated = json.load(handle)
        with self.assertRaisesRegex(ValueError, "overlap|fixed training"):
            pipeline.fit_training_mean(
                self.records,
                split_ids=contaminated["split_ids"],
                seed=contaminated["seed"],
            )

    def test_prediction_shape_mismatch_is_rejected(self):
        result = pipeline.run_baseline(self.records)
        with self.assertRaisesRegex(ValueError, "exactly match evaluation record IDs"):
            pipeline.mean_absolute_error(
                [self.records[4]], {"r6": result["baseline"]["training_mean"]}
            )

    def test_malformed_record_is_rejected(self):
        malformed = [dict(record) for record in self.records]
        del malformed[0]["target"]
        with self.assertRaisesRegex(ValueError, "exactly record_id, x1, x2, target"):
            pipeline.validate_records(malformed)


if __name__ == "__main__":
    unittest.main()
