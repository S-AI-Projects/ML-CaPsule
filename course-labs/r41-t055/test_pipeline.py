import json
import math
import pathlib
import sys
import tempfile
import unittest


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline


class PreprocessingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = pipeline.load_fixture(HERE / "fixture.json")
        cls.split_ids = pipeline.fixed_split_ids(cls.records, seed=42)

    def test_literal_fixture_and_centroid_assignments(self):
        self.assertEqual(
            [(record["x1"], record["x2"]) for record in self.records],
            [(0, 0), (0, 1), (1, 0), (5, 5), (5, 6), (6, 5)],
        )
        self.assertEqual(
            list(pipeline.nearest_centroid_assignments(self.records).values()),
            [0, 0, 0, 1, 1, 1],
        )

    def test_fixed_seed_splits_are_complete_and_disjoint(self):
        split_again = pipeline.fixed_split_ids(self.records, seed=42)
        self.assertEqual(split_again, self.split_ids)
        self.assertEqual(
            self.split_ids,
            {
                "train": ["p4", "p2", "p3", "p5"],
                "validation": ["p1"],
                "test": ["p6"],
            },
        )
        self.assertEqual(
            {name: len(ids) for name, ids in self.split_ids.items()},
            {"train": 4, "validation": 1, "test": 1},
        )
        train = set(self.split_ids["train"])
        validation = set(self.split_ids["validation"])
        test = set(self.split_ids["test"])
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)
        self.assertEqual(train | validation | test, {record["record_id"] for record in self.records})

    def test_fitted_scaler_uses_only_training_ids_and_transforms_with_float64_shape(self):
        scaler = pipeline.fit_scaler(self.records, self.split_ids, seed=42)
        self.assertEqual(set(scaler["training_ids"]), set(self.split_ids["train"]))
        self.assertTrue(set(scaler["training_ids"]).isdisjoint(self.split_ids["validation"]))
        self.assertTrue(set(scaler["training_ids"]).isdisjoint(self.split_ids["test"]))

        held_out_ids = set(self.split_ids["validation"] + self.split_ids["test"])
        held_out = [
            record for record in self.records if record["record_id"] in held_out_ids
        ]
        transformed = pipeline.transform(scaler, held_out)
        self.assertEqual(transformed["shape"], [2, 2])
        self.assertEqual(transformed["dtype"], "float64")
        self.assertTrue(
            all(type(value) is float for row in transformed["values"] for value in row)
        )
        self.assertTrue(
            all(math.isfinite(value) for row in transformed["values"] for value in row)
        )
        self.assertEqual(transformed["record_ids"], [record["record_id"] for record in held_out])
        expected_values = [
            [-1.2074068598865937, -1.1766968108291043],
            [1.4269353798659745, 0.7844645405527362],
        ]
        for actual_row, expected_row in zip(transformed["values"], expected_values):
            for actual, expected in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, expected, places=12)

    def test_saved_fitted_parameters_reproduce_held_out_transform(self):
        scaler = pipeline.fit_scaler(self.records, self.split_ids, seed=42)
        held_out_ids = set(self.split_ids["validation"] + self.split_ids["test"])
        held_out = [
            record for record in self.records if record["record_id"] in held_out_ids
        ]
        before = pipeline.transform(scaler, held_out)
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "scaler.json"
            pipeline.save_scaler(scaler, path)
            reloaded = pipeline.load_scaler(path)
        after = pipeline.transform(reloaded, held_out)
        self.assertEqual(after, before)

    def test_deliberately_contaminated_split_is_rejected(self):
        with (HERE / "contaminated_split.json").open("r", encoding="utf-8") as handle:
            contaminated = json.load(handle)
        with self.assertRaisesRegex(ValueError, "overlap|fixed seed partition"):
            pipeline.fit_scaler(
                self.records,
                split_ids=contaminated["split_ids"],
                seed=contaminated["seed"],
            )

    def test_incompatible_record_shape_is_rejected(self):
        malformed = [dict(record) for record in self.records]
        del malformed[0]["x2"]
        with self.assertRaisesRegex(ValueError, "exactly record_id, x1, x2"):
            pipeline.transform(
                pipeline.fit_scaler(self.records, self.split_ids),
                malformed,
            )


if __name__ == "__main__":
    unittest.main()
