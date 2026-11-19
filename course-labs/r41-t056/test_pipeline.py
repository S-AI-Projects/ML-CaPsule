import math
import pathlib
import sys
import unittest


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline


class DecisionTreeSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = pipeline.load_fixture(HERE / "fixture.json")

    def test_literal_simulated_fixture(self):
        self.assertEqual(
            [record["x1"] for record in self.records],
            list(range(8)),
        )
        self.assertEqual([record["x2"] for record in self.records], [1] * 8)
        self.assertEqual(
            [record["label"] for record in self.records],
            [0, 0, 0, 0, 1, 1, 1, 1],
        )
        self.assertEqual(len({record["record_id"] for record in self.records}), 8)

    def test_fixed_seed_stratified_group_split_is_disjoint(self):
        first = pipeline.fixed_stratified_splits(self.records, seed=42)
        second = pipeline.fixed_stratified_splits(self.records, seed=42)
        ids = lambda rows: [record["record_id"] for record in rows]
        self.assertEqual(
            {name: ids(rows) for name, rows in first.items()},
            {name: ids(rows) for name, rows in second.items()},
        )
        self.assertEqual(
            {name: len(rows) for name, rows in first.items()},
            {"train": 4, "validation": 2, "test": 2},
        )
        group_sets = {
            name: {record["group_id"] for record in rows}
            for name, rows in first.items()
        }
        self.assertTrue(group_sets["train"].isdisjoint(group_sets["validation"]))
        self.assertTrue(group_sets["train"].isdisjoint(group_sets["test"]))
        self.assertTrue(group_sets["validation"].isdisjoint(group_sets["test"]))
        for rows in first.values():
            self.assertEqual({record["label"] for record in rows}, {0, 1})

    def test_smoke_run_has_finite_loss_expected_shape_and_bounded_steps(self):
        result = pipeline.run_smoke(self.records, seed=42)
        self.assertTrue(math.isfinite(result["loss"]))
        self.assertTrue(math.isfinite(result["model"]["training_loss"]))
        self.assertEqual(result["prediction_shape"], [4])
        self.assertEqual(
            result["prediction_record_ids"],
            ["r3", "r7", "r0", "r4"],
        )
        self.assertEqual(
            result["predictions"],
            {"r3": 0, "r7": 1, "r0": 0, "r4": 1},
        )
        self.assertEqual(result["model"]["training_epochs"], 1)
        self.assertEqual(result["step_count"], 3)
        self.assertLessEqual(result["step_count"], pipeline.MAX_STEPS)

    def test_missing_field_is_rejected(self):
        malformed = [dict(record) for record in self.records]
        del malformed[0]["x2"]
        with self.assertRaisesRegex(ValueError, "exactly record_id, group_id"):
            pipeline.validate_records(malformed)

    def test_more_than_32_samples_are_rejected(self):
        oversized = [dict(record) for record in self.records]
        for index in range(25):
            extra = dict(self.records[index % len(self.records)])
            extra["record_id"] = "extra-%d" % index
            oversized.append(extra)
        with self.assertRaisesRegex(ValueError, "at most 32 samples"):
            pipeline.validate_records(oversized)

    def test_unknown_feature_and_excessive_steps_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_steps must be between"):
            pipeline.fit_stump(
                pipeline.fixed_stratified_splits(self.records)["train"],
                max_steps=21,
            )

        model = pipeline.fit_stump(
            pipeline.fixed_stratified_splits(self.records)["train"]
        )
        model["split_feature"] = "unknown"
        with self.assertRaisesRegex(ValueError, "split_feature is incompatible"):
            pipeline.predict(model, self.records)


if __name__ == "__main__":
    unittest.main()
