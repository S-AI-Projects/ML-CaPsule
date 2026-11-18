import json
import pathlib
import sys
import unittest


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = pipeline.load_fixture(HERE / "fixture.json")
        cls.train, cls.validation, cls.test = pipeline.fixed_splits(cls.records)
        cls.model = pipeline.train(cls.train)

    def test_literal_fixture_contract_and_splits(self):
        self.assertEqual(
            [record["record_id"] for record in self.records],
            ["t1", "t2", "t3", "t4", "t5", "t6"],
        )
        self.assertEqual(
            [record["text"] for record in self.records],
            ["red apple", "green apple", "blue sky", "dark sky", "red fruit", "blue cloud"],
        )
        self.assertEqual(
            [record["label"] for record in self.records],
            [1, 1, 0, 0, 1, 0],
        )
        self.assertEqual([record["record_id"] for record in self.train], ["t4", "t2", "t3", "t5"])
        self.assertEqual([record["record_id"] for record in self.validation], ["t1"])
        self.assertEqual([record["record_id"] for record in self.test], ["t6"])

    def test_before_and_after_save_predictions_agree(self):
        model_path = HERE / "test_saved_model.json"
        try:
            texts = [record["text"] for record in self.validation + self.test]
            before = pipeline.predict(self.model, texts)
            pipeline.save_model(self.model, model_path)
            loaded = pipeline.load_model(model_path)
            after = pipeline.predict(loaded, texts)
            self.assertEqual(before, [1, 0])
            self.assertEqual(after, before)
        finally:
            model_path.unlink(missing_ok=True)

    def test_saved_expected_predictions_match(self):
        expected = [1, 0]
        text = [record["text"] for record in self.validation + self.test]
        actual = pipeline.predict(self.model, text)
        self.assertEqual(actual, expected)

    def test_schema_mismatch_fails_clearly(self):
        bad = dict(self.model)
        bad["schema"] = "r41-t060.model.v999"
        with self.assertRaisesRegex(ValueError, "model schema mismatch"):
            pipeline.validate_model(bad)

    def test_fixture_schema_mismatch_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "fixture schema mismatch"):
            pipeline.load_fixture(HERE / "fixture_schema_mismatch.json")

    def test_inference_needs_no_training_records(self):
        model_path = HERE / "test_saved_model.json"
        try:
            pipeline.save_model(self.model, model_path)
            loaded = pipeline.load_model(model_path)
            self.assertEqual(pipeline.predict(loaded, ["red fruit"]), [1])
        finally:
            model_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
