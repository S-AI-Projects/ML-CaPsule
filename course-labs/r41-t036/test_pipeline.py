"""Unit tests for R41-T036: FIFA players CPU smoke run."""

import math
import unittest

from pipeline import CPUSmokeRegressionPipeline, load_fixture


class TestFIFACpuSmokeRun(unittest.TestCase):
    """Verifies bounded CPU smoke run guarantees for Task R41-T036."""

    def setUp(self) -> None:
        self.fixture = load_fixture()
        self.records = self.fixture["records"]
        self.train_records = [r for r in self.records if r["split"] == "train"]
        self.val_records = [r for r in self.records if r["split"] == "validation"]
        self.test_records = [r for r in self.records if r["split"] == "test"]

    def test_deterministic_formula_fixture_consistency(self) -> None:
        """Verify fixture conforms to target = 2 * x1 + 1 with stable IDs."""
        self.assertEqual(len(self.records), 6)
        for r in self.records:
            expected = 2.0 * r["x1"] + 1.0
            self.assertAlmostEqual(r["target"], expected, places=6)

    def test_smoke_run_is_bounded_cpu_only(self) -> None:
        """Verify training completes within 20 steps, loss is finite, hardware is CPU."""
        pipeline = CPUSmokeRegressionPipeline(seed=42, max_steps=15)
        summary = pipeline.fit_smoke_step(self.train_records)

        self.assertEqual(summary.hardware_used, "CPU")
        self.assertEqual(summary.sample_count, 4)
        self.assertLessEqual(summary.step_count, 20)
        self.assertEqual(summary.step_count, 15)
        self.assertTrue(summary.is_finite)
        self.assertTrue(math.isfinite(summary.final_loss))
        self.assertEqual(summary.prediction_shape, (4, 1))
        self.assertIn("baseline", summary.model_type)

    def test_inference_shape_and_finite_predictions(self) -> None:
        """Verify inference outputs finite float predictions matching input record count."""
        pipeline = CPUSmokeRegressionPipeline(seed=42, max_steps=10)
        pipeline.fit_smoke_step(self.train_records)

        preds = pipeline.predict(self.val_records)
        self.assertEqual(len(preds), 1)
        self.assertTrue(math.isfinite(preds[0]))

        test_preds = pipeline.predict(self.test_records)
        self.assertEqual(len(test_preds), 1)
        self.assertTrue(math.isfinite(test_preds[0]))

    def test_sample_bound_enforced(self) -> None:
        """Verify exceeding 32 samples raises a ValueError."""
        oversized_records = [
            {"record_id": f"r_{i}", "x1": float(i), "x2": 1.0, "target": 2.0 * i + 1.0}
            for i in range(35)
        ]
        pipeline = CPUSmokeRegressionPipeline()
        with self.assertRaises(ValueError):
            pipeline.fit_smoke_step(oversized_records)

    def test_predict_before_fit_raises_error(self) -> None:
        """Verify calling predict prior to fit raises RuntimeError."""
        pipeline = CPUSmokeRegressionPipeline()
        with self.assertRaises(RuntimeError):
            pipeline.predict(self.val_records)


if __name__ == "__main__":
    unittest.main()
