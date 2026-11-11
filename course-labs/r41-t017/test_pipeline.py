"""Unit tests for reproducible configuration and PCA projection (Task R41-T017)."""

import math
import unittest
from pipeline import PCA1DProjection, ReproduciblePCAConfig, load_fixture, run_pipeline


class TestPCAEigenfacesReproducibility(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = load_fixture()
        self.metadata = self.payload["metadata"]
        self.records = self.payload["records"]
        self.points = [[r["x1"], r["x2"]] for r in self.records]

    def test_collinear_projection_and_reconstruction(self) -> None:
        unit_vec = (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0))
        pca = PCA1DProjection(unit_vector=unit_vec, tolerance=1e-5)

        recovered, max_err = pca.verify_reconstruction(self.points)
        self.assertTrue(recovered)
        self.assertLessEqual(max_err, 1e-5)

        reconstructed = pca.reconstruct(pca.project(self.points))
        for orig, rec in zip(self.points, reconstructed):
            self.assertAlmostEqual(orig[0], rec[0], places=5)
            self.assertAlmostEqual(orig[1], rec[1], places=5)

    def test_sign_indeterminacy_invariance(self) -> None:
        # Sign of learned principal component can be reversed: -v
        reversed_unit_vec = (-1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0))
        pca_reversed = PCA1DProjection(unit_vector=reversed_unit_vec, tolerance=1e-5)

        recovered, max_err = pca_reversed.verify_reconstruction(self.points)
        self.assertTrue(recovered)
        self.assertLessEqual(max_err, 1e-5)

        reconstructed = pca_reversed.reconstruct(pca_reversed.project(self.points))
        for orig, rec in zip(self.points, reconstructed):
            self.assertAlmostEqual(orig[0], rec[0], places=5)
            self.assertAlmostEqual(orig[1], rec[1], places=5)

    def test_reproducible_config_reloads_identically(self) -> None:
        res1 = run_pipeline()
        res2 = run_pipeline()

        self.assertEqual(res1["schema_version"], "1.0.0")
        self.assertEqual(res1["seed"], 42)
        self.assertEqual(res1["projections"], res2["projections"])
        self.assertEqual(res1["reconstructed"], res2["reconstructed"])
        self.assertTrue(res1["reconstruction_recovered"])

    def test_reject_unknown_and_contradictory_settings(self) -> None:
        valid_cfg = dict(self.metadata["config"])

        # 1. Reject unknown top-level config field
        bad_cfg_unknown = dict(valid_cfg)
        bad_cfg_unknown["unexpected_param"] = "invalid"
        with self.assertRaisesRegex(ValueError, "Unknown configuration fields"):
            ReproduciblePCAConfig(bad_cfg_unknown)

        # 2. Reject contradictory n_components
        bad_cfg_components = dict(valid_cfg)
        bad_cfg_components["algorithm_settings"] = dict(valid_cfg["algorithm_settings"])
        bad_cfg_components["algorithm_settings"]["n_components"] = 2
        with self.assertRaisesRegex(ValueError, "Contradictory configuration"):
            ReproduciblePCAConfig(bad_cfg_components)

        # 3. Reject non-unit vector
        with self.assertRaisesRegex(ValueError, "not a unit vector"):
            PCA1DProjection(unit_vector=(5.0, 5.0))


if __name__ == "__main__":
    unittest.main()
