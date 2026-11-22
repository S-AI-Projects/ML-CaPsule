"""Unit tests for R41-T025 (Zoo animals – decision tree numeric preprocessing).

Task: R41-T025 · 5 of 10
Student: Sharon Robinson
Repository: S-AI-Projects/ML-CaPsule

Run offline (CPU-only, Python standard library only):
    python3 -m unittest discover -s course-labs/r41-t025 -p "test_*.py" -v

All tests use the bounded eight-row synthetic fixture for pipeline
verification only — NOT observations from the UCI Zoo dataset and NOT
evidence of real-world or clinical performance.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import os
import sys
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import pipeline  # noqa: E402

# ── Exact analytical expectations for train x1=[0, 1, 4, 5], x2=[1, 1, 1, 1] ─

EXPECTED_TRAIN_MEAN_X1 = 2.5
EXPECTED_TRAIN_MEAN_X2 = 1.0
EXPECTED_TRAIN_VAR_X1 = 4.25
EXPECTED_TRAIN_STD_X1 = math.sqrt(EXPECTED_TRAIN_VAR_X1)
EXPECTED_TRAIN_STD_X2 = 0.0
EXPECTED_TRAIN_SCALE_X1 = EXPECTED_TRAIN_STD_X1
EXPECTED_TRAIN_SCALE_X2 = 1.0

EXPECTED_TRAIN_VALUES = (
    ((0.0 - EXPECTED_TRAIN_MEAN_X1) / EXPECTED_TRAIN_SCALE_X1, 0.0),
    ((1.0 - EXPECTED_TRAIN_MEAN_X1) / EXPECTED_TRAIN_SCALE_X1, 0.0),
    ((4.0 - EXPECTED_TRAIN_MEAN_X1) / EXPECTED_TRAIN_SCALE_X1, 0.0),
    ((5.0 - EXPECTED_TRAIN_MEAN_X1) / EXPECTED_TRAIN_SCALE_X1, 0.0),
)
EXPECTED_VAL_VALUES = (
    ((2.0 - EXPECTED_TRAIN_MEAN_X1) / EXPECTED_TRAIN_SCALE_X1, 0.0),
    ((6.0 - EXPECTED_TRAIN_MEAN_X1) / EXPECTED_TRAIN_SCALE_X1, 0.0),
)
EXPECTED_TEST_VALUES = (
    ((3.0 - EXPECTED_TRAIN_MEAN_X1) / EXPECTED_TRAIN_SCALE_X1, 0.0),
    ((7.0 - EXPECTED_TRAIN_MEAN_X1) / EXPECTED_TRAIN_SCALE_X1, 0.0),
)


class TestSyntheticFixtureContract(unittest.TestCase):
    """Verify the literal eight-row synthetic fixture and schema contract."""

    def setUp(self) -> None:
        self.fixture_doc = pipeline.load_fixture()
        self.schema_doc = pipeline.load_schema()
        self.records = self.fixture_doc["records"]

    def test_explicit_synthetic_labelling(self) -> None:
        self.assertTrue(self.fixture_doc["is_synthetic"])
        self.assertIn("not observations", self.fixture_doc["disclaimer"].lower())
        self.assertIn("pipeline-verification", self.fixture_doc["disclaimer"].lower())
        self.assertTrue(self.schema_doc["synthetic_only"])

    def test_literal_eight_rows_and_values(self) -> None:
        self.assertEqual(len(self.records), 8)
        self.assertEqual([r["x1"] for r in self.records], [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual([r["x2"] for r in self.records], [1, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual([r["label"] for r in self.records], [0, 0, 0, 0, 1, 1, 1, 1])

    def test_stable_record_and_group_ids(self) -> None:
        record_ids = [r["record_id"] for r in self.records]
        group_ids = [r["group_id"] for r in self.records]
        self.assertEqual(record_ids, ["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8"])
        self.assertEqual(group_ids, ["g1", "g1", "g2", "g3", "g4", "g4", "g5", "g6"])
        self.assertEqual(len(set(record_ids)), 8)

    def test_schema_matches_pipeline_constants(self) -> None:
        self.assertEqual(
            tuple(self.schema_doc["feature_order"]),
            pipeline.EXPECTED_FEATURE_ORDER,
        )
        self.assertEqual(
            frozenset(self.schema_doc["valid_labels"]),
            pipeline.VALID_LABELS,
        )
        self.assertEqual(
            set(self.schema_doc["missing_value_policies"]),
            {p.value for p in pipeline.MissingValuePolicy},
        )


class TestStratifiedTrainValTestSplit(unittest.TestCase):
    """Verify explicit, stratified, group-isolated train/validation/test split."""

    def setUp(self) -> None:
        self.fixture_doc = pipeline.load_fixture()
        self.partitions = pipeline.validate_and_partition_fixture(
            self.fixture_doc["records"]
        )

    def test_explicit_three_way_split_membership(self) -> None:
        self.assertEqual(
            [r["record_id"] for r in self.partitions.train],
            ["r1", "r2", "r5", "r6"],
        )
        self.assertEqual(
            [r["record_id"] for r in self.partitions.validation],
            ["r3", "r7"],
        )
        self.assertEqual(
            [r["record_id"] for r in self.partitions.test],
            ["r4", "r8"],
        )

    def test_disjoint_record_and_group_ids_across_splits(self) -> None:
        split_lists = [
            self.partitions.train,
            self.partitions.validation,
            self.partitions.test,
        ]
        rec_sets = [{r["record_id"] for r in s} for s in split_lists]
        grp_sets = [{r["group_id"] for r in s} for s in split_lists]

        for i in range(len(split_lists)):
            for j in range(i + 1, len(split_lists)):
                self.assertEqual(rec_sets[i] & rec_sets[j], set())
                self.assertEqual(grp_sets[i] & grp_sets[j], set())

        self.assertEqual(set().union(*rec_sets), {f"r{k}" for k in range(1, 9)})

    def test_stratified_class_balance_in_every_split(self) -> None:
        self.assertEqual([r["label"] for r in self.partitions.train], [0, 0, 1, 1])
        self.assertEqual([r["label"] for r in self.partitions.validation], [0, 1])
        self.assertEqual([r["label"] for r in self.partitions.test], [0, 1])


class TestTrainingOnlyFitAndTransformOutputs(unittest.TestCase):
    """Assert fitted statistics, output values, dtype, shape, and determinism."""

    def setUp(self) -> None:
        self.fixture_doc = pipeline.load_fixture()
        self.partitions = pipeline.validate_and_partition_fixture(
            self.fixture_doc["records"]
        )
        self.fitted = pipeline.fit_numeric_preprocessor(
            self.partitions.train,
            held_out_records=self.partitions.validation + self.partitions.test,
            seed=pipeline.FIXED_SEED,
        )
        self.train_out = self.fitted.transform_split("train", self.partitions.train)
        self.val_out = self.fitted.transform_split(
            "validation", self.partitions.validation
        )
        self.test_out = self.fitted.transform_split("test", self.partitions.test)

    def test_fitted_statistics_reflect_training_rows_only(self) -> None:
        self.assertEqual(self.fitted.n_train_rows, 4)
        self.assertAlmostEqual(self.fitted.means[0], EXPECTED_TRAIN_MEAN_X1, places=12)
        self.assertAlmostEqual(self.fitted.means[1], EXPECTED_TRAIN_MEAN_X2, places=12)
        self.assertAlmostEqual(self.fitted.stds[0], EXPECTED_TRAIN_STD_X1, places=12)
        self.assertAlmostEqual(self.fitted.stds[1], EXPECTED_TRAIN_STD_X2, places=12)
        self.assertAlmostEqual(self.fitted.scales[0], EXPECTED_TRAIN_SCALE_X1, places=12)
        self.assertAlmostEqual(self.fitted.scales[1], EXPECTED_TRAIN_SCALE_X2, places=12)
        # Full-fixture mean of x1=[0..7] would be 3.5; confirm no leakage
        self.assertNotAlmostEqual(self.fitted.means[0], 3.5, places=6)

    def test_output_shapes_and_dtypes(self) -> None:
        self.assertEqual(self.train_out.shape, (4, 2))
        self.assertEqual(self.val_out.shape, (2, 2))
        self.assertEqual(self.test_out.shape, (2, 2))

        for matrix in (self.train_out, self.val_out, self.test_out):
            self.assertEqual(matrix.dtype, "float64")
            self.assertEqual(matrix.feature_names, ("x1", "x2"))
            for row in matrix.values:
                for val in row:
                    self.assertIs(type(val), float)
                    self.assertTrue(math.isfinite(val))

    def test_exact_transformed_values_all_splits(self) -> None:
        for actual_mat, expected_rows in (
            (self.train_out, EXPECTED_TRAIN_VALUES),
            (self.val_out, EXPECTED_VAL_VALUES),
            (self.test_out, EXPECTED_TEST_VALUES),
        ):
            self.assertEqual(len(actual_mat.values), len(expected_rows))
            for actual_row, expected_row in zip(actual_mat.values, expected_rows):
                for a_val, e_val in zip(actual_row, expected_row):
                    self.assertAlmostEqual(a_val, e_val, places=12)

    def test_train_standardized_moments_and_held_out_non_zero_mean(self) -> None:
        train_x1 = [row[0] for row in self.train_out.values]
        train_x2 = [row[1] for row in self.train_out.values]
        self.assertAlmostEqual(sum(train_x1) / len(train_x1), 0.0, places=12)
        self.assertAlmostEqual(sum(train_x2) / len(train_x2), 0.0, places=12)
        train_var_x1 = sum(v * v for v in train_x1) / len(train_x1)
        self.assertAlmostEqual(train_var_x1, 1.0, places=12)

        # Held-out splits are transformed with train stats, NOT refit to zero mean
        val_mean_x1 = sum(row[0] for row in self.val_out.values) / len(self.val_out.values)
        test_mean_x1 = sum(row[0] for row in self.test_out.values) / len(self.test_out.values)
        self.assertNotAlmostEqual(val_mean_x1, 0.0, places=6)
        self.assertNotAlmostEqual(test_mean_x1, 0.0, places=6)

    def test_deterministic_verification_pipeline(self) -> None:
        run_a = pipeline.run_verification_pipeline()
        run_b = pipeline.run_verification_pipeline()
        self.assertEqual(run_a, run_b)
        self.assertTrue(run_a["verification_only"])


class TestTransformInterfaceCannotFit(unittest.TestCase):
    """Prove fitting on held-out input is impossible via the public transform API."""

    def setUp(self) -> None:
        fixture_doc = pipeline.load_fixture()
        self.partitions = pipeline.validate_and_partition_fixture(
            fixture_doc["records"]
        )
        self.fitted = pipeline.fit_numeric_preprocessor(self.partitions.train)

    def test_no_fit_methods_exist_on_public_transform_interface(self) -> None:
        self.assertFalse(hasattr(self.fitted, "fit"))
        self.assertFalse(hasattr(self.fitted, "partial_fit"))
        self.assertFalse(hasattr(self.fitted, "fit_transform"))

    def test_fitted_statistics_are_immutable(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.fitted.means = (999.0, 999.0)  # type: ignore[misc]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.fitted.scales = (1.0, 1.0)  # type: ignore[misc]

    def test_transforming_extreme_held_out_values_does_not_change_stats(self) -> None:
        means_before = self.fitted.means
        scales_before = self.fitted.scales
        extreme_held_out = [
            {
                "record_id": "r_ext",
                "group_id": "g_ext",
                "x1": 1_000_000.0,
                "x2": 500.0,
                "label": 1,
                "split": "test",
            }
        ]
        out = self.fitted.transform_split("test", extreme_held_out)
        self.assertEqual(out.shape, (1, 2))
        self.assertEqual(self.fitted.means, means_before)
        self.assertEqual(self.fitted.scales, scales_before)


class TestDeliberateContaminationRejection(unittest.TestCase):
    """Verify deliberate contamination fixtures are explicitly rejected."""

    def setUp(self) -> None:
        self.fixture_doc = pipeline.load_fixture()
        self.records = self.fixture_doc["records"]
        self.partitions = pipeline.validate_and_partition_fixture(self.records)
        self.fitted = pipeline.fit_numeric_preprocessor(self.partitions.train)

    def test_non_train_row_injected_into_fit_is_rejected(self) -> None:
        contaminated_train = list(self.partitions.train) + [
            dict(self.partitions.validation[0])
        ]
        with self.assertRaises(pipeline.ContaminationError):
            pipeline.fit_numeric_preprocessor(contaminated_train)

    def test_overlapping_record_id_across_splits_rejected(self) -> None:
        contaminated_records = copy.deepcopy(self.records)
        # Duplicate training record_id r1 inside the test split
        contaminated_records.append(
            {
                "record_id": "r1",
                "group_id": "g99",
                "x1": 99.0,
                "x2": 1.0,
                "label": 1,
                "split": "test",
            }
        )
        with self.assertRaises(pipeline.ContaminationError):
            pipeline.validate_and_partition_fixture(contaminated_records)

    def test_overlapping_group_id_across_splits_rejected(self) -> None:
        contaminated_records = copy.deepcopy(self.records)
        # Move validation record r3 into training group g1 (group leakage)
        contaminated_records[2]["group_id"] = "g1"
        with self.assertRaises(pipeline.ContaminationError):
            pipeline.validate_and_partition_fixture(contaminated_records)

    def test_fit_with_held_out_group_overlap_rejected(self) -> None:
        bad_held_out = [
            {
                "record_id": "r_leak",
                "group_id": "g1",  # overlaps train group g1
                "x1": 9.0,
                "x2": 1.0,
                "label": 0,
                "split": "test",
            }
        ]
        with self.assertRaises(pipeline.ContaminationError):
            pipeline.fit_numeric_preprocessor(
                self.partitions.train,
                held_out_records=bad_held_out,
            )

    def test_train_row_passed_into_held_out_transform_rejected(self) -> None:
        leaked_val_batch = [dict(self.partitions.validation[0]), dict(self.partitions.train[0])]
        with self.assertRaises(pipeline.ContaminationError):
            self.fitted.transform_split("validation", leaked_val_batch)

    def test_train_group_id_disguised_as_test_rejected_in_transform(self) -> None:
        disguised_test_row = [
            {
                "record_id": "r_disguised",
                "group_id": "g4",  # belongs to train
                "x1": 4.5,
                "x2": 1.0,
                "label": 1,
                "split": "test",
            }
        ]
        with self.assertRaises(pipeline.ContaminationError):
            self.fitted.transform_split("test", disguised_test_row)


class TestFeatureOrderLabelAndMissingValueContracts(unittest.TestCase):
    """Validate feature order, label set, and missing-value policy enforcement."""

    def setUp(self) -> None:
        fixture_doc = pipeline.load_fixture()
        self.partitions = pipeline.validate_and_partition_fixture(
            fixture_doc["records"]
        )
        self.fitted = pipeline.fit_numeric_preprocessor(self.partitions.train)

    def test_swapped_feature_order_rejected_in_fit_and_transform(self) -> None:
        swapped_train = [
            {
                "record_id": r["record_id"],
                "group_id": r["group_id"],
                "x2": r["x2"],
                "x1": r["x1"],
                "label": r["label"],
                "split": r["split"],
            }
            for r in self.partitions.train
        ]
        with self.assertRaises(pipeline.FeatureOrderError):
            pipeline.fit_numeric_preprocessor(swapped_train)

        swapped_val = [
            {
                "record_id": r["record_id"],
                "group_id": r["group_id"],
                "x2": r["x2"],
                "x1": r["x1"],
                "label": r["label"],
                "split": r["split"],
            }
            for r in self.partitions.validation
        ]
        with self.assertRaises(pipeline.FeatureOrderError):
            self.fitted.transform_split("validation", swapped_val)

        with self.assertRaises(pipeline.FeatureOrderError):
            self.fitted.transform_split(
                "validation",
                self.partitions.validation,
                feature_order=("x2", "x1"),
            )

    def test_missing_or_extra_feature_column_rejected(self) -> None:
        missing_col = [
            {k: v for k, v in r.items() if k != "x2"} for r in self.partitions.train
        ]
        with self.assertRaises(pipeline.FeatureOrderError):
            pipeline.fit_numeric_preprocessor(missing_col)

        extra_col = [dict(r, x3=0.0) for r in self.partitions.validation]
        with self.assertRaises(pipeline.FeatureOrderError):
            self.fitted.transform_split("validation", extra_col)

    def test_invalid_labels_rejected(self) -> None:
        for bad_label in (2, -1, True, False, 1.0, "1", None):
            bad_train = copy.deepcopy(list(self.partitions.train))
            bad_train[0]["label"] = bad_label
            with self.assertRaises(pipeline.InvalidLabelError):
                pipeline.fit_numeric_preprocessor(bad_train)

    def test_invalid_missing_value_policy_rejected(self) -> None:
        with self.assertRaises(pipeline.InvalidPolicyError):
            pipeline.fit_numeric_preprocessor(
                self.partitions.train,
                missing_value_policy="drop_silently",
            )

    def test_missing_and_non_finite_values_rejected_under_default_policy(self) -> None:
        for bad_val in (None, float("nan"), float("inf"), float("-inf"), "", True):
            bad_train = copy.deepcopy(list(self.partitions.train))
            bad_train[0]["x1"] = bad_val
            with self.assertRaises(pipeline.MissingValueError):
                pipeline.fit_numeric_preprocessor(bad_train)

            bad_val_split = copy.deepcopy(list(self.partitions.validation))
            bad_val_split[0]["x1"] = bad_val
            with self.assertRaises(pipeline.MissingValueError):
                self.fitted.transform_split("validation", bad_val_split)

    def test_impute_train_mean_policy_uses_training_mean_only(self) -> None:
        train_with_missing = copy.deepcopy(list(self.partitions.train))
        # Observed train x1 = [0, 4, 5] -> train mean = 3.0
        train_with_missing[1]["x1"] = None
        fitted_imp = pipeline.fit_numeric_preprocessor(
            train_with_missing,
            missing_value_policy=pipeline.MissingValuePolicy.IMPUTE_TRAIN_MEAN,
        )
        self.assertAlmostEqual(fitted_imp.means[0], 3.0, places=12)

        # Held-out row with missing x1 imputes strictly with frozen train mean 3.0 -> scaled 0.0
        val_with_missing = copy.deepcopy(list(self.partitions.validation))
        val_with_missing[0]["x1"] = float("nan")
        val_out = fitted_imp.transform_split("validation", val_with_missing)
        self.assertAlmostEqual(val_out.values[0][0], 0.0, places=12)
        self.assertEqual(val_out.dtype, "float64")
        self.assertEqual(val_out.shape, (2, 2))

        # All-missing training column must still be rejected
        all_missing_train = copy.deepcopy(list(self.partitions.train))
        for row in all_missing_train:
            row["x1"] = None
        with self.assertRaises(pipeline.MissingValueError):
            pipeline.fit_numeric_preprocessor(
                all_missing_train,
                missing_value_policy=pipeline.MissingValuePolicy.IMPUTE_TRAIN_MEAN,
            )


class TestDeliberateMutationGuard(unittest.TestCase):
    """Prove assertions fail when an expected output value is deliberately altered."""

    def test_mutated_expected_value_fails_assertion(self) -> None:
        result = pipeline.run_verification_pipeline()
        actual_val_x1_first = result["splits"]["validation"]["values"][0][0]
        deliberately_wrong_expected = 0.999999
        with self.assertRaises(AssertionError):
            self.assertAlmostEqual(
                actual_val_x1_first,
                deliberately_wrong_expected,
                places=6,
            )


if __name__ == "__main__":
    unittest.main()
