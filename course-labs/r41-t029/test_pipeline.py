"""Unit tests for R41-T029 (Goodbooks-10k – content-based recommender robustness).

Task: R41-T029 · 9 of 10
Student: Sharon Robinson
Repository: S-AI-Projects/ML-CaPsule

Run offline (CPU-only, Python 3 standard library only):
    python3 -m unittest discover -s course-labs/r41-t029 -p "test_*.py" -v

All tests use six synthetic interactions strictly for pipeline and interface
robustness verification — NOT observations from Goodbooks-10k, NOT evidence
of real-world or clinical performance, and NOT a claim of broad adversarial
robustness.
"""

from __future__ import annotations

import copy
import os
import sys
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import pipeline  # noqa: E402

EXPECTED_HELD_OUT_SCORES = {
    "u1:b": 0.5,
    "u2:c": 0.0,
    "u3:c": 0.5,
}
EXPECTED_TOP2_ITEMS = {
    "u1": ["b", "c"],
    "u2": ["b", "c"],
    "u3": ["a", "c"],
}


class TestSyntheticInteractionFixtureContract(unittest.TestCase):
    """Verify the literal six synthetic interactions and schema contract."""

    def setUp(self) -> None:
        self.fixture_doc = pipeline.load_fixture()
        self.schema_doc = pipeline.load_schema()
        self.interactions = self.fixture_doc["interactions"]

    def test_explicit_synthetic_and_non_adversarial_scope_labelling(self) -> None:
        self.assertTrue(self.fixture_doc["is_synthetic"])
        self.assertIn("not observations", self.fixture_doc["disclaimer"].lower())
        self.assertIn("broad adversarial robustness", self.fixture_doc["disclaimer"].lower())
        self.assertTrue(self.schema_doc["synthetic_only"])
        self.assertFalse(
            self.schema_doc["robustness_properties"]["claims_broad_adversarial_robustness"]
        )

    def test_literal_six_interactions_and_stable_ids(self) -> None:
        tuples = [(r["user_id"], r["item_id"], r["rating"]) for r in self.interactions]
        self.assertEqual(
            tuples,
            [
                ("u1", "a", 5),
                ("u1", "b", 3),
                ("u2", "a", 4),
                ("u2", "c", 2),
                ("u3", "b", 4),
                ("u3", "c", 5),
            ],
        )

    def test_leave_last_out_reserves_final_interaction_per_user(self) -> None:
        splits = pipeline.split_leave_last_out(self.interactions)
        self.assertEqual(
            [(r["user_id"], r["item_id"], r["rating"]) for r in splits.train],
            [("u1", "a", 5.0), ("u2", "a", 4.0), ("u3", "b", 4.0)],
        )
        self.assertEqual(
            [(r["user_id"], r["item_id"], r["rating"]) for r in splits.validation],
            [("u1", "b", 3.0), ("u2", "c", 2.0)],
        )
        self.assertEqual(
            [(r["user_id"], r["item_id"], r["rating"]) for r in splits.test],
            [("u3", "c", 5.0)],
        )
        # Disjointness across train, validation, and test
        tr_ids = {r["interaction_id"] for r in splits.train}
        va_ids = {r["interaction_id"] for r in splits.validation}
        te_ids = {r["interaction_id"] for r in splits.test}
        self.assertEqual(tr_ids & va_ids, set())
        self.assertEqual(tr_ids & te_ids, set())
        self.assertEqual(va_ids & te_ids, set())


class TestValidTransformationRobustness(unittest.TestCase):
    """Inter-user reordering preserving within-user time order must preserve results."""

    def setUp(self) -> None:
        self.result = pipeline.run_robustness_verification()

    def test_permuted_order_differs_from_original_yet_preserves_aligned_results(self) -> None:
        self.assertNotEqual(
            self.result["original_order_ids"],
            self.result["permuted_order_ids"],
        )
        self.assertTrue(self.result["invariance_preserved"])
        self.assertTrue(self.result["verification_only"])
        self.assertFalse(self.result["claims_broad_adversarial_robustness"])

    def test_documented_held_out_scores_and_rankings_match_expected(self) -> None:
        combined_held_out = {
            **self.result["validation_scores"],
            **self.result["test_scores"],
        }
        for pair_key, expected_score in EXPECTED_HELD_OUT_SCORES.items():
            self.assertAlmostEqual(
                combined_held_out[pair_key]["score"],
                expected_score,
                places=12,
            )
            self.assertFalse(combined_held_out[pair_key]["is_fallback"])

        for uid, expected_items in EXPECTED_TOP2_ITEMS.items():
            actual_items = [r["item_id"] for r in self.result["recommendations"][uid]]
            self.assertEqual(actual_items, expected_items)

    def test_within_user_time_order_is_strictly_preserved_by_valid_perturbation(self) -> None:
        fixture_doc = pipeline.load_fixture()
        permuted = pipeline.permute_preserving_user_time_order(
            fixture_doc["interactions"],
            seed=pipeline.FIXED_SEED,
        )
        seen_seq: dict[str, list[int]] = {}
        for row in permuted:
            seen_seq.setdefault(row["user_id"], []).append(row["user_seq"])
        for uid, seqs in seen_seq.items():
            self.assertEqual(seqs, [1, 2], f"Within-user time order broke for {uid}")


class TestUnsupportedInputsAndExplicitFallbacks(unittest.TestCase):
    """Unsupported inputs must be rejected or mapped to explicit fallbacks."""

    def setUp(self) -> None:
        self.fixture_doc = pipeline.load_fixture()
        self.interactions = self.fixture_doc["interactions"]
        self.splits = pipeline.split_leave_last_out(self.interactions)
        self.model = pipeline.ContentBasedRecommender(
            self.fixture_doc["item_catalog"]
        ).fit(self.splits.train)

    def test_reversed_within_user_time_order_is_rejected(self) -> None:
        bad_time = copy.deepcopy(self.interactions)
        # Swap u1's seq=1 (i1) and seq=2 (i2) — forbidden time-order randomization
        bad_time[0], bad_time[1] = bad_time[1], bad_time[0]
        with self.assertRaises(pipeline.TimeOrderViolationError):
            pipeline.split_leave_last_out(bad_time)

    def test_missing_required_field_is_rejected(self) -> None:
        for missing_key in ("user_id", "item_id", "rating"):
            bad = copy.deepcopy(self.interactions)
            del bad[0][missing_key]
            with self.assertRaises(pipeline.SchemaValidationError):
                pipeline.validate_interactions(bad)

    def test_invalid_rating_values_are_rejected(self) -> None:
        for bad_rating in (None, float("nan"), 0, 6, True, "5"):
            bad = copy.deepcopy(self.interactions)
            bad[0]["rating"] = bad_rating
            with self.assertRaises(pipeline.SchemaValidationError):
                pipeline.validate_interactions(bad)

    def test_unseen_user_item_and_tokens_map_to_explicit_fallbacks(self) -> None:
        unseen_user = self.model.score_pair("u_cold", "a")
        self.assertTrue(unseen_user.is_fallback)
        self.assertEqual(unseen_user.fallback_reason, "unseen_user")
        self.assertEqual(unseen_user.score, 0.0)

        cold_recs = self.model.recommend("u_cold", top_k=2)
        self.assertEqual(len(cold_recs), 2)
        self.assertTrue(all(r.is_fallback and r.fallback_reason == "unseen_user" for r in cold_recs))

        unseen_item = self.model.score_pair("u1", "z_unknown")
        self.assertTrue(unseen_item.is_fallback)
        self.assertEqual(unseen_item.fallback_reason, "unseen_item")
        self.assertEqual(unseen_item.score, 0.0)

        unseen_tokens = self.model.score_pair(
            "u1", "d_new", override_item_tags=["unseen_genre_xyz"]
        )
        self.assertTrue(unseen_tokens.is_fallback)
        self.assertEqual(unseen_tokens.fallback_reason, "unseen_content_tokens")
        self.assertEqual(unseen_tokens.score, 0.0)

    def test_scoring_before_fit_raises(self) -> None:
        unfitted = pipeline.ContentBasedRecommender(self.fixture_doc["item_catalog"])
        with self.assertRaises(pipeline.ModelNotFittedError):
            unfitted.score_pair("u1", "b")
        with self.assertRaises(pipeline.ModelNotFittedError):
            unfitted.recommend("u1")


class TestDeliberateMutationGuard(unittest.TestCase):
    """Prove assertions fail when a documented held-out score is deliberately changed."""

    def test_mutated_expected_score_fails_assertion(self) -> None:
        res = pipeline.run_robustness_verification()
        actual_u1_b = res["validation_scores"]["u1:b"]["score"]
        with self.assertRaises(AssertionError):
            self.assertAlmostEqual(actual_u1_b, 0.99, places=6)


if __name__ == "__main__":
    unittest.main()
