"""Unit tests for MovieLens 100k minimal baseline (Task R41-T013)."""

import unittest
from pipeline import ItemMeanBaseline, evaluate_baseline, load_fixture, split_held_out_per_user


class TestMovieLensBaseline(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = load_fixture()
        self.interactions = self.payload["interactions"]

    def test_fixture_contents_and_exact_split(self) -> None:
        self.assertEqual(len(self.interactions), 6)
        train_data, test_data = split_held_out_per_user(self.interactions)
        self.assertEqual(len(train_data), 3)
        self.assertEqual(len(test_data), 3)

        expected_test = [
            {"user_id": "u1", "item_id": "b", "rating": 3.0, "order": 2},
            {"user_id": "u2", "item_id": "c", "rating": 2.0, "order": 2},
            {"user_id": "u3", "item_id": "c", "rating": 5.0, "order": 2},
        ]
        self.assertEqual(test_data, expected_test)

    def test_baseline_train_only_parameters_and_evaluation(self) -> None:
        train_data, test_data = split_held_out_per_user(self.interactions)
        model = ItemMeanBaseline().fit(train_data)

        # Baseline parameters must derive exclusively from training set
        self.assertAlmostEqual(model.item_means["a"], 4.5, places=4)
        self.assertAlmostEqual(model.item_means["b"], 4.0, places=4)
        self.assertNotIn("c", model.item_means)
        self.assertAlmostEqual(model.global_mean, 13.0 / 3.0, places=4)

        # Unseen item c in training falls back to global mean
        self.assertAlmostEqual(model.predict("u2", "c"), 13.0 / 3.0, places=4)

        res = evaluate_baseline(model, test_data)
        self.assertEqual(res["held_out_count"], 3)
        self.assertAlmostEqual(res["mae"], 1.3333, places=3)
        self.assertAlmostEqual(res["mse"], 2.2963, places=3)

    def test_failure_cases(self) -> None:
        model = ItemMeanBaseline()
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            model.predict("u1", "a")

        with self.assertRaisesRegex(ValueError, "at least 2"):
            split_held_out_per_user([{"user_id": "u1", "item_id": "a", "rating": 5.0}])


if __name__ == "__main__":
    unittest.main()
