"""Minimal baseline for MovieLens 100k collaborative filtering (Task R41-T013)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def split_held_out_per_user(interactions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Reserve the final interaction per user (by order/appearance) for held-out evaluation."""
    if not isinstance(interactions, list) or not interactions:
        raise ValueError("interactions must be a non-empty list")

    by_user: Dict[str, List[Dict[str, Any]]] = {}
    for item in interactions:
        uid = item["user_id"]
        by_user.setdefault(uid, []).append(item)

    train_set: List[Dict[str, Any]] = []
    test_set: List[Dict[str, Any]] = []

    for uid, user_records in by_user.items():
        if len(user_records) < 2:
            raise ValueError(f"user {uid} must have at least 2 interactions to reserve a held-out item")
        sorted_records = sorted(user_records, key=lambda r: r.get("order", 0))
        train_set.extend(sorted_records[:-1])
        test_set.append(sorted_records[-1])

    return train_set, test_set


class ItemMeanBaseline:
    """Baseline model predicting training item-mean rating, with fallback to global train mean."""

    def __init__(self) -> None:
        self.item_means: Dict[str, float] = {}
        self.global_mean: float = 0.0
        self.is_fitted: bool = False

    def fit(self, train_records: List[Dict[str, Any]]) -> "ItemMeanBaseline":
        if not train_records:
            raise ValueError("training records cannot be empty")

        item_totals: Dict[str, float] = {}
        item_counts: Dict[str, int] = {}
        total_rating = 0.0

        for r in train_records:
            item = r["item_id"]
            rating = float(r["rating"])
            item_totals[item] = item_totals.get(item, 0.0) + rating
            item_counts[item] = item_counts.get(item, 0) + 1
            total_rating += rating

        self.item_means = {item: item_totals[item] / item_counts[item] for item in item_totals}
        self.global_mean = total_rating / len(train_records)
        self.is_fitted = True
        return self

    def predict(self, user_id: str, item_id: str) -> float:
        if not self.is_fitted:
            raise RuntimeError("baseline model is not fitted")
        return self.item_means.get(item_id, self.global_mean)


def evaluate_baseline(model: ItemMeanBaseline, test_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate predictions strictly on the fixed held-out fixture."""
    if not test_records:
        raise ValueError("test records cannot be empty")

    predictions = []
    total_abs_error = 0.0
    total_sq_error = 0.0

    for r in test_records:
        pred = model.predict(r["user_id"], r["item_id"])
        actual = float(r["rating"])
        err = actual - pred
        total_abs_error += abs(err)
        total_sq_error += err ** 2
        predictions.append({
            "user_id": r["user_id"],
            "item_id": r["item_id"],
            "actual": actual,
            "prediction": round(pred, 4),
        })

    n = len(test_records)
    return {
        "task_id": "R41-T013",
        "baseline_strategy": "training_item_mean_with_global_fallback",
        "train_item_means": {k: round(v, 4) for k, v in model.item_means.items()},
        "train_global_mean": round(model.global_mean, 4),
        "held_out_count": n,
        "mae": round(total_abs_error / n, 4),
        "mse": round(total_sq_error / n, 4),
        "predictions": predictions,
    }


def load_fixture(path: Path = None) -> Dict[str, Any]:
    target = path or (Path(__file__).resolve().parent / "fixture.json")
    with target.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    data = load_fixture()
    train_data, test_data = split_held_out_per_user(data["interactions"])
    baseline = ItemMeanBaseline().fit(train_data)
    results = evaluate_baseline(baseline, test_data)
    print(json.dumps(results, indent=2))
