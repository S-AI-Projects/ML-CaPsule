"""Synthetic fixture for Task R41-T009: Kaggle house prices – feature engineering.

DISCLAIMER:
This dataset is an explicitly synthetic, deterministic formula fixture
(target = 2 * x1 + 1) created solely for offline pipeline verification and
robustness testing. It does NOT contain real-world Ames/Kaggle housing sales records,
real-estate transactions, or actual home valuation estimates.
"""

from typing import Any, Dict, List

DISCLAIMER: str = (
    "Synthetic pipeline-verification fixture: deterministic formula target = 2*x1 + 1. "
    "Not real-world Kaggle house prices data."
)

# Deterministic formula fixture: target = 2 * x1 + 1
# x1 = [0, 1, 2, 3, 4, 5], x2 = 1, target = [1, 3, 5, 7, 9, 11]
SYNTHETIC_RECORDS: List[Dict[str, Any]] = [
    {"record_id": "r1", "x1": 0, "x2": 1, "target": 1},
    {"record_id": "r2", "x1": 1, "x2": 1, "target": 3},
    {"record_id": "r3", "x1": 2, "x2": 1, "target": 5},
    {"record_id": "r4", "x1": 3, "x2": 1, "target": 7},
    {"record_id": "r5", "x1": 4, "x2": 1, "target": 9},
    {"record_id": "r6", "x1": 5, "x2": 1, "target": 11},
]

# Split assignment: Training r1..r4, Validation r5, Test r6
SPLIT_MANIFEST: Dict[str, List[str]] = {
    "train": ["r1", "r2", "r3", "r4"],
    "validation": ["r5"],
    "test": ["r6"],
}

FEATURE_ORDER: List[str] = ["x1", "x2"]
TARGET_FIELD: str = "target"
ID_FIELD: str = "record_id"


def get_split_records() -> Dict[str, List[Dict[str, Any]]]:
    """Return records partitioned according to SPLIT_MANIFEST."""
    records_by_id = {r[ID_FIELD]: dict(r) for r in SYNTHETIC_RECORDS}
    return {
        split_name: [dict(records_by_id[rid]) for rid in ids]
        for split_name, ids in SPLIT_MANIFEST.items()
    }
