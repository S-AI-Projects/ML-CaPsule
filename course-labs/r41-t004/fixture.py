"""Synthetic fixture and metric benchmark data for R41-T004.

DISCLAIMER:
These eight synthetic records are strictly generated for pipeline verification
and evaluation metric cross-checking. They do NOT represent real Pokémon stats,
legendary status observations, or actual game data, and must not be used as evidence
of real-world model performance.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

# Explicit statement of synthetic purpose
DATASET_DISCLAIMER: str = (
    "Synthetic verification fixture for R41-T004 metric verification. "
    "These rows do NOT represent real Pokémon stats or legendary classification observations."
)

# Canonical eight synthetic records conforming to Issue #4 specification:
# - record_id: stable, unique, non-empty string IDs
# - group_id: stable, non-empty string IDs
# - x1: [0, 1, 2, 3, 4, 5, 6, 7]
# - x2: [1, 1, 1, 1, 1, 1, 1, 1]
# - label: [0, 0, 0, 0, 1, 1, 1, 1]
SYNTHETIC_FIXTURE: List[Dict[str, Any]] = [
    {"record_id": "rec_001", "group_id": "grp_01", "x1": 0, "x2": 1, "label": 0},
    {"record_id": "rec_002", "group_id": "grp_01", "x1": 1, "x2": 1, "label": 0},
    {"record_id": "rec_003", "group_id": "grp_02", "x1": 2, "x2": 1, "label": 0},
    {"record_id": "rec_004", "group_id": "grp_02", "x1": 3, "x2": 1, "label": 0},
    {"record_id": "rec_005", "group_id": "grp_03", "x1": 4, "x2": 1, "label": 1},
    {"record_id": "rec_006", "group_id": "grp_03", "x1": 5, "x2": 1, "label": 1},
    {"record_id": "rec_007", "group_id": "grp_04", "x1": 6, "x2": 1, "label": 1},
    {"record_id": "rec_008", "group_id": "grp_04", "x1": 7, "x2": 1, "label": 1},
]

# Fixed Stratified Split Specification:
# Total: 8 records (4 with label=0, 4 with label=1)
# - Train: 4 records (50%) -> 2 negative (label=0) and 2 positive (label=1)
# - Validation: 2 records (25%) -> 1 negative (label=0) and 1 positive (label=1)
# - Test: 2 records (25%) -> 1 negative (label=0) and 1 positive (label=1)
# All splits maintain exact 50% class balance with zero overlap (stratified).
STRATIFIED_SPLIT: Dict[str, Dict[str, Any]] = {
    "train": {
        "indices": [0, 1, 4, 5],
        "record_ids": ["rec_001", "rec_002", "rec_005", "rec_006"],
        "class_distribution": {0: 2, 1: 2},
    },
    "validation": {
        "indices": [2, 6],
        "record_ids": ["rec_003", "rec_007"],
        "class_distribution": {0: 1, 1: 1},
    },
    "test": {
        "indices": [3, 7],
        "record_ids": ["rec_004", "rec_008"],
        "class_distribution": {0: 1, 1: 1},
    },
}

# Concrete verification benchmarks specified in Issue #4:
CLASSIFICATION_BENCHMARK = {
    "truth": [1, 1, 0, 0],
    "prediction": [1, 0, 1, 0],
    "expected_precision": 0.5,
    "expected_recall": 0.5,
    "expected_f1": 0.5,
    "expected_accuracy": 0.5,
    "confusion_matrix": {"tp": 1, "fp": 1, "fn": 1, "tn": 1},
}

REGRESSION_BENCHMARK = {
    "truth": [1, 2, 3],
    "prediction": [1, 3, 2],
    "expected_mae": 2.0 / 3.0,
}


def load_fixture() -> List[Dict[str, Any]]:
    """Return a fresh deep copy of the canonical eight synthetic records."""
    return copy.deepcopy(SYNTHETIC_FIXTURE)


def get_stratified_split() -> Dict[str, List[Dict[str, Any]]]:
    """Return the fixture partitioned into train, validation, and test splits."""
    fixture = load_fixture()
    return {
        "train": [fixture[i] for i in STRATIFIED_SPLIT["train"]["indices"]],
        "validation": [fixture[i] for i in STRATIFIED_SPLIT["validation"]["indices"]],
        "test": [fixture[i] for i in STRATIFIED_SPLIT["test"]["indices"]],
    }
