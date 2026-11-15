"""Synthetic fixture and stratified split specification for R41-T001.

DISCLAIMER:
These eight synthetic records are strictly generated for pipeline schema verification
and unit testing. They do NOT represent real IPL match observations, ball-by-ball
deliveries, or win probabilities, and must not be used as evidence of real-world
model performance.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

# Explicit statement of synthetic purpose
DATASET_DISCLAIMER: str = (
    "Synthetic verification data for R41-T001 data contract validation. "
    "These rows do NOT represent real IPL match observations or win probabilities."
)

# Canonical eight synthetic records conforming to Issue #1 specification:
# - record_id: stable, unique, non-empty string IDs
# - group_id: stable, non-empty string IDs
# - x1: [0, 1, 2, 3, 4, 5, 6, 7]
# - x2: 1 for every record
# - label: [0, 0, 0, 0, 1, 1, 1, 1]
SYNTHETIC_FIXTURE: List[Dict[str, Any]] = [
    {"record_id": "rec_001", "group_id": "grp_01", "x1": 0, "x2": 1, "label": 0},
    {"record_id": "rec_002", "group_id": "grp_01", "x1": 1, "x2": 1, "label": 0},
    {"record_id": "rec_003", "group_id": "grp_01", "x1": 2, "x2": 1, "label": 0},
    {"record_id": "rec_004", "group_id": "grp_01", "x1": 3, "x2": 1, "label": 0},
    {"record_id": "rec_005", "group_id": "grp_02", "x1": 4, "x2": 1, "label": 1},
    {"record_id": "rec_006", "group_id": "grp_02", "x1": 5, "x2": 1, "label": 1},
    {"record_id": "rec_007", "group_id": "grp_02", "x1": 6, "x2": 1, "label": 1},
    {"record_id": "rec_008", "group_id": "grp_02", "x1": 7, "x2": 1, "label": 1},
]

# Fixed Stratified Split Design:
# Total: 8 records (4 with label=0, 4 with label=1)
# Train split: 6 records (75%) -> 3 negative (label=0) and 3 positive (label=1)
# Test split: 2 records (25%) -> 1 negative (label=0) and 1 positive (label=1)
# Both splits preserve exact 50% class balance (stratification).
STRATIFIED_SPLIT: Dict[str, Dict[str, Any]] = {
    "train": {
        "indices": [0, 1, 2, 4, 5, 6],
        "record_ids": ["rec_001", "rec_002", "rec_003", "rec_005", "rec_006", "rec_007"],
        "class_distribution": {0: 3, 1: 3},
    },
    "test": {
        "indices": [3, 7],
        "record_ids": ["rec_004", "rec_008"],
        "class_distribution": {0: 1, 1: 1},
    },
}


def get_synthetic_fixture() -> List[Dict[str, Any]]:
    """Return a deep copy of the canonical eight synthetic records."""
    return copy.deepcopy(SYNTHETIC_FIXTURE)


def get_stratified_split() -> Dict[str, List[Dict[str, Any]]]:
    """Return the fixture partitioned into train and test splits."""
    fixture = get_synthetic_fixture()
    return {
        "train": [fixture[i] for i in STRATIFIED_SPLIT["train"]["indices"]],
        "test": [fixture[i] for i in STRATIFIED_SPLIT["test"]["indices"]],
    }

