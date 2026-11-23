"""Synthetic fixture for Task R41-T010: Stroke prediction – class imbalance with SMOTE.

DISCLAIMER:
This dataset is an explicitly synthetic verification fixture created solely
for offline pipeline verification, saved-model interface testing, and serialization
reproducibility. It does NOT contain real-world clinical stroke observations, patient health
records, or physiological measurements.
"""

from typing import Any, Dict, List

DISCLAIMER: str = (
    "Synthetic pipeline-verification fixture: eight rows with x1=[0..7], x2=1, "
    "label=[0,0,0,0,1,1,1,1]. Not observations from the real Stroke Prediction dataset."
)

# Eight synthetic classification records:
# x1 = [0, 1, 2, 3, 4, 5, 6, 7], x2 = 1, label = [0, 0, 0, 0, 1, 1, 1, 1]
SYNTHETIC_RECORDS: List[Dict[str, Any]] = [
    {"record_id": "rec_001", "group_id": "grp_01", "x1": 0, "x2": 1, "label": 0},
    {"record_id": "rec_002", "group_id": "grp_02", "x1": 1, "x2": 1, "label": 0},
    {"record_id": "rec_003", "group_id": "grp_03", "x1": 2, "x2": 1, "label": 0},
    {"record_id": "rec_004", "group_id": "grp_04", "x1": 3, "x2": 1, "label": 0},
    {"record_id": "rec_005", "group_id": "grp_01", "x1": 4, "x2": 1, "label": 1},
    {"record_id": "rec_006", "group_id": "grp_02", "x1": 5, "x2": 1, "label": 1},
    {"record_id": "rec_007", "group_id": "grp_03", "x1": 6, "x2": 1, "label": 1},
    {"record_id": "rec_008", "group_id": "grp_04", "x1": 7, "x2": 1, "label": 1},
]

# Fixed stratified train/validation/test split manifest:
# Train: 4 records (two 0s, two 1s)
# Validation: 2 records (one 0, one 1)
# Test: 2 records (one 0, one 1)
SPLIT_MANIFEST: Dict[str, List[str]] = {
    "train": ["rec_001", "rec_002", "rec_005", "rec_006"],
    "validation": ["rec_003", "rec_007"],
    "test": ["rec_004", "rec_008"],
}

FEATURE_ORDER: List[str] = ["x1", "x2"]
ID_FIELD: str = "record_id"
GROUP_FIELD: str = "group_id"
LABEL_FIELD: str = "label"
FORMAT_VERSION: str = "1.0.0"


def get_split_records() -> Dict[str, List[Dict[str, Any]]]:
    """Return records partitioned according to SPLIT_MANIFEST."""
    records_by_id = {r[ID_FIELD]: dict(r) for r in SYNTHETIC_RECORDS}
    return {
        split_name: [dict(records_by_id[rid]) for rid in ids]
        for split_name, ids in SPLIT_MANIFEST.items()
    }
