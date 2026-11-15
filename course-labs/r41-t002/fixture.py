"""Synthetic classification fixture for R41-T002.

These records are simulated examples, not observations from a real dataset.
"""


def load_fixture():
    """Return eight deterministic synthetic classification records."""
    return [
        {"record_id": "rec_001", "group_id": "grp_01", "x1": 0, "x2": 1, "label": 0},
        {"record_id": "rec_002", "group_id": "grp_02", "x1": 1, "x2": 1, "label": 0},
        {"record_id": "rec_003", "group_id": "grp_03", "x1": 2, "x2": 1, "label": 0},
        {"record_id": "rec_004", "group_id": "grp_04", "x1": 3, "x2": 1, "label": 0},
        {"record_id": "rec_005", "group_id": "grp_01", "x1": 4, "x2": 1, "label": 1},
        {"record_id": "rec_006", "group_id": "grp_02", "x1": 5, "x2": 1, "label": 1},
        {"record_id": "rec_007", "group_id": "grp_03", "x1": 6, "x2": 1, "label": 1},
        {"record_id": "rec_008", "group_id": "grp_04", "x1": 7, "x2": 1, "label": 1},
    ]
