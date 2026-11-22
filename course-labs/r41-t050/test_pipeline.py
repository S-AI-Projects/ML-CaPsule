"""Offline tests for R41-T050 saved time-series pipeline interface."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pipeline import (
    TimeSeriesValidationError,
    fit_configuration,
    load_config,
    load_fixture,
    predict,
    save_config,
    validate_splits,
)

HERE = Path(__file__).parent
FIXTURE = json.loads((HERE / "fixture.json").read_text(encoding="utf-8"))
SCHEMA = json.loads((HERE / "schema.json").read_text(encoding="utf-8"))


class SavedTimeSeriesInterfaceTests(unittest.TestCase):
    def test_literal_fixture_and_no_shuffle_splits(self):
        self.assertEqual(FIXTURE["data_kind"], "simulated")
        self.assertEqual(FIXTURE["fixture_kind"], "deterministic_timeseries")
        self.assertEqual(FIXTURE["seed"], 50)
        self.assertEqual([row["record_id"] for row in FIXTURE["records"]], [f"d{index}" for index in range(1, 11)])
        self.assertEqual([row["value"] for row in FIXTURE["records"]], [10, 12, 11, 13, 15, 14, 16, 18, 17, 19])
        self.assertEqual(SCHEMA["required"], ["record_id", "timestamp", "value"])
        self.assertEqual(FIXTURE["split_ids"], {
            "train": ["d1", "d2", "d3", "d4", "d5", "d6"],
            "validation": ["d7", "d8"],
            "test": ["d9", "d10"],
        })

    def test_save_reload_predictions_agree_without_training_data(self):
        records, splits = load_fixture()
        before = fit_configuration(records, splits)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "parameters.json"
            save_config(before, path)
            after = load_config(path)
            for record_id in ("d7", "d8", "d9", "d10"):
                record = next(row for row in records if row["record_id"] == record_id)
                self.assertLessEqual(abs(predict(record, before) - predict(record, after)), 1e-12)
                inference_only = {"record_id": record_id, "timestamp": record["timestamp"]}
                inference_only["value"] = record["value"]
                self.assertEqual(predict(inference_only, after), 12.5)

    def test_schema_version_mismatch_is_rejected(self):
        records, splits = load_fixture()
        bad = fit_configuration(records, splits)
        bad["schema_version"] = 2
        with TemporaryDirectory() as directory:
            path = Path(directory) / "parameters.json"
            with self.assertRaisesRegex(TimeSeriesValidationError, "schema_version"):
                save_config(bad, path)

    def test_missing_prediction_field_is_rejected(self):
        records, splits = load_fixture()
        config = fit_configuration(records, splits)
        bad = copy.deepcopy(records[6])
        bad.pop("timestamp")
        with self.assertRaisesRegex(TimeSeriesValidationError, "exactly"):
            predict(bad, config)

    def test_unshuffled_split_contract_is_rejected(self):
        records, splits = load_fixture()
        bad = copy.deepcopy(splits)
        bad["validation"] = ["d8", "d7"]
        with self.assertRaisesRegex(TimeSeriesValidationError, "time order"):
            validate_splits(records, bad)

    def test_unknown_split_id_is_rejected(self):
        records, splits = load_fixture()
        bad = copy.deepcopy(splits)
        bad["test"] = ["d9", "d-unseen"]
        with self.assertRaisesRegex(TimeSeriesValidationError, "unknown record_id"):
            validate_splits(records, bad)


if __name__ == "__main__":
    unittest.main()
