import pathlib
import sys
import tempfile
import unittest


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline


class PipelineConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = pipeline.load_fixture(HERE / "fixture.json")

    def test_literal_fixture_and_chronological_split_ids(self):
        self.assertEqual(
            [record["record_id"] for record in self.records],
            ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10"],
        )
        self.assertEqual(
            [record["value"] for record in self.records],
            [10, 12, 11, 13, 15, 14, 16, 18, 17, 19],
        )
        self.assertEqual(
            pipeline.expected_splits(self.records),
            {
                "train": ["d1", "d2", "d3", "d4", "d5", "d6"],
                "validation": ["d7", "d8"],
                "test": ["d9", "d10"],
            },
        )

    def test_saved_and_reloaded_config_recreates_splits_and_predictions(self):
        config_path = HERE / "pipeline_config.json"
        loaded = pipeline.load_config(config_path, self.records)
        generated = pipeline.create_config(self.records, seed=42)
        self.assertEqual(loaded, generated)

        before = pipeline.run_pipeline(self.records, generated)
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = pathlib.Path(temporary) / "config.json"
            pipeline.save_config(generated, temporary_path, self.records)
            reloaded = pipeline.load_config(temporary_path, self.records)
        after = pipeline.run_pipeline(self.records, reloaded)
        self.assertEqual(before, after)
        self.assertEqual(
            after["predictions"],
            {"d7": 14, "d8": 14, "d9": 14, "d10": 14},
        )

    def test_unknown_top_level_and_algorithm_settings_are_rejected(self):
        config = pipeline.create_config(self.records)
        unknown_top_level = dict(config, extra=True)
        with self.assertRaisesRegex(ValueError, "unknown or missing fields"):
            pipeline.validate_config(unknown_top_level, self.records)

        unknown_algorithm = dict(config)
        unknown_algorithm["algorithm"] = dict(config["algorithm"], metric="mae")
        with self.assertRaisesRegex(ValueError, "algorithm contains unknown"):
            pipeline.validate_config(unknown_algorithm, self.records)

    def test_contradictory_feature_and_split_settings_are_rejected(self):
        config = pipeline.create_config(self.records)
        incompatible_feature = dict(config, feature_order=["timestamp"])
        with self.assertRaisesRegex(ValueError, "feature_order"):
            pipeline.validate_config(incompatible_feature, self.records)

        contradictory_split = dict(config)
        contradictory_split["split_ids"] = dict(config["split_ids"])
        contradictory_split["split_ids"]["train"] = ["d1", "d2", "d3", "d4", "d5"]
        with self.assertRaisesRegex(ValueError, "chronological 6/2/2"):
            pipeline.validate_config(contradictory_split, self.records)

    def test_unsupported_algorithm_settings_are_rejected(self):
        config = pipeline.create_config(self.records)
        incompatible = dict(config)
        incompatible["algorithm"] = {"name": "last_value", "window_size": 2}
        with self.assertRaisesRegex(ValueError, "window_size 1"):
            pipeline.validate_config(incompatible, self.records)

    def test_incompatible_record_shape_is_rejected(self):
        malformed = [dict(record) for record in self.records]
        del malformed[0]["value"]
        with self.assertRaisesRegex(ValueError, "exactly record_id, timestamp, value"):
            pipeline.run_pipeline(malformed, pipeline.create_config(self.records))

    def test_wrong_schema_version_is_rejected(self):
        config = pipeline.create_config(self.records)
        config["schema_version"] = "r41-t057.config.v999"
        with self.assertRaisesRegex(ValueError, "unsupported configuration schema"):
            pipeline.validate_config(config, self.records)


if __name__ == "__main__":
    unittest.main()
