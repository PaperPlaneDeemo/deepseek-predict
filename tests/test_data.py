"""Catalog integrity, validation, and event-grain regression tests."""

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from deepseek_predict.data import DEFAULT_DATA_PATH, Release, load_releases, parse_date, releases_to_frame


ORIGINAL_RECORDS = [
    ("DeepSeek Coder", "2023-11-02"),
    ("DeepSeek-LLM", "2023-11-29"),
    ("DeepSeek-MoE", "2024-01-11"),
    ("DeepSeek-Math", "2024-02-06"),
    ("DeepSeek-V2 (May)", "2024-05-04"),
    ("DeepSeek-Coder V2 (Jun)", "2024-06-14"),
    ("DeepSeek-V2 (Jun)", "2024-06-28"),
    ("DeepSeek-Coder V2 (Jul)", "2024-07-24"),
    ("DeepSeek-V2.5 (Sep)", "2024-09-05"),
    ("DeepSeek-R1-Lite", "2024-11-20"),
    ("DeepSeek-V2.5 (Dec)", "2024-12-10"),
    ("DeepSeek-V3", "2024-12-25"),
    ("DeepSeek-R1", "2025-01-20"),
    ("DeepSeek-V3-0324", "2025-03-24"),
    ("DeepSeek-R1-0528", "2025-05-28"),
    ("DeepSeek-V3.1", "2025-08-19"),
    ("DeepSeek-V3.1-Terminus", "2025-09-22"),
    ("DeepSeek-V3.2-Exp", "2025-09-29"),
    ("DeepSeek-Math-V2", "2025-11-27"),
    ("DeepSeek-V3.2", "2025-12-01"),
    ("DeepSeek-V4-Preview", "2026-04-24"),
    ("DeepSeek-V4-Flash-0731", "2026-07-31"),
    ("DeepSeek-V4-Pro-0813", "2026-08-12"),
    ("DeepSeek-V4-Flash-Vision-Exp", "2026-08-21"),
]


def record(record_id="a", name="Model A", released_on="2024-01-01", **updates):
    value = {"id": record_id, "name": name, "date": released_on, "source_url": None, "notes": ""}
    value.update(updates)
    return value


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "releases.json"

    def load_payload(self, payload):
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        return load_releases(self.path)

    def load_records(self, records):
        return self.load_payload({"schema_version": 1, "releases": records})

    def test_migration_preserves_all_24_original_names_and_dates(self):
        releases = load_releases()
        self.assertTrue(DEFAULT_DATA_PATH.is_absolute())
        self.assertEqual([(item.name, item.date.isoformat()) for item in releases], ORIGINAL_RECORDS)
        self.assertEqual(len({item.id for item in releases}), 24)
        self.assertTrue(all(item.source_url is None for item in releases))
        payload = json.loads(DEFAULT_DATA_PATH.read_text(encoding="utf-8"))
        self.assertIn("not been independently verified", payload["notes"])

    def test_records_are_immutable_and_chronologically_sorted(self):
        releases = self.load_records([
            record("b", "Model B", "2024-01-02", source_url="https://example.com/b"),
            record(),
        ])
        self.assertEqual([item.id for item in releases], ["a", "b"])
        self.assertEqual(releases[1].source_url, "https://example.com/b")
        with self.assertRaises(FrozenInstanceError):
            releases[0].name = "changed"

    def test_rejects_invalid_schema_and_catalog_structures(self):
        invalid = [
            [], None, {},
            {"schema_version": 2, "releases": []},
            {"schema_version": True, "releases": []},
            {"schema_version": 1.0, "releases": []},
            {"schema_version": "1", "releases": []},
            {"schema_version": 1, "releases": {}},
            {"schema_version": 1, "releases": [], "notes": []},
            {"schema_version": 1, "releases": [], "release": []},
        ]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.load_payload(payload)

    def test_record_errors_identify_the_bad_array_index(self):
        invalid = [
            None, [], {},
            record(id=""), record(id="Not A Slug"), record(id=3),
            record(name=""), record(name="  "), record(name=True), record(name=" padded "),
            record(date="2024-02-30"), record(date="2024-2-01"), record(date=None),
            record(notes=None), record(notes=[]), record(extra="typo"),
            record(source_url="javascript:alert(1)"), record(source_url="ftp://example.com/x"),
            record(source_url="https://"), record(source_url="https://[bad"),
            record(source_url="https://example.com:bad"), record(source_url="https://exa mple.com"),
            record(source_url=""), record(source_url=123),
        ]
        for item in invalid:
            with self.subTest(item=item), self.assertRaisesRegex(ValueError, r"releases\[1\]"):
                self.load_records([record("valid", "Valid"), item])

    def test_duplicate_ids_and_name_date_pairs_are_rejected(self):
        for second, message in [
            (record("a", "Other", "2024-01-02"), "duplicate id"),
            (record("b", "Model A", "2024-01-01"), "duplicate name/date pair"),
        ]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.load_records([record(), second])

    def test_same_name_on_another_date_is_a_distinct_record(self):
        releases = self.load_records([record(), record("b", "Model A", "2024-01-02")])
        self.assertEqual(len(releases), 2)

    def test_missing_or_malformed_files_have_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "Cannot read release catalog"):
            load_releases(self.path)
        self.path.write_text("{broken", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Cannot read release catalog"):
            load_releases(self.path)


class DateTests(unittest.TestCase):
    def test_exact_calendar_dates(self):
        self.assertEqual(parse_date("2024-02-29"), date(2024, 2, 29))
        self.assertEqual(parse_date("0001-01-01"), date.min)
        for value in [
            "2023-02-29", "2024-13-01", "2024-00-01", "0000-01-01", "2024-01-00",
            "20240101", "2024-1-01", "2024-01-1", "2024-01-01T00:00:00",
            "2024-01-01\n", " 2024-01-01", "２０２４-０１-０１", 20240101, None, True,
        ]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_date(value)


class EventFrameTests(unittest.TestCase):
    def setUp(self):
        self.releases = [
            Release("b", "Model B", date(2024, 1, 11), None, ""),
            Release("a", "Model A", date(2024, 1, 1), None, ""),
            Release("c", "Model C", date(2024, 1, 11), None, ""),
            Release("future", "Future Model", date(2024, 2, 1), None, ""),
        ]

    def test_same_day_models_form_one_event_without_mutating_catalog(self):
        frame = releases_to_frame(self.releases, as_of=date(2024, 2, 1))
        self.assertEqual(list(frame.columns), ["version", "date", "days_since_start", "month", "interval_days"])
        self.assertEqual(frame["version"].tolist(), ["Model A", "Model B / Model C", "Future Model"])
        self.assertEqual(frame["days_since_start"].tolist(), [0, 10, 31])
        self.assertEqual(frame["month"].tolist(), [1, 1, 2])
        self.assertTrue(pd.isna(frame["interval_days"].iloc[0]))
        self.assertEqual(frame["interval_days"].dropna().tolist(), [10, 21])
        self.assertEqual(len(self.releases), 4)
        self.assertEqual(self.releases[0].id, "b")

    def test_cutoff_is_inclusive_and_future_dates_do_not_enter_training(self):
        frame = releases_to_frame(self.releases, as_of=date(2024, 1, 11))
        self.assertEqual(frame["version"].tolist(), ["Model A", "Model B / Model C"])
        self.assertEqual(frame["date"].iloc[-1].date(), date(2024, 1, 11))
        self.assertNotIn("Future Model", frame["version"].tolist())

    def test_two_distinct_days_are_required_after_cutoff(self):
        for releases, as_of in [
            ([], date(2024, 1, 1)),
            (self.releases, date(2023, 12, 31)),
            (self.releases, date(2024, 1, 1)),
            ([self.releases[0], self.releases[2]], date(2024, 1, 11)),
        ]:
            with self.subTest(releases=releases, as_of=as_of), self.assertRaisesRegex(
                ValueError, "two distinct release dates"
            ):
                releases_to_frame(releases, as_of)

    def test_as_of_rejects_ambiguous_datetime_or_string(self):
        for value in ["2024-01-11", datetime(2024, 1, 11), None]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "as_of"):
                releases_to_frame(self.releases, value)

    def test_event_frame_supports_valid_dates_outside_nanosecond_range(self):
        releases = [
            Release("a", "Old", date(1500, 1, 1), None, ""),
            Release("b", "New", date(2500, 1, 1), None, ""),
        ]
        frame = releases_to_frame(releases, date(2500, 1, 1))
        self.assertEqual(frame["interval_days"].iloc[1], (date(2500, 1, 1) - date(1500, 1, 1)).days)


if __name__ == "__main__":
    unittest.main()
