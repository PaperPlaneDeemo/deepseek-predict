"""Report serialization and command-line regression checks."""

import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from deepseek_predict.report import render_html, report_json, write_report

ROOT = Path(__file__).resolve().parent.parent


class ReportTests(unittest.TestCase):
    def test_embedded_json_cannot_close_script_or_replace_assets(self):
        payload = {"name": '</script><script>alert("x")</script>&\u2028__REPORT_SCRIPT__', "metrics": [None, 1.25]}
        rendered = render_html(payload)
        data_match = re.search(r'<script\b[^>]*id=[\"\']report-data[\"\'][^>]*>(.*?)</script>', rendered, re.S)
        self.assertIsNotNone(data_match)
        embedded = data_match.group(1)
        self.assertNotIn("<", embedded)
        self.assertEqual(json.loads(embedded), payload)
        self.assertNotIn('<script>alert("x")</script>', rendered)

    def test_nonfinite_metrics_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                report_json({"metric": value})

    def test_report_outputs_are_consistent_and_self_contained(self):
        payload = {"test": "中文", "errors": [None]}
        with tempfile.TemporaryDirectory() as temporary:
            html_path, json_path = write_report(payload, Path(temporary) / "nested")
            self.assertEqual(json.loads(json_path.read_text()), payload)
            rendered = html_path.read_text()
            self.assertNotRegex(rendered, r'<script[^>]+src=')
            self.assertNotRegex(rendered, r'<link[^>]+href=[\"\']https?://')
            self.assertNotIn("__REPORT_STYLES__", rendered)
            self.assertNotIn("__REPORT_SCRIPT__", rendered)
            self.assertNotIn("__REPORT_DATA__", rendered)
            self.assertEqual(set(html_path.parent.iterdir()), {html_path, json_path})

    def test_failed_serialization_does_not_overwrite_previous_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            html_path, json_path = write_report({"version": 1}, temporary)
            original = html_path.read_bytes(), json_path.read_bytes()
            with self.assertRaises(ValueError):
                write_report({"bad": float("nan")}, temporary)
            self.assertEqual((html_path.read_bytes(), json_path.read_bytes()), original)


class CliTests(unittest.TestCase):
    def test_legacy_command_resolves_default_data_from_other_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            process = subprocess.run([sys.executable, str(ROOT / "deepseek_predictor_modular.py"), "--validate-data"],
                                     cwd=temporary, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("24", process.stdout)

    def test_bad_arguments_fail_without_traceback(self):
        for arguments in (("--as-of", "2026-02-30"), ("--horizon", "0"),
                          ("--min-train-size", "1"), ("--as-of", "2020-01-01"),
                          ("--data", "missing-catalog.json")):
            with self.subTest(arguments=arguments):
                process = subprocess.run([sys.executable, "-m", "deepseek_predict", *arguments],
                                         cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(process.returncode, 2)
                self.assertNotIn("Traceback", process.stderr)
                self.assertIn("error:", process.stderr)


if __name__ == "__main__":
    unittest.main()
