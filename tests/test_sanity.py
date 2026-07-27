"""Tests for check_numeric_sanity's ratio thresholds.

The gate flags a year-over-year move to below ~1/3 (< 0.3x) or above 3x. These
tests pin the boundaries and the ignore-rules. No LLM, no network involved.
"""

import unittest

from agents.verifier import check_numeric_sanity


def _xbrl_entry(concept, pairs):
    """Build a get_xbrl_fact tool_log entry from (end, val) pairs."""
    values = [{"end": end, "val": val} for end, val in pairs]
    return {"tool": "get_xbrl_fact", "args": {}, "result": {"concept": concept, "values": values}}


class CheckNumericSanityTests(unittest.TestCase):
    def test_flags_growth_above_3x(self):
        log = [_xbrl_entry("Revenues", [("2021-12-31", 100), ("2022-12-31", 400)])]
        issues = check_numeric_sanity(log)
        self.assertEqual(len(issues), 1)
        self.assertIn("Revenues", issues[0])

    def test_flags_drop_below_third(self):
        log = [_xbrl_entry("Assets", [("2021-12-31", 100), ("2022-12-31", 20)])]
        self.assertEqual(len(check_numeric_sanity(log)), 1)

    def test_no_flag_for_moderate_change(self):
        log = [_xbrl_entry("Assets", [("2021-12-31", 100), ("2022-12-31", 130)])]
        self.assertEqual(check_numeric_sanity(log), [])

    def test_exactly_3x_is_not_flagged(self):
        # Threshold is strict (> 3x), so a clean tripling is allowed.
        log = [_xbrl_entry("Assets", [("2021-12-31", 100), ("2022-12-31", 300)])]
        self.assertEqual(check_numeric_sanity(log), [])

    def test_just_over_3x_is_flagged(self):
        log = [_xbrl_entry("Assets", [("2021-12-31", 100), ("2022-12-31", 301)])]
        self.assertEqual(len(check_numeric_sanity(log)), 1)

    def test_checks_every_consecutive_pair(self):
        log = [_xbrl_entry("Assets", [
            ("2020-12-31", 100),
            ("2021-12-31", 110),   # fine
            ("2022-12-31", 500),   # >3x jump -> flag
            ("2023-12-31", 10),    # <0.3x drop -> flag
        ])]
        self.assertEqual(len(check_numeric_sanity(log)), 2)

    def test_ignores_non_xbrl_tools(self):
        log = [{"tool": "list_recent_filings", "args": {}, "result": {"anything": 1}}]
        self.assertEqual(check_numeric_sanity(log), [])

    def test_handles_missing_or_none_values(self):
        log = [_xbrl_entry("Assets", [("2021-12-31", None), ("2022-12-31", 5000)])]
        # A None value should be skipped, not crash.
        self.assertEqual(check_numeric_sanity(log), [])

    def test_empty_values(self):
        log = [{"tool": "get_xbrl_fact", "args": {}, "result": {"concept": "Assets", "values": []}}]
        self.assertEqual(check_numeric_sanity(log), [])


if __name__ == "__main__":
    unittest.main()
