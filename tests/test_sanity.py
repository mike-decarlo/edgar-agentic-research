"""Tests for check_numeric_sanity's accounting-identity checks.

The gate is NetIncomeLoss-centric. It (1) flags fiscal-period spacing that isn't
~1 year apart (duplicate/misaligned entries), (2) cross-checks the
NetIncomeLoss/Revenues margin against plausible bounds, and (3) flags a sharp
NetIncomeLoss rise that coincides with a Revenues *decline*. If NetIncomeLoss is
reported with no Revenues to cross-check, it flags that too. No LLM, no network.
"""

import unittest

from agents.verifier import check_numeric_sanity


def _xbrl_entry(concept, pairs):
    """Build a get_xbrl_fact tool_log entry from (end, val) pairs."""
    values = [{"end": end, "val": val} for end, val in pairs]
    return {"tool": "get_xbrl_fact", "args": {}, "result": {"concept": concept, "values": values}}


class CheckNumericSanityTests(unittest.TestCase):
    def test_no_net_income_returns_empty(self):
        # The gate keys off NetIncomeLoss; other concepts alone are ignored.
        log = [_xbrl_entry("Revenues", [("2021-12-31", 100), ("2022-12-31", 400)])]
        self.assertEqual(check_numeric_sanity(log), [])

    def test_net_income_without_revenue_flags_missing_crosscheck(self):
        # Good ~1-year spacing, but no Revenues to check margin against.
        log = [_xbrl_entry("NetIncomeLoss", [("2021-12-31", 100), ("2022-12-31", 120)])]
        issues = check_numeric_sanity(log)
        self.assertEqual(len(issues), 1)
        self.assertIn("without Revenues", issues[0])

    def test_clean_data_with_revenue_passes(self):
        # ~1-year spacing, plausible margins, profit and revenue both growing.
        log = [
            _xbrl_entry("NetIncomeLoss", [("2021-12-31", 100), ("2022-12-31", 300)]),
            _xbrl_entry("Revenues", [("2021-12-31", 1000), ("2022-12-31", 2000)]),
        ]
        self.assertEqual(check_numeric_sanity(log), [])

    # --- Check 1: fiscal-period spacing ------------------------------------

    def test_flags_periods_not_about_a_year_apart(self):
        # Two ends only ~3 months apart -> duplicate/misaligned entry. Revenues
        # supplied (with fine margins) so the only issue is the spacing one.
        log = [
            _xbrl_entry("NetIncomeLoss", [("2022-09-30", 100), ("2022-12-31", 110)]),
            _xbrl_entry("Revenues", [("2022-09-30", 1000), ("2022-12-31", 1000)]),
        ]
        issues = check_numeric_sanity(log)
        self.assertEqual(len(issues), 1)
        self.assertIn("not ~1 year", issues[0])

    def test_spacing_within_tolerance_not_flagged(self):
        # 364 days apart is within the 330-400 day window.
        log = [
            _xbrl_entry("NetIncomeLoss", [("2021-01-01", 100), ("2021-12-31", 110)]),
            _xbrl_entry("Revenues", [("2021-01-01", 1000), ("2021-12-31", 1000)]),
        ]
        self.assertEqual(check_numeric_sanity(log), [])

    # --- Check 2: margin plausibility --------------------------------------

    def test_flags_margin_above_one(self):
        # Net income exceeding revenue is impossible -> flag. Single year, so no
        # spacing or swing checks apply.
        log = [
            _xbrl_entry("NetIncomeLoss", [("2022-12-31", 1500)]),
            _xbrl_entry("Revenues", [("2022-12-31", 1000)]),
        ]
        issues = check_numeric_sanity(log)
        self.assertEqual(len(issues), 1)
        self.assertIn("outside plausible bounds", issues[0])

    def test_flags_margin_below_negative_three(self):
        # A loss more than 3x revenue is outside the generous loss band.
        log = [
            _xbrl_entry("NetIncomeLoss", [("2022-12-31", -4000)]),
            _xbrl_entry("Revenues", [("2022-12-31", 1000)]),
        ]
        issues = check_numeric_sanity(log)
        self.assertEqual(len(issues), 1)
        self.assertIn("outside plausible bounds", issues[0])

    def test_margin_exactly_one_not_flagged(self):
        # Bound is strict (> 1.0), so margin == 1.0 is allowed.
        log = [
            _xbrl_entry("NetIncomeLoss", [("2022-12-31", 1000)]),
            _xbrl_entry("Revenues", [("2022-12-31", 1000)]),
        ]
        self.assertEqual(check_numeric_sanity(log), [])

    def test_zero_or_missing_revenue_skips_margin_safely(self):
        # rev falsy/<=0 -> margin check skipped, no ZeroDivisionError.
        log = [
            _xbrl_entry("NetIncomeLoss", [("2021-12-31", 100), ("2022-12-31", 120)]),
            _xbrl_entry("Revenues", [("2021-12-31", 0), ("2022-12-31", 0)]),
        ]
        self.assertEqual(check_numeric_sanity(log), [])

    # --- Check 3: profit-up-while-revenue-down swing -----------------------

    def test_flags_net_income_surge_with_revenue_decline(self):
        # Profit >1.5x while revenue drops >5% -> inconsistent, worth reviewing.
        log = [
            _xbrl_entry("NetIncomeLoss", [("2021-12-31", 100), ("2022-12-31", 300)]),
            _xbrl_entry("Revenues", [("2021-12-31", 1000), ("2022-12-31", 800)]),
        ]
        issues = check_numeric_sanity(log)
        self.assertEqual(len(issues), 1)
        self.assertIn("grew sharply", issues[0])

    def test_net_income_surge_with_revenue_growth_not_flagged(self):
        # Same profit surge, but revenue grew -> normal margin expansion, no flag.
        log = [
            _xbrl_entry("NetIncomeLoss", [("2021-12-31", 100), ("2022-12-31", 300)]),
            _xbrl_entry("Revenues", [("2021-12-31", 1000), ("2022-12-31", 1100)]),
        ]
        self.assertEqual(check_numeric_sanity(log), [])

    # --- Misc --------------------------------------------------------------

    def test_ignores_non_xbrl_tools(self):
        log = [{"tool": "list_recent_filings", "args": {}, "result": {"anything": 1}}]
        self.assertEqual(check_numeric_sanity(log), [])

    def test_empty_net_income_values(self):
        log = [{"tool": "get_xbrl_fact", "args": {}, "result": {"concept": "NetIncomeLoss", "values": []}}]
        self.assertEqual(check_numeric_sanity(log), [])


if __name__ == "__main__":
    unittest.main()
