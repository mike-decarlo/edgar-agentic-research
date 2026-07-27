"""Tests for the EDGAR tools: CIK resolution, dedup-by-fiscal-year, and the
real duplicate-fiscal-year regression that produced a $96.99B/$9.37B mismatch.

All SEC network calls are mocked — these never hit the wire.
"""

import os
import unittest
from unittest import mock

# Set before importing config/edgar so get_headers() never trips (calls are
# mocked regardless, but get_cik still builds real headers).
os.environ.setdefault("SEC_USER_AGENT", "Test Runner test@example.com")

from tools.edgar import (  # noqa: E402  (import after env setup, by design)
    TickerNotFoundError,
    _select_annual_facts,
    get_cik,
    get_xbrl_fact,
)


def _fact(end, val, filed, form="10-K", fp="FY"):
    """Build a single companyconcept unit entry."""
    return {"end": end, "val": val, "filed": filed, "form": form, "fp": fp}


def _mock_response(status_code=200, json_data=None):
    resp = mock.Mock()
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else {}
    resp.raise_for_status.return_value = None
    return resp


class SelectAnnualFactsTests(unittest.TestCase):
    """Dedup-by-fiscal-year logic in _select_annual_facts."""

    def test_keeps_only_annual_fy_10k_entries(self):
        values = [
            _fact("2022-12-31", 100, "2023-02-01"),
            _fact("2022-09-30", 25, "2022-10-15", fp="Q3"),      # quarterly -> drop
            _fact("2022-12-31", 100, "2023-02-01", form="10-Q"),  # wrong form -> drop
        ]
        out = _select_annual_facts(values)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["fp"], "FY")
        self.assertEqual(out[0]["form"], "10-K")

    def test_dedupes_same_year_keeping_most_recently_filed(self):
        values = [
            _fact("2022-12-31", 500, "2023-02-01"),  # original
            _fact("2022-12-31", 550, "2024-02-01"),  # restated later -> should win
        ]
        out = _select_annual_facts(values)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["val"], 550)
        self.assertEqual(out[0]["filed"], "2024-02-01")

    def test_returns_last_n_years_sorted_by_fiscal_year_end(self):
        values = [
            _fact("2019-12-31", 1, "2020-02-01"),
            _fact("2020-12-31", 2, "2021-02-01"),
            _fact("2021-12-31", 3, "2022-02-01"),
            _fact("2022-12-31", 4, "2023-02-01"),
            _fact("2023-12-31", 5, "2024-02-01"),
        ]
        out = _select_annual_facts(values, limit=4)
        self.assertEqual([v["end"] for v in out],
                         ["2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31"])
        # sorted ascending by end
        self.assertEqual([v["val"] for v in out], [2, 3, 4, 5])

    def test_empty_input(self):
        self.assertEqual(_select_annual_facts([]), [])


class DuplicateFiscalYearRegressionTests(unittest.TestCase):
    """Regression: SEC companyconcept returns comparative/duplicate entries for
    the same fiscal year across later filings. Without dedup, the same fiscal
    year surfaced two wildly different values ($96.99B and $9.37B). Dedup must
    collapse them to a single value — the most recently filed one.
    """

    def test_same_year_no_longer_reports_two_values(self):
        values = [
            # Original 10-K reported 9.37B for FY2021.
            _fact("2021-12-31", 9_370_000_000, "2022-02-01"),
            # A later 10-K's comparative column re-reported the same fiscal year
            # end with a different (96.99B) figure.
            _fact("2021-12-31", 96_990_000_000, "2023-02-01"),
        ]
        out = _select_annual_facts(values)

        ends = [v["end"] for v in out]
        self.assertEqual(ends.count("2021-12-31"), 1, "FY2021 must appear once")

        distinct_vals = {v["val"] for v in out if v["end"] == "2021-12-31"}
        self.assertEqual(len(distinct_vals), 1)
        # Most recently filed entry wins.
        self.assertEqual(out[0]["val"], 96_990_000_000)


class GetCikTests(unittest.TestCase):
    TICKER_MAP = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
    }

    @mock.patch("tools.edgar.requests.get")
    def test_resolves_and_zero_pads(self, mock_get):
        mock_get.return_value = _mock_response(json_data=self.TICKER_MAP)
        self.assertEqual(get_cik("aapl"), "0000320193")

    @mock.patch("tools.edgar.requests.get")
    def test_unknown_ticker_raises(self, mock_get):
        mock_get.return_value = _mock_response(json_data=self.TICKER_MAP)
        with self.assertRaises(TickerNotFoundError):
            get_cik("NOTREAL")


class GetXbrlFactTests(unittest.TestCase):
    @mock.patch("tools.edgar.get_cik", return_value="0000320193")
    @mock.patch("tools.edgar.requests.get")
    def test_returns_deduped_annual_values(self, mock_get, _mock_cik):
        payload = {
            "units": {
                "USD": [
                    _fact("2021-12-31", 9_370_000_000, "2022-02-01"),
                    _fact("2021-12-31", 96_990_000_000, "2023-02-01"),
                    _fact("2022-12-31", 100_000_000_000, "2023-02-01"),
                ]
            }
        }
        mock_get.return_value = _mock_response(json_data=payload)

        result = get_xbrl_fact("AAPL", "NetIncomeLoss")
        self.assertEqual(result["concept"], "NetIncomeLoss")
        self.assertEqual(result["unit"], "USD")
        self.assertEqual(len(result["values"]), 2)  # two distinct fiscal years
        self.assertEqual([v["end"] for v in result["values"]],
                         ["2021-12-31", "2022-12-31"])
        # Values are trimmed to model-friendly fields with a derived fiscal_year,
        # and the confusing raw fields (fy/frame/form/fp) are dropped.
        first = result["values"][0]
        self.assertEqual(set(first), {"fiscal_year", "end", "val", "filed"})
        self.assertEqual(first["fiscal_year"], 2021)

    @mock.patch("tools.edgar.get_cik", return_value="0000320193")
    @mock.patch("tools.edgar.requests.get")
    def test_missing_concept_returns_soft_error(self, mock_get, _mock_cik):
        mock_get.return_value = _mock_response(status_code=404)
        result = get_xbrl_fact("AAPL", "MadeUpConcept")
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
