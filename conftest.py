"""Pytest bootstrap: put the repo root on sys.path and set a dummy User-Agent.

Keeps ``from tools.edgar import ...`` working under pytest and ensures the SEC
User-Agent check never trips during tests (all network calls are mocked anyway).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("SEC_USER_AGENT", "Test Runner test@example.com")
