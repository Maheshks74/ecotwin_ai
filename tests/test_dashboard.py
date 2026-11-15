
"""Tests for Phase 9: Streamlit Dashboard AppTest."""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


def test_dashboard_startup_and_render():
    """Verify Streamlit app initializes, compiles, and renders without exceptions."""

    # Find project root (parent of tests directory)
    project_root = Path(__file__).resolve().parent.parent

    # Build the correct absolute path
    app_path = project_root / "dashboard" / "app.py"

    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()

    assert not at.exception, f"Streamlit app raised an exception: {at.exception}"
    assert len(at.tabs) == 4
