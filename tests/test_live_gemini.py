
"""Live Google Gemini API tests.

Tests:
1. Gemini client initialization and connectivity.
2. Advisor explanation.
3. Advisor response when information is missing.
4. What-if analysis.
5. Instruction parser schema validation and weight clamping.
"""

import numpy as np
import pandas as pd
import pytest

from pathlib import Path
from dotenv import load_dotenv

# --------------------------------------------------
# Load environment variables from project-root .env
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

# --------------------------------------------------
# Project imports
# --------------------------------------------------

from ecotwin.llm.client import GeminiClient
from ecotwin.llm.advisor import LLMAdvisor
from ecotwin.llm.instruction_parser import InstructionParser


# --------------------------------------------------
# 1. Gemini Client Connectivity
# --------------------------------------------------

@pytest.mark.live
def test_live_gemini_client():
    """Verify Gemini client initializes in live mode with a valid API key."""

    client = GeminiClient()

    print(f"\n[Live Test] Client badge: {client.badge}")

    assert not client.is_mock_mode, (
        "Gemini client should NOT be in mock mode. "
        "Check GEMINI_API_KEY and GEMINI_FORCE_LIVE."
    )

    assert "Live Gemini" in client.badge

    # Test basic API connectivity
    resp = client.generate_text(
        "Reply with the single word 'CONNECTED'."
    )

    print(f"[Live Test] Raw response: {resp.strip()}")

    assert len(resp.strip()) > 0


# --------------------------------------------------
# 2. Advisor Explanation
# --------------------------------------------------

@pytest.mark.live
def test_live_advisor_explanation():
    """Verify live Gemini advisor produces a telemetry-grounded explanation."""

    client = GeminiClient()
    advisor = LLMAdvisor(client=client)

    telemetry = pd.DataFrame([
        {
            "step": 8,
            "hour": 2.0,
            "price_inr_per_kwh": 4.5,
            "outside_temp_c": 22.0,
            "total_power_kw": 40.0,
            "machines_power_kw": 30.0,
            "hvac_power_kw": 10.0,
            "zone_1_temp_c": 24.0,
            "zone_2_temp_c": 24.0,
            "min_slack": 20.0,
            "completed_jobs": 5
        }
    ])

    explanation = advisor.explain(
        "Why was production scheduled at 2 AM?",
        telemetry
    )

    print(f"\n[Live Test] Advisor Explanation:\n{explanation}")

    assert len(explanation) > 15

    # Verify that the response refers to available telemetry.
    assert any(
        term in explanation.lower()
        for term in [
            "price",
            "tariff",
            "cost",
            "slack",
            "rate",
            "4.5"
        ]
    )


# --------------------------------------------------
# 3. Missing Information / Hallucination Check
# --------------------------------------------------

@pytest.mark.live
def test_live_advisor_not_enough_information():
    """Verify advisor acknowledges missing information instead of inventing facts."""

    client = GeminiClient()
    advisor = LLMAdvisor(client=client)

    telemetry = pd.DataFrame([
        {
            "step": 8,
            "hour": 2.0,
            "price_inr_per_kwh": 4.5,
            "outside_temp_c": 22.0,
            "total_power_kw": 40.0,
            "machines_power_kw": 30.0,
            "hvac_power_kw": 10.0,
            "zone_1_temp_c": 24.0,
            "zone_2_temp_c": 24.0,
            "min_slack": 20.0,
            "completed_jobs": 5
        }
    ])

    # Machine M9 and its failure are not present in the telemetry.
    missing_resp = advisor.explain(
        "Why did Machine M9 fail at 11 PM?",
        telemetry
    )

    print(
        f"\n[Live Test] Missing Fact Response:\n'{missing_resp}'"
    )

    assert "not enough information" in missing_resp.lower()


# --------------------------------------------------
# 4. What-if Analysis
# --------------------------------------------------

@pytest.mark.live
def test_live_what_if_analysis():
    """Verify what-if analysis executes comparative digital-twin simulations."""

    client = GeminiClient()
    advisor = LLMAdvisor(client=client)

    res = advisor.run_what_if_analysis(
        "What if temperature setpoint is relaxed by 1 degree to save electricity?"
    )

    print(f"\n[Live Test] What-If Summary:\n{res['summary']}")

    assert "baseline" in res
    assert "modified" in res

    assert np.isfinite(res["cost_delta_inr"])

    assert len(res["summary"]) > 20


# --------------------------------------------------
# 5. Instruction Parser / Schema Validation
# --------------------------------------------------

@pytest.mark.live
def test_live_instruction_parser_schema_and_clamping():
    """Verify live parser returns valid weights within allowed safety bounds."""

    client = GeminiClient()
    parser = InstructionParser(client=client)

    directive = (
        "Customer order 42 is critical and urgent, "
        "do not miss deadline even if electricity cost is high!"
    )

    weights, reasoning = parser.parse_instruction(directive)

    print(
        f"\n[Live Test] Compiled Weights for '{directive}':\n{weights}"
    )

    print(f"[Live Test] Reasoning:\n{reasoning}")

    # Verify every required schema key and its allowed range.
    for key, (low, high) in parser.ALLOWED_BOUNDS.items():

        assert key in weights, (
            f"Missing key '{key}' in parsed schema"
        )

        assert low <= weights[key] <= high, (
            f"Weight '{key}' = {weights[key]} is outside "
            f"allowed bounds [{low}, {high}]"
        )

    # Urgent orders should prioritize quota or late-delivery penalties.
    assert (
        weights["quota_penalty_weight"] >= 500.0
        or weights["late_penalty_weight"] >= 80.0
    ), (
        "Urgent order should increase quota or late-delivery penalty."
    )
