"""Tests for Phase 7: LLM Advisor, Logger, Instruction Parser (with Mock Mode)."""
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from ecotwin.llm.client import GeminiClient
from ecotwin.llm.logger import FloorLogger
from ecotwin.llm.advisor import LLMAdvisor
from ecotwin.llm.instruction_parser import InstructionParser
from ecotwin.twin.factory import FactoryFloorSim


def test_gemini_client_mock_mode(monkeypatch):

    monkeypatch.setenv("GEMINI_FORCE_MOCK", "1")
    monkeypatch.delenv("GEMINI_FORCE_LIVE", raising=False)

    client = GeminiClient()

    assert client.is_mock_mode is True
    assert client.badge == "Mock LLM"

    resp = client.generate_text(
        "Explain why machine was paused."
    )

    assert len(resp) > 10


def test_floor_logger_persistence(tmp_path):
    """Verify telemetry logger logs steps and saves to disk."""
    logger = FloorLogger(log_dir=str(tmp_path))
    sim = FactoryFloorSim()
    sim.reset(seed=42)

    for step in range(5):
        res = sim.step({mid: 1 for mid in sim.machines}, {zid: 2 for zid in sim.zones}, price_per_kwh=6.0, outside_temp_c=28.0)
        logger.log_step(
            result=res,
            action=np.array([1, 1, 1, 1, 1, 1, 2, 2]),
            outside_temp=28.0,
            price_per_kwh=6.0,
            reward=-1.5
        )

    df = logger.to_dataframe()
    assert len(df) == 5
    assert "total_power_kw" in df.columns
    assert "price_inr_per_kwh" in df.columns

    saved_path = logger.save("test_log.parquet")
    assert Path(saved_path).exists()


def test_advisor_factual_and_not_enough_info():
    """Verify advisor explains from facts and strictly outputs 'not enough information' for missing facts."""
    client = GeminiClient()
    advisor = LLMAdvisor(client=client)

    # Telemetry with known facts
    df = pd.DataFrame([
        {"step": 8, "hour": 2.0, "price_inr_per_kwh": 4.5, "outside_temp_c": 22.0,
         "total_power_kw": 40.0, "machines_power_kw": 30.0, "hvac_power_kw": 10.0,
         "zone_1_temp_c": 24.0, "zone_2_temp_c": 24.0, "min_slack": 20.0, "completed_jobs": 5}
    ])

    # Case A: Grounded question
    explanation = advisor.explain("Why was production scheduled at 2 AM?", df)
    assert len(explanation) > 10
    assert "not enough information" not in explanation.lower()

    # Case B: Unknown entity / unlogged question must return 'not enough information'
    missing_resp = advisor.explain("Why was Job 999 cancelled?", df)
    assert "not enough information" in missing_resp.lower()


def test_what_if_analysis():
    """Verify what-if analysis runs comparative simulation and reports cost delta."""
    client = GeminiClient()
    advisor = LLMAdvisor(client=client)

    res = advisor.run_what_if_analysis("What if temperature setpoint is relaxed by 1 degree?")
    assert "baseline" in res
    assert "modified" in res
    assert "cost_delta_inr" in res
    assert "summary" in res


def test_instruction_parser_schema_and_clamping(monkeypatch):

    monkeypatch.setenv("GEMINI_FORCE_MOCK", "1")
    monkeypatch.delenv("GEMINI_FORCE_LIVE", raising=False)

    client = GeminiClient()
    parser = InstructionParser(client=client)

    weights_urgent, reason1 = parser.parse_instruction(
        "Customer batch is urgent, ensure no late delivery!"
    )

    assert weights_urgent["quota_penalty_weight"] >= 600.0

    for k, (low, high) in parser.ALLOWED_BOUNDS.items():
        assert low <= weights_urgent[k] <= high

    weights_comfort, reason2 = parser.parse_instruction(
        "Keep indoor temperatures strictly tight!"
    )

    assert weights_comfort["comfort_violation_weight"] >= 100.0

    for k, (low, high) in parser.ALLOWED_BOUNDS.items():
        assert low <= weights_comfort[k] <= high
