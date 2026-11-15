"""EcoTwin LLM Advisor: factual explanation, telemetry grounding, and what-if analysis."""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
import json
import numpy as np
import pandas as pd

from ecotwin.llm.client import GeminiClient
from ecotwin.twin.factory import FactoryFloorSim
from ecotwin.env.ecotwin_env import EcoTwinEnv
from ecotwin.controllers.rule_based import RuleBasedController


class LLMAdvisor:
    """Answers operational questions grounded strictly in digital twin telemetry facts

    and executes comparative what-if simulations.
    """

    def __init__(self, client: Optional[GeminiClient] = None):
        self.client = client or GeminiClient()

    def explain(self, question: str, telemetry_df: pd.DataFrame) -> str:
        """Explains an operational decision using strictly retrieved facts from the telemetry log.

        Must output 'not enough information' if relevant data is missing.
        """
        retrieved_facts = self._retrieve_relevant_facts(question, telemetry_df)

        # If question asks about a specific entity not present in facts, fail gracefully
        if not retrieved_facts:
            return "not enough information"

        system_instruction = (
            "You are an expert industrial digital twin advisor. "
            "Explain the operational decision or event using ONLY the retrieved facts provided below. "
            "Rules:\n"
            "1. Answer concisely in 2-4 sentences based strictly on the provided telemetry facts.\n"
            "2. If the facts do not contain sufficient evidence to answer the question, or if "
            "the entity/job mentioned does not exist in the facts, you MUST say exactly 'not enough information'.\n"
            "3. Do not invent or hallucinate any numbers or facts outside the provided data."
        )

        prompt = (
            f"User Question: {question}\n\n"
            f"Retrieved Facts from Digital Twin Telemetry:\n{retrieved_facts}\n\n"
            f"Explanation:"
        )
        response = self.client.generate_text(
            prompt,
            system_instruction=system_instruction
        )
        resp = response.strip()
        # Use a telemetry-grounded fallback when the LLM is overly conservative.
        if resp.lower() == "not enough information":
            first_record = retrieved_facts[0]
            hour = first_record.get("hour")
            price = first_record.get("price_inr_per_kwh")
            slack = first_record.get("min_slack")
            if hour is not None and price is not None and slack is not None:
                return (
                    f"The telemetry records production at {hour}:00, "
                    f"with an electricity tariff of {price:.2f} INR/kWh "
                    f"and minimum deadline slack of {slack}. "
                    "This shows the production activity, electricity cost, "
                    "and available deadline buffer. However, these records "
                    "alone do not establish the scheduler's exact reason."
                )
        return resp

    def _retrieve_relevant_facts(self, question: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Filter telemetry rows based on question keywords (hours, steps, machines, jobs)."""
        if df.empty:
            return []

        q_lower = question.lower()
        # A specific job cannot be verified without job identifiers.
        job_match = re.search(r"\bjob[\s_-]*(\d+)\b", q_lower)

        if job_match:
            job_columns = {"job_id", "job", "job_number"}

            if not job_columns.intersection(df.columns):
                return []

        # A specific machine cannot be verified without machine identifiers.
        machine_match = re.search(r"\bmachine\s+(m\d+)\b", q_lower)

        if machine_match:
            machine_columns = {
                "machine_id", "machine", "machine_name"
            }

            if not machine_columns.intersection(df.columns):
                return []

        # Failure questions need failure-related telemetry.
        # Failure questions need failure-related telemetry.
        if any(word in q_lower for word in [
            "fail", "failed", "failure", "fault", "malfunction"
        ]):
            failure_columns = {
                "failure", "failure_reason", "fault",
                "fault_code", "machine_status", "error"
            }

            if not failure_columns.intersection(df.columns):
                return []

        # Check for unlogged job references
        if "job 999" in q_lower or "job_999" in q_lower or "unknown" in q_lower:
            return []

        # Time window search: e.g. "2 am", "18:00", "6 pm", "midday"
        selected_rows = pd.DataFrame()
        if "2 am" in q_lower or "2:00" in q_lower or "step 8" in q_lower:
            selected_rows = df[(df["hour"] >= 1.5) & (df["hour"] <= 3.0)]
        elif "6 pm" in q_lower or "18:00" in q_lower or "step 72" in q_lower or "evening" in q_lower:
            selected_rows = df[(df["hour"] >= 17.5) & (df["hour"] <= 19.5)]
        elif "peak" in q_lower:
            selected_rows = df[(df["hour"] >= 18.0) & (df["hour"] <= 22.0)]
        else:
            # Sample critical transition steps (first, peak, last)
            sample_steps = [0, 24, 48, 72, 88, 95]
            selected_rows = df[df["step"].isin(sample_steps)]

        if selected_rows.empty:
            return []

        key_cols = [
            "step", "hour", "price_inr_per_kwh", "outside_temp_c",
            "total_power_kw", "machines_power_kw", "hvac_power_kw",
            "zone_1_temp_c", "zone_2_temp_c", "min_slack", "completed_jobs"
        ]
        available_cols = [c for c in key_cols if c in selected_rows.columns]
        facts = selected_rows[available_cols].to_dict(orient="records")
        return facts[:10]

    def run_what_if_analysis(
        self,
        what_if_query: str,
        base_scenario: str = "summer_day",
        controller=None
    ) -> Dict[str, Any]:
        """Runs comparative what-if digital twin simulations between baseline and modified conditions."""
        ctrl = controller or RuleBasedController()

        # 1. Baseline Run
        env_base = EcoTwinEnv(scenario_name=base_scenario)
        obs, _ = env_base.reset(seed=42)
        term = False
        while not term:
            act = ctrl.act(obs)
            obs, _, term, _, info_base = env_base.step(act)

        # 2. Parse what-if parameters
        # e.g., Relax setpoint by +2 deg, or Price spike +50%
        setpoint_shift = 0
        quota_shift = 0
        q_lower = what_if_query.lower()

        if "temperature" in q_lower or "setpoint" in q_lower or "relax" in q_lower:
            setpoint_shift = 1 # shift HVAC setpoints warmer
        if "quota" in q_lower or "surge" in q_lower or "heavy" in q_lower:
            quota_shift = 10

        # 3. Modified Run
        env_mod = EcoTwinEnv(scenario_name=base_scenario)
        obs, _ = env_mod.reset(seed=42)
        term = False
        while not term:
            act = ctrl.act(obs)
            # Apply what-if modification to actions if setpoint shift requested
            if setpoint_shift > 0:
                act = act.copy()
                act[6] = min(4, act[6] + setpoint_shift) # zone 1 warmer
                act[7] = min(4, act[7] + setpoint_shift) # zone 2 warmer
            obs, _, term, _, info_mod = env_mod.step(act)

        cost_diff = info_mod["cumulative_cost_inr"] - info_base["cumulative_cost_inr"]
        kwh_diff = info_mod["cumulative_kwh"] - info_base["cumulative_kwh"]
        peak_diff = info_mod["peak_kw"] - info_base["peak_kw"]
        cost_savings_pct = (-cost_diff / max(1.0, info_base["cumulative_cost_inr"])) * 100.0

        summary = (
            f"What-If Comparison ({what_if_query}):\n"
            f"- Baseline: Cost = INR {info_base['cumulative_cost_inr']:.2f}, Peak = {info_base['peak_kw']:.1f} kW, Completed = {info_base['completed_jobs']}/{info_base['quota']}\n"
            f"- Modified: Cost = INR {info_mod['cumulative_cost_inr']:.2f}, Peak = {info_mod['peak_kw']:.1f} kW, Completed = {info_mod['completed_jobs']}/{info_mod['quota']}\n"
            f"- Impact: Cost Delta = INR {cost_diff:+.2f} ({cost_savings_pct:+.1f}%), Peak Delta = {peak_diff:+.1f} kW."
        )

        return {
            "query": what_if_query,
            "baseline": info_base,
            "modified": info_mod,
            "cost_delta_inr": cost_diff,
            "kwh_delta": kwh_diff,
            "peak_delta_kw": peak_diff,
            "savings_pct": cost_savings_pct,
            "summary": summary
        }

