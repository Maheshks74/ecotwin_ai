"""Instruction parser: converts natural language instructions to validated reward weight schemas."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import json
import re
import numpy as np

from ecotwin.llm.client import GeminiClient


class InstructionParser:

    ALLOWED_BOUNDS = {
        "energy_cost_weight": (0.1, 5.0),
        "demand_charge_weight": (0.1, 5.0),
        "comfort_violation_weight": (10.0, 500.0),
        "quota_penalty_weight": (100.0, 3000.0),
        "late_penalty_weight": (10.0, 500.0),
        "reward_scale": (1e-6, 1.0),  # scaling factor range
    }


    DEFAULT_WEIGHTS = {
        "energy_cost_weight": 1.2,
        "demand_charge_weight": 1.2,
        "comfort_violation_weight": 60.0,
        "quota_penalty_weight": 600.0,
        "late_penalty_weight": 80.0,
        "reward_scale": 0.005,
    }


    def __init__(self, client: Optional[GeminiClient] = None):
        self.client = client or GeminiClient()

    def parse_instruction(
        self, user_instruction: str
    ) -> Tuple[Dict[str, float], str]:

        system_instruction = """
        You are an industrial reward-weight configuration assistant.

        Convert the manager directive into reward weights.

        Return ONLY valid JSON with these keys:
        energy_cost_weight
        demand_charge_weight
        comfort_violation_weight
        quota_penalty_weight
        late_penalty_weight
        reasoning

        Higher quota and late penalties prioritize production deadlines.
        Higher comfort penalties prioritize indoor temperature constraints.
        """

        prompt = (
            f'Manager Directive: "{user_instruction}"\n\n'
            "Provide the updated reward weights JSON:"
        )

        try:
            response = self.client.generate_text(
                prompt,
                system_instruction=system_instruction
            )

            data = self._extract_json(response)

            if data is None:
                data = self._heuristic_fallback(user_instruction)

        except Exception:
            data = self._heuristic_fallback(user_instruction)

        reasoning = str(data.get("reasoning", "Weights generated from directive."))

        weights = self._validate_and_clamp(data)

        # Critical: enforce directive priorities AFTER LLM output.
        weights = self._apply_directive_constraints(
            user_instruction, weights
        )

        return weights, reasoning

    def _extract_json(self, text: str):

        if not isinstance(text, str):
            return None

        # Remove optional Markdown JSON fences.
        text = re.sub(
            r"```(?:json)?",
            "",
            text,
            flags=re.IGNORECASE
        ).replace("```", "").strip()

        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Extract JSON object if surrounded by explanatory text.
        match = re.search(r"\{[\s\S]*\}", text)

        if match:
            try:
                result = json.loads(match.group(0))
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        return None

    def _validate_and_clamp(self, data: Dict[str, Any]):

        weights = {}

        for key, (low, high) in self.ALLOWED_BOUNDS.items():

            value = data.get(key, self.DEFAULT_WEIGHTS[key])

            # Reject booleans, strings and non-finite numbers.
            if isinstance(value, bool):
                value = self.DEFAULT_WEIGHTS[key]

            try:
                value = float(value)
            except (ValueError, TypeError):
                value = self.DEFAULT_WEIGHTS[key]

            if not np.isfinite(value):
                value = self.DEFAULT_WEIGHTS[key]

            weights[key] = float(np.clip(value, low, high))

        return weights

    def _apply_directive_constraints(self, instruction, weights):

        text = instruction.lower()

        urgent_keywords = [
            "urgent",
            "critical",
            "deadline",
            "rush",
            "no late delivery",
            "do not miss",
            "don't miss",
            "on time",
            "time sensitive",
        ]

        comfort_keywords = [
            "strictly tight",
            "tight temperature",
            "strict comfort",
            "maintain comfort",
            "temperature strictly",
        ]

        is_urgent = any(word in text for word in urgent_keywords)
        is_comfort_sensitive = any(
            word in text for word in comfort_keywords
        )

        if is_urgent:
            weights["quota_penalty_weight"] = max(
                weights["quota_penalty_weight"], 600.0
            )

            weights["late_penalty_weight"] = max(
                weights["late_penalty_weight"], 80.0
            )

        if is_comfort_sensitive:
            weights["comfort_violation_weight"] = max(
                weights["comfort_violation_weight"], 100.0
            )

        return weights

    def _heuristic_fallback(self, instruction):

        text = instruction.lower()

        if any(word in text for word in [
            "urgent", "critical", "deadline", "rush",
            "no late delivery", "do not miss"
        ]):
            return {
                "energy_cost_weight": 0.8,
                "demand_charge_weight": 1.0,
                "comfort_violation_weight": 60.0,
                "quota_penalty_weight": 1200.0,
                "late_penalty_weight": 250.0,
                "reasoning": "Urgent production requires deadline prioritization."
            }

        if any(word in text for word in [
            "strictly tight", "strict comfort", "tight temperature"
        ]):
            return {
                "energy_cost_weight": 1.2,
                "demand_charge_weight": 1.2,
                "comfort_violation_weight": 150.0,
                "quota_penalty_weight": 600.0,
                "late_penalty_weight": 80.0,
                "reasoning": "Comfort constraints have higher priority."
            }

        return {
            **self.DEFAULT_WEIGHTS,
            "reasoning": "Default reward configuration."
        }
