
# Gemini client with automatic mock mode, timeout,
# retry logic, and error handling.

from __future__ import annotations

from pathlib import Path
import os
import time
import json
import yaml

from typing import Any, Dict, Optional
from dotenv import load_dotenv


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


# --------------------------------------------------
# Gemini Client
# --------------------------------------------------

class GeminiClient:
    """
    Wrapper for Google Gemini API using official google-genai SDK.

    Features:
    - Automatic Mock Mode when API key is unavailable.
    - Explicit GEMINI_FORCE_MOCK and GEMINI_FORCE_LIVE controls.
    - Configurable HTTP timeout.
    - Exponential backoff retry mechanism.
    - Authentication and API error handling.
    """

    def __init__(self, config_path: Optional[str] = None):

        # Load configuration
        self.config = self._load_config(config_path)

        self.model_name = self.config.get(
            "model_name", "gemini-3.8-flash"
        )

        self.timeout_sec = float(
            self.config.get("timeout_seconds", 30.0)
        )

        self.max_retries = max(
            1, int(self.config.get("max_retries", 3))
        )

        self.initial_backoff = float(
            self.config.get("initial_backoff_seconds", 2.0)
        )

        self.backoff_multiplier = float(
            self.config.get("backoff_multiplier", 2.0)
        )

        # API key handling
        self.api_key = (
            os.getenv("GEMINI_API_KEY", "")
            .strip()
            .strip('"\'')
        )

        # Determine mock/live mode
        force_mock = os.getenv(
            "GEMINI_FORCE_MOCK", ""
        ).lower() in {"1", "true", "yes"}

        force_live = os.getenv(
            "GEMINI_FORCE_LIVE", ""
        ).lower() in {"1", "true", "yes"}

        if force_mock:
            self.is_mock_mode = True

        elif force_live:
            self.is_mock_mode = False

        else:
            self.is_mock_mode = not bool(self.api_key)

        self._live_client = None

        # Initialize Gemini SDK
        if not self.is_mock_mode:

            if not self.api_key:
                raise RuntimeError(
                    "GEMINI_FORCE_LIVE is enabled, "
                    "but GEMINI_API_KEY is missing."
                )

            try:
                from google import genai
                from google.genai import types

                http_options = types.HttpOptions(
                    timeout=int(self.timeout_sec * 1000)
                )

                self._live_client = genai.Client(
                    api_key=self.api_key,
                    http_options=http_options
                )

                print(
                    f"[Gemini] Live client initialized: "
                    f"{self.model_name}"
                )

            except Exception as e:

                print(
                    f"[Warning] Gemini initialization failed: {e}"
                )

                self.is_mock_mode = True

        if self.is_mock_mode:
            print("[Gemini] Running in Mock Mode.")

    # --------------------------------------------------
    # Configuration Loader
    # --------------------------------------------------

    def _load_config(
        self,
        config_path: Optional[str] = None
    ) -> Dict[str, Any]:

        if config_path is not None:

            path = Path(config_path)

            if not path.is_absolute():
                path = PROJECT_ROOT / path

            if path.exists():

                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}

        candidates = [
            PROJECT_ROOT / "config" / "llm.yaml",
            PROJECT_ROOT / "ecotwin" / "config" / "llm.yaml",
            Path(__file__).parent.parent / "config" / "llm.yaml",
        ]

        for path in candidates:

            if path.exists():

                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}

        # Fallback configuration
        return {
            "model_name": "gemini-3.8-flash",
            "timeout_seconds": 30.0,
            "max_retries": 3,
            "initial_backoff_seconds": 2.0,
            "backoff_multiplier": 2.0,
            "temperature": 0.2,
            "max_output_tokens": 1024,
        }

    # --------------------------------------------------
    # Client Badge
    # --------------------------------------------------

    @property
    def badge(self) -> str:

        if self.is_mock_mode:
            return "Mock LLM"

        return f"Live Gemini ({self.model_name})"

    # --------------------------------------------------
    # Generate Text
    # --------------------------------------------------

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> str:

        # Mock Mode
        if self.is_mock_mode:
            return self._generate_mock_response(
                prompt,
                system_instruction
            )

        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=float(
                self.config.get("temperature", 0.2)
            ),
            max_output_tokens=int(
                self.config.get("max_output_tokens", 1024)
            ),
            system_instruction=system_instruction,
        )

        backoff = self.initial_backoff
        last_error = None

        # Retry loop
        for attempt in range(1, self.max_retries + 1):

            try:

                print(
                    f"[Gemini] API request: "
                    f"attempt {attempt}/{self.max_retries}"
                )

                response = (
                    self._live_client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=config,
                    )
                )

                if not response.text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                print("[Gemini] Response received successfully.")

                return response.text

            except Exception as err:

                last_error = err
                err_str = str(err).lower()

                print(
                    f"[Gemini] Request failed on attempt "
                    f"{attempt}: {err}"
                )

                # Authentication errors: do not retry
                if any(term in err_str for term in [
                    "unauthenticated",
                    "api_key",
                    "permission denied",
                    "403",
                ]):

                    raise RuntimeError(
                        f"Gemini authentication/permission error: {err}"
                    ) from err

                # Stop if maximum attempts reached
                if attempt == self.max_retries:
                    break

                # Identify temporary/network errors
                retryable = any(term in err_str for term in [
                    "429",
                    "500",
                    "502",
                    "503",
                    "504",
                    "timeout",
                    "timed out",
                    "connection",
                    "network",
                    "unavailable",
                    "overloaded",
                    "rate",
                    "ssl",
                    "temporary",
                    "getaddrinfo",
                ])

                # Do not retry permanent errors
                if not retryable:

                    raise RuntimeError(
                        f"Gemini API request failed: {err}"
                    ) from err

                print(
                    f"[Gemini] Temporary error. "
                    f"Retrying in {backoff:.1f} seconds..."
                )

                time.sleep(backoff)

                backoff *= self.backoff_multiplier

        # All retries exhausted
        raise RuntimeError(
            f"Gemini API failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        ) from last_error

    # --------------------------------------------------
    # Mock Response Generator
    # --------------------------------------------------

    def _generate_mock_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> str:

        # Case 1: Manager reward-weight instructions
        if (
            "manager directive" in prompt.lower()
            and "reward weights" in prompt.lower()
        ):

            text = prompt.lower()

            if any(word in text for word in [
                "urgent",
                "critical",
                "deadline",
                "rush",
                "no late delivery",
                "do not miss"
            ]):

                return json.dumps({
                    "energy_cost_weight": 0.8,
                    "demand_charge_weight": 1.0,
                    "comfort_violation_weight": 60.0,
                    "quota_penalty_weight": 1200.0,
                    "late_penalty_weight": 250.0,
                    "reasoning": (
                        "Urgent production requires "
                        "deadline prioritization."
                    )
                })

            if any(word in text for word in [
                "strictly tight",
                "strict comfort",
                "tight temperature"
            ]):

                return json.dumps({
                    "energy_cost_weight": 1.2,
                    "demand_charge_weight": 1.2,
                    "comfort_violation_weight": 150.0,
                    "quota_penalty_weight": 600.0,
                    "late_penalty_weight": 80.0,
                    "reasoning": (
                        "Comfort constraints have higher priority."
                    )
                })

        prompt_lower = prompt.lower()

        # Case 2: Missing information
        if any(term in prompt_lower for term in [
            "not found",
            "job 999",
            "unknown_job",
            "no matching log rows found",
            "facts: []"
        ]):

            return "not enough information"

        # Case 3: Instruction parsing
        if (
            "schema" in prompt_lower
            or "instruction" in prompt_lower
            or "parse" in prompt_lower
        ):

            if (
                "urgent" in prompt_lower
                or "rush" in prompt_lower
            ):

                return json.dumps({
                    "energy_cost_weight": 0.8,
                    "demand_charge_weight": 1.0,
                    "comfort_violation_weight": 60.0,
                    "quota_penalty_weight": 1200.0,
                    "late_penalty_weight": 250.0,
                    "reasoning": (
                        "Prioritized quota and late penalty heavily "
                        "due to urgent order instructions."
                    ),
                })

            elif any(term in prompt_lower for term in [
                "tight",
                "comfort",
                "temperature"
            ]):

                return json.dumps({
                    "energy_cost_weight": 1.0,
                    "demand_charge_weight": 1.0,
                    "comfort_violation_weight": 150.0,
                    "quota_penalty_weight": 600.0,
                    "late_penalty_weight": 80.0,
                    "reasoning": (
                        "Increased comfort violation penalty "
                        "to maintain tighter indoor temperature."
                    ),
                })

            else:

                return json.dumps({
                    "energy_cost_weight": 1.2,
                    "demand_charge_weight": 1.2,
                    "comfort_violation_weight": 60.0,
                    "quota_penalty_weight": 600.0,
                    "late_penalty_weight": 80.0,
                    "reasoning": (
                        "Adjusted energy cost and demand weights "
                        "to focus on peak demand trimming."
                    ),
                })

        # Case 4: What-if analysis
        if (
            "what-if" in prompt_lower
            or "what if" in prompt_lower
        ):

            return json.dumps({
                "modified_scenario": "price_spike",
                "hvac_setpoint_delta": 1.0,
                "quota_delta": 0,
                "reasoning": (
                    "Simulated what-if scenario adjusting "
                    "HVAC setpoint and tariff conditions."
                ),
            })

        # Case 5: Advisor explanation
        if (
            "explain" in prompt_lower
            or "why" in prompt_lower
            or "advisor" in prompt_lower
        ):

            if "job" in prompt_lower:

                return (
                    "Based on retrieved step records, the machine "
                    "began processing the job during off-peak hours "
                    "when electricity tariff was 4.50 INR/kWh. "
                    "With sufficient deadline slack available, "
                    "deferring operation to off-peak periods avoided "
                    "high tariff charges while satisfying production quota."
                )

            return (
                "Based on retrieved floor logs, total floor electrical "
                "power was balanced between machines and HVAC. "
                "The controller shifted setpoints and non-critical "
                "operations to avoid peak demand penalties."
            )

        # Default fallback
        return (
            "Based on available system telemetry, "
            "operations proceeded within nominal parameters."
        )
