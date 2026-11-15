"""Scenario loader and configurator for EcoTwin."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from ecotwin.twin.jobs import Job


class ScenarioManager:
    """Loads and provides factory simulation scenarios."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.scenarios: Dict[str, Dict[str, Any]] = self.config.get("scenarios", {})

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        if config_path is not None and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)

        candidates = [
            Path("config/scenarios.yaml"),
            Path("ecotwin/config/scenarios.yaml"),
            Path(__file__).parent.parent / "config" / "scenarios.yaml",
        ]
        for c in candidates:
            if c.exists():
                with open(c, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)

        return {}

    def list_scenarios(self) -> List[str]:
        return list(self.scenarios.keys())

    def get_scenario(self, name: str) -> Dict[str, Any]:
        if name in self.scenarios:
            return self.scenarios[name]
        # Fallback to summer_day or empty
        return self.scenarios.get("summer_day", {})

    def build_scenario_jobs(self, scenario_name: str, base_jobs: List[Job]) -> List[Job]:
        """Modifies or augments job list for rush_order or heavy_quota scenarios."""
        scen = self.get_scenario(scenario_name)
        jobs = list(base_jobs)

        if "rush_batch" in scen:
            rush_cfg = scen["rush_batch"]
            rel_step = int(rush_cfg.get("release_step", 44))
            due_step = int(rush_cfg.get("deadline_step", 56))
            count = int(rush_cfg.get("count", 6))
            priority = float(rush_cfg.get("priority", 3.0))

            for r in range(count):
                jobs.append(
                    Job(
                        job_id=f"RUSH_{r+1:02d}",
                        job_type="TypeB",
                        release_step=rel_step,
                        due_step=due_step,
                        duration_steps=2,
                        priority=priority
                    )
                )

        return jobs

