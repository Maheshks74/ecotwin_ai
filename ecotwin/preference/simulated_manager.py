"""Simulated Plant Manager with latent preferences for RLHF-style preference learning."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TrajectorySegment:
    """Represents an episode or sub-episode trajectory segment."""
    segment_id: str
    states: np.ndarray # (T, obs_dim)
    actions: np.ndarray # (T, act_dim)
    cumulative_cost: float
    total_comfort_violation: float
    peak_kw: float
    night_machine_kwh: float
    completed_jobs: int
    quota: int


class SimulatedManager:
    """Scripted Plant Manager with latent human-like operational preferences:

    - Heavy aversion to night shifts (supervision difficulties, labor cost, worker fatigue)
    - Strict comfort preference (office/shopfloor temperature control)
    - Penalty for high peak demand spikes
    - Strong requirement to fulfill production quota
    - Moderate cost sensitivity
    """

    def __init__(
        self,
        weight_cost: float = 0.5,
        weight_night_kwh: float = 2.0,
        weight_comfort: float = 120.0,
        weight_peak_kw: float = 15.0,
        weight_quota_job: float = 250.0,
        temperature: float = 1.0,
        random_seed: Optional[int] = None
    ):
        self.w_cost = weight_cost
        self.w_night = weight_night_kwh
        self.w_comfort = weight_comfort
        self.w_peak = weight_peak_kw
        self.w_quota = weight_quota_job
        self.temperature = temperature
        self.rng = np.random.default_rng(random_seed)

    def compute_latent_utility(self, segment: TrajectorySegment) -> float:
        """Calculates ground-truth hidden utility U(tau)."""
        missed = max(0, segment.quota - segment.completed_jobs)
        utility = (
            - self.w_cost * (segment.cumulative_cost / 100.0)
            - self.w_night * (segment.night_machine_kwh / 50.0)
            - self.w_comfort * segment.total_comfort_violation
            - self.w_peak * (segment.peak_kw / 50.0)
            - self.w_quota * missed
        )
        return float(utility)

    def compare_trajectories(
        self,
        seg1: TrajectorySegment,
        seg2: TrajectorySegment,
        add_noise: bool = True
    ) -> Tuple[int, float]:
        """Compares two trajectory segments using Bradley-Terry preference model.

        Returns
        -------
        Tuple[int, float]
            (winner: 1 if seg1 preferred else 0, prob_seg1: float)
        """
        u1 = self.compute_latent_utility(seg1)
        u2 = self.compute_latent_utility(seg2)

        diff = (u1 - u2) / max(0.01, self.temperature)
        prob_1 = float(1.0 / (1.0 + np.exp(-np.clip(diff, -20.0, 20.0))))

        if add_noise:
            winner = 1 if self.rng.uniform(0.0, 1.0) < prob_1 else 0
        else:
            winner = 1 if u1 >= u2 else 0

        return winner, prob_1

