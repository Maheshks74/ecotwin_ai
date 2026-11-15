"""Trajectory pair generator for preference learning."""
from __future__ import annotations
from typing import Callable, List, Optional, Tuple
import numpy as np

from ecotwin.env.ecotwin_env import EcoTwinEnv
from ecotwin.preference.simulated_manager import TrajectorySegment, SimulatedManager


class PairGenerator:
    """Rolls out policies in EcoTwinEnv to generate paired trajectory segments for reward learning."""

    def __init__(self, env: EcoTwinEnv, manager: SimulatedManager, segment_length: int = 48):
        self.env = env
        self.manager = manager
        self.segment_length = segment_length # e.g. 48 steps = 12 hours (or full 96 steps)

    def rollout_segment(
        self,
        policy_fn: Callable[[np.ndarray], np.ndarray],
        seed: Optional[int] = None,
        segment_id: str = "seg"
    ) -> TrajectorySegment:
        """Rolls out a policy for `segment_length` steps and returns a TrajectorySegment."""
        obs, info = self.env.reset(seed=seed)
        states = []
        actions = []
        night_kwh = 0.0

        for t in range(self.segment_length):
            act = policy_fn(obs)
            states.append(obs.copy())
            actions.append(np.asarray(act, dtype=np.int64).copy())

            # Step in environment
            obs, r, term, trunc, info = self.env.step(act)

            # Check if this step is during night shift (22:00 to 06:00 -> steps 0..24 and 88..96)
            step_idx = info["step"]
            if step_idx < 24 or step_idx >= 88:
                # Night machine power: sum active machine powers
                last_res = self.env.last_result
                if last_res:
                    night_kwh += last_res.machines_power_kw * 0.25

            if term or trunc:
                break

        res = self.env.last_result
        return TrajectorySegment(
            segment_id=segment_id,
            states=np.array(states, dtype=np.float32),
            actions=np.array(actions, dtype=np.int64),
            cumulative_cost=res.cumulative_cost_inr if res else 0.0,
            total_comfort_violation=float(res.cumulative_cost_inr * 0.0 + (res.total_comfort_violation if res else 0.0)),
            peak_kw=res.peak_power_kw if res else 0.0,
            night_machine_kwh=night_kwh,
            completed_jobs=info["completed_jobs"],
            quota=info["quota"]
        )

    def generate_pairs(
        self,
        num_pairs: int = 100,
        policy_pool: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None
    ) -> List[Tuple[TrajectorySegment, TrajectorySegment, int]]:
        """Generates paired trajectories labeled by the simulated manager."""
        if policy_pool is None:
            # Default diverse policy pool:
            # 1. Random policy
            p_random = lambda obs: self.env.action_space.sample()
            # 2. Daytime heavy (holds machines at night, runs during day)
            def p_daytime(obs):
                sin_t = float(obs[0])
                cos_t = float(obs[1])
                step = int(np.round(((np.arctan2(sin_t, cos_t) / (2 * np.pi)) % 1.0) * 96)) % 96
                if step < 24 or step >= 88: # night
                    return np.array([0, 0, 0, 0, 0, 0, 2, 2], dtype=np.int64)
                return np.array([1, 1, 1, 1, 1, 1, 2, 2], dtype=np.int64)
            # 3. Aggressive production (always run)
            p_always = lambda obs: np.array([1, 1, 1, 1, 1, 1, 2, 2], dtype=np.int64)
            # 4. Strict comfort (cold HVAC)
            p_cold = lambda obs: np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=np.int64)
            policy_pool = [p_random, p_daytime, p_always, p_cold]

        pairs = []
        rng = np.random.default_rng(42)

        for p_idx in range(num_pairs):
            # Select two distinct policies or same policy with different seeds
            idx1, idx2 = rng.choice(len(policy_pool), size=2, replace=True)
            pol1 = policy_pool[idx1]
            pol2 = policy_pool[idx2]

            seed1 = int(rng.integers(1, 100000))
            seed2 = int(rng.integers(1, 100000))

            seg1 = self.rollout_segment(pol1, seed=seed1, segment_id=f"pair_{p_idx}_A")
            seg2 = self.rollout_segment(pol2, seed=seed2, segment_id=f"pair_{p_idx}_B")

            label, _ = self.manager.compare_trajectories(seg1, seg2, add_noise=True)
            pairs.append((seg1, seg2, label))

        return pairs

