"""Rule-based heuristic controller with pre-cooling and tariff-aware job deferral."""
from __future__ import annotations
import numpy as np


class RuleBasedController:
    """Intelligent heuristic controller that implements engineering domain rules:

    1. HVAC Pre-cooling: Chill thermal zones to 22°C in the 2 hours before peak tariff slabs.
    2. HVAC Peak Float: Relax setpoint to 26°C during peak slabs to avoid high demand and cost.
    3. Heavy Job Deferral: Hold power-intensive machines (M4 Injection Molding, M5 Press)
       during peak tariff periods unless job deadline slack is critically tight.
    4. Off-peak Production: Fully schedule production during low-tariff off-peak periods.
    """

    def __init__(self, name: str = "Rule-Based Heuristic"):
        self.name = name

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Act method accepting the standardized 58-dim observation vector."""
        sin_t = float(observation[0])
        cos_t = float(observation[1])
        norm_price = float(observation[2])
        # Recover step index (0..95) from sin/cos
        angle = np.arctan2(sin_t, cos_t)
        fraction = (angle / (2.0 * np.pi)) % 1.0
        step = int(np.round(fraction * 96.0)) % 96

        norm_slack = float(observation[55]) # slack normalized by 48.0
        is_slack_critical = norm_slack <= 0.15 # slack under ~7 steps (~1.75 hours)

        # Price reconstruction: norm_price = (price - 8.0) / 6.0
        approx_price = norm_price * 6.0 + 8.0
        is_peak_price = (approx_price >= 10.0) or (72 <= step < 88)
        is_pre_cool_window = (60 <= step < 72) # 15:00 - 18:00 before evening peak

        # 1. HVAC Control
        if is_pre_cool_window:
            # Pre-cool to lowest comfortable level (22.0°C -> index 0)
            z1_action = 0
            z2_action = 0
        elif is_peak_price:
            # Float setpoint to upper comfortable level (26.0°C -> index 4) to slash compressor load
            z1_action = 4
            z2_action = 4
        else:
            # Standard comfort setpoint (24.0°C -> index 2)
            z1_action = 2
            z2_action = 2

        # 2. Machine Scheduling
        # Machines: M1(CNC), M2(Drill), M3(Lathe), M4(Molding, 28kW), M5(Press, 24kW), M6(Assembly, 8kW)
        if is_peak_price:
            if is_slack_critical:
                # Deadline is near: run all machines to prevent severe late / quota penalty
                m_actions = [1, 1, 1, 1, 1, 1]
            else:
                # Defer heavy machines M4 and M5 during peak electricity rate
                m_actions = [1, 1, 1, 0, 0, 1]
        else:
            # Non-peak: run machines to fulfill quota
            m_actions = [1, 1, 1, 1, 1, 1]

        action = np.array(m_actions + [z1_action, z2_action], dtype=np.int64)
        return action

    def reset(self):
        pass

