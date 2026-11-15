"""First-order RC HVAC thermal zone model for EcoTwin."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class HVACZone:
    """Models a single thermal zone governed by a first-order RC differential equation.

    dT/dt = (T_out - T_in) / (R * C) + (Q_machines - Q_cooling) / C
    """
    zone_id: str
    name: str
    thermal_resistance_R: float # deg C / kW
    thermal_capacitance_C: float # kWh / deg C
    target_temp_c: float = 24.0
    comfort_min_c: float = 22.0
    comfort_max_c: float = 26.0
    initial_temp_c: float = 24.0
    cop_nominal: float = 3.5
    cop_temp_coefficient: float = 0.04
    cop_min: float = 1.5
    cop_max: float = 4.5
    max_cooling_kw: float = 40.0
    setpoint_options: List[float] = field(default_factory=lambda: [22.0, 23.0, 24.0, 25.0, 26.0])

    current_temp_c: float = 24.0
    cooling_thermal_kw: float = 0.0
    cooling_electric_kw: float = 0.0
    step_energy_kwh: float = 0.0
    cumulative_kwh: float = 0.0
    comfort_violation_deg: float = 0.0
    cumulative_comfort_violations: float = 0.0

    def __post_init__(self):
        self.current_temp_c = self.initial_temp_c

    def reset(self, initial_temp: Optional[float] = None):
        self.current_temp_c = initial_temp if initial_temp is not None else self.initial_temp_c
        self.target_temp_c = 24.0
        self.cooling_thermal_kw = 0.0
        self.cooling_electric_kw = 0.0
        self.step_energy_kwh = 0.0
        self.cumulative_kwh = 0.0
        self.comfort_violation_deg = 0.0
        self.cumulative_comfort_violations = 0.0

    def compute_cop(self, outside_temp_c: float) -> float:
        """Computes COP as a function of outdoor ambient temperature."""
        raw_cop = self.cop_nominal - self.cop_temp_coefficient * (outside_temp_c - 25.0)
        return float(np.clip(raw_cop, self.cop_min, self.cop_max))

    def step(
        self,
        setpoint_idx: int,
        q_machines_kw: float,
        outside_temp_c: float,
        dt_hours: float = 0.25
    ) -> Tuple[float, float, float]:
        """Execute one simulation step of the HVAC zone.

        Parameters
        ----------
        setpoint_idx : int
            Index into setpoint_options (e.g. 0..4).
        q_machines_kw : float
            Total heat dissipated by active and idle machines in this zone (kW).
        outside_temp_c : float
            Current ambient outside temperature (deg C).
        dt_hours : float
            Time step duration in hours (default 0.25h = 15 min).

        Returns
        -------
        Tuple[float, float, float]
            (new_temp_c, cooling_electric_kw, comfort_violation_deg)
        """
        # Set target temperature from discrete options
        safe_idx = int(np.clip(setpoint_idx, 0, len(self.setpoint_options) - 1))
        self.target_temp_c = self.setpoint_options[safe_idx]

        # Passive temperature evolution without active cooling:
        # dT_passive/dt = (T_out - T_in) / (R * C) + Q_machines / C
        dt_rc = self.thermal_resistance_R * self.thermal_capacitance_C
        dT_ambient = (outside_temp_c - self.current_temp_c) / dt_rc
        dT_machines = q_machines_kw / self.thermal_capacitance_C
        t_uncooled = self.current_temp_c + (dT_ambient + dT_machines) * dt_hours

        # Cooling required to bring temperature down to target setpoint:
        if t_uncooled > self.target_temp_c:
            # Q_req = C * (t_uncooled - target_temp) / dt
            q_required = self.thermal_capacitance_C * (t_uncooled - self.target_temp_c) / dt_hours
            self.cooling_thermal_kw = float(np.clip(q_required, 0.0, self.max_cooling_kw))
        else:
            self.cooling_thermal_kw = 0.0

        # Resulting indoor temperature after cooling
        cooling_dT = (self.cooling_thermal_kw / self.thermal_capacitance_C) * dt_hours
        self.current_temp_c = float(t_uncooled - cooling_dT)

        # Electrical power and energy
        cop = self.compute_cop(outside_temp_c)
        self.cooling_electric_kw = self.cooling_thermal_kw / cop
        self.step_energy_kwh = self.cooling_electric_kw * dt_hours
        self.cumulative_kwh += self.step_energy_kwh

        # Check comfort band violation
        violation = 0.0
        if self.current_temp_c < self.comfort_min_c:
            violation = self.comfort_min_c - self.current_temp_c
        elif self.current_temp_c > self.comfort_max_c:
            violation = self.current_temp_c - self.comfort_max_c

        self.comfort_violation_deg = float(violation)
        self.cumulative_comfort_violations += self.comfort_violation_deg

        return self.current_temp_c, self.cooling_electric_kw, self.comfort_violation_deg

