"""Tests for Phase 1: Digital Twin (SimPy model, machines, HVAC, jobs)."""
import pytest
import numpy as np

from ecotwin.twin.factory import FactoryFloorSim
from ecotwin.twin.machines import Machine, MachineState
from ecotwin.twin.hvac import HVACZone
from ecotwin.twin.jobs import Job, JobQueue


def test_idle_machines_draw_idle_power():
    """Verify idle machines draw rated_power * idle_power_fraction."""
    sim = FactoryFloorSim()
    sim.reset(seed=42)

    # Instruct all machines to HOLD (action=0)
    actions = {mid: 0 for mid in sim.machines}
    hvac_actions = {zid: 2 for zid in sim.zones}

    res = sim.step(machine_actions=actions, hvac_actions=hvac_actions, price_per_kwh=5.0, outside_temp_c=25.0)

    for mid, m in sim.machines.items():
        assert m.state == MachineState.HELD
        expected_idle_kw = m.rated_power_kw * m.idle_power_fraction
        assert np.isclose(m.step_power_kw, expected_idle_kw), f"{mid} idle power mismatch"
        assert np.isclose(m.step_energy_kwh, expected_idle_kw * sim.dt_hours), f"{mid} energy mismatch"


def test_energy_units_accounting():
    """Verify kW vs kWh units: floor kWh must equal floor kW * dt_hours."""
    sim = FactoryFloorSim()
    sim.reset(seed=123)

    for step in range(10):
        # Action 1 (start jobs)
        m_actions = {mid: 1 for mid in sim.machines}
        h_actions = {zid: 1 for zid in sim.zones}
        res = sim.step(m_actions, h_actions, price_per_kwh=6.5, outside_temp_c=30.0)

        # Check floor power vs energy
        assert np.isclose(res.step_energy_kwh, res.total_floor_power_kw * sim.dt_hours)
        assert np.isclose(res.step_cost_inr, res.step_energy_kwh * 6.5)
        assert res.total_floor_power_kw >= res.machines_power_kw


def test_temperatures_stay_physical():
    """Verify temperatures stay within realistic bounds under extreme outside temps."""
    sim = FactoryFloorSim()
    sim.reset(seed=999)

    # Test extreme summer heat (45 C)
    for _ in range(40):
        m_actions = {mid: 1 for mid in sim.machines}
        h_actions = {zid: 0 for zid in sim.zones} # max cooling setpoint (22 C)
        res = sim.step(m_actions, h_actions, price_per_kwh=4.0, outside_temp_c=45.0)
        for zid, temp in res.zone_temperatures.items():
            assert 18.0 <= temp <= 35.0, f"Zone {zid} temp {temp} out of physical range in heatwave"

    # Test extreme winter cold (5 C)
    sim.reset(seed=999)
    for _ in range(40):
        m_actions = {mid: 0 for mid in sim.machines}
        h_actions = {zid: 4 for zid in sim.zones} # relaxed setpoint (26 C)
        res = sim.step(m_actions, h_actions, price_per_kwh=4.0, outside_temp_c=5.0)
        for zid, temp in res.zone_temperatures.items():
            assert 10.0 <= temp <= 30.0, f"Zone {zid} temp {temp} out of physical range in cold"


def test_simulation_determinism():
    """Verify identical random seed and action sequence produce identical outputs."""
    sim1 = FactoryFloorSim()
    sim2 = FactoryFloorSim()

    sim1.reset(seed=2026)
    sim2.reset(seed=2026)

    for step in range(25):
        m_act = {mid: (step + i) % 2 for i, mid in enumerate(sim1.machines)}
        h_act = {zid: (step % 5) for zid in sim1.zones}
        temp = 25.0 + 5.0 * np.sin(step / 10.0)
        price = 4.0 + 2.0 * (step % 4)

        r1 = sim1.step(m_act, h_act, price_per_kwh=price, outside_temp_c=temp)
        r2 = sim2.step(m_act, h_act, price_per_kwh=price, outside_temp_c=temp)

        assert np.isclose(r1.total_floor_power_kw, r2.total_floor_power_kw)
        assert np.isclose(r1.step_cost_inr, r2.step_cost_inr)
        assert r1.completed_jobs_count == r2.completed_jobs_count
        for zid in sim1.zones:
            assert np.isclose(r1.zone_temperatures[zid], r2.zone_temperatures[zid])

