"""Machine models for EcoTwin manufacturing simulation."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from ecotwin.twin.jobs import Job, JobQueue


class MachineState(str, Enum):
    IDLE = "IDLE"
    PROCESSING = "PROCESSING"
    HELD = "HELD"


@dataclass
class Machine:
    """Represents an industrial machine on the factory floor."""
    id: str
    name: str
    zone_id: str
    rated_power_kw: float
    idle_power_fraction: float
    heat_emission_fraction: float
    processing_steps: Dict[str, int]
    state: MachineState = MachineState.IDLE
    current_job: Optional[Job] = None
    step_energy_kwh: float = 0.0
    step_power_kw: float = 0.0
    step_heat_kw: float = 0.0
    cumulative_kwh: float = 0.0
    jobs_processed: int = 0

    @property
    def idle_power_kw(self) -> float:
        return self.rated_power_kw * self.idle_power_fraction

    def reset(self):
        self.state = MachineState.IDLE
        self.current_job = None
        self.step_energy_kwh = 0.0
        self.step_power_kw = self.idle_power_kw
        self.step_heat_kw = self.step_power_kw * self.heat_emission_fraction
        self.cumulative_kwh = 0.0
        self.jobs_processed = 0

    def step(self, action: int, current_step: int, job_queue: JobQueue, dt_hours: float = 0.25) -> Tuple[float, float]:
        """Execute one discrete time step (15 min).

        Parameters
        ----------
        action : int
            0 = HOLD (stay idle or hold off starting new job), 1 = START next job if idle.
            If machine is already in the middle of processing a job, it continues until finished.
        current_step : int
            Current simulation step index (0..95).
        job_queue : JobQueue
            Global job queue to pull next job from when idle.
        dt_hours : float
            Step duration in hours (default 0.25h = 15 min).

        Returns
        -------
        Tuple[float, float]
            (step_energy_kwh, step_heat_kw)
        """
        completed_this_step = False

        if self.current_job is not None:
            # Continue existing job
            self.state = MachineState.PROCESSING
            self.step_power_kw = self.rated_power_kw
            self.current_job.progress_steps += 1
            if self.current_job.is_completed:
                job_queue.mark_completed(self.current_job.job_id, current_step)
                self.jobs_processed += 1
                self.current_job = None
                self.state = MachineState.IDLE
                completed_this_step = True
        else:
            # Machine is currently free
            if action == 1:
                # Attempt to start next job
                candidate = job_queue.pop_next_job(
                    machine_id=self.id,
                    current_step=current_step,
                    allowed_types=list(self.processing_steps.keys())
                )
                if candidate is not None:
                    # Update candidate duration specific to this machine's processing speed
                    candidate.duration_steps = self.processing_steps.get(candidate.job_type, 2)
                    self.current_job = candidate
                    self.state = MachineState.PROCESSING
                    self.step_power_kw = self.rated_power_kw
                    self.current_job.progress_steps += 1
                    if self.current_job.is_completed:
                        job_queue.mark_completed(self.current_job.job_id, current_step)
                        self.jobs_processed += 1
                        self.current_job = None
                        self.state = MachineState.IDLE
                        completed_this_step = True
                else:
                    # No job ready/available; remain idle
                    self.state = MachineState.IDLE
                    self.step_power_kw = self.idle_power_kw
            else:
                # Explicit hold
                self.state = MachineState.HELD
                self.step_power_kw = self.idle_power_kw

        self.step_energy_kwh = self.step_power_kw * dt_hours
        self.step_heat_kw = self.step_power_kw * self.heat_emission_fraction
        self.cumulative_kwh += self.step_energy_kwh

        return self.step_energy_kwh, self.step_heat_kw

