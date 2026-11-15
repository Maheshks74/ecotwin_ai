"""Job and JobQueue models for EcoTwin manufacturing simulation."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class Job:
    """Represents a discrete manufacturing job."""
    job_id: str
    job_type: str
    release_step: int
    due_step: int
    duration_steps: int
    priority: float = 1.0
    progress_steps: int = 0
    assigned_machine_id: Optional[str] = None
    started_step: Optional[int] = None
    completed_step: Optional[int] = None

    @property
    def is_completed(self) -> bool:
        return self.progress_steps >= self.duration_steps

    @property
    def remaining_steps(self) -> int:
        return max(0, self.duration_steps - self.progress_steps)

    def slack(self, current_step: int) -> int:
        """Slack time (steps) before deadline violation."""
        return self.due_step - (current_step + self.remaining_steps)

    def is_late(self, current_step: int) -> bool:
        """True if past due date without completion, or completed after due step."""
        if self.completed_step is not None:
            return self.completed_step > self.due_step
        return current_step > self.due_step


class JobQueue:
    """Manages pool of pending and active jobs in the factory floor."""

    def __init__(self, quota: int = 30, random_seed: Optional[int] = None):
        self.quota = quota
        self.rng = np.random.default_rng(random_seed)
        self.all_jobs: Dict[str, Job] = {}
        self.pending_queue: List[str] = []
        self.active_jobs: Dict[str, str] = {} # machine_id -> job_id
        self.completed_jobs: List[str] = []

    def reset(self, initial_jobs: Optional[List[Job]] = None, quota: Optional[int] = None):
        if quota is not None:
            self.quota = quota
        self.all_jobs.clear()
        self.pending_queue.clear()
        self.active_jobs.clear()
        self.completed_jobs.clear()

        if initial_jobs:
            for job in initial_jobs:
                self.add_job(job)

    def add_job(self, job: Job):
        self.all_jobs[job.job_id] = job
        if not job.is_completed:
            self.pending_queue.append(job.job_id)

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.all_jobs.get(job_id)

    def pop_next_job(self, machine_id: str, current_step: int, allowed_types: Optional[List[str]] = None) -> Optional[Job]:
        """Pops next candidate job for a machine, prioritizing slack / urgency and priority."""
        candidates = []
        for jid in self.pending_queue:
            job = self.all_jobs[jid]
            if job.release_step <= current_step:
                if allowed_types is None or job.job_type in allowed_types:
                    candidates.append(job)

        if not candidates:
            return None

        # Sort candidates: smallest slack first, then highest priority
        candidates.sort(key=lambda j: (j.slack(current_step), -j.priority))
        chosen = candidates[0]
        self.pending_queue.remove(chosen.job_id)
        chosen.assigned_machine_id = machine_id
        chosen.started_step = current_step
        self.active_jobs[machine_id] = chosen.job_id
        return chosen

    def mark_completed(self, job_id: str, step: int):
        if job_id in self.all_jobs:
            job = self.all_jobs[job_id]
            job.progress_steps = job.duration_steps
            job.completed_step = step
            if job.assigned_machine_id and job.assigned_machine_id in self.active_jobs:
                del self.active_jobs[job.assigned_machine_id]
            if job_id in self.pending_queue:
                self.pending_queue.remove(job_id)
            if job_id not in self.completed_jobs:
                self.completed_jobs.append(job_id)

    @property
    def num_completed(self) -> int:
        return len(self.completed_jobs)

    @property
    def quota_achieved(self) -> bool:
        return self.num_completed >= self.quota

    def min_slack(self, current_step: int) -> float:
        """Computes minimum slack among all uncompleted jobs."""
        uncompleted = [j for j in self.all_jobs.values() if not j.is_completed]
        if not uncompleted:
            return 96.0 # Max slack
        return float(min(j.slack(current_step) for j in uncompleted))

    def late_count(self, current_step: int) -> int:
        """Count of jobs that are currently late or completed late."""
        return sum(1 for j in self.all_jobs.values() if j.is_late(current_step))

