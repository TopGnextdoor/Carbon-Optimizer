"""Workload and ML job generation for simulation."""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import random


JOB_TYPES = [
    ("LLM Fine-Tuning",     "high",   60, 240, 24),   # (name, priority, min_min, max_min, vram_req_gb)
    ("Image Classification","medium", 10,  60,  8),
    ("Object Detection",    "medium", 20,  90, 16),
    ("GAN Training",        "high",   90, 300, 16),
    ("NLP BERT Pretraining","high",   120,480, 32),
    ("RL Simulation",       "low",    30, 120, 12),
    ("Diffusion Model",     "high",   60, 180, 24),
    ("Time-Series LSTM",    "low",    10,  40,  8),
    ("ResNet Training",     "medium", 15,  80, 16),
    ("Transformer Inference","low",    5,  20,  8),
]

PRIORITY_SLA_HEADROOM = {"high": 1.1, "medium": 1.3, "low": 1.8}


@dataclass
class Job:
    job_id: int
    name: str
    priority: str
    duration_minutes: float       # actual compute time on a reference GPU
    deadline_minutes: float       # hard deadline from submission
    vram_required_gb: float
    compute_intensity: float      # fraction of GPU TFLOPS needed
    submitted_at: float = 0.0     # simulation time (minutes)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    assigned_gpu_id: Optional[int] = None
    sla_violated: bool = False
    carbon_cost_kg: float = 0.0
    energy_kwh: float = 0.0

    @property
    def is_pending(self):
        return self.started_at is None

    @property
    def is_running(self):
        return self.started_at is not None and self.completed_at is None

    @property
    def is_done(self):
        return self.completed_at is not None

    def actual_duration_on_gpu(self, gpu_tflops: float, reference_tflops: float = 14.13) -> float:
        """Scale duration based on GPU TFLOPS vs reference (V100 16GB)."""
        speedup = gpu_tflops / reference_tflops
        return self.duration_minutes / speedup


def generate_job_queue(n_jobs: int = 30, seed: int = 42) -> list[Job]:
    """Generate a realistic ML job queue."""
    rng = np.random.default_rng(seed)
    jobs = []
    # Submit jobs spread over a 4-hour window
    submission_times = np.sort(rng.uniform(0, 240, n_jobs))

    for i in range(n_jobs):
        jtype = JOB_TYPES[rng.integers(0, len(JOB_TYPES))]
        name, priority, min_dur, max_dur, vram = jtype
        duration = float(rng.uniform(min_dur, max_dur))
        headroom = PRIORITY_SLA_HEADROOM[priority]
        deadline = submission_times[i] + duration * headroom + rng.uniform(10, 60)
        compute_intensity = float(rng.uniform(0.55, 0.95))

        job = Job(
            job_id=i,
            name=f"{name} #{i}",
            priority=priority,
            duration_minutes=duration,
            deadline_minutes=deadline,
            vram_required_gb=float(vram),
            compute_intensity=compute_intensity,
            submitted_at=float(submission_times[i]),
        )
        jobs.append(job)
    return jobs
