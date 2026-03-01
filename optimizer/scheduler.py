"""Baseline and carbon-aware GPU schedulers."""
import numpy as np
from copy import deepcopy
from typing import Optional

from simulator.gpu import GPU, create_gpu_cluster
from simulator.workload import Job
from simulator.carbon import CarbonIntensityModel
from utils.metrics import SimulationMetrics


class SchedulerResult:
    def __init__(self, name: str):
        self.name = name
        self.jobs: list[Job] = []
        self.gpus: list[GPU] = []
        self.timeline: list[dict] = []  # per-timestep records
        self.metrics: Optional[SimulationMetrics] = None


def _find_eligible_gpu(job: Job, gpus: list[GPU], prefer_efficient: bool = False) -> Optional[GPU]:
    """Find an available GPU that can run this job."""
    candidates = [g for g in gpus if g.is_available() and g.vram_gb >= job.vram_required_gb]
    if not candidates:
        return None
    if prefer_efficient:
        # Sort by efficiency (TFLOPS/watt) descending
        candidates.sort(key=lambda g: g.efficiency_ratio, reverse=True)
    else:
        # FCFS: just pick first available (sorted by gpu_id)
        candidates.sort(key=lambda g: g.gpu_id)
    return candidates[0]


def _carbon_score(job: Job, gpu: GPU, current_intensity: float, forecast_intensities: list) -> float:
    """
    Carbon score for scheduling job on gpu now vs. deferring.
    Lower = better to schedule now.
    """
    actual_duration = job.actual_duration_on_gpu(gpu.compute_tflops)
    avg_future_intensity = np.mean(forecast_intensities) if forecast_intensities else current_intensity
    # Energy if scheduled now
    power_draw = gpu.power_at_utilization(job.compute_intensity)
    energy_now = (power_draw / 1000.0) * (actual_duration / 60.0)
    carbon_now = energy_now * current_intensity
    # Estimated carbon if deferred (use future average intensity)
    carbon_future_est = energy_now * avg_future_intensity
    # Urgency factor: penalize if deadline is tight
    return carbon_now - carbon_future_est  # negative = defer is better


def run_simulation(
    jobs_template: list[Job],
    scheduler_type: str,
    carbon_model: CarbonIntensityModel,
    sim_duration_minutes: float = 480,
    tick_minutes: float = 5.0,
    seed: int = 42,
) -> SchedulerResult:
    """
    Run a full scheduling simulation.
    scheduler_type: 'baseline' | 'carbon_aware'
    """
    # Deep copy jobs and create fresh GPUs
    jobs = deepcopy(jobs_template)
    gpus = create_gpu_cluster()
    result = SchedulerResult(scheduler_type)

    pending_jobs = [j for j in jobs if j.submitted_at <= 0]
    future_jobs = [j for j in jobs if j.submitted_at > 0]
    running_jobs: list[Job] = []
    completed_jobs: list[Job] = []

    current_time = 0.0
    dt = tick_minutes

    while current_time <= sim_duration_minutes:
        # Add newly submitted jobs
        newly_arrived = [j for j in future_jobs if j.submitted_at <= current_time]
        pending_jobs.extend(newly_arrived)
        future_jobs = [j for j in future_jobs if j.submitted_at > current_time]

        # Complete running jobs
        for job in list(running_jobs):
            gpu = gpus[job.assigned_gpu_id]
            actual_dur = job.actual_duration_on_gpu(gpu.compute_tflops)
            if job.started_at + actual_dur <= current_time:
                job.completed_at = current_time
                if job.completed_at > job.deadline_minutes:
                    job.sla_violated = True
                gpu.release_job()
                running_jobs.remove(job)
                completed_jobs.append(job)

        carbon_intensity = carbon_model.intensity_at(current_time)

        # Schedule pending jobs
        if scheduler_type == "baseline":
            # First-Come-First-Served on first available GPU
            priority_order = {"high": 0, "medium": 1, "low": 2}
            pending_jobs.sort(key=lambda j: (priority_order[j.priority], j.submitted_at))
            for job in list(pending_jobs):
                gpu = _find_eligible_gpu(job, gpus, prefer_efficient=False)
                if gpu:
                    gpu.assign_job(job.job_id, job.compute_intensity)
                    job.started_at = current_time
                    job.assigned_gpu_id = gpu.gpu_id
                    pending_jobs.remove(job)
                    running_jobs.append(job)

        elif scheduler_type == "carbon_aware":
            # Carbon-aware: prefer low-carbon windows + efficient GPUs
            _, forecast_intensities = carbon_model.get_forecast(current_time, horizon_minutes=60)
            forecast_list = list(forecast_intensities)
            priority_order = {"high": 0, "medium": 1, "low": 2}
            pending_jobs.sort(key=lambda j: (priority_order[j.priority], j.submitted_at))

            for job in list(pending_jobs):
                urgency = job.deadline_minutes - current_time
                time_to_run = job.duration_minutes * 1.5
                must_schedule = urgency <= time_to_run or job.priority == "high"

                gpu = _find_eligible_gpu(job, gpus, prefer_efficient=True)
                if not gpu:
                    continue

                score = _carbon_score(job, gpu, carbon_intensity, forecast_list)
                # Schedule if: carbon is favorable now, or urgency demands it
                if must_schedule or carbon_intensity < np.mean(forecast_list) * 1.05:
                    gpu.assign_job(job.job_id, job.compute_intensity)
                    job.started_at = current_time
                    job.assigned_gpu_id = gpu.gpu_id
                    pending_jobs.remove(job)
                    running_jobs.append(job)

        # Tick all GPUs
        for gpu in gpus:
            gpu.tick(dt / 60.0, carbon_intensity)
            if gpu.assigned_job_id is not None:
                job = next((j for j in running_jobs if j.job_id == gpu.assigned_job_id), None)
                if job:
                    energy_step = (gpu.power_at_utilization(gpu.current_utilization) / 1000.0) * (dt / 60.0)
                    job.energy_kwh += energy_step
                    job.carbon_cost_kg += energy_step * carbon_intensity / 1000.0

        # Record timeline snapshot
        active_count = len(running_jobs)
        total_util = np.mean([g.current_utilization for g in gpus])
        total_power = sum(g.power_at_utilization(g.current_utilization) for g in gpus)
        result.timeline.append({
            "time_min": current_time,
            "carbon_intensity": carbon_intensity,
            "active_jobs": active_count,
            "avg_utilization": total_util,
            "total_power_kw": total_power / 1000.0,
            "pending_jobs": len(pending_jobs),
        })

        current_time += dt

    # Mark any still-running or pending jobs as SLA violated
    for job in running_jobs + pending_jobs:
        if current_time > job.deadline_minutes:
            job.sla_violated = True
        completed_jobs.append(job)

    result.jobs = completed_jobs
    result.gpus = gpus

    from utils.metrics import SimulationMetrics
    result.metrics = SimulationMetrics.compute(result.jobs, result.gpus, result.timeline)
    return result
