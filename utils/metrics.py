"""Energy, carbon, and SLA metrics calculations."""
import numpy as np
from dataclasses import dataclass


@dataclass
class SimulationMetrics:
    total_energy_kwh: float
    total_carbon_kg: float
    avg_carbon_intensity: float
    sla_violation_rate: float
    jobs_completed: int
    jobs_total: int
    avg_job_latency_min: float
    peak_power_kw: float
    avg_gpu_utilization: float
    carbon_efficiency: float  # jobs per kg CO2

    @classmethod
    def compute(cls, jobs: list, gpus: list, timeline: list) -> "SimulationMetrics":
        completed = [j for j in jobs if j.completed_at is not None]
        total_energy = sum(g.total_energy_kwh for g in gpus)
        total_carbon = sum(g.carbon_emitted_kg for g in gpus)
        n_violated = sum(1 for j in jobs if j.sla_violated)
        sla_rate = n_violated / len(jobs) if jobs else 0.0

        latencies = []
        for j in completed:
            if j.started_at is not None:
                latencies.append(j.completed_at - j.submitted_at)

        avg_intensity = np.mean([t["carbon_intensity"] for t in timeline]) if timeline else 0.0
        peak_power = max((t["total_power_kw"] for t in timeline), default=0.0)
        avg_util = np.mean([t["avg_utilization"] for t in timeline]) if timeline else 0.0

        return cls(
            total_energy_kwh=total_energy,
            total_carbon_kg=total_carbon,
            avg_carbon_intensity=avg_intensity,
            sla_violation_rate=sla_rate,
            jobs_completed=len(completed),
            jobs_total=len(jobs),
            avg_job_latency_min=float(np.mean(latencies)) if latencies else 0.0,
            peak_power_kw=peak_power,
            avg_gpu_utilization=avg_util,
            carbon_efficiency=len(completed) / total_carbon if total_carbon > 0 else 0.0,
        )

    def compare_to(self, other: "SimulationMetrics") -> dict:
        """Compare this result vs another (baseline)."""
        def pct_change(new, old):
            if old == 0:
                return 0.0
            return (new - old) / old * 100

        return {
            "energy_savings_pct": pct_change(other.total_energy_kwh, self.total_energy_kwh),
            "carbon_savings_pct": pct_change(other.total_carbon_kg, self.total_carbon_kg),
            "energy_savings_kwh": other.total_energy_kwh - self.total_energy_kwh,
            "carbon_savings_kg": other.total_carbon_kg - self.total_carbon_kg,
            "sla_delta_pct": (self.sla_violation_rate - other.sla_violation_rate) * 100,
            "latency_delta_min": self.avg_job_latency_min - other.avg_job_latency_min,
        }
