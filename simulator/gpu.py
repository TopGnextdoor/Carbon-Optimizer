"""GPU simulation logic for heterogeneous GPU cluster."""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPU:
    gpu_id: int
    model: str
    compute_tflops: float      # FP32 TFLOPS
    tdp_watts: float           # Thermal Design Power
    vram_gb: float
    efficiency_ratio: float    # TFLOPS per watt
    current_utilization: float = 0.0
    current_power_draw: float = 0.0
    assigned_job_id: Optional[int] = None
    total_energy_kwh: float = 0.0
    carbon_emitted_kg: float = 0.0
    idle_power_watts: float = 0.0

    def __post_init__(self):
        self.idle_power_watts = self.tdp_watts * 0.12  # ~12% idle power

    def power_at_utilization(self, utilization: float) -> float:
        """Realistic power model: idle + (TDP - idle) * util^1.4"""
        return self.idle_power_watts + (self.tdp_watts - self.idle_power_watts) * (utilization ** 1.4)

    def is_available(self) -> bool:
        return self.assigned_job_id is None

    def assign_job(self, job_id: int, utilization: float):
        self.assigned_job_id = job_id
        self.current_utilization = utilization
        self.current_power_draw = self.power_at_utilization(utilization)

    def release_job(self):
        self.assigned_job_id = None
        self.current_utilization = 0.0
        self.current_power_draw = self.idle_power_watts

    def tick(self, dt_hours: float, carbon_intensity: float):
        """Advance simulation by dt_hours, accumulate energy and carbon."""
        power = self.power_at_utilization(self.current_utilization)
        energy_kwh = (power / 1000.0) * dt_hours
        self.total_energy_kwh += energy_kwh
        self.carbon_emitted_kg += energy_kwh * carbon_intensity / 1000.0  # intensity in gCO2/kWh → kg


GPU_CATALOG = [
    # (model, tflops, tdp_watts, vram_gb)
    ("NVIDIA A100 80GB", 77.97, 400, 80),
    ("NVIDIA A100 40GB", 77.97, 300, 40),
    ("NVIDIA V100 32GB", 14.13, 300, 32),
    ("NVIDIA V100 16GB", 14.13, 250, 16),
    ("NVIDIA A40",       37.42, 300, 48),
    ("NVIDIA RTX 3090",  35.58, 350, 24),
    ("NVIDIA RTX 4090",  82.58, 450, 24),
    ("AMD MI250X",       95.70, 500, 128),
    ("AMD RX 7900 XTX",  61.44, 355, 24),
    ("NVIDIA T4",        8.14,  70,  16),
]


def create_gpu_cluster() -> list[GPU]:
    """Create 10 heterogeneous GPUs."""
    gpus = []
    for i, (model, tflops, tdp, vram) in enumerate(GPU_CATALOG):
        efficiency = tflops / tdp
        gpu = GPU(
            gpu_id=i,
            model=model,
            compute_tflops=tflops,
            tdp_watts=tdp,
            vram_gb=vram,
            efficiency_ratio=efficiency,
        )
        gpus.append(gpu)
    return gpus
