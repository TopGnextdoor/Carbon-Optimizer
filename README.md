# ⚡ Carbon Optimizer
### AI-Based Data Center Carbon-Aware GPU Workload Scheduling Simulator

> **A Hardware-Agnostic Carbon-Aware GPU Scheduling Simulation Engine Designed for Future Integration with AMD ROCm Telemetry.**

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red) ![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-orange) ![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Key Features](#2-key-features)
3. [Architecture](#3-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Installation & Setup](#5-installation--setup)
6. [Usage Guide](#6-usage-guide)
7. [Metrics Reference](#7-metrics-reference)
8. [SLA & Scheduling Constraints](#8-sla--scheduling-constraints)
9. [Extending the Project](#9-extending-the-project)
10. [Design Decisions & Limitations](#10-design-decisions--limitations)
11. [Positioning & Roadmap](#11-positioning--roadmap)

---

## 1. Overview

Carbon Optimizer is a fully self-contained, hardware-agnostic simulation engine that demonstrates how carbon-aware scheduling can reduce the carbon footprint of GPU-intensive machine learning workloads in a data center environment.

The system simulates a 10-node heterogeneous GPU cluster, generates a realistic ML job queue, predicts short-term workload demand using a PyTorch LSTM model, and compares two scheduling strategies — a **baseline FCFS scheduler** and a **carbon-aware scheduler** — across real-world grid carbon intensity profiles for six regions.

All data is synthetically generated. No external APIs, hardware access, or cloud credentials are required. The entire application runs offline with a single command.

```
Carbon (kg) = Energy (kWh) × Carbon Intensity (gCO₂/kWh) ÷ 1000
```

---

## 2. Key Features

### 2.1 Heterogeneous GPU Cluster Simulation

Ten real-world GPU specifications are modelled, spanning consumer, prosumer, and enterprise tiers:

| GPU Model | TFLOPS (FP32) | TDP (W) | VRAM (GB) | Efficiency (TFLOPS/W) |
|---|---|---|---|---|
| NVIDIA A100 80GB | 77.97 | 400 | 80 | 0.1949 |
| NVIDIA A100 40GB | 77.97 | 300 | 40 | 0.2599 |
| NVIDIA V100 32GB | 14.13 | 300 | 32 | 0.0471 |
| NVIDIA V100 16GB | 14.13 | 250 | 16 | 0.0565 |
| NVIDIA A40 | 37.42 | 300 | 48 | 0.1247 |
| NVIDIA RTX 3090 | 35.58 | 350 | 24 | 0.1017 |
| NVIDIA RTX 4090 | 82.58 | 450 | 24 | 0.1835 |
| AMD MI250X | 95.70 | 500 | 128 | 0.1914 |
| AMD RX 7900 XTX | 61.44 | 355 | 24 | 0.1730 |
| NVIDIA T4 | 8.14 | 70 | 16 | 0.1163 |

**Power model:** `P(u) = P_idle + (TDP − P_idle) × u^1.4` — a realistic non-linear curve where idle power is ~12% of TDP. Job duration on any GPU scales proportionally to its TFLOPS relative to the V100 16GB reference.

---

### 2.2 ML Job Queue Generator

Thirty ML jobs are generated at simulation start with randomised parameters drawn from ten realistic workload templates:

| Workload Type | Priority | Duration Range | VRAM Required |
|---|---|---|---|
| LLM Fine-Tuning | High | 60–240 min | 24 GB |
| Image Classification | Medium | 10–60 min | 8 GB |
| Object Detection | Medium | 20–90 min | 16 GB |
| GAN Training | High | 90–300 min | 16 GB |
| NLP BERT Pretraining | High | 120–480 min | 32 GB |
| RL Simulation | Low | 30–120 min | 12 GB |
| Diffusion Model Training | High | 60–180 min | 24 GB |
| Time-Series LSTM | Low | 10–40 min | 8 GB |
| ResNet Training | Medium | 15–80 min | 16 GB |
| Transformer Inference | Low | 5–20 min | 8 GB |

Each job carries: a unique ID, submission time, deadline (priority-scaled headroom × duration + jitter), VRAM requirement, and compute intensity (fraction of GPU TFLOPS utilised).

---

### 2.3 Carbon Intensity Model

A physics-inspired diurnal model simulates real-world grid carbon intensity for six regions. Each region has a unique base intensity, amplitude, and phase offset:

| Region | Base (gCO₂/kWh) | Amplitude | Profile |
|---|---|---|---|
| Texas (ERCOT) | 420 | 160 | High fossil dependency, strong solar midday dip |
| California (CAISO) | 240 | 110 | Significant solar + renewables penetration |
| UK (National Grid) | 210 | 90 | Mixed gas/wind, moderate intensity |
| Germany (ENTSO-E) | 320 | 130 | Coal + wind mix, notable diurnal swing |
| France (RTE) | 85 | 40 | Nuclear-dominant, very low and stable |
| Australia (NEM) | 580 | 180 | Coal-heavy, highest baseline intensity |

The model combines a Gaussian solar generation dip centred at 13:00, an evening demand peak at 19:00, and per-slot random noise for continuity. Intensity is clamped to a realistic range of 10–1200 gCO₂/kWh.

---

### 2.4 LSTM Workload Predictor

A two-layer PyTorch LSTM is trained on synthetically generated historical GPU cluster utilisation data to predict future load. The model helps the carbon-aware scheduler assess whether deferring a job to a lower-carbon window is worthwhile.

| Parameter | Value |
|---|---|
| Architecture | LSTM → FC(64 → 32 → 1) |
| Input features | `time_sin`, `time_cos`, `normalised_utilisation` |
| Hidden size | 64 units |
| LSTM layers | 2 (dropout = 0.2) |
| Sequence length | 12 timesteps |
| Optimiser | Adam (lr = 1e-3) |
| Loss function | Mean Squared Error (MSE) |
| Training data | Synthetic 24h profile at 5-min resolution |

---

### 2.5 Scheduling Algorithms

#### Baseline Scheduler (FCFS + Priority)

The baseline scheduler processes jobs in order of priority (high → medium → low) then submission time. It assigns the first available GPU that satisfies the VRAM requirement, with no regard for carbon intensity or energy efficiency. This serves as the benchmark.

#### Carbon-Aware Scheduler

The carbon-aware scheduler introduces three optimisation levers:

- **Efficient GPU preference** — Available GPUs are ranked by efficiency (TFLOPS/W) and the most efficient eligible GPU is selected first, reducing energy consumption per unit of compute.
- **Carbon-window deferral** — Low and medium-priority jobs are deferred if the current carbon intensity exceeds the 60-minute forecast average by more than 5%, shifting load toward greener windows.
- **Deadline urgency override** — If a job must complete within 1.5× its expected runtime, or it carries high priority, it schedules immediately regardless of current carbon intensity to protect SLA.

Carbon deferral score: `score = carbon_now − carbon_if_deferred` — a negative score signals deferral is advantageous.

---

## 3. Architecture

### 3.1 Folder Structure

```
carbon_optimizer/
├── app.py                    # Main Streamlit dashboard
├── requirements.txt          # Python dependencies
├── simulator/
│   ├── __init__.py
│   ├── gpu.py                # GPU dataclass, power model, cluster factory
│   ├── workload.py           # Job dataclass, job queue generator
│   └── carbon.py             # Diurnal carbon intensity model
├── models/
│   ├── __init__.py
│   └── predictor.py          # PyTorch LSTM trainer & inference
├── optimizer/
│   ├── __init__.py
│   └── scheduler.py          # Baseline + carbon-aware schedulers
└── utils/
    ├── __init__.py
    └── metrics.py            # SimulationMetrics dataclass & comparison
```

---

### 3.2 Module Responsibilities

| Module | Responsibility |
|---|---|
| `simulator/gpu.py` | Defines the `GPU` dataclass with a non-linear power curve. Exposes `create_gpu_cluster()` which instantiates all 10 GPUs. |
| `simulator/workload.py` | Defines the `Job` dataclass. `generate_job_queue()` produces a seeded, reproducible job queue with heterogeneous workload types, priorities, deadlines, and compute requirements. |
| `simulator/carbon.py` | `CarbonIntensityModel` simulates regional grid carbon intensity using Gaussian solar/demand curves with per-slot noise. Provides `intensity_at(t)`, `get_forecast()`, and `get_full_day_profile()`. |
| `models/predictor.py` | `WorkloadLSTM` is a 2-layer LSTM PyTorch module. `WorkloadPredictor` wraps it with `train()` and `predict()` methods. `generate_training_data()` creates synthetic historical utilisation for bootstrapped training. |
| `optimizer/scheduler.py` | `run_simulation()` advances the simulation in 5-minute ticks, handling job arrivals, completions, and scheduling decisions. Supports `'baseline'` and `'carbon_aware'` modes. Returns a `SchedulerResult` with full timeline, jobs, and GPU state. |
| `utils/metrics.py` | `SimulationMetrics.compute()` aggregates energy, carbon, SLA, latency, and throughput from a completed run. `compare_to()` produces percentage and absolute deltas vs. a baseline. |
| `app.py` | Streamlit dashboard with sidebar controls, pre-simulation GPU cluster and carbon profile views, and a 5-tab result dashboard (Comparison, Timeline, LSTM Predictor, GPU Analysis, Job Table). |

---

### 3.3 Data Flow

```
User configures parameters (region, job count, duration, seed)
        │
        ▼
WorkloadPredictor trains LSTM on synthetic utilisation history
        │
        ▼
generate_job_queue() + CarbonIntensityModel initialise simulation inputs
        │
        ├──► run_simulation('baseline')
        │         │
        │         └── Per 5-min tick:
        │               • New jobs arrive
        │               • Completed jobs released from GPUs
        │               • FCFS scheduler assigns pending jobs
        │               • All GPUs accumulate energy + carbon
        │
        └──► run_simulation('carbon_aware')
                  │
                  └── Per 5-min tick:
                        • New jobs arrive
                        • Completed jobs released from GPUs
                        • Carbon-aware scheduler (efficiency + deferral + urgency)
                        • All GPUs accumulate energy + carbon
                              │
                              ▼
                  SimulationMetrics.compute() → KPIs for both runs
                              │
                              ▼
                  Streamlit renders charts + comparison tables
```

---

## 4. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend / UI | Streamlit ≥ 1.32 | Web dashboard, sidebar controls, multi-tab layout |
| Visualisation | Plotly ≥ 5.19 | Interactive charts: bar, scatter, time-series, subplots |
| Machine Learning | PyTorch ≥ 2.2 | LSTM workload predictor — training, inference, gradient descent |
| Data Processing | Pandas ≥ 2.2 | DataFrames for job logs, GPU tables, comparison tables |
| Numerical Simulation | NumPy ≥ 1.26 | Power curves, carbon models, time-series generation |
| Simulation Engine | Pure Python 3.11 | GPU cluster, scheduler, job queue — zero external hardware |
| Styling | Custom CSS | Dark industrial theme with Share Tech Mono + Barlow fonts |

---

## 5. Installation & Setup

### 5.1 Prerequisites

- Python 3.11 or higher
- pip (bundled with Python)
- 4 GB RAM minimum (LSTM training is CPU-based)
- No GPU required — all simulation is software-only

### 5.2 Quick Start

```bash
# Clone or unzip the project
cd carbon_optimizer

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

### 5.3 Dependencies

```
streamlit>=1.32.0
torch>=2.2.0
numpy>=1.26.0
pandas>=2.2.0
plotly>=5.19.0
scikit-learn>=1.4.0
```

### 5.4 Platform Notes

| Platform | Notes |
|---|---|
| macOS / Linux | Runs natively. Use `pip3` if Python 3 is not the default. |
| Windows | Use `pip install` and `streamlit run` normally. No WSL required. |
| Docker | Base image: `python:3.11-slim`. Install requirements and expose port `8501`. |
| Google Colab | Install with `!pip install`, then use `streamlit` with an ngrok tunnel. |

---

## 6. Usage Guide

### 6.1 Sidebar Configuration

| Parameter | Default | Range | Description |
|---|---|---|---|
| Grid Region | Texas (ERCOT) | 6 regions | Selects carbon intensity profile for the simulation |
| Number of ML Jobs | 30 | 10–60 | Total jobs submitted to the scheduler |
| Simulation Duration | 8 hours | 2–12 hours | Total simulated time window |
| Random Seed | 42 | Any integer | Ensures fully reproducible results |
| LSTM Training Epochs | 20 | 5–50 | Training iterations for the workload predictor |

### 6.2 Dashboard Tabs

| Tab | Contents |
|---|---|
| 📊 Comparison | Side-by-side bar charts, percentage improvement chart, full metrics comparison table |
| 🕐 Timeline | 3-panel time-series: carbon intensity, active jobs, and total power draw across the simulation |
| 🔬 LSTM Predictor | Training loss curve, actual vs. predicted utilisation plot, model architecture summary |
| 🖥️ GPU Analysis | Per-GPU energy and carbon bar charts, efficiency vs. carbon scatter plot |
| 📋 Job Table | Full job execution log with start/complete times, energy, carbon, GPU assignment, and SLA status |

### 6.3 Interpreting Results

- **Carbon Savings %** — the primary headline metric. Positive values mean the carbon-aware scheduler emitted less CO₂.
- **SLA Violation Rate** — must stay below 5% for the carbon-aware scheduler to be considered valid. The urgency override is the primary mechanism keeping this in check.
- **Carbon Efficiency (jobs/kg CO₂)** — a composite throughput-per-emission metric. Higher is better.
- **Timeline panel** — look for the carbon-aware job curve shifting away from peaks in the carbon intensity trace.

---

## 7. Metrics Reference

| Metric | Formula / Definition | Unit |
|---|---|---|
| Total Energy | `Σ (P(u) × dt)` over all GPU ticks | kWh |
| Total Carbon | `Σ (energy_step × carbon_intensity) ÷ 1000` | kg CO₂eq |
| Avg Carbon Intensity | Mean of `carbon_intensity` across all timeline ticks | gCO₂/kWh |
| SLA Violation Rate | Jobs completed after deadline ÷ total jobs | % |
| Avg Job Latency | Mean of `(completed_at − submitted_at)` over completed jobs | minutes |
| Peak Power | Max total cluster power draw across all ticks | kW |
| Avg GPU Utilisation | Mean of per-GPU utilisation across all ticks | fraction (0–1) |
| Carbon Efficiency | Jobs completed ÷ total carbon emitted | jobs / kg CO₂ |
| Carbon Savings | `baseline_carbon − aware_carbon` | kg CO₂ / % |

---

## 8. SLA & Scheduling Constraints

The simulator enforces the following hard and soft constraints at every scheduling decision:

- **SLA target** — fewer than 5% of jobs may violate their deadline. The carbon-aware scheduler monitors urgency to prevent cascading SLA failures.
- **GPU overload prevention** — each GPU holds at most one job simultaneously. Jobs queue until a compatible GPU (sufficient VRAM) becomes available.
- **VRAM matching** — a job may only be assigned to a GPU with equal or greater VRAM capacity than its requirement.
- **Urgency override** — any job within 1.5× its expected runtime of its deadline, or marked high priority, bypasses carbon deferral and schedules immediately.
- **Priority ordering** — high → medium → low. Same-priority jobs are sorted by submission time (FCFS within tier).

---

## 9. Extending the Project

### 9.1 Adding Real Hardware Telemetry

The `GPU` dataclass in `simulator/gpu.py` is designed to accept live telemetry. To integrate AMD ROCm or NVIDIA NVML:

- Replace `create_gpu_cluster()` with a function that polls `rocm-smi` or `pynvml` for real utilisation and power.
- Feed live power readings into `gpu.tick()` in place of the modelled power curve.
- No changes to the scheduler or metrics logic are required.

```python
# Example: replace power model with live NVML reading
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu.gpu_id)
live_power_watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
```

### 9.2 Adding Real Carbon Data

The `CarbonIntensityModel` can be replaced with a live API client:

- [Electricity Maps API](https://electricitymaps.com) provides real-time and forecast carbon intensity globally.
- [WattTime API](https://watttime.org) offers marginal emissions data for US regions.
- Implement `intensity_at(t)` and `get_forecast(t, horizon)` with cached API responses to respect rate limits.

### 9.3 Advanced Scheduling Strategies

The `scheduler.py` module is structured to allow plug-in strategies. Possible extensions:

- Multi-GPU job parallelism (data-parallel training across 2–8 GPUs).
- Preemptible jobs with checkpointing to pause/resume at carbon intensity peaks.
- Reinforcement learning agent that learns optimal deferral policies from historical traces.
- Deadline-aware bin-packing for higher GPU utilisation density.

### 9.4 Improving the LSTM Predictor

- Replace synthetic training data with real historical cluster logs (e.g. from Kubernetes metrics or Prometheus).
- Add carbon intensity as an additional input feature for joint prediction.
- Experiment with Temporal Fusion Transformers (TFT) for longer forecast horizons.
- Add uncertainty estimation via MC Dropout to make deferral decisions probabilistic.

---

## 10. Design Decisions & Limitations

### 10.1 Design Decisions

| Decision | Rationale |
|---|---|
| 5-minute simulation tick | Balances resolution and runtime. Fine enough to capture carbon intensity variation; coarse enough for fast execution. |
| Synthetic but realistic data | Enables fully offline execution, reproducible benchmarks, and safe demonstrations without API costs or credentials. |
| Non-linear power model (`u^1.4`) | GPU power draw is superlinear with utilisation. Linear models underestimate energy at high load. |
| PyTorch LSTM over statistical models | Demonstrates a production-grade ML pipeline within the project. Kept small intentionally for CPU-only training speed. |
| VRAM as primary resource constraint | VRAM is the most common binding constraint in ML workloads; compute can be time-multiplexed but VRAM cannot. |
| Seeded randomness throughout | Every component accepts a `seed` parameter, ensuring results are fully reproducible across runs. |

### 10.2 Known Limitations

- The simulation assumes single-GPU jobs only — multi-GPU parallelism is not modelled.
- Carbon intensity uses a smooth diurnal model; real grids exhibit more irregular, event-driven spikes.
- The LSTM is trained on synthetic utilisation patterns; real cluster logs would yield more accurate predictions.
- GPU-to-GPU memory transfer costs and PCIe bandwidth are not modelled.
- Cooling overhead and PUE (Power Usage Effectiveness) are not included in the energy model.
- The scheduler does not yet support preemption — once a job starts, it runs to completion.

---

## 11. Positioning & Roadmap

> A Hardware-Agnostic Carbon-Aware GPU Scheduling Simulation Engine Designed for Future Integration with AMD ROCm Telemetry.

### Near-Term Roadmap

- AMD ROCm telemetry integration via `rocm-smi` Python bindings.
- Live Electricity Maps API connector for real-time carbon intensity.
- Multi-GPU job support with data-parallel scheduling.
- Persistent run history and simulation comparison across seeds and regions.

### Long-Term Roadmap

- Reinforcement learning scheduler trained on real cluster traces.
- Kubernetes operator for carbon-aware pod scheduling in production clusters.
- Carbon budget enforcement: hard cap on kg CO₂ per billing period.
- Integration with MLflow / Weights & Biases for carbon-annotated experiment tracking.
- PUE-aware energy model incorporating cooling efficiency by region and season.

---

*Carbon Optimizer MVP · Python 3.11 / Streamlit / PyTorch · Hardware-Agnostic · AMD ROCm-Ready*
