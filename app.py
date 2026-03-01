"""
Carbon-Aware GPU Workload Scheduling Simulator
A Hardware-Agnostic Carbon-Aware GPU Scheduling Simulation Engine
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from simulator.gpu import create_gpu_cluster
from simulator.workload import generate_job_queue
from simulator.carbon import CarbonIntensityModel
from models.predictor import WorkloadPredictor
from optimizer.scheduler import run_simulation

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Carbon Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&family=Barlow+Condensed:wght@700;800&display=swap');

:root {
    --bg-deep:    #080c10;
    --bg-card:    #0d1520;
    --bg-panel:   #111d2b;
    --border:     #1a3a5c;
    --accent-g:   #00ff88;
    --accent-b:   #00aaff;
    --accent-r:   #ff4455;
    --accent-y:   #ffcc00;
    --text-pri:   #e8f4ff;
    --text-sec:   #6a9bc2;
    --text-dim:   #334f6a;
}

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: var(--bg-deep);
    color: var(--text-pri);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg-card);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text-pri) !important; }

/* Headers */
h1, h2, h3 { font-family: 'Barlow Condensed', sans-serif; letter-spacing: 0.05em; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 16px;
    box-shadow: 0 0 20px rgba(0,170,255,0.05);
}
div[data-testid="metric-container"] label { color: var(--text-sec) !important; font-size: 0.72rem !important; letter-spacing: 0.1em; text-transform: uppercase; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family: 'Share Tech Mono', monospace; font-size: 1.6rem !important; color: var(--accent-b) !important; }
div[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0a3d62, #1a5c8a);
    color: var(--accent-b);
    border: 1px solid var(--accent-b);
    border-radius: 4px;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 0.1em;
    font-size: 0.85rem;
    padding: 8px 20px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1a5c8a, #0a3d62);
    box-shadow: 0 0 15px rgba(0,170,255,0.3);
    color: #fff;
}

/* Selectbox / Slider labels */
.stSelectbox label, .stSlider label, .stNumberInput label {
    color: var(--text-sec) !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Dividers */
hr { border-color: var(--border); }

/* Title bar */
.title-bar {
    background: linear-gradient(90deg, #0a1929 0%, #0d2240 50%, #0a1929 100%);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 18px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 0 40px rgba(0,170,255,0.08);
}
.title-bar h1 { margin: 0; font-size: 1.9rem; color: var(--accent-b); letter-spacing: 0.12em; }
.title-bar p { margin: 0; color: var(--text-sec); font-size: 0.82rem; letter-spacing: 0.06em; }
.badge {
    background: rgba(0,255,136,0.12);
    border: 1px solid var(--accent-g);
    color: var(--accent-g);
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    padding: 3px 8px;
    border-radius: 3px;
    letter-spacing: 0.15em;
}

/* Section headers */
.section-header {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-sec);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
    margin: 20px 0 14px;
}

/* Status pills */
.pill-green { background: rgba(0,255,136,0.12); color: var(--accent-g); border: 1px solid var(--accent-g); padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; font-family: 'Share Tech Mono', monospace; }
.pill-red   { background: rgba(255,68,85,0.12);  color: var(--accent-r); border: 1px solid var(--accent-r); padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; font-family: 'Share Tech Mono', monospace; }
.pill-blue  { background: rgba(0,170,255,0.12);  color: var(--accent-b); border: 1px solid var(--accent-b); padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; font-family: 'Share Tech Mono', monospace; }
.pill-yellow{ background: rgba(255,204,0,0.12);  color: var(--accent-y); border: 1px solid var(--accent-y); padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; font-family: 'Share Tech Mono', monospace; }

/* Info box */
.info-box {
    background: var(--bg-panel);
    border-left: 3px solid var(--accent-b);
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    font-size: 0.85rem;
    color: var(--text-sec);
    margin: 8px 0;
}

/* Dataframe */
.stDataFrame { border: 1px solid var(--border); border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#080c10",
    plot_bgcolor="#0d1520",
    font=dict(family="Barlow, sans-serif", color="#6a9bc2"),
)

def fmt_number(v, decimals=2):
    return f"{v:,.{decimals}f}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙ SIMULATION PARAMETERS")
    st.markdown("---")

    region = st.selectbox(
        "Grid Region",
        list(CarbonIntensityModel.REGION_PRESETS.keys()),
        index=0,
    )
    n_jobs = st.slider("Number of ML Jobs", 10, 60, 30, step=5)
    sim_hours = st.slider("Simulation Duration (hours)", 2, 12, 8)
    seed = st.number_input("Random Seed", value=42, step=1)

    st.markdown("---")
    st.markdown("### 🧠 LSTM PREDICTOR")
    lstm_epochs = st.slider("Training Epochs", 5, 50, 20, step=5)

    st.markdown("---")
    st.markdown(
        '<div class="info-box">Carbon = Energy × CarbonIntensity<br>'
        'SLA Target: &lt; 5% violations</div>',
        unsafe_allow_html=True,
    )

    run_btn = st.button("▶  RUN SIMULATION", use_container_width=True)

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-bar">
  <div>
    <h1>⚡ CARBON OPTIMIZER</h1>
    <p>AI-Based Data Center Carbon-Aware GPU Workload Scheduling Simulator</p>
  </div>
  <div style="margin-left:auto; text-align:right">
    <div class="badge">AMD ROCm Ready</div><br>
    <div class="badge" style="margin-top:4px">Hardware-Agnostic</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

# ── Pre-simulation: show cluster info ─────────────────────────────────────────
if st.session_state.results is None:
    st.markdown('<div class="section-header">GPU CLUSTER — 10 HETEROGENEOUS NODES</div>', unsafe_allow_html=True)
    gpus = create_gpu_cluster()
    gpu_data = pd.DataFrame([{
        "GPU": g.model,
        "TFLOPS": g.compute_tflops,
        "TDP (W)": g.tdp_watts,
        "VRAM (GB)": g.vram_gb,
        "Efficiency (TFLOPS/W)": round(g.efficiency_ratio, 4),
    } for g in gpus])

    fig_cluster = go.Figure(go.Bar(
        x=gpu_data["GPU"],
        y=gpu_data["TFLOPS"],
        marker=dict(
            color=gpu_data["Efficiency (TFLOPS/W)"],
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="TFLOPS/W"),
        ),
        text=[f"{v:.1f}" for v in gpu_data["TFLOPS"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>TFLOPS: %{y:.1f}<br>Efficiency: %{marker.color:.4f}<extra></extra>",
    ))
    fig_cluster.update_layout(
        **PLOTLY_THEME,
        title="GPU Compute Performance (TFLOPS) — Color = Efficiency",
        xaxis_tickangle=-35,
        height=360,
        margin=dict(t=50, b=120),
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.markdown('<div class="section-header">CARBON INTENSITY FORECAST — 24H PROFILE</div>', unsafe_allow_html=True)
    carbon_model = CarbonIntensityModel(region=region)
    times, intensities = carbon_model.get_full_day_profile()
    hours = times / 60

    fig_carbon = go.Figure()
    fig_carbon.add_trace(go.Scatter(
        x=hours, y=intensities,
        mode="lines",
        fill="tozeroy",
        line=dict(color="#00aaff", width=2),
        fillcolor="rgba(0,170,255,0.08)",
        name="Carbon Intensity",
        hovertemplate="Hour %{x:.1f}: <b>%{y:.0f} gCO₂/kWh</b><extra></extra>",
    ))
    # Mark low-carbon window
    low_threshold = np.percentile(intensities, 25)
    low_mask = intensities <= low_threshold
    fig_carbon.add_hrect(y0=0, y1=low_threshold, fillcolor="rgba(0,255,136,0.05)",
                          line_width=0, annotation_text="Low Carbon Window",
                          annotation_position="top left",
                          annotation_font_color="#00ff88", annotation_font_size=10)
    fig_carbon.update_layout(
        **PLOTLY_THEME,
        title=f"Grid Carbon Intensity — {region}",
        xaxis_title="Hour of Day",
        yaxis_title="gCO₂eq / kWh",
        height=300,
        showlegend=False,
    )
    st.plotly_chart(fig_carbon, use_container_width=True)

    st.markdown(
        '<div class="info-box">👆 Configure simulation parameters in the sidebar and click <b>▶ RUN SIMULATION</b> to start.</div>',
        unsafe_allow_html=True,
    )

# ── Run simulation ─────────────────────────────────────────────────────────────
if run_btn:
    carbon_model = CarbonIntensityModel(region=region, seed=int(seed))
    jobs = generate_job_queue(n_jobs=n_jobs, seed=int(seed))

    progress_bar = st.progress(0, text="Initializing...")

    # Train LSTM predictor
    progress_bar.progress(10, text="Training LSTM workload predictor...")
    predictor = WorkloadPredictor(seed=int(seed))
    train_times, train_utils = predictor.generate_training_data(seed=int(seed))
    lstm_losses = predictor.train(train_times, train_utils, epochs=lstm_epochs)

    # Run baseline
    progress_bar.progress(40, text="Running baseline (FCFS) scheduler...")
    baseline_result = run_simulation(
        jobs, "baseline", carbon_model,
        sim_duration_minutes=sim_hours * 60,
        seed=int(seed),
    )

    # Run carbon-aware
    progress_bar.progress(70, text="Running carbon-aware scheduler...")
    aware_result = run_simulation(
        jobs, "carbon_aware", carbon_model,
        sim_duration_minutes=sim_hours * 60,
        seed=int(seed),
    )

    progress_bar.progress(100, text="Computing metrics...")
    progress_bar.empty()

    st.session_state.results = {
        "baseline": baseline_result,
        "aware": aware_result,
        "lstm_losses": lstm_losses,
        "carbon_model": carbon_model,
        "predictor": predictor,
        "region": region,
    }
    st.rerun()

# ── Results dashboard ──────────────────────────────────────────────────────────
if st.session_state.results:
    R = st.session_state.results
    bl = R["baseline"]
    ca = R["aware"]
    bm = bl.metrics
    cam = ca.metrics
    diff = cam.compare_to(bm)

    # ── KPI row ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">SIMULATION RESULTS — KEY METRICS</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Carbon Savings", f"{diff['carbon_savings_kg']:.2f} kg",
                  delta=f"{diff['carbon_savings_pct']:.1f}%",
                  delta_color="normal" if diff['carbon_savings_pct'] > 0 else "inverse")
    with c2:
        st.metric("Energy Savings", f"{diff['energy_savings_kwh']:.2f} kWh",
                  delta=f"{diff['energy_savings_pct']:.1f}%")
    with c3:
        sla_ok = cam.sla_violation_rate <= 0.05
        st.metric("SLA Violations (C-Aware)",
                  f"{cam.sla_violation_rate*100:.1f}%",
                  delta=f"{'✓ OK' if sla_ok else '⚠ OVER'} (<5% target)")
    with c4:
        st.metric("Jobs Completed (C-Aware)",
                  f"{cam.jobs_completed}/{cam.jobs_total}",
                  delta=f"vs {bm.jobs_completed}/{bm.jobs_total} baseline")
    with c5:
        st.metric("Carbon Efficiency", f"{cam.carbon_efficiency:.2f} jobs/kg",
                  delta=f"vs {bm.carbon_efficiency:.2f} baseline")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Comparison", "🕐 Timeline", "🔬 LSTM Predictor", "🖥️ GPU Analysis", "📋 Job Table"
    ])

    # ── Tab 1: Comparison ────────────────────────────────────────────────────
    with tab1:
        col_a, col_b = st.columns(2)

        # Carbon comparison bar
        with col_a:
            fig_compare = go.Figure()
            categories = ["Total Carbon (kg)", "Total Energy (kWh)", "SLA Violation %", "Avg Latency (min)"]
            baseline_vals = [bm.total_carbon_kg, bm.total_energy_kwh,
                             bm.sla_violation_rate*100, bm.avg_job_latency_min]
            aware_vals = [cam.total_carbon_kg, cam.total_energy_kwh,
                          cam.sla_violation_rate*100, cam.avg_job_latency_min]

            fig_compare.add_trace(go.Bar(name="Baseline (FCFS)", x=categories, y=baseline_vals,
                                         marker_color="#ff4455", opacity=0.85))
            fig_compare.add_trace(go.Bar(name="Carbon-Aware", x=categories, y=aware_vals,
                                         marker_color="#00ff88", opacity=0.85))
            fig_compare.update_layout(
                **PLOTLY_THEME, barmode="group",
                title="Baseline vs Carbon-Aware: Key Metrics",
                height=380, legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_compare, use_container_width=True)

        # Savings summary
        with col_b:
            labels = ["Carbon Reduction", "Energy Reduction", "SLA Delta", "Throughput"]
            savings_pct = [
                diff["carbon_savings_pct"],
                diff["energy_savings_pct"],
                -diff["sla_delta_pct"],
                (cam.jobs_completed - bm.jobs_completed) / max(bm.jobs_completed, 1) * 100,
            ]
            colors = ["#00ff88" if v >= 0 else "#ff4455" for v in savings_pct]

            fig_savings = go.Figure(go.Bar(
                x=labels, y=savings_pct,
                marker_color=colors,
                text=[f"{v:+.1f}%" for v in savings_pct],
                textposition="outside",
            ))
            fig_savings.update_layout(
                **PLOTLY_THEME,
                title="Carbon-Aware vs Baseline: % Improvement",
                yaxis_title="% Change (positive = better)",
                height=380,
            )
            st.plotly_chart(fig_savings, use_container_width=True)

        # Detailed comparison table
        st.markdown('<div class="section-header">DETAILED METRICS COMPARISON</div>', unsafe_allow_html=True)
        comparison_df = pd.DataFrame({
            "Metric": ["Total Carbon (kg)", "Total Energy (kWh)", "Avg Carbon Intensity (gCO₂/kWh)",
                       "SLA Violation Rate", "Jobs Completed", "Avg Job Latency (min)",
                       "Peak Power (kW)", "Avg GPU Utilization", "Carbon Efficiency (jobs/kg CO₂)"],
            "Baseline": [
                fmt_number(bm.total_carbon_kg), fmt_number(bm.total_energy_kwh),
                fmt_number(bm.avg_carbon_intensity, 0), f"{bm.sla_violation_rate*100:.1f}%",
                f"{bm.jobs_completed}/{bm.jobs_total}", fmt_number(bm.avg_job_latency_min),
                fmt_number(bm.peak_power_kw), f"{bm.avg_gpu_utilization*100:.1f}%",
                fmt_number(bm.carbon_efficiency, 3),
            ],
            "Carbon-Aware": [
                fmt_number(cam.total_carbon_kg), fmt_number(cam.total_energy_kwh),
                fmt_number(cam.avg_carbon_intensity, 0), f"{cam.sla_violation_rate*100:.1f}%",
                f"{cam.jobs_completed}/{cam.jobs_total}", fmt_number(cam.avg_job_latency_min),
                fmt_number(cam.peak_power_kw), f"{cam.avg_gpu_utilization*100:.1f}%",
                fmt_number(cam.carbon_efficiency, 3),
            ],
            "Δ (C-Aware vs Baseline)": [
                f"{diff['carbon_savings_kg']:+.2f} kg saved",
                f"{diff['energy_savings_kwh']:+.2f} kWh saved",
                "—",
                f"{diff['sla_delta_pct']:+.1f}pp",
                "—",
                f"{diff['latency_delta_min']:+.1f} min",
                "—", "—", "—",
            ],
        })
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # ── Tab 2: Timeline ───────────────────────────────────────────────────────
    with tab2:
        bl_tl = pd.DataFrame(bl.timeline)
        ca_tl = pd.DataFrame(ca.timeline)

        fig_tl = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=("Carbon Intensity (gCO₂/kWh)", "Active Jobs", "Total Power Draw (kW)"),
            vertical_spacing=0.08,
        )

        fig_tl.add_trace(go.Scatter(x=bl_tl["time_min"], y=bl_tl["carbon_intensity"],
                                     line=dict(color="#00aaff", width=1.5), name="Carbon Intensity",
                                     fill="tozeroy", fillcolor="rgba(0,170,255,0.06)"), row=1, col=1)

        fig_tl.add_trace(go.Scatter(x=bl_tl["time_min"], y=bl_tl["active_jobs"],
                                     line=dict(color="#ff4455", width=2), name="Baseline Jobs"), row=2, col=1)
        fig_tl.add_trace(go.Scatter(x=ca_tl["time_min"], y=ca_tl["active_jobs"],
                                     line=dict(color="#00ff88", width=2), name="C-Aware Jobs"), row=2, col=1)

        fig_tl.add_trace(go.Scatter(x=bl_tl["time_min"], y=bl_tl["total_power_kw"],
                                     line=dict(color="#ff4455", width=2, dash="dash"), name="Baseline Power"), row=3, col=1)
        fig_tl.add_trace(go.Scatter(x=ca_tl["time_min"], y=ca_tl["total_power_kw"],
                                     line=dict(color="#00ff88", width=2), name="C-Aware Power"), row=3, col=1)

        fig_tl.update_layout(**PLOTLY_THEME, height=560, showlegend=True,
                              legend=dict(orientation="h", y=-0.05),
                              xaxis3_title="Simulation Time (minutes)")
        st.plotly_chart(fig_tl, use_container_width=True)

    # ── Tab 3: LSTM Predictor ─────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">LSTM WORKLOAD PREDICTOR — TRAINING & INFERENCE</div>', unsafe_allow_html=True)

        col_l, col_r = st.columns(2)
        with col_l:
            losses = R["lstm_losses"]
            fig_loss = go.Figure(go.Scatter(
                y=losses, x=list(range(1, len(losses)+1)),
                mode="lines+markers",
                line=dict(color="#00aaff", width=2),
                marker=dict(size=5, color="#00ff88"),
            ))
            fig_loss.update_layout(**PLOTLY_THEME, title="LSTM Training Loss", height=300,
                                    xaxis_title="Epoch", yaxis_title="MSE Loss")
            st.plotly_chart(fig_loss, use_container_width=True)

        with col_r:
            predictor = R["predictor"]
            train_times, train_utils = predictor.generate_training_data(seed=42)
            # Show prediction vs actual on last 60 points
            n = 60
            t_slice = train_times[-n:]
            u_slice = train_utils[-n:]
            preds = []
            for i in range(predictor.SEQ_LEN, n):
                p = predictor.predict(t_slice[:i], u_slice[:i])
                preds.append(p)

            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(y=u_slice[predictor.SEQ_LEN:],
                                           x=t_slice[predictor.SEQ_LEN:] / 60,
                                           name="Actual", line=dict(color="#00aaff", width=2)))
            fig_pred.add_trace(go.Scatter(y=preds,
                                           x=t_slice[predictor.SEQ_LEN:] / 60,
                                           name="LSTM Predicted", line=dict(color="#ffcc00", width=2, dash="dot")))
            fig_pred.update_layout(**PLOTLY_THEME, title="LSTM: Actual vs Predicted Utilization",
                                    xaxis_title="Hour of Day", yaxis_title="GPU Utilization",
                                    height=300, legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_pred, use_container_width=True)

        # Model architecture info
        st.markdown('<div class="section-header">MODEL ARCHITECTURE</div>', unsafe_allow_html=True)
        arch_cols = st.columns(4)
        arch_cols[0].metric("Input Features", "3")
        arch_cols[1].metric("Hidden Size", "64")
        arch_cols[2].metric("LSTM Layers", "2")
        arch_cols[3].metric("Sequence Length", "12")
        st.markdown("""
        <div class="info-box">
        Input features: [time_sin, time_cos, normalized_utilization] → 
        LSTM(64 hidden, 2 layers, dropout=0.2) → FC(64→32→1) → predicted_utilization
        </div>""", unsafe_allow_html=True)

    # ── Tab 4: GPU Analysis ───────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">PER-GPU ENERGY & CARBON BREAKDOWN</div>', unsafe_allow_html=True)

        def gpu_df(result, label):
            rows = []
            for gpu in result.gpus:
                rows.append({
                    "GPU": gpu.model[:22],
                    "Energy (kWh)": round(gpu.total_energy_kwh, 3),
                    "Carbon (kg)": round(gpu.carbon_emitted_kg, 4),
                    "TDP (W)": gpu.tdp_watts,
                    "TFLOPS": gpu.compute_tflops,
                    "Efficiency": round(gpu.efficiency_ratio, 4),
                    "Scheduler": label,
                })
            return pd.DataFrame(rows)

        bl_df = gpu_df(bl, "Baseline")
        ca_df = gpu_df(ca, "Carbon-Aware")

        fig_gpu = go.Figure()
        fig_gpu.add_trace(go.Bar(x=bl_df["GPU"], y=bl_df["Energy (kWh)"], name="Baseline Energy",
                                  marker_color="#ff4455", opacity=0.8))
        fig_gpu.add_trace(go.Bar(x=ca_df["GPU"], y=ca_df["Energy (kWh)"], name="C-Aware Energy",
                                  marker_color="#00ff88", opacity=0.8))
        fig_gpu.update_layout(**PLOTLY_THEME, barmode="group",
                               title="Per-GPU Energy Consumption (kWh)",
                               xaxis_tickangle=-35, height=380,
                               margin=dict(b=120), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_gpu, use_container_width=True)

        # Scatter: Efficiency vs Carbon emitted
        merged = pd.concat([bl_df.assign(sched="Baseline"), ca_df.assign(sched="Carbon-Aware")])
        fig_scatter = px.scatter(merged, x="Efficiency", y="Carbon (kg)", color="Scheduler",
                                  size="Energy (kWh)", hover_data=["GPU"],
                                  color_discrete_map={"Baseline": "#ff4455", "Carbon-Aware": "#00ff88"})
        fig_scatter.update_layout(**PLOTLY_THEME, title="GPU Efficiency vs Carbon Emitted",
                                   height=320)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Tab 5: Job Table ─────────────────────────────────────────────────────
    with tab5:
        st.markdown('<div class="section-header">JOB EXECUTION LOG — CARBON-AWARE SCHEDULER</div>', unsafe_allow_html=True)

        job_rows = []
        for j in ca.jobs:
            gpu_model = ca.gpus[j.assigned_gpu_id].model if j.assigned_gpu_id is not None else "—"
            job_rows.append({
                "Job": j.name,
                "Priority": j.priority.upper(),
                "Submitted": f"{j.submitted_at:.0f} min",
                "Started": f"{j.started_at:.0f} min" if j.started_at else "—",
                "Completed": f"{j.completed_at:.0f} min" if j.completed_at else "—",
                "Deadline": f"{j.deadline_minutes:.0f} min",
                "Energy (kWh)": round(j.energy_kwh, 4),
                "Carbon (kg)": round(j.carbon_cost_kg, 5),
                "GPU": gpu_model[:22] if gpu_model != "—" else "—",
                "SLA": "✓ OK" if not j.sla_violated else "✗ VIOLATED",
            })

        jobs_df = pd.DataFrame(job_rows)
        st.dataframe(jobs_df, use_container_width=True, hide_index=True, height=450)

        # Priority breakdown
        st.markdown('<div class="section-header">PRIORITY × SLA BREAKDOWN</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, priority in enumerate(["high", "medium", "low"]):
            pjobs = [j for j in ca.jobs if j.priority == priority]
            violated = sum(1 for j in pjobs if j.sla_violated)
            cols[i].metric(f"{priority.upper()} Priority Jobs",
                           f"{len(pjobs) - violated}/{len(pjobs)} met SLA",
                           delta=f"{violated} violated")

    # ── Reset button ──────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔄  RESET SIMULATION"):
        st.session_state.results = None
        st.rerun()
