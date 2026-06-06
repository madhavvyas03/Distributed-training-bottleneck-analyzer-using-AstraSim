<div align="center">

# 🚀 Distributed Training Bottleneck Analyzer

### An AI-Powered Performance Diagnostic Engine for ASTRA-Sim Distributed Training Logs

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-LLaMA3-purple?style=for-the-badge)](https://ollama.ai)
[![ASTRA-Sim](https://img.shields.io/badge/ASTRA--Sim-Compatible-green?style=for-the-badge)](https://astra-sim.github.io)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

> **Parse → Compute → Classify → Visualize → Recommend**
> 
> A complete, rule-driven bottleneck analysis pipeline for multi-GPU distributed deep learning workloads, powered by local LLM expert commentary.

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Repository Structure](#-repository-structure)
- [How It Works](#-how-it-works)
- [Key Metrics & Formulas](#-key-metrics--formulas)
- [Bottleneck Classification Rules (R1–R10)](#-bottleneck-classification-rules-r1r10)
- [Rule Thresholds](#-rule-thresholds)
- [Congestion & Severity Scoring](#-congestion--severity-scoring)
- [What-If Simulation](#-what-if-simulation)
- [Fingerprint Database & Case Matching](#-fingerprint-database--case-matching)
- [Output Visualizations](#-output-visualizations)
- [Output Sample](#-output-sample)
- [Dashboard PDF Report](#-dashboard-pdf-report)
- [Installation](#-installation)
- [Usage](#-usage)
- [Parameters Reference](#-parameters-reference)
- [Supported Models](#-supported-models)
- [Dependencies](#-dependencies)

---

## 🔍 Overview

Training large deep learning models across multiple GPUs introduces complex performance dynamics. Communication overhead, compute-communication imbalance, straggler ranks, and bandwidth saturation can silently degrade scaling efficiency — often in ways that are hard to diagnose from raw logs.

This tool takes **ASTRA-Sim simulation output logs** and, through a multi-stage analysis pipeline, automatically:

- Parses per-rank timing data (wall time, communication time, compute time)
- Computes 15+ derived performance metrics
- Applies a **rule-based expert system (R1–R10)** to classify the dominant bottleneck
- Generates congestion heatmaps, critical-path breakdowns, C2C ratio plots, and more
- Produces a **landscape PDF dashboard** summarizing all findings
- Queries a **local Ollama LLM (LLaMA3)** for human-readable expert analysis and mitigation recommendations
- Stores case **fingerprints** for similarity matching against historical runs

The system is split into two well-separated layers: a **pure-Python core engine** (`core_engine.py`) containing zero AI and zero visualization logic, and a **full-featured analyzer** (`ollama_analyzer.py`) that orchestrates everything.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ASTRA-Sim Log File (.log)                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LOG PARSER                                  │
│  Extracts per-rank: wall_time, comm_time, compute_time          │
│  Detects topology from [system::topology::*] tags               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    METRIC COMPUTATION                            │
│  Computes all 15+ performance metrics (see Formulas section)    │
│  Throughput · C2C · Comm Fraction · GPU Util · Scaling Eff ...  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│               RULE ENGINE  (R1 – R10)                           │
│  Applies threshold-based rules against computed metrics         │
│  Returns all triggered rules + dominant bottleneck via          │
│  priority hierarchy                                             │
└─────────────┬─────────────────────────────┬─────────────────────┘
              │                             │
              ▼                             ▼
┌─────────────────────┐         ┌───────────────────────────────┐
│  VISUALIZATION      │         │  OLLAMA LLM EXPERT            │
│  9 chart types      │         │  Sends structured prompt to   │
│  (heatmaps, pies,   │         │  local LLaMA3 model for       │
│  bars, line plots)  │         │  root-cause analysis and      │
│  + causal graph     │         │  mitigation strategies        │
└─────────┬───────────┘         └──────────────┬────────────────┘
          │                                    │
          └──────────────┬─────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              PDF DASHBOARD + JSON/CSV EXPORT                    │
│  Landscape A4 report with all metrics, charts, bottleneck       │
│  diagnosis, triggered rules, and LLM recommendations           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
distributed-training-bottleneck-analyzer/
│
├── core_engine.py              # Pure analysis engine — zero AI, zero viz
│   ├── PerformanceMetrics      # Dataclass holding all computed metrics
│   ├── TraceEngine             # Main engine class
│   │   ├── parse_trace()       # Log parsing via regex
│   │   ├── compute_metrics()   # Metric derivation
│   │   └── classify_bottleneck() # R1–R9 rule application
│   └── run_core_engine()       # Standalone entry-point function
│
├── ollama_analyzer.py          # Full-featured analyzer (1151 lines)
│   ├── parse_astrasim_log()    # Robust multi-pattern log parser
│   ├── compute_metrics()       # Extended metric computation
│   ├── apply_rules()           # Full R1–R10 rule engine with thresholds
│   ├── simulate_what_if()      # What-If topology/batch/GA simulator
│   ├── generate_all_graphs()   # All 9 visualization functions
│   ├── generate_dashboard_pdf()# ReportLab landscape PDF builder
│   ├── call_ollama()           # Subprocess wrapper for local LLM
│   ├── find_similar_cases()    # Cosine similarity fingerprint search
│   └── generate_recommendation() # Full expert LLM prompt builder
│
├── astrasim_analyzer/          # Supporting analysis modules + PDF report
│   └── *.pdf                   # Technical report explaining the system
│
├── analysis_20251128_133731/   # Sample analysis output folder
│   ├── *.png                   # Generated plots
│   ├── *.json                  # Metrics export
│   ├── *.csv                   # Per-rank data
│   └── *.pdf                   # Generated dashboard
│
├── fingerprints.json           # Historical case database (auto-generated)
├── log_dlrm_comm.log           # Sample DLRM log file for testing
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## ⚙️ How It Works

### Stage 1 — Log Parsing

The parser handles two log formats from ASTRA-Sim:

**Format A — Statistics lines:**
```
[statistics] [info] sys[0], Wall time: 123456
[statistics] [info] sys[0], Comm time: 98765
```

**Format B — Workload finish lines (fallback):**
```
[workload] [info] sys[0] finished, 123456 cycles, exposed communication 98765 cycles.
```

Both formats are tried for every rank. If a value is missing, it defaults to `0.0` with a warning — the engine never crashes on incomplete logs.

Topology is extracted from:
```
[system::topology::Ring]
```

### Stage 2 — Metric Computation

All metrics are derived from per-rank wall, communication, and compute times. See the [Formulas](#-key-metrics--formulas) section for the exact equations.

### Stage 3 — Rule Application

Ten rules (R1–R10) are evaluated in sequence against the metric values and rule thresholds defined in `RULE_THRESHOLDS`. All triggered rules are returned. The **dominant bottleneck** is selected using a strict priority ordering (see [Priority Hierarchy](#priority-hierarchy)).

### Stage 4 — Visualization & Reporting

Nine charts are generated as PNGs, assembled into a landscape A4 PDF dashboard alongside the metrics table, triggered rules, and LLM commentary.

### Stage 5 — LLM Expert Analysis

A structured prompt containing all metrics, triggered rules, and dominant bottleneck is sent to a locally running **Ollama LLaMA3** instance. The model returns a concise expert analysis with root-cause explanation and mitigation steps.

---

## 📐 Key Metrics & Formulas

All metrics are computed from raw ASTRA-Sim cycle counts. Cycles are converted to milliseconds by dividing by 1,000.

### Time Metrics

| Metric | Formula |
|--------|---------|
| **Wall Time (ms)** | `avg(wall_time_cycles) / 1000` |
| **Comm Time (ms)** | `avg(comm_time_cycles) / 1000` |
| **Compute Time (ms)** | `avg(wall_time - comm_time) / 1000` |
| **Idle Time (ms)** | `wall_ms - comm_ms - compute_ms` |

### Fraction Metrics

| Metric | Formula | What It Means |
|--------|---------|---------------|
| **Comm Fraction** | `avg_comm / avg_wall` | Share of total time spent on communication |
| **Compute Fraction** | `avg_compute / avg_wall` | Share of total time spent on computation |
| **Overlap Ratio** | `max(0, 1 − comm_fraction − compute_fraction)` | Fraction of time where comm and compute overlap |

### Performance Ratios

| Metric | Formula | What It Means |
|--------|---------|---------------|
| **C2C Ratio** | `avg_comm / avg_compute` | Communication-to-Compute ratio; >1.0 means comm dominates |
| **Rank Imbalance (%)** | `(max_wall − min_wall) / avg_wall × 100` | How unevenly distributed work is across ranks |

### Throughput

```
Global Batch Size = num_gpus × batch_size_per_gpu

Iteration Time (sec) = wall_time_ms / 1000

Throughput (samples/sec) = Global Batch Size / Iteration Time
```

This is the primary scaling metric — it measures how many training samples the entire cluster processes per second.

### GPU Utilization

```
GPU Utilization (%) = compute_fraction × 100
```

Where `compute_fraction` is the fraction of wall time actually spent doing arithmetic. Values below 60% suggest kernel scheduling issues, memory-bound operations, or excessive idling.

### Scaling Efficiency

```
Scaling Efficiency (%) = max(0, 100 − (comm_fraction × 100 × 0.75))
```

Compares expected linear scaling against observed throughput. Reference values by model type:

| Model | 1 GPU | 2 GPUs | 4 GPUs | 8 GPUs |
|-------|-------|--------|--------|--------|
| ResNet | 100% | 98% | 95% | 98% |
| BERT | 100% | 92% | 85% | 82% |
| DLRM | 100% | 88% | 77% | 51.25% |

### Congestion Score

```
Congestion Score = (comm_fraction × 0.6) + (rank_imbalance / 100 × 0.4)
```

A composite score blending how much time is spent communicating and how uneven the load is across ranks. Higher values indicate more severe network pressure.

### Activation Memory Pressure

```
Activation Pressure = compute_fraction × (1 − overlap_ratio)
```

Predicts how much activation memory pressure exists during pipeline stages with no overlap.

### Bisection Bandwidth Stress

```
BW Stress = comm_fraction × C2C_ratio × (rank_imbalance / 100)
```

Estimates the stress placed on bisection bandwidth — the cross-cluster link capacity — by combining communication dominance, the communication-to-compute ratio, and rank imbalance.

### Temporal Instability

```
thr_stdev = standard deviation of per-rank throughputs
temporal_instability = thr_stdev / thr_mean
```

Measures whether throughput fluctuates across ranks, which can indicate transient congestion or jitter.

---

## 🔬 Bottleneck Classification Rules (R1–R10)

Each rule checks a specific combination of metrics against configurable thresholds. Multiple rules can fire simultaneously; the dominant one is selected by the priority hierarchy.

### R1 — Network Congestion
**Condition:** `comm_fraction > 0.45` AND `rank_imbalance > 5.0%`

High communication fraction paired with notable rank imbalance implies network link saturation or poorly implemented collectives. The network is the critical path — compute is waiting for gradients to be exchanged.

**Mitigation:** Switch to Tree or Fully-Connected topology; increase gradient accumulation steps; apply gradient compression.

---

### R2 — Compute Inefficiency
**Condition:** `compute_fraction > 0.70` AND `gpu_utilization < 60%`

Compute occupies most of wall time, but GPUs aren't being efficiently utilized — a sign of kernel scheduling stalls, poor memory access patterns, or underutilized tensor cores.

**Mitigation:** Increase per-GPU batch size; enable mixed-precision (FP16/BF16); reduce kernel launch overhead.

---

### R3 — Mixed Bottleneck
**Condition:** `comm_fraction > 0.35` AND `compute_fraction > 0.55`

Both communication and computation significantly affect iteration latency — neither alone is the clear bottleneck. This often happens in mid-scale deployments where the model is large but the network is not fast enough to overlap.

**Mitigation:** Use Tree topology to reduce communication; tune gradient accumulation to balance the two phases.

---

### R4 — Poor Overlap Efficiency
**Condition:** `overlap_ratio < 0.25` AND `comm_fraction > 0.40`

Communication is not being effectively hidden behind computation. The training framework is serializing comm and compute instead of overlapping them.

**Mitigation:** Enable asynchronous collectives; prefetch activations during backward pass; reorder compute operations to increase concurrency.

---

### R5 — Straggler-Induced Synchronization Delay
**Condition:** `rank_imbalance > 7.0%`

Significant differences in per-rank wall times mean some ranks (stragglers) are holding back the entire cluster during the synchronization barrier at the end of each iteration.

**Mitigation:** Rebalance dataset shards; ensure homogeneous hardware; implement adaptive load balancing.

---

### R6 — Temporal Instability
**Condition:** `temporal_instability > 0.10` AND `gpu_util_variance > 10.0`

Throughput varies significantly across iterations or ranks, likely due to transient network congestion, background system load, or NCCL jitter.

**Mitigation:** Increase gradient accumulation to amortize per-iteration variance; improve comm-compute overlap; stabilize network conditions (dedicated RDMA fabric).

---

### R7 — Arithmetic Bound
**Condition:** `compute_fraction > 0.75` AND `C2C_ratio < 0.2`

Compute is the overwhelming bottleneck and communication is cheap (very low C2C). The arithmetic throughput of the GPUs is the limiting factor — this is the ideal case for model parallelism.

**Mitigation:** Increase model parallelism; scale batch size; no major communication optimizations needed.

---

### R8 — Pipeline Imbalance
**Condition:** `comm_fraction > 0.55` AND `overlap_ratio > 0.5` AND `gpu_utilization < 65%`

Some overlap is present but GPU utilization is still low — indicating that pipeline stages are imbalanced and some stages are stalling.

**Mitigation:** Reduce batch size per stage; re-partition pipeline stages for more equal compute load.

---

### R9 — Bandwidth Saturation
**Condition:** `scaling_efficiency < 70%` AND `comm_fraction > 0.5`

Scaling efficiency is well below linear and communication dominates — the interconnect bandwidth is saturated and cannot keep up with gradient exchange demand.

**Mitigation:** Switch to higher-bandwidth topology (Fully-Connected); apply gradient compression; increase gradient accumulation steps.

---

### R10 — Underutilized GPUs
**Condition:** `gpu_utilization < 45%`

Average GPU utilization is critically low — hardware is severely underused regardless of bottleneck type.

**Mitigation:** Increase global batch size; reduce synchronization frequency; investigate kernel-level stalls with profiling tools like Nsight.

---

## ⚖️ Rule Thresholds

All thresholds are configurable via the `RULE_THRESHOLDS` dictionary:

```python
RULE_THRESHOLDS = {
    "R1_comm_fraction":     0.45,
    "R1_imbalance":         5.0,
    "R2_compute_fraction":  0.70,
    "R2_gpu_util":          60.0,
    "R3_comm_fraction":     0.35,
    "R3_compute_fraction":  0.55,
    "R4_overlap_ratio":     0.25,
    "R4_comm_fraction":     0.40,
    "R5_imbalance":         7.0,
    "R6_temporal_instability": 0.10,
    "R6_gpu_util_var":      10.0,
    "R7_compute_fraction":  0.75,
    "R7_c2c":               0.2,
    "R8_comm_fraction":     0.55,
    "R8_overlap_ratio":     0.5,
    "R8_gpu_util":          65.0,
    "R9_scaling_eff":       70.0,
    "R9_comm_fraction":     0.5,
    "R10_gpu_util":         45.0,
}
```

---

## 🌡️ Congestion & Severity Scoring

### Congestion Type Classification

After computing the congestion score, the system classifies its spatial pattern:

| Condition | Congestion Type |
|-----------|----------------|
| `rank_imbalance < 5%` AND `comm_fraction > 0.9` | `Uniform_Global` — all ranks equally congested |
| `rank_imbalance > 10%` | `Localized` — some ranks much worse than others |
| Otherwise | `Moderate_Global` |

### Severity Classification

| Condition | Severity |
|-----------|---------|
| `comm_fraction ≥ 0.95` OR `C2C > 5` | **CATASTROPHIC** |
| `comm_fraction ≥ 0.80` OR `C2C > 1.5` | **CRITICAL** |
| Otherwise | **MEDIUM** |

### Priority Hierarchy

When multiple rules fire, the dominant bottleneck is selected in this priority order:

```
1. Network Congestion
2. Bandwidth Saturation
3. Mixed Bottleneck
4. Compute Inefficiency
5. Poor Overlap Efficiency
6. Straggler Delay
7. Pipeline Imbalance
8. Temporal Instability
9. Underutilized GPUs
10. Arithmetic Bound
```

---

## 🔮 What-If Simulation

The `simulate_what_if()` function lets you model hypothetical hardware/config changes without re-running ASTRA-Sim.

**Usage:**
```
Enter: what-if topology=tree,ga=4,batch=64
```

### What-If Transformations

| Parameter | Effect |
|-----------|--------|
| `topology=tree` | Reduces `comm_fraction` by 18% (tree topology reduces collective latency) |
| `topology=fc` | Reduces `comm_fraction` by 25% and `rank_imbalance` by 10% |
| `topology=ring` | Increases `comm_fraction` by 5% (ring has higher per-message latency) |
| `ga=N` | Reduces `comm_fraction` by `1/√N` and increases throughput by `√N` |
| `batch=B` | GPU utilization scales by `min(2.0, B / original_batch_per_gpu)` |

**Example output:**
```
Original throughput: 1240.5 → 2104.3 samples/sec
Comm fraction:       0.72   → 0.51
GPU utilization:     38.2%  → 64.1%
Overlap ratio:       0.05   → 0.24
```

---

## 🗃️ Fingerprint Database & Case Matching

Every analysis run saves a compact **fingerprint** to `fingerprints.json`:

```json
{
  "timestamp": 1732790251.4,
  "model_type": "dlrm",
  "dominant_bottleneck": "Network Congestion",
  "comm_fraction": 0.847,
  "compute_fraction": 0.121,
  "c2c_ratio": 7.0,
  "rank_imbalance_pct": 12.3,
  "gpu_utilization_pct": 12.1
}
```

On subsequent runs, the system searches the fingerprint database using **cosine similarity** on a 4-dimensional feature vector:

```
v = [comm_fraction, C2C_ratio, rank_imbalance_pct, gpu_utilization_pct]
```

The top-3 most similar historical cases are retrieved and included in the LLM prompt, enabling the model to reference patterns from past runs.

---

## 📊 Output Visualizations
## Example AI-Generated Performance Analysis Report

The system automatically transforms raw Astra-Sim execution traces into a comprehensive performance engineering report.

For each distributed training run, the AI Agent:

- Extracts communication and compute metrics
- Computes scaling and utilization statistics
- Identifies dominant bottlenecks
- Applies explainable reasoning rules (R1–R9)
- Generates root-cause analysis
- Produces optimization recommendations
- Visualizes system behavior through charts and heatmaps

### Sample Output

![Performance Analysis Report](assets/output.png)

### Key Findings

| Metric | Value |
|----------|----------|
| Average Wall Time | 407,625 cycles |
| Communication Fraction | 49.4% |
| Compute Fraction | 50.6% |
| C2C Ratio | 0.98 |
| Rank Imbalance | 9.57% |

### AI Diagnosis

**Dominant Bottleneck:** Network Congestion

The agent detected elevated communication overhead combined with rank imbalance, indicating congestion during collective communication operations.

### Triggered Rules

| Rule | Description |
|--------|--------|
| R1 | Network Congestion |
| R4 | Poor Overlap Efficiency |
| R5 | Straggler Delay |

### Recommended Actions

1. Switch to Tree or Fully Connected topology
2. Increase gradient accumulation
3. Apply gradient compression
4. Improve compute-communication overlap
5. Rebalance workload across ranks

### Generated Insight

Unlike traditional monitoring tools that only expose metrics, the AI Agent explains:

- Why the bottleneck occurs
- Which system components are responsible
- Which optimization strategies are likely to help
- The expected impact of each recommendation

This enables engineers to move from trace collection to actionable performance optimization in a single workflow.
The analyzer generates **9 chart types**, all saved as PNG files:

| Chart | File | What It Shows |
|-------|------|---------------|
| Per-Rank Wall Time | `wall_times.png` | Bar chart of iteration duration per GPU rank |
| GPU Utilization | `gpu_utilization.png` | Line plot of (compute + comm) / wall per rank |
| Comm vs Compute | `comm_vs_compute.png` | Stacked bar: communication and compute breakdown |
| Congestion Heatmap | `congestion_heatmap.png` | Color-mapped comm fraction across all ranks |
| C2C Ratio Heatmap | `c2c_heatmap.png` | Per-rank communication-to-compute ratio |
| Critical Path Pie | `critical_path_pie.png` | Comm / Compute / Overlap Lost proportions |
| Scaling Efficiency | `scaling_efficiency_curve.png` | Expected efficiency curve by GPU count |
| Activation Pressure | `activation_pressure.png` | Bar chart of memory pressure predictor |
| BW Stress Score | `bisection_bw_stress.png` | Bisection bandwidth stress indicator |

Additionally, a **Bottleneck Causal Graph** is generated (via Graphviz if available, otherwise Matplotlib fallback) showing the dominant bottleneck and the four key metrics that led to it.

---

## 🖼️ Output Sample

> ⬇️ **Add your output screenshot here**

```
<!-- Replace this comment with your output image -->
![Analysis Output](./assets/output_screenshot.png)
```

### Understanding the Output

When you run the analyzer, you will see a structured console report followed by file outputs. Here is how to read the key values:

**Comm Fraction** — If this is above `0.45`, communication is dominating your iteration time. For DLRM workloads at 8 GPUs, values above `0.80` are common and indicate near-CATASTROPHIC bandwidth bottleneck.

**C2C Ratio** — Values above `1.5` mean your GPUs are spending more time waiting for gradients than doing actual compute. A ratio of `7.0` (as seen in the sample DLRM log) means the communication phase is 7× longer than the compute phase.

**Rank Imbalance** — Above `5%` triggers R1; above `7%` triggers R5 (straggler detection). High imbalance at 8+ GPUs usually means one rank is slower due to load imbalance or hardware variance.

**GPU Utilization** — For a healthy run, this should be above `70%`. Values below `45%` trigger R10 (Underutilized GPUs).

**Throughput** — The most actionable metric. Compare this against your single-GPU baseline to compute actual scaling efficiency.

**Severity** — If the output says `CATASTROPHIC`, your training is spending nearly all iteration time on gradient synchronization. The first fix is almost always increasing gradient accumulation steps.

---

## 📄 Dashboard PDF Report

The generated PDF (`dashboard.pdf`) uses a **landscape A4 layout** with two columns:

**Left column:** All 5 core plots (wall times, GPU utilization, comm/compute stacked, critical-path pie, congestion heatmap)

**Right column:**
- Summary metadata (model, GPUs, batch size, dominant bottleneck)
- Metrics table
- All triggered rules (R1–R10) with mitigations
- What-if simulation results (if requested)
- LLM expert commentary

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/madhavvyas03/distributed-training-bottleneck-analyzer.git
cd distributed-training-bottleneck-analyzer
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama (optional, for LLM commentary)

```bash
# macOS / Linux
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3
```

> If Ollama is not installed or unavailable, the tool will still run — LLM sections will show a fallback message and all other outputs (metrics, charts, PDF) are unaffected.

---

## ▶️ Usage

### Basic Run

```bash
python ollama_analyzer.py
```

You will be prompted interactively:

```
Path to .log file: log_dlrm_comm.log
Is this distributed training? (yes/no): yes
Model type (resnet/bert/dlrm): dlrm
Per-GPU batch size: 32
```

### What-If Mode

At the file path prompt, type a what-if string:

```
Enter: what-if topology=tree,ga=4,batch=128
```

### Core Engine Only (No Visualization / No LLM)

```python
from core_engine import run_core_engine

with open("log_dlrm_comm.log") as f:
    text = f.read()

metrics, df = run_core_engine(text, batch_size=32)
print(metrics)
```

### Output Files

All outputs are saved to a timestamped folder:
```
analysis_YYYYMMDD_HHMMSS/
├── metrics.json
├── per_rank.csv
├── dashboard.pdf
├── wall_times.png
├── gpu_utilization.png
├── comm_vs_compute.png
├── critical_path_pie.png
├── congestion_heatmap.png
├── c2c_heatmap.png
├── scaling_efficiency_curve.png
├── activation_pressure.png
├── bisection_bw_stress.png
└── bottleneck_causal_graph.png
```

---

## 📋 Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_path` | `str` | — | Path to ASTRA-Sim `.log` file |
| `batch_size_per_gpu` | `int` | `32` | Per-GPU mini-batch size for throughput calculation |
| `model_type` | `str` | `"dlrm"` | Model type: `resnet`, `bert`, or `dlrm` |
| `is_distributed` | `bool` | `True` | Whether this is a multi-GPU distributed run |
| `ollama_model` | `str` | `"llama3"` | Ollama model name to use for LLM commentary |
| `ollama_timeout` | `int` | `40` | Timeout in seconds for Ollama subprocess |
| `whatif_str` | `str` | `None` | What-If scenario string (e.g., `"topology=tree,ga=4"`) |

### What-If Parameters

| Key | Values | Effect |
|-----|--------|--------|
| `topology` | `tree`, `fc`, `ring` | Changes interconnect topology and adjusts comm_fraction |
| `ga` | integer | Gradient accumulation steps — reduces per-iteration communication |
| `batch` | integer | Per-GPU batch size — affects GPU utilization estimate |

---

## 🤖 Supported Models

The scaling efficiency reference curves are built in for three model families:

**ResNet** — CNN image classification. Communication-efficient; near-linear scaling at small GPU counts. Comm fraction stays low due to relatively small gradient sizes.

**BERT** — Large transformer for NLP. Moderate communication overhead. Scaling efficiency degrades noticeably above 4 GPUs due to all-reduce on large embedding layers.

**DLRM** — Deep Learning Recommendation Model. Embedding tables create very large, irregular communication patterns. Scaling efficiency at 8 GPUs can be as low as 51.25% — the most communication-bound of the three.

---

## 📦 Dependencies

```
numpy          — Numerical computation, metric arrays
matplotlib     — All chart generation
pandas         — Per-rank DataFrame construction
reportlab      — PDF dashboard generation
regex          — Enhanced log parsing
python-dateutil — Timestamp handling
Pillow         — Image processing for PDF layout
tabulate       — Console table formatting
ollama         — Python client for local LLM
scikit-learn   — (Available for future ML-based classification)
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🔗 Related Work & References

- [ASTRA-Sim](https://astra-sim.github.io/) — Georgia Tech / Meta / Intel distributed training simulator
- [Is Network the Bottleneck of Distributed Training?](https://arxiv.org/abs/2006.10103) — Foundational measurement study
- [Parameter Hub](https://arxiv.org/abs/1805.07891) — Rack-scale parameter server for distributed DNN training
- [Ollama](https://ollama.ai) — Local LLM runtime

---

<div align="center">

Made with ⚡ for systems researchers and ML engineers who need answers, not just logs.

**[⭐ Star this repo](https://github.com/madhavvyas03/distributed-training-bottleneck-analyzer)** if it saved you debugging time!

</div>
