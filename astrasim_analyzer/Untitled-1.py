
# ============================================================
# 🟦 PART 1: IMPORTS + GLOBAL DEFAULTS + INSTRUCTIONS PRINTER
# ============================================================

import os
import re
import json
import math
import sys
import time
import statistics
import subprocess
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# PDF generation library
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib import colors
from reportlab.lib.units import inch

# ============================================================
# 1. CRISP INSTRUCTIONS (printed at program start)
# ============================================================

def print_instructions():
    print("\n" + "="*82)
    print("   ADVANCED ASTRA-SIM AI-AGENT — QUICK INSTRUCTIONS")
    print("="*82)
    print("""
How to use:
1. Run:    python ai_agent_advanced.py
2. Read these instructions.
3. Then provide:
      • Path to .log file (Astra-Sim output)
      • Whether it is for distributed training (yes/no)
      • Model type (resnet / bert / dlrm)
      • Per-GPU batch size
4. Optional advanced mode:
      • Enter 'what-if' at prompt to simulate:
          topology=tree, ga=4, batch=64
        Example input: what-if topology=fc,ga=4,batch=128
5. Output:
      • Detailed metrics
      • R1–R10 bottleneck classification
      • Bottleneck causal graph
      • Congestion heatmap
      • Critical-path pie chart
      • Throughput / GPU util graphs
      • Dashboard-style PDF report
      • JSON + CSV + PNG files
      • Ollama-enhanced expert analysis

You CAN press ENTER at the first file prompt and provide the log later.
""")
    print("="*82 + "\n")


# ============================================================
# 2. GLOBAL DEFAULT SCALING EFFICIENCY (from your PDFs)
# ============================================================

DEFAULT_SCALING_EFF = {
    "resnet": {1: 100.0, 2: 98.0, 4: 95.0, 8: 98.0},
    "bert":   {1: 100.0, 2: 92.0, 4: 85.0, 8: 82.0},
    "dlrm":   {1: 100.0, 2: 88.0, 4: 77.0, 8: 51.25}
}

# ============================================================
# 3. RULE R1–R10 THRESHOLDS (PDF-informed)
# ============================================================

RULE_THRESHOLDS = {
    "R1_comm_fraction": 0.45,
    "R1_imbalance": 5.0,
    "R2_compute_fraction": 0.70,
    "R2_gpu_util": 60.0,
    "R3_comm_fraction": 0.35,
    "R3_compute_fraction": 0.55,
    "R4_overlap_ratio": 0.25,
    "R4_comm_fraction": 0.40,
    "R5_imbalance": 7.0,
    "R6_temporal_instability": 0.10,
    "R6_gpu_util_var": 10.0,
    "R7_compute_fraction": 0.75,
    "R7_c2c": 0.2,
    "R8_comm_fraction": 0.55,
    "R8_overlap_ratio": 0.5,
    "R8_gpu_util": 65.0,
    "R9_scaling_eff": 70.0,
    "R9_comm_fraction": 0.5,
    "R10_gpu_util": 45.0
}

# ============================================================
# 4. OLLAMA CALLER
# ============================================================

def call_ollama(prompt: str, model: str = "llama3", timeout: int = 40) -> str:
    """
    Calls local Ollama instance. If unavailable, returns fallback message.
    """
    try:
        proc = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        out, err = proc.communicate(prompt, timeout=timeout)
        if out.strip():
            return out.strip()
        if err.strip():
            return f"[Ollama Error] {err.strip()}"
        return "[Ollama returned no output]"
    except Exception as e:
        return f"[Ollama unavailable: {e}]"

# ============================================================
# 🟦 PART 2: LOG PARSER + METRICS COMPUTATION + WHAT-IF ENGINE
# ============================================================


# ============================================================
# 5. LOG PARSER
# ============================================================

def parse_astrasim_log(log_path: Path) -> dict:
    """
    Parses Astra-Sim style logs.
    Looks for patterns like:
        sys[0], Wall time: 12345
        sys[0], Comm time: 6789

    Returns:
        { rank: {"wall_time": float, "comm_time": float, "compute_time": float}, ... }
    """
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    text = log_path.read_text(errors="ignore")

    wall_pattern = r"sys\[(\d+)\].*?Wall time[:,]?\s*([0-9]+(?:\.[0-9]+)?)"
    comm_pattern = r"sys\[(\d+)\].*?Comm time[:,]?\s*([0-9]+(?:\.[0-9]+)?)"

    walls = re.findall(wall_pattern, text, flags=re.IGNORECASE | re.DOTALL)
    comms = re.findall(comm_pattern, text, flags=re.IGNORECASE | re.DOTALL)

    ranks = {}

    # Extract wall times
    for r, w in walls:
        r = int(r)
        ranks.setdefault(r, {})["wall_time"] = float(w)

    # Extract comm times
    for r, c in comms:
        r = int(r)
        ranks.setdefault(r, {})["comm_time"] = float(c)

    # Compute time
    for r, v in ranks.items():
        wall = v.get("wall_time", 0.0)
        comm = v.get("comm_time", 0.0)
        v["compute_time"] = max(0.0, wall - comm)

    return ranks


# ============================================================
# 6. METRICS COMPUTATION
# ============================================================

def compute_metrics(ranks_data: dict, batch_size: int, model_type: str, scaling_eff: dict) -> dict:
    """
    Computes:
        • average / max / min wall time
        • comm and compute fractions
        • c2c ratio
        • overlap ratio
        • rank imbalance
        • throughput
        • GPU utilization metrics
        • temporal instability
        • scaling efficiency (from defaults or file)
        • activation memory + overlap pressure predictor
        • bisection bandwidth stress score
    """

    # Convert lists
    ranks = sorted(ranks_data.keys())
    wall = [ranks_data[r]["wall_time"] for r in ranks]
    comm = [ranks_data[r].get("comm_time", 0.0) for r in ranks]
    compute = [ranks_data[r]["compute_time"] for r in ranks]

    avg_wall = statistics.mean(wall)
    max_wall = max(wall)
    min_wall = min(wall)

    avg_comm = statistics.mean(comm)
    avg_compute = statistics.mean(compute)

    # Fractions
    comm_fraction = avg_comm / avg_wall if avg_wall > 0 else 0
    compute_fraction = avg_compute / avg_wall if avg_wall > 0 else 0

    # C2C
    c2c = avg_comm / avg_compute if avg_compute > 0 else float("inf")

    # Overlap ratio
    overlap_ratio = max(0, 1 - (comm_fraction + compute_fraction))

    # Rank imbalance
    rank_imbalance = ((max_wall - min_wall) / avg_wall * 100) if avg_wall > 0 else 0

    # Number of GPUs
    num_gpus = len(ranks)
    global_batch = batch_size * num_gpus

    # Convert cycles → seconds (heuristic)
    iteration_time_sec = avg_wall / 1_000_000 if avg_wall > 1_000 else avg_wall / 1000

    # Throughput
    throughput = global_batch / iteration_time_sec if iteration_time_sec > 0 else 0

    # Per-rank throughput for instability
    thr_samples = []
    for w in wall:
        t = w / 1_000_000 if w > 1_000 else w / 1000
        thr_samples.append(global_batch / t if t > 0 else 0)

    thr_mean = statistics.mean(thr_samples)
    thr_stdev = statistics.stdev(thr_samples) if len(thr_samples) > 1 else 0
    temporal_instability = thr_stdev / thr_mean if thr_mean > 0 else 0

    # GPU utilization
    gpu_utils = [((c + cm) / w) * 100 if w > 0 else 0 for w, cm, c in zip(wall, comm, compute)]
    gpu_util_mean = statistics.mean(gpu_utils)
    gpu_util_var = statistics.stdev(gpu_utils) if len(gpu_utils) > 1 else 0

    # Scaling efficiency
    model_eff = scaling_eff.get(model_type.lower(), {})
    scaling_eff_at_num = model_eff.get(num_gpus, None)

    # Additional advanced metrics
    activation_pressure = compute_fraction * (1 - overlap_ratio)
    bb_stress = comm_fraction * c2c * (rank_imbalance / 100)

    return {
        "num_ranks": num_gpus,
        "avg_wall_time": avg_wall,
        "max_wall_time": max_wall,
        "min_wall_time": min_wall,
        "avg_comm_time": avg_comm,
        "avg_compute_time": avg_compute,
        "comm_fraction": comm_fraction,
        "compute_fraction": compute_fraction,
        "c2c_ratio": c2c,
        "overlap_ratio": overlap_ratio,
        "rank_imbalance_pct": rank_imbalance,
        "throughput": throughput,
        "throughput_samples_mean": thr_mean,
        "temporal_instability": temporal_instability,
        "gpu_utilization_pct": gpu_util_mean,
        "gpu_util_variance": gpu_util_var,
        "global_batch_size": global_batch,
        "scaling_efficiency_pct": scaling_eff_at_num,
        "scaling_eff_curve": model_eff,
        "activation_pressure": activation_pressure,
        "bisection_bw_stress": bb_stress
    }


# ============================================================
# 7. WHAT-IF ANALYZER  (Feature #7)
# ============================================================

def simulate_what_if(metrics: dict, whatif_str: str) -> dict:
    """
    Example input:
        "topology=tree,ga=4,batch=64"

    What-It Can Modify:
        - GA (grad accumulation)
        - topology (tree, fc, ring)
        - batch size

    These heuristics are derived from your PDFs:
      - Tree topology often reduces comm ~15–25%
      - FC topology reduces R1 (congestion)
      - Increasing batch improves compute efficiency & GPU utilization
      - GA reduces comm per iteration
    """

    new = metrics.copy()

    # Parse
    parts = whatif_str.replace(" ", "").split(",")
    params = {}
    for p in parts:
        if "=" in p:
            k, v = p.split("=")
            params[k.strip()] = v.strip()

    # Apply hypothetical changes
    if "ga" in params:
        ga = int(params["ga"])
        # GA reduces comm cost inversely
        new["comm_fraction"] *= (1 / math.sqrt(ga))
        new["throughput"] *= math.sqrt(ga)

    if "batch" in params:
        b = int(params["batch"])
        # Larger batch → better GPU util
        factor = min(2.0, b / (metrics["global_batch_size"] / metrics["num_ranks"]))
        new["gpu_utilization_pct"] *= factor

    if "topology" in params:
        topo = params["topology"].lower()
        if topo == "tree":
            new["comm_fraction"] *= 0.82     # ~18% reduction
        elif topo == "fc":
            new["comm_fraction"] *= 0.75     # more aggressive improvement
            new["rank_imbalance_pct"] *= 0.9
        elif topo == "ring":
            new["comm_fraction"] *= 1.05     # ring usually worse than tree/fc

    # Recompute dependent metrics
    new["overlap_ratio"] = max(0, 1 - (new["comm_fraction"] + new["compute_fraction"]))
    return new

# ============================================================
# 🟦 PART 3: RULE ENGINE (R1–R10) + PRIORITIZER + CAUSAL GRAPH
# ============================================================


# ============================================================
# 8. RULE ENGINE (R1–R10)
# ============================================================

def apply_rules(metrics: dict, T: dict = RULE_THRESHOLDS) -> list:
    """
    Applies the 10 diagnostic rules:
      R1  – Network Congestion
      R2  – Compute Inefficiency
      R3  – Mixed Bottleneck
      R4  – Poor Overlap
      R5  – Straggler Delay
      R6  – Temporal Instability
      R7  – Arithmetic Bound
      R8  – Pipeline Imbalance
      R9  – Bandwidth Saturation
      R10 – Underutilized GPUs
    """

    cf = metrics["comm_fraction"]
    cpf = metrics["compute_fraction"]
    imb = metrics["rank_imbalance_pct"]
    ovl = metrics["overlap_ratio"]
    c2c = metrics["c2c_ratio"]
    gpu = metrics["gpu_utilization_pct"]
    eff = metrics.get("scaling_efficiency_pct", None)
    inst = metrics["temporal_instability"]
    var = metrics["gpu_util_variance"]

    rules = []

    # ---------------- R1: Network Congestion ----------------
    if cf > T["R1_comm_fraction"] and imb > T["R1_imbalance"]:
        rules.append({
            "rule": "R1",
            "type": "Network Congestion",
            "interpretation": "High communication fraction with notable rank imbalance implies link saturation or congested collectives.",
            "mitigation": "Try Tree or Fully-Connected topology; increase gradient accumulation; compress gradients."
        })

    # ---------------- R2: Compute Inefficiency ----------------
    if cpf > T["R2_compute_fraction"] and gpu < T["R2_gpu_util"]:
        rules.append({
            "rule": "R2",
            "type": "Compute Inefficiency",
            "interpretation": "High compute fraction but low GPU utilization suggests kernel scheduling issues or memory-bound operations.",
            "mitigation": "Increase per-GPU batch, enable mixed precision, reduce kernel launch overhead."
        })

    # ---------------- R3: Mixed Bottleneck ----------------
    if cf > T["R3_comm_fraction"] and cpf > T["R3_compute_fraction"]:
        rules.append({
            "rule": "R3",
            "type": "Mixed Bottleneck",
            "interpretation": "Both computation and communication significantly affect iteration latency.",
            "mitigation": "Use Tree topology and tune gradient accumulation to balance comm and compute."
        })

    # ---------------- R4: Poor Overlap Efficiency ----------------
    if ovl < T["R4_overlap_ratio"] and cf > T["R4_comm_fraction"]:
        rules.append({
            "rule": "R4",
            "type": "Poor Overlap Efficiency",
            "interpretation": "Communication is not overlapping effectively with computation.",
            "mitigation": "Enable async collectives; prefetch activations; reorder compute to increase concurrency."
        })

    # ---------------- R5: Straggler-Induced Delay ----------------
    if imb > T["R5_imbalance"]:
        rules.append({
            "rule": "R5",
            "type": "Straggler Delay",
            "interpretation": "Significant difference in per-rank wall times — some ranks act as stragglers.",
            "mitigation": "Rebalance data; ensure uniform hardware; use adaptive load balancing."
        })

    # ---------------- R6: Temporal Instability ----------------
    if inst > T["R6_temporal_instability"] and var > T["R6_gpu_util_var"]:
        rules.append({
            "rule": "R6",
            "type": "Temporal Instability",
            "interpretation": "Throughput fluctuates across iterations, likely due to transient congestion or jitter.",
            "mitigation": "Increase gradient accumulation; improve comm-compute overlap; stabilize network conditions."
        })

    # ---------------- R7: Arithmetic Bound ----------------
    if cpf > T["R7_compute_fraction"] and c2c < T["R7_c2c"]:
        rules.append({
            "rule": "R7",
            "type": "Arithmetic Bound",
            "interpretation": "Compute-bound workload; arithmetic throughput is the critical path.",
            "mitigation": "Increase model parallelism or batch size; no major communication optimizations needed."
        })

    # ---------------- R8: Pipeline Imbalance ----------------
    if cf > T["R8_comm_fraction"] and ovl > T["R8_overlap_ratio"] and gpu < T["R8_gpu_util"]:
        rules.append({
            "rule": "R8",
            "type": "Pipeline Imbalance",
            "interpretation": "Some overlap exists but compute underutilized — indicates pipeline stage imbalance.",
            "mitigation": "Reduce batch size per stage; optimize pipeline stage partitioning."
        })

    # ---------------- R9: Bandwidth Saturation ----------------
    if eff is not None and eff < T["R9_scaling_eff"] and cf > T["R9_comm_fraction"]:
        rules.append({
            "rule": "R9",
            "type": "Bandwidth Saturation",
            "interpretation": "Scaling efficiency is low and communication dominates — bandwidth likely saturated.",
            "mitigation": "Switch to higher-bandwidth topology (FC), compress communication, or increase gradient accumulation."
        })

    # ---------------- R10: Low GPU Utilization ----------------
    if gpu < T["R10_gpu_util"]:
        rules.append({
            "rule": "R10",
            "type": "Underutilized GPUs",
            "interpretation": "Average GPU utilization is low — hardware underused.",
            "mitigation": "Increase global batch size; reduce synchronization frequency; investigate kernel-level stalls."
        })

    return rules


# ============================================================
# 9. PRIORITIZER — Select Dominant Bottleneck
# ============================================================

PRIORITY_ORDER = [
    "Network Congestion",
    "Bandwidth Saturation",
    "Mixed Bottleneck",
    "Compute Inefficiency",
    "Poor Overlap Efficiency",
    "Straggler Delay",
    "Pipeline Imbalance",
    "Temporal Instability",
    "Underutilized GPUs",
    "Arithmetic Bound"
]

def prioritize(rules: list) -> dict:
    """Selects the dominant bottleneck using priority hierarchy."""
    if not rules:
        return None
    for t in PRIORITY_ORDER:
        for r in rules:
            if r["type"] == t:
                return r
    return rules[0]


# ============================================================
# 10. BOTTLENECK CAUSAL GRAPH (Mermaid + Graphviz fallback)
# ============================================================

def generate_causal_graph(metrics: dict, dominant: dict, outdir: Path) -> Path:
    """
    Creates a bottleneck causal graph:
       Dominant Bottleneck
            |
            +--> evidence nodes
            +--> metric thresholds crossed

    First attempts Mermaid (text-only), then Graphviz if available.
    Output is saved as PNG.
    """

    graph_txt = f"""
flowchart TD
    A[Dominant Bottleneck: {dominant['type']}]:::b
    A --> B[Comm Fraction = {metrics['comm_fraction']:.2f}]
    A --> C[C2C Ratio = {metrics['c2c_ratio']:.2f}]
    A --> D[Rank Imbalance = {metrics['rank_imbalance_pct']:.2f}%]
    A --> E[GPU Util = {metrics['gpu_utilization_pct']:.2f}%]
    classDef b fill:#ff9999,stroke:#333,stroke-width:2px;
"""

    mermaid_path = outdir / "bottleneck_causal_graph.mmd"
    mermaid_path.write_text(graph_txt)

    # Try Graphviz
    png_path = outdir / "bottleneck_causal_graph.png"
    try:
        import graphviz
        g = graphviz.Digraph()
        g.node("A", f"Dominant Bottleneck:\n{dominant['type']}", color="red", style="filled")
        g.node("B", f"Comm Fraction\n{metrics['comm_fraction']:.2f}")
        g.node("C", f"C2C Ratio\n{metrics['c2c_ratio']:.2f}")
        g.node("D", f"Rank Imbalance\n{metrics['rank_imbalance_pct']:.2f}%")
        g.node("E", f"GPU Util\n{metrics['gpu_utilization_pct']:.2f}%")

        g.edges([("A","B"),("A","C"),("A","D"),("A","E")])
        g.render(str(png_path), format="png", cleanup=True)
    except Exception:
        # If Graphviz unavailable, create a placeholder PNG
        plt.figure(figsize=(6,4))
        plt.text(0.01, 0.5, graph_txt, fontsize=8, va="center", ha="left")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()

    return png_path

# ============================================================
# 🟦 PART 4: GRAPH + HEATMAP GENERATION
# ============================================================

def save_plot(fig, outpath: Path):
    """Helper to save plots cleanly."""
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# ============================================================
# 11. GPU UTILIZATION PLOT
# ============================================================

def plot_gpu_utilization(ranks_data: dict, outdir: Path) -> Path:
    ranks = sorted(ranks_data.keys())
    utils = [
        ((ranks_data[r]['compute_time'] + ranks_data[r]['comm_time']) /
         ranks_data[r]['wall_time']) * 100
        if ranks_data[r]['wall_time'] > 0 else 0
        for r in ranks
    ]

    fig = plt.figure(figsize=(8, 4))
    plt.plot(ranks, utils, marker='o')
    plt.title("GPU Utilization per Rank")
    plt.xlabel("Rank")
    plt.ylabel("Utilization (%)")
    plt.grid(True)

    outpath = outdir / "gpu_utilization.png"
    save_plot(fig, outpath)
    return outpath


# ============================================================
# 12. PER-RANK WALL TIME BAR
# ============================================================

def plot_wall_times(ranks_data: dict, outdir: Path) -> Path:
    ranks = sorted(ranks_data.keys())
    walls = [ranks_data[r]['wall_time'] for r in ranks]

    fig = plt.figure(figsize=(8, 4))
    plt.bar(ranks, walls)
    plt.ylabel("Wall Time (cycles)")
    plt.xlabel("Rank")
    plt.title("Per-Rank Wall Time")
    plt.grid(axis='y', alpha=0.4)

    outpath = outdir / "wall_times.png"
    save_plot(fig, outpath)
    return outpath


# ============================================================
# 13. COMM VS COMPUTE STACKED BAR
# ============================================================

def plot_comm_vs_compute(ranks_data: dict, outdir: Path) -> Path:
    ranks = sorted(ranks_data.keys())
    comm = [ranks_data[r].get("comm_time", 0) for r in ranks]
    comp = [ranks_data[r]['compute_time'] for r in ranks]

    fig = plt.figure(figsize=(8, 4))
    plt.bar(ranks, comm, label="Comm")
    plt.bar(ranks, comp, bottom=comm, label="Compute")
    plt.xlabel("Rank")
    plt.ylabel("Cycles")
    plt.title("Communication + Compute Breakdown")
    plt.legend()

    outpath = outdir / "comm_vs_compute.png"
    save_plot(fig, outpath)
    return outpath


# ============================================================
# 14. CONGESTION HEATMAP
# ============================================================

def plot_congestion_heatmap(ranks_data: dict, outdir: Path) -> Path:
    """
    Uses rank-imbalance-like pattern to visualize congestion.
    Heatmap values = normalized (wall_time - compute_time).
    """
    ranks = sorted(ranks_data.keys())
    vals = [
        ranks_data[r]['comm_time'] / max(1, ranks_data[r]['wall_time'])
        for r in ranks
    ]

    fig = plt.figure(figsize=(7, 3))
    arr = np.array([vals])
    plt.imshow(arr, cmap="Reds", aspect="auto")
    plt.colorbar(label="Comm Fraction")
    plt.yticks([])
    plt.xticks(range(len(ranks)), ranks)
    plt.title("Communication Pressure Heatmap")

    outpath = outdir / "congestion_heatmap.png"
    save_plot(fig, outpath)
    return outpath


# ============================================================
# 15. C2C MATRIX HEATMAP
# ============================================================

def plot_c2c_matrix(ranks_data: dict, outdir: Path) -> Path:
    """
    Communication-to-Compute ratio per-rank shown as heatmap.
    """
    ranks = sorted(ranks_data.keys())
    c2c_vals = [
        ranks_data[r]['comm_time'] / ranks_data[r]['compute_time']
        if ranks_data[r]['compute_time'] > 0 else 0
        for r in ranks
    ]

    fig = plt.figure(figsize=(7, 3))
    arr = np.array([c2c_vals])
    plt.imshow(arr, cmap="viridis", aspect="auto")
    plt.colorbar(label="C2C Ratio")
    plt.yticks([])
    plt.xticks(range(len(ranks)), ranks)
    plt.title("C2C Ratio Heatmap")

    outpath = outdir / "c2c_heatmap.png"
    save_plot(fig, outpath)
    return outpath


# ============================================================
# 16. CRITICAL PATH PIE CHART
# ============================================================

def plot_critical_path(metrics: dict, outdir: Path) -> Path:
    labels = ["Communication", "Compute", "Overlap Lost"]
    comm = metrics["comm_fraction"]
    comp = metrics["compute_fraction"]
    lost = max(0, 1 - comm - comp)

    fig = plt.figure(figsize=(6, 6))
    plt.pie([comm, comp, lost], labels=labels, autopct="%1.1f%%",
            colors=["#ff9999", "#99ff99", "#9999ff"])
    plt.title("Critical Path Breakdown")

    outpath = outdir / "critical_path_pie.png"
    save_plot(fig, outpath)
    return outpath


# ============================================================
# 17. SCALING EFFICIENCY CURVE
# ============================================================

def plot_scaling_efficiency(metrics: dict, outdir: Path) -> Path:
    curve = metrics.get("scaling_eff_curve", {})
    if not curve:
        return None

    gpus = sorted(curve.keys())
    vals = [curve[g] for g in gpus]

    fig = plt.figure(figsize=(7, 4))
    plt.plot(gpus, vals, marker='o')
    plt.title("Scaling Efficiency Curve")
    plt.xlabel("GPUs")
    plt.ylabel("Efficiency (%)")
    plt.grid(True)

    outpath = outdir / "scaling_efficiency_curve.png"
    save_plot(fig, outpath)
    return outpath


# ============================================================
# 18. ACTIVATION PRESSURE BAR
# ============================================================

def plot_activation_pressure(metrics: dict, outdir: Path) -> Path:
    fig = plt.figure(figsize=(6, 4))
    plt.bar(["Pressure"], [metrics["activation_pressure"]], color="#ffcc88")
    plt.title("Activation Memory Pressure")
    plt.ylabel("Pressure Score")

    outpath = outdir / "activation_pressure.png"
    save_plot(fig, outpath)
    return outpath


# ============================================================
# 19. BISECTION BANDWIDTH STRESS
# ============================================================

def plot_bisection_stress(metrics: dict, outdir: Path) -> Path:
    fig = plt.figure(figsize=(6, 4))
    plt.bar(["BW Stress"], [metrics["bisection_bw_stress"]], color="#ff8888")
    plt.title("Bisection Bandwidth Stress Score")
    plt.ylabel("Stress Value")

    outpath = outdir / "bisection_bw_stress.png"
    save_plot(fig, outpath)
    return outpath


# ============================================================
# 20. MASTER GRAPH GENERATOR
# ============================================================

def generate_all_graphs(ranks_data: dict, metrics: dict, outdir: Path) -> dict:
    """
    Returns dict containing paths to all generated graphs.
    """

    paths = {}

    paths["wall_times"]        = plot_wall_times(ranks_data, outdir)
    paths["gpu_util"]          = plot_gpu_utilization(ranks_data, outdir)
    paths["comm_compute"]      = plot_comm_vs_compute(ranks_data, outdir)
    paths["congestion"]        = plot_congestion_heatmap(ranks_data, outdir)
    paths["c2c_heatmap"]       = plot_c2c_matrix(ranks_data, outdir)
    paths["critical_path"]     = plot_critical_path(metrics, outdir)
    paths["activation_press"]  = plot_activation_pressure(metrics, outdir)
    paths["bw_stress"]         = plot_bisection_stress(metrics, outdir)

    scale_path = plot_scaling_efficiency(metrics, outdir)
    if scale_path:
        paths["scaling_eff"] = scale_path

    return paths

# ============================================================
# 🟦 PART 5: LLaMA REASONING + FINGERPRINT DB + RECOMMENDATION ENGINE
# ============================================================


# ============================================================
# 21. FINGERPRINT DATABASE (Case Memory)
# ============================================================

FINGERPRINT_DB = "fingerprints.json"

def load_fingerprint_db():
    if Path(FINGERPRINT_DB).exists():
        try:
            return json.loads(Path(FINGERPRINT_DB).read_text())
        except:
            return []
    return []

def save_fingerprint(case: dict):
    db = load_fingerprint_db()
    db.append(case)
    Path(FINGERPRINT_DB).write_text(json.dumps(db, indent=2))


def compute_case_fingerprint(metrics: dict, model_type: str, dominant: dict):
    """
    Stores a compact 'case' record.
    Does NOT perform automatic model detection (as requested).
    """
    return {
        "timestamp": time.time(),
        "model_type": model_type,
        "dominant_bottleneck": dominant["type"] if dominant else None,
        "comm_fraction": metrics["comm_fraction"],
        "compute_fraction": metrics["compute_fraction"],
        "c2c_ratio": metrics["c2c_ratio"],
        "rank_imbalance_pct": metrics["rank_imbalance_pct"],
        "gpu_utilization_pct": metrics["gpu_utilization_pct"]
    }


def find_similar_cases(metrics: dict, top_k: int = 3):
    """
    Compares (CF, C2C, imbalance, GPU util) with past fingerprints.
    Returns the most similar cases (cosine distance on 4D vector).
    """

    db = load_fingerprint_db()
    if not db:
        return []

    import numpy as np

    v = np.array([
        metrics["comm_fraction"],
        metrics["c2c_ratio"],
        metrics["rank_imbalance_pct"],
        metrics["gpu_utilization_pct"]
    ], dtype=float)

    scored = []
    for case in db:
        vc = np.array([
            case["comm_fraction"],
            case["c2c_ratio"],
            case["rank_imbalance_pct"],
            case["gpu_utilization_pct"]
        ], dtype=float)

        # Cosine similarity
        if np.linalg.norm(v) * np.linalg.norm(vc) == 0:
            score = 0
        else:
            score = float(np.dot(v, vc) / (np.linalg.norm(v) * np.linalg.norm(vc)))

        scored.append((score, case))

    # Highest similarity first
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


# ============================================================
# 22. WHAT-IF EXPLANATION GENERATOR
# ============================================================

def whatif_explanation(original: dict, simulated: dict) -> str:
    return (
        f"Original throughput: {original['throughput']:.2f} → "
        f"{simulated['throughput']:.2f} samples/sec\n"
        f"Comm fraction: {original['comm_fraction']:.2f} → "
        f"{simulated['comm_fraction']:.2f}\n"
        f"GPU util: {original['gpu_utilization_pct']:.2f}% → "
        f"{simulated['gpu_utilization_pct']:.2f}%\n"
        f"Overlap ratio: {original['overlap_ratio']:.2f} → "
        f"{simulated['overlap_ratio']:.2f}\n"
    )


# ============================================================
# 23. LLaMA EXPERT ANALYSIS GENERATOR
# ============================================================

def generate_llama_expert_commentary(metrics: dict, rules: list, dominant: dict, similar_cases: list) -> str:
    """
    Sends a structured prompt to Ollama.
    Hidden chain-of-thought is ignored; model outputs a short expert analysis.
    """

    prompt = f"""
You are an expert in distributed deep learning performance.

Metrics:
{json.dumps(metrics, indent=2)}

Rules triggered:
{json.dumps(rules, indent=2)}

Dominant bottleneck:
{json.dumps(dominant, indent=2)}

Similar historical cases:
{json.dumps(similar_cases, indent=2)}

Write a concise expert analysis (max 200 words).
Include:
1. What the metrics tell us.
2. Why the dominant bottleneck happened.
3. Practical, low-risk next steps.
Avoid disclaimers. Use technical language.
"""

    return call_ollama(prompt)


# ============================================================
# 24. COMBINED RECOMMENDATION REPORTER
# ============================================================

def generate_recommendation(metrics: dict, rules: list, dominant: dict, similar_cases: list) -> str:
    """
    Merges:
      - R1–R10 rule outputs
      - Similar case memories
      - LLaMA expert commentary
    """
    text = ""

    # Rule summary
    if rules:
        text += "Triggered Bottleneck Rules:\n"
        for r in rules:
            text += f"• [{r['rule']}] {r['type']}: {r['interpretation']}\n"
            text += f"  Mitigation: {r['mitigation']}\n"
        text += "\n"

    # Similar cases
    if similar_cases:
        text += "Similar Past Cases:\n"
        for c in similar_cases:
            text += (
                f"• {c['model_type']} case with {c['dominant_bottleneck']}, "
                f"CF={c['comm_fraction']:.2f}, C2C={c['c2c_ratio']:.2f}\n"
            )
        text += "\n"

    # LLaMA expert
    llama_comment = generate_llama_expert_commentary(metrics, rules, dominant, similar_cases)
    text += "Expert Analysis:\n" + llama_comment + "\n"

    return text

# ============================================================
# 🟦 PART 6: DASHBOARD PDF (Landscape A4, Single Page)
# ============================================================

def generate_dashboard_pdf(
    output_path: Path,
    metrics: dict,
    dominant: dict,
    rules: list,
    recommendations: str,
    graph_paths: dict,
    whatif_text: str = None
):
    """
    Creates a landscape A4 single-page dashboard using reportlab.

    Layout:
    ------------------------------------------------------------
    | Title + metadata                                         |
    ------------------------------------------------------------
    | Column 1 (Graphs)        | Column 2 (Metrics + Text)     |
    ------------------------------------------------------------
    """

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(A4),
        rightMargin=20, leftMargin=20,
        topMargin=20, bottomMargin=20
    )

    styles = getSampleStyleSheet()
    body = []

    # ------------------------------------------------------------
    # Title
    # ------------------------------------------------------------
    title = f"<para align='center'><b>Distributed Training Bottleneck Dashboard</b></para>"
    body.append(Paragraph(title, styles['Title']))
    body.append(Spacer(1, 12))

    # ------------------------------------------------------------
    # Summary line
    # ------------------------------------------------------------
    summary = (
        f"Model: <b>{metrics.get('model_type', 'unknown').upper()}</b> | "
        f"GPUs: <b>{metrics['num_ranks']}</b> | "
        f"Batch: <b>{metrics['global_batch_size']}</b> | "
        f"Dominant Bottleneck: <b>{dominant['type'] if dominant else 'None'}</b>"
    )
    body.append(Paragraph(summary, styles['Heading3']))
    body.append(Spacer(1, 10))

    # ------------------------------------------------------------
    # 2-column dashboard
    # ------------------------------------------------------------
    col1 = []
    col2 = []

    # ---------------- COL 1: Graphs ----------------------------

    def add_graph_if_exists(name):
        if name in graph_paths:
            col1.append(Image(str(graph_paths[name]), width=270, height=150))
            col1.append(Spacer(1, 8))

    add_graph_if_exists("wall_times")
    add_graph_if_exists("gpu_util")
    add_graph_if_exists("comm_compute")
    add_graph_if_exists("critical_path")
    add_graph_if_exists("congestion")
    add_graph_if_exists("c2c_heatmap")

    # ---------------- COL 2: Metrics / Text ---------------------

    # Metrics Table
    data = [
        ["Metric", "Value"],
        ["Avg Wall Time", f"{metrics['avg_wall_time']:.2f}"],
        ["Comm Fraction", f"{metrics['comm_fraction']:.3f}"],
        ["Compute Fraction", f"{metrics['compute_fraction']:.3f}"],
        ["C2C Ratio", f"{metrics['c2c_ratio']:.2f}"],
        ["Rank Imbalance (%)", f"{metrics['rank_imbalance_pct']:.2f}"],
        ["GPU Util (%)", f"{metrics['gpu_utilization_pct']:.2f}"],
        ["Temporal Instability", f"{metrics['temporal_instability']:.3f}"],
        ["Activation Pressure", f"{metrics['activation_pressure']:.3f}"],
        ["BW Stress", f"{metrics['bisection_bw_stress']:.3f}"],
    ]

    tbl = Table(data, colWidths=[150, 200])
    tbl.setStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("BACKGROUND", (0,1), (-1,-1), colors.whitesmoke),
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
    ])
    col2.append(tbl)
    col2.append(Spacer(1, 12))

    # Dominant bottleneck description
    if dominant:
        col2.append(Paragraph("<b>Dominant Bottleneck:</b>", styles['Heading4']))
        col2.append(Paragraph(f"{dominant['type']}", styles['BodyText']))
        col2.append(Paragraph(f"{dominant['interpretation']}", styles['BodyText']))
        col2.append(Spacer(1, 10))

    # Rules triggered
    if rules:
        col2.append(Paragraph("<b>Triggered Rules:</b>", styles['Heading4']))
        for r in rules:
            col2.append(Paragraph(f"<b>[{r['rule']}] {r['type']}</b>", styles['BodyText']))
            col2.append(Paragraph(r["mitigation"], styles['BodyText']))
        col2.append(Spacer(1, 10))

    # What-if section
    if whatif_text:
        col2.append(Paragraph("<b>What-If Prediction:</b>", styles['Heading4']))
        col2.append(Paragraph(whatif_text.replace("\n", "<br/>"), styles['BodyText']))
        col2.append(Spacer(1, 10))

    # Expert analysis
    col2.append(Paragraph("<b>Expert Analysis (LLM):</b>", styles['Heading4']))
    col2.append(Paragraph(recommendations.replace("\n", "<br/>"), styles['BodyText']))
    col2.append(Spacer(1, 10))

    # ------------------------------------------------------------
    # Assemble columns in a table
    # ------------------------------------------------------------
    dash = Table(
        [[col1, col2]],
        colWidths=[300, 450]
    )
    dash.setStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ])
    body.append(dash)

    # ------------------------------------------------------------
    # Build PDF
    # ------------------------------------------------------------
    doc.build(body)

    return output_path
# ============================================================
# 🟦 PART 7: MAIN CLI + ASSEMBLY LOGIC
# ============================================================

def main():
    print_instructions()

    # ------------------------------------------------------------
    # Ask for .log file (allow ENTER to skip)
    # ------------------------------------------------------------
    log_file = input("Enter path to Astra-Sim .log file (or press ENTER to provide later): ").strip()
    while not log_file:
        log_file = input("Please enter path to .log file: ").strip()

    log_path = Path(log_file)
    if not log_path.exists():
        print(f"[ERROR] File not found: {log_path}")
        return

    is_distributed = input("Is this for distributed training? (yes/no): ").strip().lower()
    if is_distributed != "yes":
        print("This tool is for distributed training analysis only.")
        return

    model_type = input("Model type (resnet / bert / dlrm): ").strip().lower()
    if model_type not in ["resnet", "bert", "dlrm"]:
        print("[ERROR] Unsupported model type.")
        return

    try:
        batch_size = int(input("Per-GPU batch size: ").strip())
    except:
        print("[ERROR] Invalid batch size.")
        return

    whatif_str = None
    do_whatif = input("Do you want to enter What-If mode? (type 'what-if' or press ENTER to skip): ").strip()
    if do_whatif.lower() == "what-if":
        whatif_str = input("Enter what-if parameters (e.g., topology=tree,ga=4,batch=64): ").strip()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(f"analysis_{timestamp}")
    outdir.mkdir(parents=True, exist_ok=True)

    print("\nParsing traces...")
    ranks_data = parse_astrasim_log(log_path)
    if not ranks_data:
        print("[ERROR] No ranks found in log.")
        return

    print("Computing metrics...")
    metrics = compute_metrics(ranks_data, batch_size, model_type, DEFAULT_SCALING_EFF)
    metrics["model_type"] = model_type 

    print("Applying R1–R10 rules...")
    rules = apply_rules(metrics)
    dominant = prioritize(rules)

    whatif_output = None
    if whatif_str:
        print("Simulating What-If scenario...")
        simulated = simulate_what_if(metrics, whatif_str)
        whatif_output = whatif_explanation(metrics, simulated)

    similar_cases = find_similar_cases(metrics)

    print("Querying LLaMA for expert analysis...")
    recommendations = generate_recommendation(metrics, rules, dominant, similar_cases)

    save_fingerprint(compute_case_fingerprint(metrics, model_type, dominant))

    print("Generating graphs...")
    graph_paths = generate_all_graphs(ranks_data, metrics, outdir)
    graph_paths["causal_graph"] = generate_causal_graph(metrics, dominant, outdir)

    json_path = outdir / "analysis.json"
    json_path.write_text(json.dumps({
        "metrics": metrics,
        "rules": rules,
        "dominant_bottleneck": dominant,
        "recommendations": recommendations
    }, indent=2))

    df = pd.DataFrame([metrics])
    csv_path = outdir / "metrics.csv"
    df.to_csv(csv_path, index=False)

    txt_path = outdir / "report.txt"
    with open(txt_path, "w") as f:
        f.write("=== AI-Agent Distributed Training Analysis ===\n\n")
        f.write("Model: " + model_type.upper() + "\n")
        f.write("Num GPUs: " + str(metrics["num_ranks"]) + "\n")
        f.write("\n--- Metrics ---\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\n--- Dominant Bottleneck ---\n")
        if dominant:
            f.write(dominant["type"] + "\n")
            f.write(dominant["interpretation"] + "\n")
        f.write("\n--- Rules Triggered ---\n")
        for r in rules:
            f.write(f"[{r['rule']}] {r['type']} → {r['mitigation']}\n")
        if whatif_output:
            f.write("\n--- What-If Scenario ---\n" + whatif_output + "\n")
        f.write("\n--- Expert Analysis ---\n")
        f.write(recommendations)

    print("Generating PDF dashboard...")
    pdf_path = outdir / "dashboard.pdf"
    generate_dashboard_pdf(
        pdf_path, metrics, dominant, rules, recommendations, graph_paths, whatif_output
    )

    print("\n===================================================")
    print(" Analysis complete!")
    print(" Output folder:", outdir)
    print(" PDF dashboard:", pdf_path)
    print("===================================================\n")

if __name__ == "__main__":
    main()

