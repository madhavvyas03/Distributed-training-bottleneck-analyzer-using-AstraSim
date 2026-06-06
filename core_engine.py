# =====================================================================
#  CORE ENGINE FOR ASTRA-SIM PERFORMANCE ANALYSIS
#  (LOG PARSING + METRIC COMPUTATION + BOTTLENECK CLASSIFICATION)
#  THIS BLOCK CONTAINS ZERO AI / ZERO VISUALIZATION
# =====================================================================

import re
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict


# ============================================================
#               METRICS STRUCTURE (AS PER REPORT)
# ============================================================

@dataclass
class PerformanceMetrics:
    total_nodes: int
    topology_type: str

    wall_time_ms: float
    comm_time_ms: float
    compute_time_ms: float
    idle_time_ms: float

    comm_fraction: float
    compute_fraction: float
    c2c_ratio: float

    rank_imbalance_percent: float

    throughput_samples_per_sec: float
    scaling_efficiency_percent: float
    gpu_utilization_percent: float

    congestion_score: float
    congestion_type: str

    synchrony_state: str

    bottleneck_type: str
    severity: str


# =====================================================================
#                           CORE ENGINE
# =====================================================================

class TraceEngine:
    """
    Core engine:
    - Parses ASTRA-sim log
    - Computes all metrics required for the report
    - Computes throughput = Global Batch Size / wall_time(sec)
    - Detects bottleneck based on R1–R9 rules
    - Computes congestion score + synchrony + severity
    """

    def __init__(self, batch_size_per_gpu: int = 32):
        self.batch_size = batch_size_per_gpu

    # ---------------------------------------------------------------
    #                    LOG PARSING
    # ---------------------------------------------------------------

    def parse_trace(self, text: str):
        """Returns (topology, DataFrame[ranks])"""

        topo = "Unknown"
        topo_match = re.search(r"\[system::topology::(\w+)\]", text)
        if topo_match:
            topo = topo_match.group(1)

        pattern = r'sys\[(\d+)\].*?Wall time:\s*(\d+).*?Comm time:\s*(\d+)'
        matches = re.findall(pattern, text, re.DOTALL)

        rank_data = []
        for rank_id, wall, comm in matches:
            wall = int(wall)
            comm = int(comm)
            rank_data.append({
                "rank": int(rank_id),
                "wall_time": wall,
                "comm_time": comm,
                "compute_time": wall - comm
            })

        df = pd.DataFrame(rank_data)
        return topo, df

    # ---------------------------------------------------------------
    #                    METRIC COMPUTATION
    # ---------------------------------------------------------------

    def compute_metrics(self, topo: str, df: pd.DataFrame):
        """Computes ALL metrics exactly as per your report"""

        avg_wall = df["wall_time"].mean()
        avg_comm = df["comm_time"].mean()
        avg_comp = df["compute_time"].mean()

        # → convert cycles → ms
        wall_ms = avg_wall / 1000
        comm_ms = avg_comm / 1000
        comp_ms = avg_comp / 1000
        idle_ms = wall_ms - comm_ms - comp_ms

        comm_frac = avg_comm / avg_wall if avg_wall > 0 else 0
        compute_frac = avg_comp / avg_wall if avg_wall > 0 else 0

        c2c = avg_comm / avg_comp if avg_comp > 0 else float("inf")

        imbalance = (
            (df["wall_time"].max() - df["wall_time"].min()) / avg_wall * 100
            if avg_wall > 0 else 0
        )

        # -----------------------------------------------------------
        # THROUGHPUT USING YOUR FORMULA:
        #   throughput = GlobalBatchSize / wall_time(sec)
        # -----------------------------------------------------------
        wall_sec = wall_ms / 1000
        gbs = len(df) * self.batch_size
        throughput = gbs / wall_sec if wall_sec > 0 else 0

        gpu_util = compute_frac * 100

        scaling_eff = max(0, 100 - (comm_frac * 100 * 0.75))

        congestion_score = (comm_frac * 0.6) + (imbalance / 100 * 0.4)

        if imbalance < 5 and comm_frac > 0.9:
            congestion_type = "Uniform_Global"
        elif imbalance > 10:
            congestion_type = "Localized"
        else:
            congestion_type = "Moderate_Global"

        synchrony_state = "Fully Synchronous" if df["wall_time"].std() < 1 else "Asynchronous"

        # ============================================================
        #           BOTTLENECK CLASSIFICATION (R1–R9)
        # ============================================================

        bottleneck = self.classify_bottleneck(
            comm_frac, compute_frac, gpu_util, imbalance, c2c, scaling_eff
        )

        # Severity
        if comm_frac >= 0.95 or c2c > 5:
            severity = "CATASTROPHIC"
        elif comm_frac >= 0.8 or c2c > 1.5:
            severity = "CRITICAL"
        else:
            severity = "MEDIUM"

        return PerformanceMetrics(
            total_nodes=len(df),
            topology_type=topo,

            wall_time_ms=wall_ms,
            comm_time_ms=comm_ms,
            compute_time_ms=comp_ms,
            idle_time_ms=idle_ms,

            comm_fraction=comm_frac,
            compute_fraction=compute_frac,
            c2c_ratio=c2c,

            rank_imbalance_percent=imbalance,

            throughput_samples_per_sec=throughput,
            scaling_efficiency_percent=scaling_eff,
            gpu_utilization_percent=gpu_util,

            congestion_score=congestion_score,
            congestion_type=congestion_type,

            synchrony_state=synchrony_state,

            bottleneck_type=bottleneck,
            severity=severity
        )

    # ---------------------------------------------------------------
    #                BOTTLENECK CLASSIFICATION RULES
    # ---------------------------------------------------------------

    def classify_bottleneck(
        self, comm_frac, comp_frac, gpu_util, imbalance, c2c, scaling_eff
    ):

        # R1 — Network Congestion
        if comm_frac > 0.45 and imbalance > 5:
            return "Network Congestion"

        # R2 — Compute Inefficiency
        if comp_frac > 0.70 and gpu_util < 60:
            return "Compute Inefficiency"

        # R3 — Mixed Bottleneck
        if comm_frac > 0.35 and comp_frac > 0.55:
            return "Mixed Bottleneck"

        # R4 — Straggler Sync Delay
        if imbalance > 7:
            return "Straggler-Induced Synchronization Delay"

        # R5 — Temporal Instability
        # removed p99/p50 as requested
        # (you removed these metrics in the report)

        # R6 — Arithmetic Bound
        if comp_frac > 0.75 and c2c < 0.2:
            return "Arithmetic Bound"

        # R7 — Pipeline Imbalance
        if comm_frac > 0.55 and gpu_util < 65:
            return "Pipeline Imbalance"

        # R8 — Bandwidth Saturation
        if scaling_eff < 70 and comm_frac > 0.5:
            return "Bandwidth Saturation"

        return "Balanced / No Dominant Bottleneck"


# =====================================================================
#                    STANDALONE ENGINE FUNCTION
# =====================================================================

def run_core_engine(trace_text: str, batch_size: int = 32):
    engine = TraceEngine(batch_size)
    topo, df = engine.parse_trace(trace_text)
    metrics = engine.compute_metrics(topo, df)
    return metrics, df
