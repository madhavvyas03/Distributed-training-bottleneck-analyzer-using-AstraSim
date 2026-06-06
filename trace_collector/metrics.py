# astrasim_analyzer/metrics.py

from dataclasses import dataclass, asdict
from typing import Dict
import pandas as pd


@dataclass
class PerformanceMetrics:
    total_nodes: int
    topology_type: str
    batch_size: int

    wall_time_ms: float
    comm_time_ms: float
    compute_time_ms: float
    idle_time_ms: float

    comm_fraction: float
    compute_fraction: float
    idle_fraction: float

    c2c_ratio: float
    overlap_ratio: float
    rank_imbalance_percent: float

    throughput_samples_per_sec: float
    scaling_efficiency_percent: float
    gpu_utilization_percent: float

    congestion_score: float
    congestion_type: str

    synchrony_state: str
    bottleneck_type: str
    severity: str

    temporal_instability_index: float


class MetricsEngine:

    def compute(self, parsed, iter_time_sec: float) -> PerformanceMetrics:

        df = parsed.rank_df
        topology = parsed.topology
        batch_size = parsed.batch_size

        num_nodes = len(df)

        avg_wall_cycles = df["wall_time_cycles"].mean()
        avg_comm_cycles = df["comm_time_cycles"].mean()
        avg_compute_cycles = df["compute_time_cycles"].mean()

        wall_ms = avg_wall_cycles * 1e-6
        comm_ms = avg_comm_cycles * 1e-6
        compute_ms = avg_compute_cycles * 1e-6
        idle_ms = wall_ms - comm_ms - compute_ms

        comm_frac = comm_ms / wall_ms if wall_ms > 0 else 0
        compute_frac = compute_ms / wall_ms if wall_ms > 0 else 0
        idle_frac = max(0.0, 1.0 - comm_frac - compute_frac)

        c2c = (comm_ms / compute_ms) if compute_ms > 0 else float("inf")

        overlap_ratio = max(0.0, 1.0 - comm_frac)

        min_w = df["wall_time_cycles"].min()
        max_w = df["wall_time_cycles"].max()
        imbalance = (max_w - min_w) / avg_wall_cycles * 100 if avg_wall_cycles > 0 else 0

        global_batch = batch_size * num_nodes
        throughput = global_batch / iter_time_sec

        gpu_util = compute_frac * 100
        scaling_eff = max(0.0, 100 - (comm_frac * 100 * 0.75))

        congestion_score = (comm_frac * 0.6) + ((imbalance / 100) * 0.4)

        if imbalance < 5 and comm_frac > 0.9:
            congestion_type = "Uniform_Global"
        elif imbalance > 10:
            congestion_type = "Localized"
        else:
            congestion_type = "Moderate_Global"

        std_cycles = df["wall_time_cycles"].std()
        synchrony = "Fully Synchronous" if std_cycles < 1 else "Asynchronous"

        if comm_frac > 0.7:
            bottleneck = "Communication"
        elif compute_frac > 0.7:
            bottleneck = "Compute"
        else:
            bottleneck = "Mixed"

        if comm_frac >= 0.92 or c2c > 4.0:
            severity = "CATASTROPHIC"
        elif comm_frac >= 0.75 or c2c > 2.0:
            severity = "CRITICAL"
        elif comm_frac >= 0.55:
            severity = "HIGH"
        else:
            severity = "MEDIUM"

        temporal_instability = round(imbalance * comm_frac, 3)

        return PerformanceMetrics(
            total_nodes=num_nodes,
            topology_type=topology,
            batch_size=batch_size,

            wall_time_ms=wall_ms,
            comm_time_ms=comm_ms,
            compute_time_ms=compute_ms,
            idle_time_ms=idle_ms,

            comm_fraction=comm_frac,
            compute_fraction=compute_frac,
            idle_fraction=idle_frac,

            c2c_ratio=c2c,
            overlap_ratio=overlap_ratio,
            rank_imbalance_percent=imbalance,

            throughput_samples_per_sec=throughput,
            scaling_efficiency_percent=scaling_eff,
            gpu_utilization_percent=gpu_util,

            congestion_score=congestion_score,
            congestion_type=congestion_type,

            synchrony_state=synchrony,
            bottleneck_type=bottleneck,
            severity=severity,

            temporal_instability_index=temporal_instability
        )


def compute_metrics(parsed, iter_time_sec: float) -> PerformanceMetrics:
    engine = MetricsEngine()
    return engine.compute(parsed, iter_time_sec)


def serialize_metrics(metrics: PerformanceMetrics) -> Dict:
    return asdict(metrics)
