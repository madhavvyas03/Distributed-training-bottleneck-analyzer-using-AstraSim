# astrasim_analyzer/rule_engine.py

from typing import Dict, List
from dataclasses import dataclass
from metrics import PerformanceMetrics
from parser import ParsedLog


@dataclass
class RuleResult:
    rule_id: str
    triggered: bool
    explanation: str
    metrics_used: Dict


@dataclass
class RuleEngineOutput:
    triggered_rules: List[RuleResult]
    dominant_bottleneck: str
    mitigation_cluster: str
    summary: str
    parsed_data: ParsedLog        # <-- REQUIRED BY AI AGENT


class RuleEngine:
    """
    Implements R1–R10 from the report exactly.
    """

    def evaluate(self, m: PerformanceMetrics, parsed: ParsedLog) -> RuleEngineOutput:
        rules = []

        # ---- R1 ------------------------------------------------------------
        r1 = m.comm_fraction > 0.45 and m.rank_imbalance_percent > 5
        rules.append(RuleResult(
            "R1", r1,
            "Network Congestion: High comm_fraction with noticeable rank imbalance.",
            {"comm_fraction": m.comm_fraction, "rank_imbalance_percent": m.rank_imbalance_percent}
        ))

        # ---- R2 ------------------------------------------------------------
        r2 = m.compute_fraction > 0.70 and m.gpu_utilization_percent < 60
        rules.append(RuleResult(
            "R2", r2,
            "Compute Inefficiency: compute_fraction high but GPU util low.",
            {"compute_fraction": m.compute_fraction, "gpu_utilization": m.gpu_utilization_percent}
        ))

        # ---- R3 ------------------------------------------------------------
        r3 = m.comm_fraction > 0.35 and m.compute_fraction > 0.55
        rules.append(RuleResult(
            "R3", r3,
            "Mixed Bottleneck: both compute and communication heavily loaded.",
            {"comm_fraction": m.comm_fraction, "compute_fraction": m.compute_fraction}
        ))

        # ---- R4 ------------------------------------------------------------
        r4 = m.overlap_ratio < 0.25 and m.comm_fraction > 0.40
        rules.append(RuleResult(
            "R4", r4,
            "Poor Overlap Efficiency.",
            {"overlap_ratio": m.overlap_ratio, "comm_fraction": m.comm_fraction}
        ))

        # ---- R5 ------------------------------------------------------------
        r5 = m.rank_imbalance_percent > 7
        rules.append(RuleResult(
            "R5", r5,
            "Straggler-Induced Synchronization Delay.",
            {"rank_imbalance_percent": m.rank_imbalance_percent}
        ))

        # ---- R6 ------------------------------------------------------------
        r6 = m.temporal_instability_index > 0.10
        rules.append(RuleResult(
            "R6", r6,
            "Temporal Instability.",
            {"temporal_instability_index": m.temporal_instability_index}
        ))

        # ---- R7 ------------------------------------------------------------
        r7 = m.compute_fraction > 0.75 and m.c2c_ratio < 0.20
        rules.append(RuleResult(
            "R7", r7,
            "Arithmetic/FLOP-Limited (no mitigation).",
            {"compute_fraction": m.compute_fraction, "c2c_ratio": m.c2c_ratio}
        ))

        # ---- R8 ------------------------------------------------------------
        r8 = m.comm_fraction > 0.55 and m.overlap_ratio > 0.5 and m.gpu_utilization_percent < 65
        rules.append(RuleResult(
            "R8", r8,
            "Pipeline Imbalance.",
            {"comm_fraction": m.comm_fraction, "overlap_ratio": m.overlap_ratio,
             "gpu_utilization": m.gpu_utilization_percent}
        ))

        # ---- R9 ------------------------------------------------------------
        r9 = m.scaling_efficiency_percent < 70 and m.comm_fraction > 50
        rules.append(RuleResult(
            "R9", r9,
            "Bandwidth Saturation.",
            {"scaling_efficiency": m.scaling_efficiency_percent, "comm_fraction": m.comm_fraction}
        ))

        # ---- R10 -----------------------------------------------------------
        r10 = m.idle_fraction > 0.05
        rules.append(RuleResult(
            "R10", r10,
            "Synchronization Barrier Stall.",
            {"idle_fraction": m.idle_fraction}
        ))

        # ------------------------------------------------------------------
        # Determine dominant bottleneck priority
        # ------------------------------------------------------------------
        triggered = [r for r in rules if r.triggered]

        if not triggered:
            dominant = "No Major Bottleneck"
            mitigation = "No Action Required"
        else:
            priority = ["R1", "R3", "R9", "R4", "R5", "R8", "R2", "R10", "R6", "R7"]
            dominant_rule = next((p for p in priority if any(r.rule_id == p and r.triggered for r in rules)), None)
            dominant = dominant_rule

            if dominant_rule in ["R1", "R9"]:
                mitigation = "Network / Topology Optimization"
            elif dominant_rule in ["R3", "R4", "R8"]:
                mitigation = "Overlap & Pipeline Optimization"
            elif dominant_rule == "R2":
                mitigation = "Compute Kernel Optimization"
            elif dominant_rule in ["R5", "R10"]:
                mitigation = "Load Balancing"
            elif dominant_rule == "R7":
                mitigation = "No Mitigation Required"
            else:
                mitigation = "General Optimization"

        # ------------------------------------------------------------------
        # Summary text
        # ------------------------------------------------------------------
        if not triggered:
            summary = "No rules triggered. System performing normally."
        else:
            lines = ["Triggered Rules:"]
            for r in triggered:
                lines.append(f" - {r.rule_id}: {r.explanation}")
            lines.append(f"\nDominant Bottleneck: {dominant}")
            lines.append(f"Mitigation Cluster: {mitigation}")
            summary = "\n".join(lines)

        # ------------------------------------------------------------------
        return RuleEngineOutput(
            triggered_rules=triggered,
            dominant_bottleneck=dominant,
            mitigation_cluster=mitigation,
            summary=summary,
            parsed_data=parsed    # <-- REQUIRED FIX
        )


    # ----------------------------------------------------------------------
    def _build_summary(self, triggered, dominant, mitigation):
        if not triggered:
            return "No rules triggered. System performing within normal operational envelope."

        lines = []
        lines.append("Triggered Rules:")
        for r in triggered:
            lines.append(f" - {r.rule_id}: {r.explanation}")

        lines.append(f"\nDominant Bottleneck: {dominant}")
        lines.append(f"Recommended Mitigation: {mitigation}")

        return "\n".join(lines)
