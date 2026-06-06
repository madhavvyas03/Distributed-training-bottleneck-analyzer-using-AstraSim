import requests
import time
from typing import Dict
from metrics import PerformanceMetrics
from rule_engine import RuleEngineOutput


class AIAgent:

    def __init__(self, model="llama3", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.available = self._check_ollama()


    def generate(self,
                 metrics: PerformanceMetrics,
                 rules: RuleEngineOutput):
        parsed_data = rules.parsed_data
        return self.analyze(metrics, rules, parsed_data)

    # ----------------------------------------------------------------------
    def _check_ollama(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except:
            return False

    # ----------------------------------------------------------------------
    def _ask(self, prompt: str) -> str:
        if not self.available:
            return "LLM unavailable — fallback to rule-based reasoning."

        for _ in range(3):
            try:
                r = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.25,
                            "top_p": 0.9,
                            "num_predict": 500
                        }
                    },
                    timeout=60
                )
                if r.status_code == 200:
                    return r.json().get("response", "").strip()
            except:
                time.sleep(1)
        return "LLM unavailable — fallback reasoning."

    # ----------------------------------------------------------------------
    def analyze(self,
                metrics: PerformanceMetrics,
                rules: RuleEngineOutput,
                parsed_data: Dict):

        ctx = self._build_context(metrics, rules, parsed_data)

        return {
            "bottleneck_analysis": self._ask(self._prompt_bottleneck(ctx)),
            "root_cause": self._ask(self._prompt_root_cause(ctx)),
            "recommendations": self._ask(self._prompt_recommendations(ctx)),
            "implementation": self._ask(self._prompt_implementation(ctx)),
            "risks": self._ask(self._prompt_risks(ctx))
        }

    # ----------------------------------------------------------------------
    def _build_context(self, m, rules, parsed):
        rank_df = parsed.rank_df

        if rules.triggered_rules:
            tr = "\n".join([f"- {r.rule_id}: {r.explanation}"
                            for r in rules.triggered_rules])
        else:
            tr = "None"

        return f"""
==============================
PERFORMANCE ANALYSIS CONTEXT
==============================

Topology: {m.topology_type}
Nodes: {m.total_nodes}

Performance:
- Wall Time: {m.wall_time_ms:.2f} ms
- Comm Time: {m.comm_time_ms:.2f} ms ({m.comm_fraction:.2f})
- Compute Time: {m.compute_time_ms:.2f} ms ({m.compute_fraction:.2f})
- Idle Time: {m.idle_time_ms:.2f} ms ({m.idle_fraction:.2f})

- C2C Ratio: {m.c2c_ratio:.3f}
- Overlap Ratio: {m.overlap_ratio:.3f}
- Rank Imbalance: {m.rank_imbalance_percent:.2f}%
- Throughput: {m.throughput_samples_per_sec:.2f} samples/sec
- GPU Utilization: {m.gpu_utilization_percent:.2f}%
- Scaling Efficiency: {m.scaling_efficiency_percent:.2f}%

Congestion:
- Score: {m.congestion_score:.3f}
- Type: {m.congestion_type}

Synchrony State: {m.synchrony_state}
Severity: {m.severity}
Detected Bottleneck: {m.bottleneck_type}

Triggered Rules:
{tr}

Dominant Bottleneck (Rule Engine): {rules.dominant_bottleneck}
Mitigation Cluster: {rules.mitigation_cluster}

Rank Stats:
- Mean: {rank_df['wall_time_cycles'].mean():.2f}
- Std: {rank_df['wall_time_cycles'].std():.2f}
"""

    # ----------------------------------------------------------------------
    def _prompt_bottleneck(self, ctx):
        return f"""
{ctx}

Provide a 3–4 sentence **Bottleneck Analysis** using ONLY these rules:

R1: comm_fraction > 0.45 AND imbalance > 5%
R2: compute_fraction > 0.70 AND gpu_util < 60%
R3: comm_fraction > 0.35 AND compute_fraction > 0.55
R4: overlap_ratio < 0.25 AND comm_fraction > 0.40
R5: rank_imbalance > 7%
R6: temporal_instability_index > 0.10
R7: compute_fraction > 0.75 AND c2c < 0.20
R8: comm_fraction > 0.55 AND overlap_ratio > 0.5 AND gpu_util < 65%
R9: scaling_eff < 70% AND comm_fraction > 0.5
R10: idle_fraction > 0.05
"""

    def _prompt_root_cause(self, ctx):
        return f"""
{ctx}


Provide a Root Cause Analysis (3–4 sentences) using ONLY the concepts, terminology, 
and system behaviors described in the project report.

Your explanation MUST explicitly reference the following:

1. How the chosen topology (Ring / Tree / FC) shapes the communication dependency chain 
   and exposes or hides communication, as described in the report.

2. How the communication pattern (especially all-reduce) interacts with per-hop latency, 
   link saturation, and synchronization behavior in Astra-Sim’s analytical backend.

3. How the measured C2C ratio aligns with typical model behavior from the report:
   - BERT → balanced but communication-sensitive in Ring
   - ResNet → compute-dominant with low C2C
   - DLRM → communication-heavy with high exposed communication

4. Frame the root cause ONLY using mechanisms described in the report such as:
   “Moderate_Global congestion,” “serialization of exposed communication,” 
   “imbalance amplification across ranks,” “synchronization sensitivity,” 
   “pipeline non-overlap,” or “topology-induced constraint.”

Do NOT generalize. The analysis must sound like it came directly from the report.

"""

    # --------------------------------------------------------
    def _prompt_recommendations(self, ctx):
        return f"""
{ctx}

Provide **3 prioritized recommendations**:

For each:
1. Exact action  
2. Expected qualitative improvement  
3. Technical justification tied to RULE IDs  

Allowed tunables:
- GPUs / Ranks  
- Topology (Ring / Tree / FC / Hierarchical)  
- Batch Size  
- Gradient Accumulation  
- Compute Intensity  
- Communication Volume  
- Overlap Ratio  
PROVIDE EXACT VALUES WHERE POSSIBLE. 
Keep recommendations short, actionable, non-redundant.
"""

    # --------------------------------------------------------
    def _prompt_implementation(self, ctx):
        return f"""
{ctx}

Provide a **Implementation Strategy** (1–2 sentences):

- First parameter to tune  
- How to check and validate improvement  
"""

    # --------------------------------------------------------
    def _prompt_risks(self, ctx):
        return f"""
{ctx}

Provide a **Risk Assessment**:

- Risks of NOT resolving this  
- Risks introduced by proposed fixes  
- Mitigation steps in bullet points  

"""
