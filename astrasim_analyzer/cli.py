# astrasim_analyzer/cli.py

import argparse
import json
from pathlib import Path
from dataclasses import asdict

from parser import AstraSimParser
from metrics import MetricsEngine, serialize_metrics
from rule_engine import RuleEngine
from ai_agent import AIAgent
from visualizer import Visualizer


# ----------------------------------------------------------------------
# JSON SAFETY CONVERTER
# ----------------------------------------------------------------------
def make_json_safe(obj):
    import numpy as np

    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]

    # numpy scalar → python scalar
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # normal python types
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj

    return str(obj)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Astra-Sim AI Analyzer")

    parser.add_argument("logfile", type=str, help="Path to Astra-Sim log file")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=str, default="result")

    args = parser.parse_args()

    print("\n==========================================================================================")
    print("                            ASTRA-SIM AI PERFORMANCE ANALYZER")
    print("                                (BERT / ResNet / DLRM)")
    print("==========================================================================================\n")

    # ----------------------------------------------------------------------
    # 1) Parse
    # ----------------------------------------------------------------------
    print("Parsing trace...")

    parser_obj = AstraSimParser()
    parsed = parser_obj.parse(args.logfile)

    parsed.batch_size = args.batch_size

    # ----------------------------------------------------------------------
    # 2) Compute Metrics
    # ----------------------------------------------------------------------
    print("Computing metrics...")

    metrics_engine = MetricsEngine()
    metrics = metrics_engine.compute(parsed=parsed, iter_time_sec=parsed.iter_time_sec)

    # ----------------------------------------------------------------------
    # 3) Rule Engine
    # ----------------------------------------------------------------------
    print("Running rule engine...")

    rule_engine = RuleEngine()
    rule_results = rule_engine.evaluate(metrics, parsed)

    # ----------------------------------------------------------------------
    # 4) AI Agent Output
    # ----------------------------------------------------------------------
    print("Running AI Agent...")

    agent = AIAgent()
    ai_output = agent.generate(metrics, rule_results)

    # ----------------------------------------------------------------------
    # 5) Save JSON
    # ----------------------------------------------------------------------
    print("\nSaving results...")

    # Rule engine → dict
    rule_results_dict = {
        "triggered_rules": [
            {
                "rule_id": r.rule_id,
                "triggered": bool(r.triggered),
                "explanation": r.explanation,
                "metrics_used": make_json_safe(r.metrics_used)
            }
            for r in rule_results.triggered_rules
        ],
        "dominant_bottleneck": rule_results.dominant_bottleneck,
        "mitigation_cluster": rule_results.mitigation_cluster,
        "summary": rule_results.summary
    }

    # Parsed data → dict
    parsed_dict = {
        "topology": parsed.topology,
        "batch_size": parsed.batch_size,
        "iter_time_sec": parsed.iter_time_sec,
        "rank_data": make_json_safe(parsed.rank_df.to_dict(orient="records"))
    }

    # Metrics → dict
    serialized_metrics = make_json_safe(serialize_metrics(metrics))

    # Combine
    final_obj = {
        "metrics": serialized_metrics,
        "rules": rule_results_dict,
        "parsed": parsed_dict,
        "ai_agent": make_json_safe(ai_output)
    }
    
    print("\n==================== ANALYSIS SUMMARY ====================\n")

    # ---- METRICS ----
    print("---- METRICS ----")
    for k, v in serialized_metrics.items():
        print(f"{k:30s} : {v}")
    print()

    # ---- RULE ENGINE ----
    print("---- TRIGGERED RULES ----")
    for r in rule_results_dict["triggered_rules"]:
        print(f"{r['rule_id']}  ->  {r['explanation']}")
    print()

    print("Dominant Bottleneck       :", rule_results_dict["dominant_bottleneck"])
    print("Mitigation Cluster        :", rule_results_dict["mitigation_cluster"])
    print()

    # ---- PARSED TRACE ----
    print("---- PARSED TRACE ----")
    print("Topology                  :", parsed_dict["topology"])
    print("Batch Size                :", parsed_dict["batch_size"])
    print("Iteration Time (sec)      :", parsed_dict["iter_time_sec"])
    print("Ranks Parsed              :", len(parsed_dict["rank_data"]))
    print()

    # ---- AI AGENT OUTPUT ----
    print("---- AI AGENT OUTPUT ----")
    print("\n[BOTTLENECK ANALYSIS]\n")
    print(ai_output.get("bottleneck_analysis", ""))

    print("\n[ROOT CAUSE]\n")
    print(ai_output.get("root_cause", ""))

    print("\n[RECOMMENDATIONS]\n")
    print(ai_output.get("recommendations", ""))
    

    print("\n==========================================================\n")


    # Save JSON
    output_file = Path(f"{args.output}.json")
    with open(output_file, "w") as f:
        json.dump(make_json_safe(final_obj), f, indent=4)

    print(f"\n✔ Results written to {output_file}")

    # ----------------------------------------------------------------------
    # 6) Visualizations
    # ----------------------------------------------------------------------
    viz = Visualizer(
    rank_df=parsed.rank_df,
    metrics=metrics,
    rules=rule_results
)
    print("\nGenerating visualizations...")

    viz.plot_metrics_dashboard(f"{args.output}_metrics.png")
    viz.plot_rank_analysis(f"{args.output}_rank.png")
    viz.plot_time_heatmap(f"{args.output}_time.png")

    print("✔ Visualization images saved.")



    print("\n✔ Visualization images saved.")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
