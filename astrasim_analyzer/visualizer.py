# astrasim_analyzer/visualizer.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from metrics import PerformanceMetrics
from rule_engine import RuleEngineOutput

sns.set_palette("husl")
plt.style.use("seaborn-v0_8-darkgrid")


class Visualizer:

    def __init__(self, rank_df, metrics: PerformanceMetrics, rules: RuleEngineOutput):
        self.df = rank_df
        self.m = metrics
        self.rules = rules

    # ----------------------------------------------------------------------
    def generate_all(self, output_prefix: str):
        outdir = f"{output_prefix}_viz"
        os.makedirs(outdir, exist_ok=True)

        self.plot_metrics_dashboard(f"{outdir}/metrics_dashboard.png")
        self.plot_rank_analysis(f"{outdir}/rank_analysis.png")
        self.plot_time_heatmap(f"{outdir}/time_heatmap.png")

        print(f"✔ Visualizations saved to: {outdir}")

    # ----------------------------------------------------------------------
    def plot_metrics_dashboard(self, out):

        m = self.m
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.35)

        # ----------------------------------------------------------
        # Radar Chart
        ax1 = fig.add_subplot(gs[0, 0], projection="polar")

        categories = [
            "Throughput\n(norm)",
            "Scaling Eff",
            "GPU Util",
            "Balance",
            "Network Health"
        ]

        throughput_norm = min(100, m.throughput_samples_per_sec / 10)
        balance = max(0, 100 - m.rank_imbalance_percent)
        net_health = max(0, 100 - m.congestion_score * 100)

        values = [
            throughput_norm,
            m.scaling_efficiency_percent,
            m.gpu_utilization_percent,
            balance,
            net_health
        ]

        angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False)
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))

        ax1.plot(angles, values, "o-", linewidth=2, color="#3b82f6")
        ax1.fill(angles, values, alpha=0.25, color="#3b82f6")
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories)
        ax1.set_ylim(0, 100)
        ax1.set_title("Performance Profile", fontsize=14, weight="bold")

        # ----------------------------------------------------------
        # Efficiency Indicators Bar Chart
        ax2 = fig.add_subplot(gs[0, 1])

        names = ["Throughput", "GPU Util", "Scaling Eff", "Balance"]
        vals = [
            throughput_norm,
            m.gpu_utilization_percent,
            m.scaling_efficiency_percent,
            balance
        ]

        bars = ax2.bar(names, vals, color=sns.color_palette("husl", 4))
        ax2.set_title("Efficiency Indicators", fontsize=14, weight="bold")
        ax2.set_ylim(0, 100)

        for bar, v in zip(bars, vals):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                v + 2,
                f"{v:.1f}%",
                ha="center"
            )

        # ----------------------------------------------------------
        # C2C Ratio Classification
        ax3 = fig.add_subplot(gs[1, 0])

        ranges = [
            "<0.3\nCompute",
            "0.3–0.7\nBalanced",
            "0.7–1.2\nComm\nSensitive",
            ">1.2\nNetwork"
        ]

        colors = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444"]

        c2c = min(m.c2c_ratio, 5)
        idx = 0
        if c2c < 0.3: idx = 0
        elif c2c < 0.7: idx = 1
        elif c2c < 1.2: idx = 2
        else: idx = 3

        highlight = []
        for i, col in enumerate(colors):
            if i == idx:
                highlight.append(col)
            else:
                highlight.append(col + "55")   # faded hex

        ax3.bar(range(4), [1]*4, color=highlight)
        ax3.set_xticks(range(4))
        ax3.set_xticklabels(ranges)
        ax3.set_ylim(0, 1.2)
        ax3.set_title(f"C2C Classification (Current: {m.c2c_ratio:.2f})")

        # ----------------------------------------------------------
        # Key Metrics Table
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")

        table_data = [
            ["Throughput", f"{m.throughput_samples_per_sec:.1f}"],
            ["GPU Utilization", f"{m.gpu_utilization_percent:.1f}%"],
            ["Scaling Eff.", f"{m.scaling_efficiency_percent:.1f}%"],
            ["Rank Imbalance", f"{m.rank_imbalance_percent:.1f}%"],
            ["Comm Fraction", f"{m.comm_fraction:.2f}"],
            ["Compute Fraction", f"{m.compute_fraction:.2f}"]
        ]

        table = ax4.table(
            cellText=table_data,
            colLabels=["Metric", "Value"],
            loc="center"
        )
        table.scale(1, 2)
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        fig.suptitle("Metrics Dashboard", fontsize=18, weight="bold")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()


    # ----------------------------------------------------------------------
    def plot_rank_analysis(self, out):

        df = self.df
        m = self.m

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.35)

        # ----------------------------------------------------------
        # Stacked Breakdown (Compute + Comm)
        ax1 = fig.add_subplot(gs[0, 0])

        ranks = df["rank"].values
        comp = df["compute_time_cycles"].values * 1e-6
        comm = df["comm_time_cycles"].values * 1e-6

        ax1.bar(ranks, comp, label="Compute", color="#10b981")
        ax1.bar(ranks, comm, bottom=comp, label="Communication", color="#ef4444")
        ax1.set_title("Per-Rank Time Breakdown")
        ax1.set_xlabel("Rank")
        ax1.set_ylabel("Time (ms)")
        ax1.legend()

        # ----------------------------------------------------------
        # Rank Imbalance Curve
        ax2 = fig.add_subplot(gs[0, 1])

        wall = df["wall_time_cycles"].values * 1e-6
        mean_w = wall.mean()

        ax2.plot(ranks, wall, "o-", color="#3b82f6")
        ax2.axhline(mean_w, ls="--", color="red", label=f"Mean: {mean_w:.2f} ms")
        ax2.fill_between(ranks, wall, mean_w, alpha=0.2)
        ax2.set_title(f"Rank Imbalance: {m.rank_imbalance_percent:.2f}%")
        ax2.legend()

        # ----------------------------------------------------------
        # Communication Heatmap
        ax3 = fig.add_subplot(gs[1, 0])

        comm_matrix = comm.reshape(1, -1)
        sns.heatmap(comm_matrix, cmap="YlOrRd", ax=ax3, cbar=True)
        ax3.set_title("Communication Time Heatmap")
        ax3.set_xlabel("Rank")
        ax3.set_yticks([])

        # ----------------------------------------------------------
        # Statistics Table
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")

        stats = [
            ["Ranks", len(df)],
            ["Mean Wall (ms)", f"{wall.mean():.2f}"],
            ["Std Wall (ms)", f"{wall.std():.2f}"],
            ["Min Wall (ms)", f"{wall.min():.2f}"],
            ["Max Wall (ms)", f"{wall.max():.2f}"],
            ["Imbalance %", f"{m.rank_imbalance_percent:.2f}%"]
        ]

        table = ax4.table(
            cellText=stats,
            colLabels=["Metric", "Value"],
            loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.1, 2)

        fig.suptitle("Rank-Level Analysis", fontsize=18, weight="bold")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()


    # ----------------------------------------------------------------------
    def plot_time_heatmap(self, out):

        m = self.m
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.35)

        # ----------------------------------------------------------
        # Main Pie Chart
        ax1 = fig.add_subplot(gs[0, 0])

        sizes = [m.comm_time_ms, m.compute_time_ms, m.idle_time_ms]
        labels = ["Comm", "Compute", "Idle"]
        colors = ["#ef4444", "#10b981", "#6b7280"]

        ax1.pie(sizes, labels=labels, autopct="%1.1f%%",
                startangle=90, colors=colors)
        ax1.set_title("Time Breakdown")

        # ----------------------------------------------------------
        # Fractions Bar
        ax2 = fig.add_subplot(gs[0, 1])

        fracs = [
            m.comm_fraction,
            m.compute_fraction,
            1 - m.comm_fraction - m.compute_fraction
        ]

        bars = ax2.bar(labels, fracs, color=colors)
        ax2.set_title("Time Fractions")
        ax2.set_ylim(0, 1)

        for bar, v in zip(bars, fracs):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.02,
                f"{v:.2f}",
                ha="center"
            )

        # ----------------------------------------------------------
        # Stacked Time
        ax3 = fig.add_subplot(gs[1, 0])

        comp = m.compute_time_ms
        comm = m.comm_time_ms
        idle = m.idle_time_ms

        ax3.bar(["Current"], [comp], color="#10b981")
        ax3.bar(["Current"], [comm], bottom=[comp], color="#ef4444")
        ax3.bar(["Current"], [idle], bottom=[comp + comm], color="#6b7280")

        ax3.set_title("Stacked Time Breakdown")
        ax3.set_ylabel("ms")

        # ----------------------------------------------------------
        # Summary Table
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")

        summary = [
            ["Comm %", f"{m.comm_fraction:.2f}"],
            ["Compute %", f"{m.compute_fraction:.2f}"],
            ["Idle %", f"{1 - m.compute_fraction - m.comm_fraction:.2f}"],
            ["C2C Ratio", f"{m.c2c_ratio:.2f}"],
            ["Severity", m.severity]
        ]

        table = ax4.table(
            cellText=summary,
            colLabels=["Metric", "Value"],
            loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.1, 2)

        fig.suptitle("Time Breakdown & Heatmap", fontsize=18, weight="bold")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
