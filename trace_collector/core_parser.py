# astrasim_analyzer/core_parser.py

import re
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple


class AstraSimParser:
    """
    Parses ASTRA-Sim training logs and extracts rank-level performance data.
    This module produces *only raw data*. No metric computation is done here.
    """

    # ----------------------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------------------
    def parse(self, log_path: str, batch_size: int) -> Dict:
        """
        Main entry point used by CLI.

        Returns dict:
        {
            "topology": str,
            "batch_size": int,
            "rank_df": DataFrame(rank, wall_time, comm_time, compute_time),
            "num_ranks": int,
        }
        """
        log_text = self._load_file(log_path)

        topology = self._extract_topology(log_text)
        rank_df = self._extract_rank_data(log_text)

        return {
            "topology": topology,
            "batch_size": batch_size,
            "rank_df": rank_df,
            "num_ranks": len(rank_df),
        }

    # ----------------------------------------------------------------------
    # INTERNAL HELPERS
    # ----------------------------------------------------------------------

    def _load_file(self, path: str) -> str:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Trace file not found: {path}")

        return path.read_text()

    def _extract_topology(self, text: str) -> str:
        """
        Detects topology from lines like:
        [system::topology::RingTopology]
        """

        pattern = r"\[system::topology::(\w+)\]"
        m = re.search(pattern, text)

        if m:
            return m.group(1)

        # fallback: unknown topology
        return "Unknown"

    def _extract_rank_data(self, text: str) -> pd.DataFrame:
        """
        Extracts per-rank wall_time and comm_time.

        Example log lines:
            sys[0], Wall time: 460300
            sys[0], Comm time: 210200
        """

        pattern = (
            r"sys\[(\d+)\].*?Wall time:\s*(\d+).*?Comm time:\s*(\d+)"
        )

        matches = re.findall(pattern, text, re.DOTALL)

        if not matches:
            raise RuntimeError(
                "Could not find rank statistics in ASTRA-Sim log."
            )

        rows = []
        for rank_id, wall_cycles, comm_cycles in matches:
            wall_cycles = int(wall_cycles)
            comm_cycles = int(comm_cycles)
            compute_cycles = wall_cycles - comm_cycles

            rows.append({
                "rank": int(rank_id),
                "wall_time_cycles": wall_cycles,
                "comm_time_cycles": comm_cycles,
                "compute_time_cycles": compute_cycles,
            })

        df = pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)

        return df


# --------------------------------------------------------------------------
# Utility function used by metrics engine
# --------------------------------------------------------------------------

def compute_iteration_time_seconds(df: pd.DataFrame) -> float:
    """
    ASTRA-Sim has 1 cycle = 1 ns = 1e-9 sec.
    -> 1e6 cycles = 1 ms.
    """

    avg_cycles = df["wall_time_cycles"].mean()
    iteration_time_sec = avg_cycles * 1e-9
    return iteration_time_sec
