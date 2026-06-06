# astrasim_analyzer/parser.py

import re
import pandas as pd
from dataclasses import dataclass

@dataclass
class ParsedLog:
    topology: str
    rank_df: pd.DataFrame
    iter_time_sec: float
    batch_size: int = 32


class AstraSimParser:
    def parse(self, path: str) -> ParsedLog:
        with open(path, "r") as f:
            content = f.read()

        # -----------------------------
        # Parse topology
        # -----------------------------
        topo_match = re.search(r'\[system::topology::(\w+)\]', content)
        topology = topo_match.group(1) if topo_match else "Unknown"

        # -----------------------------
        # Parse rank data
        # -----------------------------
        pattern = r'sys\[(\d+)\].*?Wall time:\s*(\d+).*?Comm time:\s*(\d+)'
        matches = re.findall(pattern, content, re.DOTALL)

        rows = []
        for rank, wall, comm in matches:
            wall = int(wall)
            comm = int(comm)
            rows.append({
                "rank": int(rank),
                "wall_time_cycles": wall,
                "comm_time_cycles": comm,
                "compute_time_cycles": wall - comm
            })

        df = pd.DataFrame(rows)

        # -----------------------------
        # Compute iteration time (sec)
        # -----------------------------
        # All ranks in Astra-Sim finish same iteration
        # So we use average cycles → seconds
        if len(df) > 0:
            avg_wall_cycles = df["wall_time_cycles"].mean()
        else:
            avg_wall_cycles = 1

        iter_time_sec = avg_wall_cycles / 1_000_000  # cycles→seconds

        # -----------------------------
        # Return ParsedLog object
        # -----------------------------
        return ParsedLog(
            topology=topology,
            rank_df=df,
            iter_time_sec=iter_time_sec
        )
