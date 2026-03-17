"""
Phase 3: Empirical transition counts from Q-learning training (train windows only, no leak).
Aggregates N_sa[s,a], N_sas[(s,a,s')], R_sum_sa[s,a] for building DP_empirical.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


class EmpiricalCounts:
    """
    Accumulate (s, a, r, s_next, done) from training; aggregate N_sa, N_sas, R_sum_sa.
    Train-only: no eval data.
    """

    def __init__(self, nS: int, nA: int):
        self.nS = nS
        self.nA = nA
        self.N_sa = np.zeros((nS, nA), dtype=np.float64)
        self.R_sum_sa = np.zeros((nS, nA), dtype=np.float64)
        self.R2_sum_sa = np.zeros((nS, nA), dtype=np.float64)  # for variance
        # N_sas: sparse. Key (s,a,s') -> count
        self._N_sas: Dict[Tuple[int, int, int], float] = {}

    def add(self, s: int, a: int, r: float, s_next: int, done: bool) -> None:
        if not (0 <= s < self.nS and 0 <= a < self.nA):
            return
        self.N_sa[s, a] += 1
        self.R_sum_sa[s, a] += r
        self.R2_sum_sa[s, a] += r * r
        if not done and 0 <= s_next < self.nS:
            key = (s, a, s_next)
            self._N_sas[key] = self._N_sas.get(key, 0) + 1

    def get_N_sas(self, s: int, a: int, s_next: int) -> float:
        return self._N_sas.get((s, a, s_next), 0.0)

    def get_N_sas_row(self, s: int, a: int) -> Dict[int, float]:
        """Return dict s_next -> count for (s,a)."""
        out = {}
        for (ss, aa, sn), c in self._N_sas.items():
            if ss == s and aa == a:
                out[sn] = out.get(sn, 0) + c
        return out

    def coverage_stats(self) -> dict:
        """Fraction (s,a) visited, top states, entropy, fallback count."""
        visited_sa = np.sum(self.N_sa > 0)
        total_sa = self.nS * self.nA
        frac_sa = visited_sa / max(1, total_sa)
        state_visits = self.N_sa.sum(axis=1)
        top_states = np.argsort(-state_visits)[:20]
        state_visits_list = state_visits.tolist()
        total_visits = state_visits.sum()
        probs = state_visits / max(1, total_visits)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs + 1e-20))
        fallback_sa = total_sa - visited_sa
        return {
            "visited_sa": int(visited_sa),
            "total_sa": int(total_sa),
            "fraction_sa_visited": float(frac_sa),
            "top_20_states": [int(x) for x in top_states],
            "state_visit_counts_top20": [float(state_visits[i]) for i in top_states],
            "entropy_state_visitation": float(entropy),
            "fallback_sa_pairs": int(fallback_sa),
            "total_transitions": int(self.N_sa.sum()),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        # N_sas as list of (s, a, s_next, count) for npz
        sas_list = np.array([(s, a, sn, c) for (s, a, sn), c in self._N_sas.items()], dtype=np.float64)
        if len(sas_list) == 0:
            sas_list = np.zeros((0, 4), dtype=np.float64)
        np.savez_compressed(
            path,
            nS=np.array(self.nS),
            nA=np.array(self.nA),
            N_sa=self.N_sa,
            R_sum_sa=self.R_sum_sa,
            R2_sum_sa=self.R2_sum_sa,
            N_sas_rows=sas_list,
        )
        # Also save coverage for quick load
        stats = self.coverage_stats()
        import json
        (path.parent / (path.stem + "_coverage.json")).write_text(json.dumps(stats, indent=2))

    @classmethod
    def load(cls, path: Path) -> "EmpiricalCounts":
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        nS, nA = int(data["nS"]), int(data["nA"])
        ec = cls(nS, nA)
        ec.N_sa = data["N_sa"].copy()
        ec.R_sum_sa = data["R_sum_sa"].copy()
        ec.R2_sum_sa = data["R2_sum_sa"].copy() if "R2_sum_sa" in data else np.zeros_like(ec.N_sa)
        rows = data["N_sas_rows"]
        if rows.size > 0:
            for i in range(rows.shape[0]):
                s, a, sn, c = int(rows[i, 0]), int(rows[i, 1]), int(rows[i, 2]), float(rows[i, 3])
                ec._N_sas[(s, a, sn)] = ec._N_sas.get((s, a, sn), 0) + c
        return ec
