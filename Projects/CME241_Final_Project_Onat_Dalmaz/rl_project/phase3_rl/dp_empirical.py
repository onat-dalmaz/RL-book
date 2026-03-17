"""
Phase 3: DP on empirical MDP built from Q-learning training transitions only (fair baseline).
P_hat(s'|s,a) from N_sas with smoothing; R_hat(s,a) = R_sum_sa/N_sa; unvisited (s,a): self-loop, R=0.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import json
import numpy as np

from . import state as st
from .empirical_model import EmpiricalCounts


def build_empirical_mdp(
    ec: EmpiricalCounts,
    alpha_smooth: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    P_hat[s,a,s'] with Dirichlet smoothing on observed support per (s,a).
    R_hat[s,a] = R_sum_sa / max(1, N_sa). Unvisited (s,a): P(s->s)=1, R=0.
    Returns (P_hat, R_hat, model_meta).
    """
    nS, nA = ec.nS, ec.nA
    P_hat = np.zeros((nS, nA, nS))
    R_hat = np.zeros((nS, nA))

    fallback_count = 0
    for s in range(nS):
        for a in range(nA):
            n_sa = ec.N_sa[s, a]
            if n_sa > 0:
                row = ec.get_N_sas_row(s, a)
                support = list(row.keys())
                if not support:
                    # visited but no s' recorded (e.g. all done) -> self-loop
                    P_hat[s, a, s] = 1.0
                    R_hat[s, a] = ec.R_sum_sa[s, a] / n_sa
                    continue
                denom = n_sa + alpha_smooth * len(support)
                for s_next in support:
                    P_hat[s, a, s_next] = (row[s_next] + alpha_smooth) / denom
                # normalize in case of float
                tot = P_hat[s, a, :].sum()
                if tot > 0:
                    P_hat[s, a, :] /= tot
                R_hat[s, a] = ec.R_sum_sa[s, a] / n_sa
            else:
                P_hat[s, a, s] = 1.0
                R_hat[s, a] = 0.0
                fallback_count += 1

    coverage = ec.coverage_stats()
    model_meta = {
        "alpha_smooth": alpha_smooth,
        "visited_sa": coverage["visited_sa"],
        "total_sa": coverage["total_sa"],
        "fraction_sa_visited": coverage["fraction_sa_visited"],
        "fallback_sa_pairs": fallback_count,
        "entropy_state_visitation": coverage["entropy_state_visitation"],
        "total_transitions": coverage["total_transitions"],
    }
    return P_hat, R_hat, model_meta


def solve_dp_empirical(
    P_hat: np.ndarray,
    R_hat: np.ndarray,
    gamma: float = 0.99,
    tol: float = 1e-8,
    max_iter: int = 50_000,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Value iteration. pi = argmax_a Q with random tie-break (seeded). Returns V, pi, Q."""
    from . import q_learning as ql

    nS, nA = P_hat.shape[0], P_hat.shape[1]
    V = np.zeros(nS)
    for _ in range(max_iter):
        V_old = V.copy()
        next_V = (P_hat.reshape(nS * nA, nS) @ V).reshape(nS, nA)
        Q = R_hat + gamma * next_V
        V = np.max(Q, axis=1)
        if np.max(np.abs(V - V_old)) < tol:
            break
    next_V = (P_hat.reshape(nS * nA, nS) @ V).reshape(nS, nA)
    Q = R_hat + gamma * next_V
    rng = np.random.default_rng(seed)
    pi = np.array([ql.argmax_random_tiebreak(Q[s, :], rng) for s in range(nS)], dtype=int)
    return V, pi, Q


def export_dp_empirical_artifacts(
    outdir: Path,
    pi: np.ndarray,
    Q: np.ndarray,
    V: np.ndarray,
    Imax: int,
    z_bins: int,
    model_meta: dict,
) -> None:
    outdir = Path(outdir)
    rows = []
    for s in range(len(pi)):
        z, i, o_side, o_age, v_bin = st.index_to_state(s, Imax, z_bins)
        rows.append({
            "state_idx": s, "z": z, "i": i, "o_side": o_side, "o_age": o_age,
            "v_bin": v_bin, "action": int(pi[s]),
        })
    import pandas as pd
    pd.DataFrame(rows).to_csv(outdir / "DP_EMPIRICAL_POLICY_TABLE.csv", index=False)
    np.save(outdir / "DP_EMPIRICAL_Q_TABLE.npy", Q)
    np.save(outdir / "DP_EMPIRICAL_V.npy", V)


def write_dp_empirical_model_json(outdir: Path, model_meta: dict) -> None:
    outdir = Path(outdir)
    (outdir / "DP_EMPIRICAL_MODEL.json").write_text(json.dumps(model_meta, indent=2))
