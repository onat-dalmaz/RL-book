"""
Phase 3 DP baseline: model-based optimal policy on same 5D state, 5 actions, same fill model.
Estimate P(z'|z), P(v'|v) from fit data only; build P(s'|s,a), R(s,a); solve with value iteration.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import state as st
from .env import ExecutionEnvConfig, p_fill


def _z_to_idx(z: int, n_z: int) -> int:
    if n_z == 3:
        return int(np.clip(z + 1, 0, 2))
    return int(np.clip(z, 0, n_z - 1))


def _idx_to_z(zx: int, n_z: int) -> int:
    if n_z == 3:
        return zx - 1
    return zx


def estimate_exogenous_markov(
    z_seq: np.ndarray,
    v_seq: np.ndarray,
    n_z: int = 3,
    smooth: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Estimate P(z'|z) and P(v'|v) from fit sequences. Fit-only data; no eval leak.
    Returns (P_z shape (n_z,n_z), P_v shape (2,2), metadata with counts).
    """
    z_seq = np.asarray(z_seq, dtype=int)
    v_seq = np.asarray(v_seq, dtype=int)
    v_seq = np.clip(v_seq, 0, 1)
    T = min(len(z_seq), len(v_seq))
    if T < 2:
        P_z = np.ones((n_z, n_z)) / n_z
        P_v = np.ones((2, 2)) / 2
        return P_z, P_v, {"n_samples": 0}

    # z: map to 0..n_z-1
    z_idx = np.array([_z_to_idx(z, n_z) for z in z_seq[:T]])
    v_idx = v_seq[:T]

    count_z = np.zeros((n_z, n_z))
    for t in range(T - 1):
        i, j = z_idx[t], z_idx[t + 1]
        if 0 <= i < n_z and 0 <= j < n_z:
            count_z[i, j] += 1
    P_z = (count_z + smooth) / (count_z.sum(axis=1, keepdims=True) + n_z * smooth)
    P_z = P_z / P_z.sum(axis=1, keepdims=True)

    count_v = np.zeros((2, 2))
    for t in range(T - 1):
        i, j = int(v_idx[t]), int(v_idx[t + 1])
        if 0 <= i < 2 and 0 <= j < 2:
            count_v[i, j] += 1
    P_v = (count_v + smooth) / (count_v.sum(axis=1, keepdims=True) + 2 * smooth)
    P_v = P_v / P_v.sum(axis=1, keepdims=True)

    metadata = {"n_samples": T - 1, "count_z": count_z.tolist(), "count_v": count_v.tolist()}
    return P_z, P_v, metadata


def estimate_ey_given_zv(
    z_seq: np.ndarray,
    v_seq: np.ndarray,
    y_seq: np.ndarray,
    n_z: int = 3,
) -> np.ndarray:
    """E[y | z, v] from fit data. Returns array shape (n_z, 2)."""
    z_seq = np.asarray(z_seq, dtype=int)
    v_seq = np.clip(np.asarray(v_seq, dtype=int), 0, 1)
    y_seq = np.asarray(y_seq, dtype=float)
    T = min(len(z_seq), len(v_seq), len(y_seq))
    sums = np.zeros((n_z, 2))
    counts = np.zeros((n_z, 2))
    for t in range(T):
        zx = _z_to_idx(int(z_seq[t]), n_z)
        vx = int(v_seq[t])
        if 0 <= zx < n_z and 0 <= vx < 2:
            sums[zx, vx] += y_seq[t]
            counts[zx, vx] += 1
    ey = np.zeros((n_z, 2))
    for i in range(n_z):
        for j in range(2):
            ey[i, j] = sums[i, j] / counts[i, j] if counts[i, j] > 0 else 0.0
    return ey


def build_transition_model(
    cfg: ExecutionEnvConfig,
    P_z: np.ndarray,
    P_v: np.ndarray,
    Ey_zv: np.ndarray,
    Imax: int,
    z_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build P[s,a,s'] and R[s,a] for value iteration.
    P[s,a,s'] = probability of next state s' from (s,a).
    R[s,a] = expected immediate reward (using E[y|z,v] for current state's z,v).
    """
    nS = st.n_states(Imax, z_bins)
    nA = st.N_ACTIONS
    n_z, n_i, n_os, n_oa, n_v = st.state_dimensions(Imax, z_bins)

    P = np.zeros((nS, nA, nS))
    R = np.zeros((nS, nA))

    for s in range(nS):
        z, i, o_side, o_age, v_bin = st.index_to_state(s, Imax, z_bins)
        zx = _z_to_idx(z, n_z)
        vx = int(np.clip(v_bin, 0, 1))
        ey = Ey_zv[zx, vx]

        for a in range(nA):
            # Enumerate endogenous outcomes (i', o_side', o_age') and their probabilities
            outcomes = []  # (prob, di, i_next, o_side_next, o_age_next, cost_bps)
            i_prev = i
            if a == st.A_HOLD:
                if o_side != 0:
                    pf = p_fill(o_side, z, v_bin, o_age, cfg)
                    i_fill = np.clip(i_prev + o_side, -Imax, Imax)
                    if i_fill == i_prev + o_side:
                        outcomes.append((pf, o_side, i_fill, 0, 0, cfg.c_maker_bps))
                    else:
                        outcomes.append((pf, 0, i_prev, o_side, 0, 0.0))
                    outcomes.append((1 - pf, 0, i_prev, o_side, 2, 0.0))
                else:
                    outcomes.append((1.0, 0, i_prev, 0, 0, 0.0))
            elif a == st.A_PLACE_BUY:
                pf = p_fill(1, z, v_bin, 1, cfg)
                i_next = int(np.clip(i_prev + 1, -Imax, Imax))
                if i_next == i_prev + 1:
                    outcomes.append((pf, 1, i_next, 0, 0, cfg.c_maker_bps))
                    outcomes.append((1 - pf, 0, i_prev, 1, 1, 0.0))
                else:
                    outcomes.append((1.0, 0, i_prev, 1, 1, 0.0))
            elif a == st.A_PLACE_SELL:
                pf = p_fill(-1, z, v_bin, 1, cfg)
                i_next = int(np.clip(i_prev - 1, -Imax, Imax))
                if i_next == i_prev - 1:
                    outcomes.append((pf, -1, i_next, 0, 0, cfg.c_maker_bps))
                    outcomes.append((1 - pf, 0, i_prev, -1, 1, 0.0))
                else:
                    outcomes.append((1.0, 0, i_prev, -1, 1, 0.0))
            elif a == st.A_BUY_MARKET:
                i_next = np.clip(i_prev + 1, -Imax, Imax)
                outcomes.append((1.0, 1 if i_next == i_prev + 1 else 0, i_next, 0, 0, cfg.c_taker_bps if i_next == i_prev + 1 else 0.0))
            else:  # A_SELL_MARKET
                i_next = np.clip(i_prev - 1, -Imax, Imax)
                outcomes.append((1.0, -1 if i_next == i_prev - 1 else 0, i_next, 0, 0, cfg.c_taker_bps if i_next == i_prev - 1 else 0.0))

            # Normalize outcome probs
            total_p = sum(o[0] for o in outcomes)
            if total_p <= 0:
                total_p = 1.0
            outcomes = [(p / total_p, di, i_n, os_n, oa_n, cost) for p, di, i_n, os_n, oa_n, cost in outcomes]

            r_total = 0.0
            for prob, di, i_next, o_side_next, o_age_next, cost_bps in outcomes:
                r_total += prob * (
                    i_next * ey - cost_bps * abs(di) - cfg.lambda_inv * (i_next ** 2) - cfg.eta_turnover * (1.0 if di != 0 else 0.0)
                )
                for z_next_idx in range(n_z):
                    for v_next_idx in range(2):
                        p_zv = P_z[zx, z_next_idx] * P_v[vx, v_next_idx]
                        z_next = _idx_to_z(z_next_idx, n_z)
                        s_next = st.state_to_index(z_next, i_next, o_side_next, o_age_next, v_next_idx, Imax, z_bins)
                        P[s, a, s_next] += prob * p_zv
            R[s, a] = r_total

    # Normalize P[s,a,:]
    for s in range(nS):
        for a in range(nA):
            row = P[s, a, :]
            total = row.sum()
            if total > 0:
                P[s, a, :] /= total
            else:
                P[s, a, s] = 1.0
    return P, R


def solve_dp(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float = 0.99,
    tol: float = 1e-5,
    max_iter: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Value iteration. Returns V[s], pi[s], Q[s,a]."""
    nS, nA = P.shape[0], P.shape[1]
    V = np.zeros(nS)
    for _ in range(max_iter):
        V_old = V.copy()
        # (nS, nA, nS) @ V -> (nS, nA): for each (s,a) sum_s' P[s,a,s']*V[s']
        next_V = (P.reshape(nS * nA, nS) @ V).reshape(nS, nA)
        Q = R + gamma * next_V
        V = np.max(Q, axis=1)
        if np.max(np.abs(V - V_old)) < tol:
            break
    next_V = (P.reshape(nS * nA, nS) @ V).reshape(nS, nA)
    Q = R + gamma * next_V
    pi = np.argmax(Q, axis=1)
    return V, pi, Q


def export_dp_artifacts(
    outdir: Path,
    pi: np.ndarray,
    Q: np.ndarray,
    V: np.ndarray,
    Imax: int,
    z_bins: int,
    metadata: dict,
) -> None:
    outdir = Path(outdir)
    from . import state as st
    # DP_POLICY_TABLE.csv
    rows = []
    for s in range(len(pi)):
        z, i, o_side, o_age, v_bin = st.index_to_state(s, Imax, z_bins)
        rows.append({"state_idx": s, "z": z, "i": i, "o_side": o_side, "o_age": o_age, "v_bin": v_bin, "action": int(pi[s])})
    import pandas as pd
    pd.DataFrame(rows).to_csv(outdir / "DP_POLICY_TABLE.csv", index=False)
    np.save(outdir / "DP_Q_TABLE.npy", Q)
    np.save(outdir / "DP_V.npy", V)
    import json
    (outdir / "DP_METADATA.json").write_text(json.dumps(metadata, indent=2))


def write_dp_model_summary(outdir: Path, P_z: np.ndarray, P_v: np.ndarray, metadata: dict) -> None:
    outdir = Path(outdir)
    lines = [
        "# DP model summary (fit-only data)",
        "",
        "## Exogenous transitions (estimated from fit split)",
        "",
        "P(z'|z) 3x3:",
        str(P_z.tolist()),
        "",
        "P(v'|v) 2x2:",
        str(P_v.tolist()),
        "",
        "Metadata: " + str(metadata),
    ]
    (outdir / "DP_MODEL_SUMMARY.md").write_text("\n".join(lines))
