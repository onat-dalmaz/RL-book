"""
Phase 3: Phase2-style DP baseline — reduced state (z, i) and actions HOLD, BUY_MKT, SELL_MKT.
Fit-only P(z'|z) and E[y|z]. Solves DP on this abstraction so "RL beats DP_PHASE2" is a fair claim.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# Phase2 action set: 0=HOLD, 1=BUY_MARKET, 2=SELL_MARKET (map to Phase3 action ids 0, 3, 4)
N_ACTIONS_P2 = 3
A_P2_HOLD = 0
A_P2_BUY_MKT = 1
A_P2_SELL_MKT = 2

# Map Phase2 action index -> Phase3 action id
P2_TO_P3_ACTION = [0, 3, 4]  # HOLD, BUY_MARKET, SELL_MARKET
P2_ACTION_NAMES = ["HOLD", "BUY_MARKET", "SELL_MARKET"]


def n_states_phase2(Imax: int, n_z: int = 3) -> int:
    """State (z, i): z in {0..n_z-1} internal, i in {-Imax..Imax} -> n_z * (2*Imax+1)."""
    return n_z * (2 * Imax + 1)


def state_p2_to_index(z: int, i: int, Imax: int, n_z: int = 3) -> int:
    """Map (z, i) to state index. z in {-1,0,1} for n_z=3."""
    zx = int(np.clip(z + 1, 0, 2)) if n_z == 3 else int(np.clip(z, 0, n_z - 1))
    ix = int(np.clip(i + Imax, 0, 2 * Imax))
    return zx * (2 * Imax + 1) + ix


def index_to_state_p2(idx: int, Imax: int, n_z: int = 3) -> Tuple[int, int]:
    """State index -> (z, i)."""
    n_i = 2 * Imax + 1
    zx = idx // n_i
    ix = idx % n_i
    z = (zx - 1) if n_z == 3 else zx
    i = ix - Imax
    return (z, i)


def estimate_pz_and_ey_z(
    z_seq: np.ndarray,
    y_seq: np.ndarray,
    n_z: int = 3,
    smooth: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Fit-only. P(z'|z) and E[y|z]. Returns (P_z (n_z,n_z), E_y_z (n_z,), metadata).
    """
    z_seq = np.asarray(z_seq, dtype=int)
    y_seq = np.asarray(y_seq, dtype=float)
    T = min(len(z_seq), len(y_seq))
    if T < 2:
        P_z = np.ones((n_z, n_z)) / n_z
        E_y_z = np.zeros(n_z)
        return P_z, E_y_z, {"n_samples": 0}

    z_idx = np.array([int(np.clip(z + 1, 0, 2)) for z in z_seq[:T]]) if n_z == 3 else np.clip(z_seq[:T], 0, n_z - 1)
    count_z = np.zeros((n_z, n_z))
    for t in range(T - 1):
        i, j = z_idx[t], z_idx[t + 1]
        if 0 <= i < n_z and 0 <= j < n_z:
            count_z[i, j] += 1
    row_sum = count_z.sum(axis=1, keepdims=True) + n_z * smooth
    P_z = (count_z + smooth) / row_sum
    P_z = P_z / P_z.sum(axis=1, keepdims=True)

    sum_y = np.zeros(n_z)
    cnt_y = np.zeros(n_z)
    for t in range(T):
        zx = int(z_idx[t])
        if 0 <= zx < n_z:
            sum_y[zx] += y_seq[t]
            cnt_y[zx] += 1
    E_y_z = np.where(cnt_y > 0, sum_y / cnt_y, 0.0)

    meta = {"n_samples": T - 1, "count_z": count_z.tolist(), "T": T, "smooth": smooth}
    return P_z, E_y_z, meta


def build_phase2_transition_reward(
    P_z: np.ndarray,
    E_y_z: np.ndarray,
    Imax: int,
    c_taker_bps: float,
    lambda_inv: float,
    eta_turnover: float,
    n_z: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    P[s,a,s'], R[s,a] for Phase2 state space (z,i). Actions: HOLD, BUY_MKT, SELL_MKT.
    Reward: R = sum_{z'} P(z'|z) * [ i'*E[y|z'] - cost - lambda*i'^2 - eta*1{Δi≠0} ].
    """
    nS = n_states_phase2(Imax, n_z)
    nA = N_ACTIONS_P2
    P = np.zeros((nS, nA, nS))
    R = np.zeros((nS, nA))

    n_i = 2 * Imax + 1
    for idx in range(nS):
        z, i = index_to_state_p2(idx, Imax, n_z)
        zx = (z + 1) if n_z == 3 else z
        zx = int(np.clip(zx, 0, n_z - 1))

        for a in range(nA):
            if a == A_P2_HOLD:
                i_next = i
                cost = 0.0
                d_i = 0
            elif a == A_P2_BUY_MKT:
                i_next = int(np.clip(i + 1, -Imax, Imax))
                cost = c_taker_bps if (i_next != i) else 0.0
                d_i = 1 if (i_next != i) else 0
            else:  # A_P2_SELL_MKT
                i_next = int(np.clip(i - 1, -Imax, Imax))
                cost = c_taker_bps if (i_next != i) else 0.0
                d_i = 1 if (i_next != i) else 0

            r_expect = 0.0
            for z_next_idx in range(n_z):
                p_z = P_z[zx, z_next_idx]
                ey = E_y_z[z_next_idx]
                term = i_next * ey - cost - lambda_inv * (i_next ** 2) - eta_turnover * (1.0 if d_i else 0.0)
                r_expect += p_z * term
                idx_next = state_p2_to_index((z_next_idx - 1) if n_z == 3 else z_next_idx, i_next, Imax, n_z)
                P[idx, a, idx_next] = p_z
            R[idx, a] = r_expect

    # Normalize P rows
    for idx in range(nS):
        for a in range(nA):
            row = P[idx, a, :]
            s = row.sum()
            if s > 0:
                P[idx, a, :] /= s
            else:
                P[idx, a, idx] = 1.0
    return P, R


def solve_dp_phase2(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float = 0.99,
    tol: float = 1e-6,
    max_iter: int = 2000,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Value iteration. pi = argmax_a Q with random tie-break. Returns V, pi, Q."""
    from . import q_learning as ql

    nS, nA = P.shape[0], P.shape[1]
    V = np.zeros(nS)
    for _ in range(max_iter):
        V_old = V.copy()
        next_V = (P.reshape(nS * nA, nS) @ V).reshape(nS, nA)
        Q = R + gamma * next_V
        V = np.max(Q, axis=1)
        if np.max(np.abs(V - V_old)) < tol:
            break
    next_V = (P.reshape(nS * nA, nS) @ V).reshape(nS, nA)
    Q = R + gamma * next_V
    rng = np.random.default_rng(seed)
    pi = np.array([ql.argmax_random_tiebreak(Q[s, :], rng) for s in range(nS)], dtype=int)
    return V, pi, Q


def export_dp_phase2_artifacts(
    outdir: Path,
    pi: np.ndarray,
    Q: np.ndarray,
    Imax: int,
    meta: dict,
) -> None:
    """Write DP_PHASE2_POLICY_TABLE.csv (z, i, action_id, action_name), DP_PHASE2_Q_TABLE.npy."""
    outdir = Path(outdir)
    n_z = 3
    rows = []
    for idx in range(len(pi)):
        z, i = index_to_state_p2(idx, Imax, n_z)
        a_p2 = int(pi[idx])
        action_id = P2_TO_P3_ACTION[a_p2]
        action_name = P2_ACTION_NAMES[a_p2]
        rows.append({"z": z, "i": i, "action_id": action_id, "action_name": action_name})
    import pandas as pd
    pd.DataFrame(rows).to_csv(outdir / "DP_PHASE2_POLICY_TABLE.csv", index=False)
    np.save(outdir / "DP_PHASE2_Q_TABLE.npy", Q)


def write_dp_phase2_model_summary(outdir: Path, P_z: np.ndarray, E_y_z: np.ndarray, meta: dict) -> None:
    """DP_PHASE2_MODEL_SUMMARY.md with P(z'|z), E[y|z], fit counts."""
    outdir = Path(outdir)
    lines = [
        "# DP Phase2 baseline (fit-only)",
        "",
        "State: (z, i). Actions: HOLD, BUY_MARKET, SELL_MARKET.",
        "",
        "## P(z'|z) 3x3",
        str(P_z.tolist()),
        "",
        "## E[y|z]",
        str(E_y_z.tolist()),
        "",
        "## Metadata",
        str(meta),
    ]
    (outdir / "DP_PHASE2_MODEL_SUMMARY.md").write_text("\n".join(lines))


def write_dp_phase2_model_json(outdir: Path, meta: dict) -> None:
    """DP_PHASE2_MODEL.json: reward_mode, P(z'|z), E_y_z used, E_y_z_fitted (if any), counts."""
    import json
    outdir = Path(outdir)
    export = {
        "data_scope": "fit_or_train_only",
        "reward_mode": meta.get("reward_mode", "drift"),
        "E_y_z": meta.get("E_y_z", []),
        "E_y_z_fitted": meta.get("E_y_z_fitted", meta.get("E_y_z", [])),
        "P_z": meta.get("P_z", []),
        "count_z": meta.get("count_z", []),
        "n_samples": meta.get("n_samples", 0),
        "T": meta.get("T", 0),
        "smooth": meta.get("smooth", 1e-4),
    }
    (outdir / "DP_PHASE2_MODEL.json").write_text(json.dumps(export, indent=2))


def build_and_solve_dp_phase2(
    z_train: np.ndarray,
    y_train: np.ndarray,
    Imax: int,
    c_taker_bps: float,
    lambda_inv: float,
    eta_turnover: float,
    gamma: float = 0.99,
    seed: Optional[int] = None,
    reward_mode: str = "zero_mean",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Estimate P(z'|z), E[y|z] from train; optionally zero E[y|z] for fair baseline.
    reward_mode: "drift" = use fitted E[y|z]; "zero_mean" = E[y|z]=0 (risk/cost-only, no alpha).
    Returns (V, pi, Q, meta).
    """
    P_z, E_y_z, meta = estimate_pz_and_ey_z(z_train, y_train, n_z=3)
    E_y_z_used = E_y_z.copy()
    if reward_mode == "zero_mean":
        E_y_z_used = np.zeros_like(E_y_z)
    meta["reward_mode"] = reward_mode
    meta["E_y_z_fitted"] = E_y_z.tolist()
    P, R = build_phase2_transition_reward(
        P_z, E_y_z_used, Imax, c_taker_bps, lambda_inv, eta_turnover, n_z=3
    )
    V, pi, Q = solve_dp_phase2(P, R, gamma=gamma, seed=seed)
    meta["P_z"] = P_z.tolist()
    meta["E_y_z"] = E_y_z_used.tolist()
    return V, pi, Q, meta


def policy_phase2_from_pi(pi_p2: np.ndarray, Imax: int, z_bins: int = 3):
    """
    Return a policy function policy(s) -> action_id for Phase 3 evaluator.
    s is Phase3 state index; we reduce to (z, i), look up pi_p2, map to Phase3 action (0, 3, or 4).
    """
    from . import state as st

    def fn(s: int) -> int:
        z, i, _, _, _ = st.index_to_state(s, Imax, z_bins)
        idx_p2 = state_p2_to_index(z, i, Imax, 3)
        a_p2 = int(pi_p2[idx_p2])
        return P2_TO_P3_ACTION[a_p2]
    return fn
