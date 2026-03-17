"""
Phase 2 DP v3: Tabular MDP definition, value iteration, policy tables.
Tie-breaking: hold (0) > reduce |i| > follow z.
"""

from pathlib import Path
import numpy as np
import pandas as pd

Z_LIST = [-1, 0, 1]


def build_R_table(mu_long: dict, mu_short: dict, eta_turnover_bps: float = 0.0):
    """R(z, a) = base_edge(z,a) - eta*|a|. Inventory penalty applied in VI. (one_step_bet legacy.)"""
    R = {}
    for z in Z_LIST:
        for a in (-1, 0, 1):
            if a == 1:
                base = mu_long.get(z, 0.0)
            elif a == -1:
                base = mu_short.get(z, 0.0)
            else:
                base = 0.0
            R[(z, a)] = base - eta_turnover_bps * abs(a)
    return R


def mu_y_from_reward_stats(R_stats_df: pd.DataFrame, dataset_id: str, fee_bps: float) -> dict:
    """E[y|z] from reward stats: mean_bps long = E[y|z]-fee => mu_y[z] = mean_bps_long[z] + fee_bps."""
    mu_y = {z: 0.0 for z in Z_LIST}
    if R_stats_df is None or R_stats_df.empty:
        return mu_y
    grp = R_stats_df[R_stats_df["dataset_id"] == dataset_id]
    for _, r in grp.iterrows():
        z = int(r["z"])
        action = r["action"].lower() if hasattr(r["action"], "lower") else str(r["action"])
        if action == "long":
            mu_y[z] = float(r["mean_bps"]) + fee_bps
    return mu_y


def value_iteration(P_z: np.ndarray, R_table: dict, Imax: int, gamma: float = 0.99,
                    lambda_inv: float = 0.1, eta_turnover_bps: float = 0.0,
                    tol: float = 1e-10, max_iters: int = 50000,
                    reward_mode: str = "one_step_bet", mu_y: dict = None, fee_bps: float = 0.0):
    """
    Infinite-horizon value iteration.
    reward_mode one_step_bet: r = R_table(z,a) - lambda*i'^2 (legacy).
    reward_mode inventory_mtm: r = i'*mu_y[z] - fee_bps*|a| - eta*|a| - lambda*i'^2.
    Tie-breaking: prefer 0 (hold), then action that reduces |i|, then follow z.
    Returns V, policy (3 x (2*Imax+1)), Z, I.
    """
    I = list(range(-Imax, Imax + 1))
    nZ, nI = 3, len(I)
    V = np.zeros((nZ, nI))
    policy = np.zeros((nZ, nI), dtype=int)
    mu_y = mu_y or {z: 0.0 for z in Z_LIST}

    for it in range(max_iters):
        V_old = V.copy()
        for zi, z in enumerate(Z_LIST):
            for ii, i in enumerate(I):
                candidates = []
                for a in (-1, 0, 1):
                    i_next = np.clip(i + a, -Imax, Imax)
                    if reward_mode == "inventory_mtm":
                        r = i_next * mu_y.get(z, 0.0) - fee_bps * abs(a) - eta_turnover_bps * abs(a) - lambda_inv * (i_next ** 2)
                    else:
                        r = R_table.get((z, a), 0.0) - lambda_inv * (i_next ** 2)
                    ev = 0.0
                    for zj in range(3):
                        ev += P_z[zi, zj] * V_old[zj, I.index(i_next)]
                    v = r + gamma * ev
                    candidates.append((v, a, i_next))
                best_v = max(c[0] for c in candidates)
                tied = [c for c in candidates if abs(c[0] - best_v) < 1e-12]
                def key_tie(c):
                    v, a, i_next = c
                    return (0 if a == 0 else 1, abs(i_next))
                best_c = min(tied, key=key_tie)
                best_a = best_c[1]
                V[zi, ii] = best_v
                policy[zi, ii] = best_a
        if np.abs(V - V_old).max() < tol:
            break
    return V, policy, Z_LIST, I


def write_economics_sanity(outdir: Path, dataset_id: str, mu_y: dict, Imax: int):
    """ECONOMICS_SANITY.csv: per z, mu_y[z], and i'*mu_y[z] at i' in {±1, ±Imax}."""
    outdir = Path(outdir)
    rows = []
    for z in Z_LIST:
        m = mu_y.get(z, 0.0)
        for i_prime in [-Imax, -1, 1, Imax]:
            rows.append({"dataset_id": dataset_id, "z": z, "i_prime": i_prime, "mu_y": m, "i_prime_mu_y": i_prime * m})
    if rows:
        pd.DataFrame(rows).to_csv(outdir / "ECONOMICS_SANITY.csv", index=False)


def policy_diagnostics(policy: np.ndarray, Z_list=None, I_list=None) -> dict:
    """policy_depends_on_z: fraction of i states where action differs across z. policy_nontrivial: fraction where a != 0."""
    Z_list = Z_list or Z_LIST
    I_list = I_list or list(range(-policy.shape[1] // 2, policy.shape[1] // 2 + 1))
    if policy.shape[1] != len(I_list):
        I_list = list(range(-(policy.shape[1] - 1) // 2, (policy.shape[1] - 1) // 2 + 1))
    n_i = policy.shape[1]
    depends = 0
    nontrivial = 0
    for ii in range(n_i):
        actions_z = [policy[zi, ii] for zi in range(3)]
        if len(set(actions_z)) > 1:
            depends += 1
        if any(a != 0 for a in actions_z):
            nontrivial += 1
    return {
        "policy_depends_on_z": depends / max(1, n_i),
        "policy_nontrivial": nontrivial / max(1, n_i),
    }


def write_policy_value_tables(outdir: Path, dataset_id: str, V: np.ndarray, policy: np.ndarray,
                              lambda_val: float, Z_list: list, I_list: list):
    """Write VALUE_TABLE_lambda=<x>.csv, POLICY_TABLE_lambda=<x>.csv, POLICY_HEATMAP_lambda=<x>.csv."""
    outdir = Path(outdir)
    lam_str = str(lambda_val).replace(".", "_")
    value_rows = []
    policy_rows = []
    for zi, z in enumerate(Z_list):
        for ii, i in enumerate(I_list):
            value_rows.append({"dataset_id": dataset_id, "z": z, "i": i, "V": float(V[zi, ii])})
            policy_rows.append({"dataset_id": dataset_id, "z": z, "i": i, "action": int(policy[zi, ii])})
    pd.DataFrame(value_rows).to_csv(outdir / f"VALUE_TABLE_lambda_{lam_str}.csv", index=False)
    pd.DataFrame(policy_rows).to_csv(outdir / f"POLICY_TABLE_lambda_{lam_str}.csv", index=False)
    heatmap = pd.DataFrame(policy.astype(int), index=Z_list, columns=I_list)
    heatmap.to_csv(outdir / f"POLICY_HEATMAP_lambda_{lam_str}.csv")
    return value_rows, policy_rows


def read_policy_from_csv(outdir: Path, lambda_val: float, dataset_id: str = None) -> tuple:
    """
    Load policy from POLICY_TABLE_lambda_<x>.csv. Returns (policy 3xnI, I_list).
    If dataset_id is given, filter by it; else use first dataset_id in file.
    """
    outdir = Path(outdir)
    lam_str = str(lambda_val).replace(".", "_")
    path = outdir / f"POLICY_TABLE_lambda_{lam_str}.csv"
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    if df.empty or "z" not in df.columns or "i" not in df.columns or "action" not in df.columns:
        return None, None
    if dataset_id is not None and "dataset_id" in df.columns:
        df = df[df["dataset_id"] == dataset_id]
    if df.empty and "dataset_id" in df.columns:
        all_df = pd.read_csv(path)
        if not all_df.empty:
            first_id = all_df["dataset_id"].iloc[0]
            df = all_df[all_df["dataset_id"] == first_id]
    if df.empty:
        return None, None
    I_list = sorted(df["i"].unique().tolist())
    nI = len(I_list)
    policy = np.zeros((3, nI), dtype=int)
    for _, r in df.iterrows():
        zi = int(r["z"]) + 1
        if zi < 0 or zi > 2:
            continue
        i_val = int(r["i"])
        if i_val in I_list:
            policy[zi, I_list.index(i_val)] = int(r["action"])
    return policy, I_list


def P_z_from_dataframe(P_z_df: pd.DataFrame, dataset_id: str) -> np.ndarray:
    """Build 3x3 P(z'|z) from P_Z_GIVEN_Z.csv for given dataset_id."""
    if P_z_df is None or P_z_df.empty:
        return np.ones((3, 3)) / 3
    grp = P_z_df[P_z_df["dataset_id"] == dataset_id]
    if grp.empty:
        return np.ones((3, 3)) / 3
    P = np.zeros((3, 3))
    for _, r in grp.iterrows():
        i = int(r["z"]) + 1
        j = int(r["z_next"]) + 1
        P[i, j] = r["P"]
    row_sums = P.sum(axis=1, keepdims=True)
    P = P / np.where(row_sums > 0, row_sums, 1)
    return P


def mu_from_reward_stats(R_stats_df: pd.DataFrame, dataset_id: str) -> tuple:
    """(mu_long dict z->float, mu_short dict z->float)."""
    mu_long = {z: 0.0 for z in Z_LIST}
    mu_short = {z: 0.0 for z in Z_LIST}
    if R_stats_df is None or R_stats_df.empty:
        return mu_long, mu_short
    grp = R_stats_df[R_stats_df["dataset_id"] == dataset_id]
    for _, r in grp.iterrows():
        z = int(r["z"])
        action = r["action"].lower() if hasattr(r["action"], "lower") else str(r["action"])
        mean_bps = float(r["mean_bps"])
        if action == "long":
            mu_long[z] = mean_bps
        elif action == "short":
            mu_short[z] = mean_bps
    return mu_long, mu_short
