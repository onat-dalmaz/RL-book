"""
Phase 2 DP v3: Replay Monte Carlo rollouts.
Step1: counterfactual reward samples by (z, action).
S7: proxy evaluation from observed side-specific reward distributions.
"""

from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

import data_io


def run_rollouts(
    dataset_id: str,
    z_seq: np.ndarray,
    policy: np.ndarray,
    Imax: int,
    lambda_inv: float,
    kind: str,
    n_rollouts: int,
    max_steps: int,
    seed: int,
    # Step1: samples_long[z] = list of r_long, samples_short[z] = list of r_short
    samples_long: dict = None,
    samples_short: dict = None,
    # S7: net_long[z], net_short[z] lists (observed side only; proxy)
    net_long_by_z: dict = None,
    net_short_by_z: dict = None,
    eta_turnover_bps: float = 0.0,
) -> list:
    """
    Run n_rollouts over z_seq. Return list of dicts with keys:
    cum_bps, turnover_pct, avg_abs_inv, p05, p50, p95 (per rollout then aggregate).
    kind in ("step1", "s7").
    """
    rng = np.random.default_rng(seed)
    samples_long = samples_long or {}
    samples_short = samples_short or {}
    net_long_by_z = net_long_by_z or {}
    net_short_by_z = net_short_by_z or {}
    T_full = len(z_seq)
    steps_used = min(T_full, max_steps) if max_steps > 0 else T_full
    T = steps_used
    I_list = list(range(-Imax, Imax + 1))

    rewards_list = []
    turnover_list = []
    inv_list = []

    for _ in range(n_rollouts):
        i = 0
        cum = 0.0
        turn = 0
        inv_abs = []
        for t in range(T):
            z = int(z_seq[t])
            zi = z + 1
            ii = I_list.index(i)
            a = int(policy[zi, ii])
            i_next = np.clip(i + a, -Imax, Imax)
            if a != 0:
                turn += 1
            inv_abs.append(abs(i_next))

            if kind == "step1":
                if a == 1 and samples_long.get(z):
                    r_t = float(rng.choice(samples_long[z]))
                elif a == -1 and samples_short.get(z):
                    r_t = float(rng.choice(samples_short[z]))
                else:
                    r_t = 0.0
            else:
                if a == 1 and net_long_by_z.get(z):
                    r_t = float(rng.choice(net_long_by_z[z]))
                elif a == -1 and net_short_by_z.get(z):
                    r_t = float(rng.choice(net_short_by_z[z]))
                else:
                    r_t = 0.0

            cum += r_t - lambda_inv * (i_next ** 2) - eta_turnover_bps * abs(a)
            i = i_next

        rewards_list.append(cum)
        turnover_list.append(turn / max(1, T))
        inv_list.append(np.mean(inv_abs))

    mean_cum = float(np.mean(rewards_list))
    std_cum = float(np.std(rewards_list))
    mean_steps_used = float(steps_used)
    mean_bps_per_step = mean_cum / mean_steps_used if mean_steps_used > 0 else 0.0
    std_bps_per_step = std_cum / mean_steps_used if mean_steps_used > 0 else 0.0
    max_steps_per_rollout = steps_used
    return {
        "mean_cum_bps": mean_cum,
        "std_cum_bps": std_cum,
        "turnover_pct": float(np.mean(turnover_list)) * 100,
        "avg_abs_inv": float(np.mean(inv_list)),
        "p05": float(np.percentile(rewards_list, 5)),
        "p50": float(np.percentile(rewards_list, 50)),
        "p95": float(np.percentile(rewards_list, 95)),
        "rewards_list": rewards_list,
        "mean_steps_used": mean_steps_used,
        "mean_bps_per_step": mean_bps_per_step,
        "std_bps_per_step": std_bps_per_step,
        "max_steps_per_rollout": max_steps_per_rollout,
    }


def sample_window_starts(z_seq_len: int, window_len: int, num_windows: int, seed: int) -> list:
    """
    Deterministic: pick K start indices uniformly from [0, z_seq_len - window_len].
    Returns list of start indices (length <= num_windows; may be fewer if range too small).
    """
    max_start = z_seq_len - window_len
    if max_start <= 0:
        return []
    rng = np.random.default_rng(seed)
    starts = rng.choice(max_start + 1, size=min(num_windows, max_start + 1), replace=False)
    return sorted(starts.tolist())


def run_rollouts_windowed(
    dataset_id: str,
    z_seq: np.ndarray,
    policy: np.ndarray,
    Imax: int,
    lambda_inv: float,
    kind: str,
    window_len: int,
    num_windows: int,
    n_rollouts_total: int,
    seed: int,
    samples_long: dict = None,
    samples_short: dict = None,
    net_long_by_z: dict = None,
    net_short_by_z: dict = None,
    eta_turnover_bps: float = 0.0,
):
    """
    Evaluate on K contiguous windows of length L. Rollouts are split across windows
    (n_per_window = n_rollouts_total // num_windows). Aggregate mean/std across windows.
    Returns same-shaped dict as run_rollouts, plus "window_starts" (list of int).
    """
    if len(z_seq) < window_len:
        # Fallback: single window = full sequence
        out = run_rollouts(
            dataset_id, z_seq, policy, Imax, lambda_inv, kind,
            n_rollouts_total, 0, seed,
            samples_long=samples_long, samples_short=samples_short,
            net_long_by_z=net_long_by_z, net_short_by_z=net_short_by_z,
            eta_turnover_bps=eta_turnover_bps,
        )
        out["window_starts"] = []
        return out
    window_starts = sample_window_starts(len(z_seq), window_len, num_windows, seed)
    if not window_starts:
        out = run_rollouts(
            dataset_id, z_seq, policy, Imax, lambda_inv, kind,
            n_rollouts_total, 0, seed,
            samples_long=samples_long, samples_short=samples_short,
            net_long_by_z=net_long_by_z, net_short_by_z=net_short_by_z,
            eta_turnover_bps=eta_turnover_bps,
        )
        out["window_starts"] = []
        return out
    n_per_window = max(1, n_rollouts_total // len(window_starts))
    per_window_means = []
    per_window_std = []
    per_turnover = []
    per_abs_inv = []
    per_p05 = []
    per_p50 = []
    per_p95 = []

    for wi, start in enumerate(window_starts):
        z_slice = z_seq[start : start + window_len]
        res = run_rollouts(
            dataset_id, z_slice, policy, Imax, lambda_inv, kind,
            n_per_window, 0, seed + wi + 1,
            samples_long=samples_long, samples_short=samples_short,
            net_long_by_z=net_long_by_z, net_short_by_z=net_short_by_z,
            eta_turnover_bps=eta_turnover_bps,
        )
        per_window_means.append(res["mean_cum_bps"])
        per_window_std.append(res["std_cum_bps"])
        per_turnover.append(res["turnover_pct"])
        per_abs_inv.append(res["avg_abs_inv"])
        per_p05.append(res["p05"])
        per_p50.append(res["p50"])
        per_p95.append(res["p95"])

    mean_cum_bps = float(np.mean(per_window_means))
    std_cum_bps = float(np.std(per_window_means)) if len(per_window_means) > 1 else float(per_window_std[0]) if per_window_std else 0.0
    mean_steps_used = float(window_len)
    mean_bps_per_step = mean_cum_bps / mean_steps_used if mean_steps_used > 0 else 0.0
    std_bps_per_step = std_cum_bps / mean_steps_used if mean_steps_used > 0 else 0.0
    return {
        "mean_cum_bps": mean_cum_bps,
        "std_cum_bps": std_cum_bps,
        "turnover_pct": float(np.mean(per_turnover)),
        "avg_abs_inv": float(np.mean(per_abs_inv)),
        "p05": float(np.mean(per_p05)),
        "p50": float(np.mean(per_p50)),
        "p95": float(np.mean(per_p95)),
        "mean_steps_used": mean_steps_used,
        "mean_bps_per_step": mean_bps_per_step,
        "std_bps_per_step": std_bps_per_step,
        "max_steps_per_rollout": int(mean_steps_used),
        "window_starts": window_starts,
    }


def run_deterministic_replay(
    z_seq: np.ndarray,
    y_seq: np.ndarray,
    policy: np.ndarray,
    Imax: int,
    fee_bps: float,
    lambda_inv: float,
    eta_turnover_bps: float = 0.0,
) -> dict:
    """
    Single trajectory: r_t = i_{t+1}*y_t - fee*|a_t| - lambda*i_{t+1}^2 - eta*|a_t|.
    Returns cum_bps, turnover_pct, avg_abs_inv, (and bps_per_step = cum_bps/T).
    """
    T = min(len(z_seq), len(y_seq))
    if T == 0:
        return {"cum_bps": 0.0, "turnover_pct": 0.0, "avg_abs_inv": 0.0, "bps_per_step": 0.0}
    I_list = list(range(-Imax, Imax + 1))
    i = 0
    cum = 0.0
    turn = 0
    inv_abs = []
    for t in range(T):
        z = int(z_seq[t])
        zi = z + 1
        ii = I_list.index(i)
        a = int(policy[zi, ii])
        i_next = np.clip(i + a, -Imax, Imax)
        y_t = float(y_seq[t])
        r_t = i_next * y_t - fee_bps * abs(a) - lambda_inv * (i_next ** 2) - eta_turnover_bps * abs(a)
        cum += r_t
        if a != 0:
            turn += 1
        inv_abs.append(abs(i_next))
        i = i_next
    return {
        "cum_bps": cum,
        "turnover_pct": (turn / T) * 100,
        "avg_abs_inv": float(np.mean(inv_abs)),
        "bps_per_step": cum / T if T > 0 else 0.0,
        "steps_used": T,
    }


def run_deterministic_replay_windowed(
    dataset_id: str,
    z_seq: np.ndarray,
    y_seq: np.ndarray,
    policy: np.ndarray,
    Imax: int,
    fee_bps: float,
    lambda_inv: float,
    eta_turnover_bps: float,
    window_len: int,
    num_windows: int,
    seed: int,
    bootstrap_iters: int = 1000,
) -> dict:
    """
    Run deterministic replay on K windows; aggregate; bootstrap over windows for CI.
    Returns dict with mean_cum_bps, std_cum_bps, mean_bps_per_step, turnover_pct, avg_abs_inv,
    window_starts, per_window_rows (for EVAL_WINDOWS_DETERMINISTIC), bootstrap_ci (for EVAL_BOOTSTRAP_CI).
    """
    T = min(len(z_seq), len(y_seq))
    if T < window_len:
        single = run_deterministic_replay(z_seq, y_seq, policy, Imax, fee_bps, lambda_inv, eta_turnover_bps)
        return {
            "mean_cum_bps": single["cum_bps"],
            "std_cum_bps": 0.0,
            "mean_bps_per_step": single["bps_per_step"],
            "std_bps_per_step": 0.0,
            "turnover_pct": single["turnover_pct"],
            "avg_abs_inv": single["avg_abs_inv"],
            "p05": single["cum_bps"], "p50": single["cum_bps"], "p95": single["cum_bps"],
            "mean_steps_used": single["steps_used"],
            "max_steps_per_rollout": single["steps_used"],
            "window_starts": [],
            "per_window_rows": [],
            "bootstrap_mean": single["cum_bps"],
            "bootstrap_ci_low": single["cum_bps"],
            "bootstrap_ci_high": single["cum_bps"],
        }
    window_starts = sample_window_starts(T, window_len, num_windows, seed)
    if not window_starts:
        single = run_deterministic_replay(z_seq, y_seq, policy, Imax, fee_bps, lambda_inv, eta_turnover_bps)
        return {
            "mean_cum_bps": single["cum_bps"], "std_cum_bps": 0.0,
            "mean_bps_per_step": single["bps_per_step"], "std_bps_per_step": 0.0,
            "turnover_pct": single["turnover_pct"], "avg_abs_inv": single["avg_abs_inv"],
            "p05": single["cum_bps"], "p50": single["cum_bps"], "p95": single["cum_bps"],
            "mean_steps_used": single["steps_used"], "max_steps_per_rollout": single["steps_used"],
            "window_starts": [], "per_window_rows": [],
            "bootstrap_mean": single["cum_bps"], "bootstrap_ci_low": single["cum_bps"], "bootstrap_ci_high": single["cum_bps"],
        }
    per_window = []
    for start in window_starts:
        end = min(start + window_len, T)
        z_slice = z_seq[start:end]
        y_slice = y_seq[start:end]
        res = run_deterministic_replay(z_slice, y_slice, policy, Imax, fee_bps, lambda_inv, eta_turnover_bps)
        per_window.append({
            "cum_bps": res["cum_bps"],
            "bps_per_step": res["bps_per_step"],
            "turnover_pct": res["turnover_pct"],
            "avg_abs_inv": res["avg_abs_inv"],
        })
    cum_list = [w["cum_bps"] for w in per_window]
    bps_list = [w["bps_per_step"] for w in per_window]
    rng = np.random.default_rng(seed)
    boot_means = []
    nw = len(per_window)
    for _ in range(bootstrap_iters):
        idx = rng.integers(0, nw, size=nw)
        boot_means.append(float(np.mean([cum_list[i] for i in idx])))
    boot_means = np.array(boot_means)
    return {
        "mean_cum_bps": float(np.mean(cum_list)),
        "std_cum_bps": float(np.std(cum_list)) if len(cum_list) > 1 else 0.0,
        "mean_bps_per_step": float(np.mean(bps_list)),
        "std_bps_per_step": float(np.std(bps_list)) if len(bps_list) > 1 else 0.0,
        "turnover_pct": float(np.mean([w["turnover_pct"] for w in per_window])),
        "avg_abs_inv": float(np.mean([w["avg_abs_inv"] for w in per_window])),
        "p05": float(np.percentile(cum_list, 5)),
        "p50": float(np.percentile(cum_list, 50)),
        "p95": float(np.percentile(cum_list, 95)),
        "mean_steps_used": float(window_len),
        "max_steps_per_rollout": window_len,
        "window_starts": window_starts,
        "per_window_rows": per_window,
        "bootstrap_mean": float(np.mean(boot_means)),
        "bootstrap_ci_low": float(np.percentile(boot_means, 2.5)),
        "bootstrap_ci_high": float(np.percentile(boot_means, 97.5)),
    }


def build_z_seq_and_reward_pools(
    manifest_df: pd.DataFrame,
    run_root: Path,
    dataset_id: str,
    split: str,
    z_thresholds: list,
    kind: str,
    fee_bps: float = 0.0,
):
    """
    Load test (or given split) data, compute z sequence, and build reward pools for rollouts.
    Returns (z_seq, samples_long, samples_short) for step1, or (z_seq, net_long_by_z, net_short_by_z) for s7.
    """
    run_root = Path(run_root)
    thresh = {r["dataset_id"]: (r["q_lo"], r["q_hi"]) for r in z_thresholds}
    q_lo, q_hi = thresh.get(dataset_id, (-0.01, 0.01))
    parts = dataset_id.split("_")
    coin = parts[0] if parts else "UNK"
    variant = parts[1] if len(parts) > 1 else coin
    fold = 0
    for p in parts:
        if p.startswith("fold") and len(p) > 4:
            try:
                fold = int(p[4:])
            except ValueError:
                pass
            break

    grp = manifest_df[(manifest_df["coin"] == coin) & (manifest_df["variant"] == variant) & (manifest_df["fold"] == fold)]
    path_row = grp[grp["split"] == split]
    if path_row.empty:
        return None, None, None
    path = run_root / path_row.iloc[0]["path"]
    k = path_row.iloc[0]["kind"]
    df, err = data_io.load_and_canonicalize(path, run_root, k, allow_no_pred_s7=True)
    if err or df is None or len(df) < 10:
        return None, None, None

    pred = df["pred"].values
    z_seq = np.zeros(len(pred), dtype=int)
    z_seq[pred >= q_hi] = 1
    z_seq[pred <= q_lo] = -1

    if kind == "step1":
        y = df["y"].values
        r_long = y - fee_bps
        r_short = -y - fee_bps
        samples_long = defaultdict(list)
        samples_short = defaultdict(list)
        for zi in (-1, 0, 1):
            mask = z_seq == zi
            samples_long[zi] = r_long[mask].tolist()
            samples_short[zi] = r_short[mask].tolist()
        return z_seq, dict(samples_long), dict(samples_short)
    else:
        net = df["net_bps"].values
        side = df["side"].values
        net_long_by_z = defaultdict(list)
        net_short_by_z = defaultdict(list)
        for zi in (-1, 0, 1):
            mask = (z_seq == zi) & (side == 1)
            net_long_by_z[zi] = net[mask].tolist()
            mask = (z_seq == zi) & (side == -1)
            net_short_by_z[zi] = net[mask].tolist()
        return z_seq, dict(net_long_by_z), dict(net_short_by_z)


def build_z_seq_and_y_seq(
    manifest_df: pd.DataFrame,
    run_root: Path,
    dataset_id: str,
    split: str,
    z_thresholds: list,
    kind: str,
    stride: int = 1,
):
    """
    Load split data, optionally downsample by stride, return (z_seq, y_seq) for deterministic replay.
    Step1 only; returns (None, None) for s7.
    """
    run_root = Path(run_root)
    thresh = {r["dataset_id"]: (r["q_lo"], r["q_hi"]) for r in z_thresholds}
    q_lo, q_hi = thresh.get(dataset_id, (-0.01, 0.01))
    parts = dataset_id.split("_")
    coin = parts[0] if parts else "UNK"
    variant = parts[1] if len(parts) > 1 else coin
    fold = 0
    for p in parts:
        if p.startswith("fold") and len(p) > 4:
            try:
                fold = int(p[4:])
            except ValueError:
                pass
            break
    grp = manifest_df[(manifest_df["coin"] == coin) & (manifest_df["variant"] == variant) & (manifest_df["fold"] == fold)]
    path_row = grp[grp["split"] == split]
    if path_row.empty:
        return None, None
    path = run_root / path_row.iloc[0]["path"]
    k = path_row.iloc[0]["kind"]
    df, err = data_io.load_and_canonicalize(path, run_root, k, allow_no_pred_s7=True)
    if err or df is None or len(df) < 10 or kind != "step1":
        return None, None
    df = df.sort_values("t").reset_index(drop=True)
    if stride > 1:
        df = df.iloc[::stride].reset_index(drop=True)
    pred = df["pred"].values
    y_seq = np.asarray(df["y"].values, dtype=float)
    z_seq = np.zeros(len(pred), dtype=int)
    z_seq[pred >= q_hi] = 1
    z_seq[pred <= q_lo] = -1
    return z_seq, y_seq
