"""
Phase 3 RL: Evaluation — deterministic market replay + stochastic fills; multiple seeds per window; bootstrap CI.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data_io

from .env import ExecutionEnv, ExecutionEnvConfig
from . import state as st
from . import policies as pol


def build_v_bin_seq(y_seq: np.ndarray, v_bin_threshold: float) -> np.ndarray:
    """v_bin = 1 if |y| >= v_bin_threshold else 0."""
    abs_y = np.abs(y_seq).astype(float)
    return (abs_y >= v_bin_threshold).astype(int)


def sample_window_starts(seq_len: int, window_len: int, num_windows: int, seed: int) -> List[int]:
    """Pick up to num_windows start indices from [0, seq_len - window_len]."""
    max_start = seq_len - window_len
    if max_start <= 0:
        return []
    rng = np.random.default_rng(seed)
    n = min(num_windows, max_start + 1)
    starts = rng.choice(max_start + 1, size=n, replace=False)
    return sorted(starts.tolist())


def run_eval_window(
    env_config: ExecutionEnvConfig,
    z_seq: np.ndarray,
    y_seq: np.ndarray,
    v_bin_seq: np.ndarray,
    start: int,
    window_len: int,
    policy: Callable[[int], int],
    fill_seeds: List[int],
) -> dict:
    """Run policy on one window with multiple fill seeds; return mean/std cum_bps, bps_per_step, turnover_pct."""
    cum_list = []
    bps_list = []
    turn_list = []
    for seed in fill_seeds:
        env = ExecutionEnv(env_config, z_seq, y_seq, v_bin_seq)
        res = pol.run_policy_on_env(env, policy, start_idx=start, seed=seed)
        cum_list.append(res["cum_bps"])
        bps_list.append(res["bps_per_step"])
        turn_list.append(res["turnover_pct"])
    return {
        "cum_bps_mean": float(np.mean(cum_list)),
        "cum_bps_std": float(np.std(cum_list)) if len(cum_list) > 1 else 0.0,
        "bps_per_step_mean": float(np.mean(bps_list)),
        "turnover_pct_mean": float(np.mean(turn_list)),
        "n_seeds": len(fill_seeds),
    }


def run_eval_window_with_visitation(
    env_config: ExecutionEnvConfig,
    z_seq: np.ndarray,
    y_seq: np.ndarray,
    v_bin_seq: np.ndarray,
    start: int,
    window_len: int,
    policy: Callable[[int], int],
    fill_seed: int,
) -> dict:
    """Run policy on one window with one seed; return cum_bps and visitation (action counts, fill_type counts, state distribution)."""
    env = ExecutionEnv(env_config, z_seq, y_seq, v_bin_seq)
    res = pol.run_policy_on_env_with_trajectory(env, policy, start_idx=0, seed=fill_seed)
    actions = res["actions"]
    fill_types = res["fill_types"]
    states_z = res["states_z"]
    states_i = res["states_i"]
    n = len(actions)
    action_counts = [0] * st.N_ACTIONS
    for a in actions:
        action_counts[a] += 1
    maker_fills = sum(1 for f in fill_types if f == "maker")
    taker_fills = sum(1 for f in fill_types if f == "taker")
    total_fills = maker_fills + taker_fills
    maker_share = maker_fills / total_fills if total_fills > 0 else 0.0
    taker_share = taker_fills / total_fills if total_fills > 0 else 0.0
    avg_abs_inv = float(np.mean(np.abs(states_i))) if len(states_i) else 0.0
    return {
        "cum_bps": res["cum_bps"],
        "steps": res["steps"],
        "turnover_pct": res["turnover_pct"],
        "avg_abs_inv": avg_abs_inv,
        "action_counts": action_counts,
        "action_frac_HOLD": action_counts[st.A_HOLD] / n if n else 0,
        "action_frac_PLACE_BUY": action_counts[st.A_PLACE_BUY] / n if n else 0,
        "action_frac_PLACE_SELL": action_counts[st.A_PLACE_SELL] / n if n else 0,
        "action_frac_BUY_MARKET": action_counts[st.A_BUY_MARKET] / n if n else 0,
        "action_frac_SELL_MARKET": action_counts[st.A_SELL_MARKET] / n if n else 0,
        "maker_fills": maker_fills,
        "taker_fills": taker_fills,
        "maker_share": maker_share,
        "taker_share": taker_share,
        "states_z": states_z,
        "states_i": states_i,
        "state_idxs": res["state_idxs"],
    }


def run_eval_all(
    env_config: ExecutionEnvConfig,
    z_seq: np.ndarray,
    y_seq: np.ndarray,
    v_bin_seq: np.ndarray,
    window_len: int,
    num_windows: int,
    fill_seeds: List[int],
    bootstrap_iters: int,
    seed: int,
    Q: Optional[np.ndarray] = None,
    policy_name: str = "baseline",
    policy_fn: Optional[Callable[[int], int]] = None,
    deterministic_replay: bool = False,
) -> dict:
    """
    Run one policy on K windows with multiple fill seeds each; bootstrap over windows for CI.
    deterministic_replay: if True, use only the first fill_seed per window for reproducible eval.
    """
    T = min(len(z_seq), len(y_seq), len(v_bin_seq))
    if T < window_len:
        return {}
    window_starts = sample_window_starts(T, window_len, num_windows, seed)
    if not window_starts:
        return {}
    if policy_fn is None and Q is not None:
        rng = np.random.default_rng(seed)
        policy_fn = pol.policy_greedy_from_q(Q, env_config.Imax, rng=rng)
    elif policy_fn is None:
        return {}
    # deterministic = market path fixed; still average over all fill_seeds for stable CI
    seeds_to_use = fill_seeds if fill_seeds else [seed]
    per_window = []
    for start in window_starts:
        z_slice = z_seq[start : start + window_len]
        y_slice = y_seq[start : start + window_len]
        v_slice = v_bin_seq[start : start + window_len]
        res = run_eval_window(env_config, z_slice, y_slice, v_slice, 0, window_len, policy_fn, seeds_to_use)
        per_window.append({**res, "start": start})
    cum_means = [w["cum_bps_mean"] for w in per_window]
    rng = np.random.default_rng(seed)
    boot_means = []
    nw = len(per_window)
    for _ in range(bootstrap_iters):
        idx = rng.integers(0, nw, size=nw)
        boot_means.append(float(np.mean([cum_means[i] for i in idx])))
    boot_means = np.array(boot_means)
    return {
        "policy_name": policy_name,
        "mean_cum_bps": float(np.mean(cum_means)),
        "std_cum_bps": float(np.std(cum_means)) if nw > 1 else 0.0,
        "mean_bps_per_step": float(np.mean([w["bps_per_step_mean"] for w in per_window])),
        "turnover_pct": float(np.mean([w["turnover_pct_mean"] for w in per_window])),
        "bootstrap_ci_low": float(np.percentile(boot_means, 2.5)),
        "bootstrap_ci_high": float(np.percentile(boot_means, 97.5)),
        "window_starts": window_starts,
        "per_window": per_window,
        "per_window_cum_bps": [w["cum_bps_mean"] for w in per_window],
    }


def run_eval_ql_with_visitation(
    env_config: ExecutionEnvConfig,
    z_seq: np.ndarray,
    y_seq: np.ndarray,
    v_bin_seq: np.ndarray,
    window_len: int,
    num_windows: int,
    fill_seeds: List[int],
    seed: int,
    Q: np.ndarray,
    deterministic_replay: bool = False,
) -> Tuple[dict, List[dict], List[dict]]:
    """Run QL on eval windows; return (eval_result_dict, per_window_visitation_rows, state_topk_rows)."""
    T = min(len(z_seq), len(y_seq), len(v_bin_seq))
    if T < window_len or Q is None:
        return {}, [], []
    window_starts = sample_window_starts(T, window_len, num_windows, seed)
    if not window_starts:
        return {}, [], []
    rng = np.random.default_rng(seed)
    policy_fn = pol.policy_greedy_from_q(Q, env_config.Imax, rng=rng)
    # deterministic = market path fixed; average over all fill_seeds per window for stable CI
    seeds_to_use = fill_seeds if fill_seeds else [seed]
    per_window_cum = []
    visitation_rows = []
    all_z_i_counts: Dict[Tuple[int, int], int] = {}
    for wi, start in enumerate(window_starts):
        z_slice = z_seq[start : start + window_len]
        y_slice = y_seq[start : start + window_len]
        v_slice = v_bin_seq[start : start + window_len]
        # Window reward = mean over seeds (report-grade)
        res_multi = run_eval_window(env_config, z_slice, y_slice, v_slice, 0, window_len, policy_fn, seeds_to_use)
        per_window_cum.append(res_multi["cum_bps_mean"])
        # Visitation from one trajectory (first seed) for diagnostics
        res = run_eval_window_with_visitation(env_config, z_slice, y_slice, v_slice, 0, window_len, policy_fn, seeds_to_use[0])
        for (z, i) in zip(res["states_z"], res["states_i"]):
            all_z_i_counts[(z, i)] = all_z_i_counts.get((z, i), 0) + 1
        visitation_rows.append({
            "window_idx": wi,
            "start": start,
            "cum_bps": res["cum_bps"],
            "steps": res["steps"],
            "turnover_pct": res["turnover_pct"],
            "avg_abs_inv": res.get("avg_abs_inv", 0.0),
            "action_frac_HOLD": res["action_frac_HOLD"],
            "action_frac_PLACE_BUY": res["action_frac_PLACE_BUY"],
            "action_frac_PLACE_SELL": res["action_frac_PLACE_SELL"],
            "action_frac_BUY_MARKET": res["action_frac_BUY_MARKET"],
            "action_frac_SELL_MARKET": res["action_frac_SELL_MARKET"],
            "maker_fills": res["maker_fills"],
            "taker_fills": res["taker_fills"],
            "maker_share": res["maker_share"],
            "taker_share": res["taker_share"],
        })
    nw = len(per_window_cum)
    # State top-K: sort (z,i) by count, take top 50
    state_topk = sorted(all_z_i_counts.items(), key=lambda x: -x[1])[:50]
    state_topk_rows = [{"z": z, "i": i, "count": c} for (z, i), c in state_topk]
    return {
        "mean_cum_bps": float(np.mean(per_window_cum)),
        "std_cum_bps": float(np.std(per_window_cum)) if nw > 1 else 0.0,
        "per_window_cum_bps": per_window_cum,
        "visitation_rows": visitation_rows,
    }, visitation_rows, state_topk_rows


def _bootstrap_ci(values: List[float], bootstrap_iters: int, seed: int) -> Tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    boot = [float(np.mean([values[i] for i in rng.integers(0, n, size=n)])) for _ in range(bootstrap_iters)]
    return float(np.mean(values)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def bootstrap_delta_fixed_and_oracle(
    ql_per_window: List[float],
    baseline_per_window: Dict[str, List[float]],
    baseline_means: Dict[str, float],
    bootstrap_iters: int,
    seed: int,
) -> dict:
    """
    Delta_fixed: QL - (fixed best baseline by mean). best_fixed = argmax baseline_means.
    Delta_oracle: QL - (window-wise max(A,B,C)); upper bound.
    Returns dict with delta_fixed_*, delta_oracle_*, best_fixed_baseline (name).
    """
    nw = len(ql_per_window)
    if nw == 0 or not baseline_per_window or not baseline_means:
        return {
            "best_fixed_baseline": "",
            "delta_fixed_mean": 0.0, "delta_fixed_ci_low": 0.0, "delta_fixed_ci_high": 0.0,
            "delta_oracle_mean": 0.0, "delta_oracle_ci_low": 0.0, "delta_oracle_ci_high": 0.0,
        }
    best_fixed_name = max(baseline_means.keys(), key=lambda k: baseline_means[k])
    fixed_per_window = baseline_per_window.get(best_fixed_name, [0.0] * nw)
    if len(fixed_per_window) < nw:
        fixed_per_window = (fixed_per_window + [0.0] * nw)[:nw]
    delta_fixed = [ql_per_window[w] - fixed_per_window[w] for w in range(nw)]
    oracle_per_window = []
    for w in range(nw):
        vals = [baseline_per_window[n][w] for n in baseline_per_window if w < len(baseline_per_window.get(n, []))]
        oracle_per_window.append(max(vals) if vals else 0.0)
    delta_oracle = [ql_per_window[w] - oracle_per_window[w] for w in range(nw)]
    m_f, lo_f, hi_f = _bootstrap_ci(delta_fixed, bootstrap_iters, seed)
    m_o, lo_o, hi_o = _bootstrap_ci(delta_oracle, bootstrap_iters, seed + 1)
    return {
        "best_fixed_baseline": best_fixed_name,
        "delta_fixed_mean": m_f, "delta_fixed_ci_low": lo_f, "delta_fixed_ci_high": hi_f,
        "delta_oracle_mean": m_o, "delta_oracle_ci_low": lo_o, "delta_oracle_ci_high": hi_o,
    }


def bootstrap_delta_fair_and_gap_to_oracle(
    ql_per_window: List[float],
    baseline_per_window: Dict[str, List[float]],
    baseline_means: Dict[str, float],
    fair_baseline_names: List[str],
    dp_exact_name: str,
    bootstrap_iters: int,
    seed: int,
) -> dict:
    """
    Delta_fair (primary): QL - best fixed among fair_baseline_names only (A, B, Hold, DP_empirical).
    Gap_to_oracle (secondary): QL - DP_exact. DP_exact not in fair set.
    Returns dict with best_fair_baseline, delta_fair_*, gap_to_oracle_*.
    """
    nw = len(ql_per_window)
    out = {
        "best_fair_baseline": "",
        "delta_fair_mean": 0.0, "delta_fair_ci_low": 0.0, "delta_fair_ci_high": 0.0,
        "gap_to_oracle_mean": 0.0, "gap_to_oracle_ci_low": 0.0, "gap_to_oracle_ci_high": 0.0,
    }
    if nw == 0 or not baseline_per_window or not baseline_means:
        return out
    fair_means = {k: baseline_means[k] for k in fair_baseline_names if k in baseline_means}
    if not fair_means:
        return out
    best_fair_name = max(fair_means.keys(), key=lambda k: fair_means[k])
    fair_per_window = baseline_per_window.get(best_fair_name, [0.0] * nw)
    if len(fair_per_window) < nw:
        fair_per_window = (fair_per_window + [0.0] * nw)[:nw]
    delta_fair = [ql_per_window[w] - fair_per_window[w] for w in range(nw)]
    dp_exact_per_window = baseline_per_window.get(dp_exact_name, [0.0] * nw)
    if len(dp_exact_per_window) < nw:
        dp_exact_per_window = (dp_exact_per_window + [0.0] * nw)[:nw]
    gap_to_oracle = [ql_per_window[w] - dp_exact_per_window[w] for w in range(nw)]
    m_f, lo_f, hi_f = _bootstrap_ci(delta_fair, bootstrap_iters, seed)
    m_g, lo_g, hi_g = _bootstrap_ci(gap_to_oracle, bootstrap_iters, seed + 1)
    out["best_fair_baseline"] = best_fair_name
    out["delta_fair_mean"] = m_f
    out["delta_fair_ci_low"] = lo_f
    out["delta_fair_ci_high"] = hi_f
    out["gap_to_oracle_mean"] = m_g
    out["gap_to_oracle_ci_low"] = lo_g
    out["gap_to_oracle_ci_high"] = hi_g
    return out


def bootstrap_delta_to_best_baseline(
    ql_per_window: List[float],
    baseline_per_window: Dict[str, List[float]],
    bootstrap_iters: int,
    seed: int,
) -> dict:
    """Legacy: oracle delta (window-wise max). Prefer bootstrap_delta_fixed_and_oracle."""
    nw = len(ql_per_window)
    if nw == 0:
        return {"ql_minus_best_baseline_mean": 0.0, "ql_minus_best_baseline_ci_low": 0.0, "ql_minus_best_baseline_ci_high": 0.0}
    best_per_window = []
    for w in range(nw):
        vals = [baseline_per_window[name][w] for name in baseline_per_window if w < len(baseline_per_window.get(name, []))]
        best_per_window.append(max(vals) if vals else 0.0)
    diffs = [ql_per_window[w] - best_per_window[w] for w in range(nw)]
    m, lo, hi = _bootstrap_ci(diffs, bootstrap_iters, seed)
    return {
        "ql_minus_best_baseline_mean": m,
        "ql_minus_best_baseline_ci_low": lo,
        "ql_minus_best_baseline_ci_high": hi,
    }


def build_phase3_sequences(
    manifest_df,
    run_root: Path,
    dataset_id: str,
    split: str,
    z_thresholds: list,
    v_bin_threshold: float,
    stride: int = 1,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load split, compute z_seq, y_seq, v_bin_seq. Returns (z_seq, y_seq, v_bin_seq) or (None, None, None)."""
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
    if err or df is None or len(df) < 10 or k != "step1":
        return None, None, None
    df = df.sort_values("t").reset_index(drop=True)
    if stride > 1:
        df = df.iloc[::stride].reset_index(drop=True)
    pred = df["pred"].values
    y_seq = np.asarray(df["y"].values, dtype=float)
    z_seq = np.zeros(len(pred), dtype=int)
    z_seq[pred >= q_hi] = 1
    z_seq[pred <= q_lo] = -1
    v_bin_seq = build_v_bin_seq(y_seq, v_bin_threshold)
    return z_seq, y_seq, v_bin_seq
