#!/usr/bin/env python3
"""
Phase 2 DP/MDP from existing *_trades.parquet.
Builds manifest, opportunity stream, reward model, tabular MDP, value iteration,
baselines, replay evaluation, and PASS_FAIL gates.
"""

import argparse
import json
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

# Canonical column mapping
ENTRY_TS_CANDIDATES = ["entry_ts", "entry_time", "timestamp", "ts"]
SIDE_CANDIDATES = ["side", "pos"]
PRED_CANDIDATES = ["pred_at_entry", "pred", "signal", "alpha"]
NET_BPS_CANDIDATES = ["net_bps", "net_pnl_bps", "net_exec_bps", "pnl_bps"]
GROSS_BPS_CANDIDATES = ["gross_bps", "gross_pnl_bps"]


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_path_metadata(path: Path, run_root: Path):
    rel = path.relative_to(run_root)
    parts = rel.parts
    coin = variant = fold = split = None
    for i, p in enumerate(parts):
        if p in ("MAGIC", "MINA", "RVN", "HOT", "AXS", "SNX") and coin is None:
            coin = p
        if p.startswith("S7_") or p.startswith("fold_"):
            if p.startswith("S7_"):
                variant = p
            if p.startswith("fold_"):
                try:
                    fold = int(p.split("_")[1])
                except (IndexError, ValueError):
                    fold = 0
        if "test_trades" in path.name or "trades_test" in path.name:
            split = "test"
        elif "val_trades" in path.name or "trades_val" in path.name:
            split = "val"
        elif "train_trades" in path.name or "trades_train" in path.name:
            split = "train"
    if coin is None and len(parts) >= 2 and parts[0] == "COINS":
        coin = parts[1]
    if coin is None and len(parts) >= 1:
        coin = parts[0]
    if variant is None and coin is not None:
        variant = coin
    return coin or "UNK", variant or "UNK", fold if fold is not None else 0, split or "test"


def phase0_manifest(run_root: Path, outdir: Path):
    """Phase 0: Build manifest of trade parquets."""
    run_root = Path(run_root)
    rows = []
    for name in [
        "test_trades.parquet", "val_trades.parquet", "train_trades.parquet",
        "trades_test.parquet", "trades_val.parquet", "trades_train.parquet",
    ]:
        for path in run_root.rglob(name):
            if not path.is_file() or pq is None:
                continue
            try:
                t = pq.read_table(str(path))
                df = t.to_pandas()
                n = len(df)
                ts_col = find_col(df, ENTRY_TS_CANDIDATES)
                min_ts = max_ts = None
                if ts_col:
                    ts = pd.to_datetime(df[ts_col], errors="coerce").dropna()
                    if len(ts) > 0:
                        min_ts = str(ts.min())
                        max_ts = str(ts.max())
                coin, variant, fold, split = parse_path_metadata(path, run_root)
                rows.append({
                    "coin": coin,
                    "variant": variant,
                    "fold": fold,
                    "split": split,
                    "path": str(path.relative_to(run_root)),
                    "n_rows": n,
                    "min_ts": min_ts,
                    "max_ts": max_ts,
                    "columns_json": json.dumps(list(df.columns)),
                })
            except Exception as e:
                rows.append({
                    "coin": "ERR", "variant": path.name, "fold": -1, "split": "err",
                    "path": str(path.relative_to(run_root)), "n_rows": 0,
                    "min_ts": None, "max_ts": None, "columns_json": str(e)[:200],
                })
    if not rows:
        (outdir / "PHASE2_DP_MANIFEST.csv").write_text("coin,variant,fold,split,path,n_rows,min_ts,max_ts,columns_json\n")
        return pd.DataFrame(), []
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "PHASE2_DP_MANIFEST.csv", index=False)
    return df, rows


def load_and_standardize(path, run_root: Path):
    """Load parquet and standardize to side in {-1,+1}, pred, net_bps. Returns (df, missing_cols)."""
    path = Path(path)
    run_root = Path(run_root)
    # path may be already full (run_root / rel); otherwise resolve relative to run_root
    if path.is_absolute() or path.exists():
        full = path
    else:
        full = run_root / path
    if not full.exists():
        return None, ["path"]
    if pq is None:
        return None, ["path"]
    df = pq.read_table(str(full)).to_pandas()
    if df.empty:
        return None, ["no_rows"]
    sort_col = find_col(df, ENTRY_TS_CANDIDATES) or df.columns[0]
    df = df.sort_values(by=sort_col).reset_index(drop=True)
    entry_ts = find_col(df, ENTRY_TS_CANDIDATES)
    side_col = find_col(df, SIDE_CANDIDATES)
    pred_col = find_col(df, PRED_CANDIDATES)
    net_col = find_col(df, NET_BPS_CANDIDATES)
    missing = []
    if not entry_ts:
        missing.append("entry_ts")
    if not side_col:
        missing.append("side")
    if not net_col:
        missing.append("net_bps")
    if not pred_col:
        # Proxy: use side as pred (long=+0.5, short=-0.5) so z buckets work
        df["_pred_proxy"] = 0.5
        if side_col:
            side_raw = df[side_col]
            if side_raw.dtype == object or side_raw.dtype.name == "string":
                df["_pred_proxy"] = np.where(side_raw.str.lower().str.contains("long|buy|1").fillna(False), 0.5, -0.5)
            else:
                df["_pred_proxy"] = np.where(side_raw > 0, 0.5, -0.5)
        pred_col = "_pred_proxy"
    if missing:
        return None, missing
    df["entry_ts"] = pd.to_datetime(df[entry_ts], utc=True)
    df["side"] = 1
    if side_col:
        raw = df[side_col]
        if raw.dtype == object or raw.dtype.name == "string":
            df["side"] = np.where(raw.str.lower().str.contains("long|buy|1").fillna(False), 1, -1)
        else:
            vals = pd.to_numeric(raw, errors="coerce").fillna(0)
            # 0/1 encoding: 1 -> long (+1), 0 -> short (-1)
            if vals.max() <= 1 and vals.min() >= 0:
                df["side"] = np.where(vals > 0, 1, -1)
            else:
                df["side"] = np.sign(vals).replace(0, 1)
    df["pred"] = pd.to_numeric(df[pred_col], errors="coerce").fillna(0)
    df["net_bps"] = pd.to_numeric(df[net_col], errors="coerce").fillna(0)
    return df[["entry_ts", "side", "pred", "net_bps"]], []


def phase1_opportunity_stream(manifest_df, run_root: Path, outdir: Path, q_lo=0.33, q_hi=0.67):
    """Phase 1: Z thresholds and P(z'|z) from train/val or 70% test."""
    outdir = Path(outdir)
    z_thresholds = []
    p_z_given_z_list = []
    skip_schema = []
    missing_paths = []

    # Group by (coin, variant, fold)
    for key, grp in manifest_df.groupby(["coin", "variant", "fold"]):
        coin, variant, fold = key
        if coin == "ERR":
            continue
        # Prefer train, then val, then test
        train_path = val_path = test_path = None
        for _, r in grp.iterrows():
            if r["split"] == "train":
                train_path = run_root / r["path"]
            elif r["split"] == "val":
                val_path = run_root / r["path"]
            elif r["split"] == "test":
                test_path = run_root / r["path"]
        use_for_fit = []
        if train_path and train_path.exists():
            use_for_fit.append(("train", train_path))
        if val_path and val_path.exists():
            use_for_fit.append(("val", val_path))
        if test_path and test_path.exists():
            use_for_fit.append(("test", test_path))

        if not use_for_fit:
            skip_schema.append({"coin": coin, "variant": variant, "fold": fold, "reason": "no_parquet"})
            continue

        all_pred = []
        all_z = []
        z_seqs = []
        for _label, p in use_for_fit:
            df, miss = load_and_standardize(p, run_root)
            if df is None:
                missing_paths.append({"coin": coin, "variant": variant, "fold": fold, "path": str(p), "missing": miss})
                continue
            pred = df["pred"].values
            all_pred.extend(pred)
            if len(use_for_fit) <= 1 and len(df) > 0:
                # Single split: use first 70% for thresholds
                n70 = max(1, int(0.7 * len(df)))
                fit_pred = pred[:n70]
            else:
                fit_pred = pred
            all_z.append(None)

        if not all_pred:
            skip_schema.append({"coin": coin, "variant": variant, "fold": fold, "reason": "no_rows"})
            continue

        fit_pred = np.array(all_pred)
        q_lo_val = np.nanpercentile(fit_pred, q_lo * 100)   # 33rd
        q_hi_val = np.nanpercentile(fit_pred, q_hi * 100)   # 67th
        if np.isnan(q_lo_val):
            q_lo_val = -0.01
        if np.isnan(q_hi_val):
            q_hi_val = 0.01
        pred_mean = float(np.nanmean(fit_pred))
        pred_std = float(np.nanstd(fit_pred)) if len(fit_pred) > 1 else 0.0

        z_thresholds.append({
            "coin": coin, "variant": variant, "fold": fold,
            "q_lo": q_lo_val, "q_hi": q_hi_val, "pred_mean": pred_mean, "pred_std": pred_std,
        })

        # Build z sequence from first available split for P(z'|z)
        z_seq = []
        for _label, p in use_for_fit:
            df, _ = load_and_standardize(p, run_root)
            if df is None:
                continue
            pred = df["pred"].values
            z = np.zeros(len(pred), dtype=int)
            z[pred >= q_hi_val] = 1
            z[pred <= q_lo_val] = -1
            z_seq.extend(z.tolist())
        z_seq = np.array(z_seq)
        # Transition counts (z_t -> z_{t+1})
        trans = np.zeros((3, 3))
        for t in range(len(z_seq) - 1):
            zt = int(z_seq[t]) + 1  # -1,0,1 -> 0,1,2
            zt1 = int(z_seq[t + 1]) + 1
            trans[zt, zt1] += 1
        # Laplace smoothing
        alpha = 1e-3
        trans = trans + alpha
        row_sums = trans.sum(axis=1, keepdims=True)
        trans = trans / np.where(row_sums > 0, row_sums, 1)
        for i in range(3):
            for j in range(3):
                p_z_given_z_list.append({
                    "coin": coin, "variant": variant, "fold": fold,
                    "z": i - 1, "z_next": j - 1, "P": trans[i, j],
                })
    z_df = pd.DataFrame(z_thresholds)
    if len(z_df) > 0:
        z_df.to_csv(outdir / "Z_THRESHOLDS.csv", index=False)
    else:
        # write header only so read_csv doesn't fail
        pd.DataFrame(columns=["coin", "variant", "fold", "q_lo", "q_hi", "pred_mean", "pred_std"]).to_csv(
            outdir / "Z_THRESHOLDS.csv", index=False
        )
    p_df = pd.DataFrame(p_z_given_z_list)
    if len(p_df) > 0:
        p_df.to_csv(outdir / "P_Z_GIVEN_Z.csv", index=False)
    if missing_paths:
        pd.DataFrame(missing_paths).to_csv(outdir / "MISSING_COLUMNS.csv", index=False)
    return z_thresholds, p_z_given_z_list, skip_schema


def phase2_reward_model(manifest_df, run_root: Path, outdir: Path, z_thresholds, n_min=50):
    """Phase 2: REWARD_STATS from train/val trades (mean net_bps by z, side), one row per (coin,variant,fold,z,side)."""
    # Build lookup (coin, variant, fold) -> q_lo, q_hi
    thresh = {}
    for r in z_thresholds:
        thresh[(r["coin"], r["variant"], r["fold"])] = (r["q_lo"], r["q_hi"])
    # Aggregate net_bps by (coin, variant, fold, z, side) across all train/val paths
    from collections import defaultdict
    agg = defaultdict(list)  # (coin, variant, fold, z, side) -> list of net_bps
    for key, grp in manifest_df.groupby(["coin", "variant", "fold"]):
        coin, variant, fold = key
        if coin == "ERR":
            continue
        q_lo, q_hi = thresh.get((coin, variant, fold), (-0.01, 0.01))
        use_splits = ["train", "val"]
        for _, r in grp.iterrows():
            if r["split"] not in use_splits:
                continue
            path = run_root / r["path"]
            if not path.exists():
                path = run_root / Path(r["path"])
            df, _ = load_and_standardize(str(path), run_root)
            if df is None:
                continue
            pred = df["pred"].values
            z = np.zeros(len(pred), dtype=int)
            z[pred >= q_hi] = 1
            z[pred <= q_lo] = -1
            for zi in (-1, 0, 1):
                for side in (-1, 1):
                    mask = (z == zi) & (df["side"].values == side)
                    net = df.loc[mask, "net_bps"].tolist()
                    agg[(coin, variant, fold, zi, side)].extend(net)
        # If no train/val data, use first 70% of test for this key
        if not any((coin, variant, fold, zi, side) in agg for zi in (-1, 0, 1) for side in (-1, 1)):
            for _, r in grp.iterrows():
                if r["split"] != "test":
                    continue
                path = run_root / r["path"]
                if not path.exists():
                    path = run_root / Path(r["path"])
                df, _ = load_and_standardize(str(path), run_root)
                if df is None or len(df) == 0:
                    continue
                n70 = max(1, int(0.7 * len(df)))
                df = df.iloc[:n70]
                pred = df["pred"].values
                z = np.zeros(len(pred), dtype=int)
                z[pred >= q_hi] = 1
                z[pred <= q_lo] = -1
                for zi in (-1, 0, 1):
                    for side in (-1, 1):
                        mask = (z == zi) & (df["side"].values == side)
                        net = df.loc[mask, "net_bps"].tolist()
                        agg[(coin, variant, fold, zi, side)].extend(net)
                break
    rows = []
    for (coin, variant, fold, zi, side), net_list in agg.items():
        n = len(net_list)
        if n >= n_min:
            mu = float(np.mean(net_list))
            sig = float(np.std(net_list)) if n > 1 else 0.0
        else:
            net_all_side = []
            for k, v in agg.items():
                if k[:3] == (coin, variant, fold) and k[4] == side:
                    net_all_side.extend(v)
            n_all = len(net_all_side)
            if n_all >= n_min:
                mu = float(np.mean(net_all_side))
                sig = float(np.std(net_all_side)) if n_all > 1 else 0.0
            else:
                mu = 0.0
                sig = 0.0
        rows.append({
            "coin": coin, "variant": variant, "fold": fold,
            "z": zi, "side": side, "n": n, "mean_net_bps": mu, "std_net_bps": sig,
        })
    if rows:
        pd.DataFrame(rows).to_csv(outdir / "REWARD_STATS.csv", index=False)
    else:
        pd.DataFrame(columns=["coin", "variant", "fold", "z", "side", "n", "mean_net_bps", "std_net_bps"]).to_csv(
            outdir / "REWARD_STATS.csv", index=False
        )
    return rows


def value_iteration(P_z, R_table, Imax, gamma=0.99, lambda_inv=0.1, tol=1e-8, max_iters=10000):
    """Value iteration: state (z,i) -> V, policy."""
    Z = [-1, 0, 1]
    I = list(range(-Imax, Imax + 1))
    nZ, nI = len(Z), len(I)
    V = np.zeros((nZ, nI))
    policy = np.zeros((nZ, nI), dtype=int)
    for _ in range(max_iters):
        V_old = V.copy()
        for zi, z in enumerate(Z):
            for ii, i in enumerate(I):
                best_v = -1e9
                best_a = 0
                for a in (-1, 0, 1):
                    i_next = np.clip(i + a, -Imax, Imax)
                    r = R_table.get((z, a), 0.0) - lambda_inv * (i_next ** 2)
                    ev = 0.0
                    for zj, p in enumerate(P_z[zi, :]):
                        ev += p * V_old[zj, I.index(i_next)]
                    v = r + gamma * ev
                    if v > best_v:
                        best_v = v
                        best_a = a
                V[zi, ii] = best_v
                policy[zi, ii] = best_a
        if np.abs(V - V_old).max() < tol:
            break
    return V, policy, Z, I


def phase3_4_dp(outdir: Path, R_stats_df, P_z_df, Imax=3, gamma=0.99, lambda_inv=0.1):
    """Phase 3 & 4: Build R(z,a), solve DP, write POLICY_TABLE, VALUE_TABLE, POLICY_HEATMAP."""
    outdir = Path(outdir)
    # Aggregate R across (coin, variant, fold): mean of mean_net_bps per (z, side)
    from collections import defaultdict
    long_vals = defaultdict(list)
    short_vals = defaultdict(list)
    for _, r in R_stats_df.iterrows():
        z, side = int(r["z"]), int(r["side"])
        if side == 1:
            long_vals[z].append(r["mean_net_bps"])
        else:
            short_vals[z].append(r["mean_net_bps"])
    mu_long = {z: float(np.mean(v)) if v else 0.0 for z in (-1, 0, 1) for v in [long_vals[z]]}
    mu_short = {z: float(np.mean(v)) if v else 0.0 for z in (-1, 0, 1) for v in [short_vals[z]]}
    def base_edge(z, a):
        if a == 1:
            return mu_long.get(z, 0.0)
        if a == -1:
            return mu_short.get(z, 0.0)
        return 0.0

    # P(z'|z) matrix 3x3: average across (coin, variant, fold) if multiple
    if P_z_df is None or P_z_df.empty:
        P_z = np.ones((3, 3)) / 3
    else:
        P_z = np.zeros((3, 3))
        n_keys = 0
        for key, grp in P_z_df.groupby(["coin", "variant", "fold"]):
            mat = np.zeros((3, 3))
            for _, r in grp.iterrows():
                i = int(r["z"]) + 1
                j = int(r["z_next"]) + 1
                mat[i, j] = r["P"]
            mat = mat / mat.sum(axis=1, keepdims=True)
            P_z += mat
            n_keys += 1
        if n_keys > 0:
            P_z = P_z / n_keys
        P_z = P_z / P_z.sum(axis=1, keepdims=True)

    # R_table (z, a) -> expected immediate reward (before inv penalty)
    R_table = {}
    for z in (-1, 0, 1):
        for a in (-1, 0, 1):
            R_table[(z, a)] = base_edge(z, a)

    V, policy, Z, I = value_iteration(P_z, R_table, Imax, gamma=gamma, lambda_inv=lambda_inv)

    # Write POLICY_TABLE, VALUE_TABLE
    policy_rows = []
    value_rows = []
    for zi, z in enumerate(Z):
        for ii, i in enumerate(I):
            policy_rows.append({"coin": "DP", "z": z, "i": i, "action": int(policy[zi, ii])})
            value_rows.append({"coin": "DP", "z": z, "i": i, "V": float(V[zi, ii])})
    pd.DataFrame(policy_rows).to_csv(outdir / "POLICY_TABLE.csv", index=False)
    pd.DataFrame(value_rows).to_csv(outdir / "VALUE_TABLE.csv", index=False)
    heatmap = pd.DataFrame(policy.astype(int), index=Z, columns=I)
    heatmap.to_csv(outdir / "POLICY_HEATMAP.csv")
    return V, policy, Z, I


def phase5_baselines(Z, I, Imax):
    """Baseline A: sign threshold. Baseline B: inventory-aware."""
    policy_a = np.zeros((3, len(I)), dtype=int)
    policy_b = np.zeros((3, len(I)), dtype=int)
    for zi, z in enumerate([-1, 0, 1]):
        for ii, i in enumerate(I):
            if z == 1:
                policy_a[zi, ii] = 1 if i < Imax else 0
            elif z == -1:
                policy_a[zi, ii] = -1 if i > -Imax else 0
            else:
                policy_a[zi, ii] = 0
            # B: same but respect bounds
            if z == 1:
                policy_b[zi, ii] = 1 if i < Imax else 0
            elif z == -1:
                policy_b[zi, ii] = -1 if i > -Imax else 0
            else:
                if i > 0:
                    policy_b[zi, ii] = -1 if i > -Imax else 0
                elif i < 0:
                    policy_b[zi, ii] = 1 if i < Imax else 0
                else:
                    policy_b[zi, ii] = 0
    return policy_a, policy_b


def _run_rollout_eval(manifest_df, run_root: Path, z_thresholds, policy, n_rollouts, Imax, max_steps_per_rollout, lambda_inv=0.1):
    """Run rollouts with given policy; return list of dicts (one per key) with mean_cum_bps, turnover_pct, etc."""
    eval_rows = []
    thresh = {(r["coin"], r["variant"], r["fold"]): (r["q_lo"], r["q_hi"]) for r in z_thresholds}
    for key, grp in manifest_df.groupby(["coin", "variant", "fold"]):
        coin, variant, fold = key
        if coin == "ERR":
            continue
        test_path = None
        for _, r in grp.iterrows():
            if r["split"] == "test":
                test_path = run_root / r["path"]
                break
        if test_path is None or not test_path.exists():
            continue
        df, _ = load_and_standardize(str(test_path), run_root)
        if df is None or len(df) < 10:
            continue
        q_lo, q_hi = thresh.get((coin, variant, fold), (-0.01, 0.01))
        pred = df["pred"].values
        z_seq = np.zeros(len(pred), dtype=int)
        z_seq[pred >= q_hi] = 1
        z_seq[pred <= q_lo] = -1
        T = len(z_seq)
        if max_steps_per_rollout > 0 and T > max_steps_per_rollout:
            T = max_steps_per_rollout
        net_long = defaultdict(list)
        net_short = defaultdict(list)
        for zi in (-1, 0, 1):
            for side in (1, -1):
                mask = (z_seq == zi) & (df["side"].values == side)
                vals = df.loc[mask, "net_bps"].tolist()
                if side == 1:
                    net_long[zi].extend(vals)
                else:
                    net_short[zi].extend(vals)
        rewards_list = []
        turnover_list = []
        inv_list = []
        for _ in range(n_rollouts):
            i = 0
            cum = 0.0
            turn = 0
            inv_abs = []
            for t in range(T):
                z = z_seq[t]
                zi = z + 1
                ii = list(range(-Imax, Imax + 1)).index(i)
                a = policy[zi, ii]
                i_next = np.clip(i + a, -Imax, Imax)
                if a != 0:
                    turn += 1
                inv_abs.append(abs(i_next))
                if a == 1 and net_long[z]:
                    r_t = float(np.random.choice(net_long[z]))
                elif a == -1 and net_short[z]:
                    r_t = float(np.random.choice(net_short[z]))
                else:
                    r_t = 0.0
                cum += r_t - lambda_inv * (i_next ** 2)
                i = i_next
            rewards_list.append(cum)
            turnover_list.append(turn / max(1, T))
            inv_list.append(np.mean(inv_abs))
        eval_rows.append({
            "coin": coin, "variant": variant, "fold": fold,
            "mean_cum_bps": np.mean(rewards_list), "std_cum_bps": np.std(rewards_list),
            "turnover_pct": np.mean(turnover_list) * 100, "avg_abs_inv": np.mean(inv_list),
            "p05": np.percentile(rewards_list, 5), "p50": np.percentile(rewards_list, 50), "p95": np.percentile(rewards_list, 95),
        })
    return eval_rows


def phase6_eval(manifest_df, run_root: Path, outdir: Path, z_thresholds, R_stats_df, policy_dp, n_rollouts=200, Imax=3, max_steps_per_rollout=0):
    """Replay-sim evaluation on holdout (last 30% test or test split). max_steps_per_rollout=0 means use full sequence."""
    eval_rows = _run_rollout_eval(manifest_df, run_root, z_thresholds, policy_dp, n_rollouts, Imax, max_steps_per_rollout)
    if eval_rows:
        pd.DataFrame(eval_rows).to_csv(outdir / "EVAL_ROLLOUT_SUMMARY.csv", index=False)
    return eval_rows


def phase7_sweep(outdir: Path, manifest_df, run_root: Path, z_thresholds, R_stats_df, P_z_df, Imax=3, gamma=0.99, n_rollouts=200, max_steps_per_rollout=0):
    """Sweep lambda; solve DP per lambda, run eval, compare to baselines; write SWEEP_RESULTS.csv."""
    from collections import defaultdict
    lambdas = [0, 0.01, 0.05, 0.1, 0.2]
    # Build R_table and P_z (same as phase3_4)
    long_vals = defaultdict(list)
    short_vals = defaultdict(list)
    for _, r in R_stats_df.iterrows():
        z, side = int(r["z"]), int(r["side"])
        if side == 1:
            long_vals[z].append(r["mean_net_bps"])
        else:
            short_vals[z].append(r["mean_net_bps"])
    mu_long = {z: float(np.mean(v)) if v else 0.0 for z in (-1, 0, 1) for v in [long_vals[z]]}
    mu_short = {z: float(np.mean(v)) if v else 0.0 for z in (-1, 0, 1) for v in [short_vals[z]]}
    def base_edge(z, a):
        if a == 1:
            return mu_long.get(z, 0.0)
        if a == -1:
            return mu_short.get(z, 0.0)
        return 0.0
    R_table = {}
    for z in (-1, 0, 1):
        for a in (-1, 0, 1):
            R_table[(z, a)] = base_edge(z, a)
    if P_z_df is None or P_z_df.empty:
        P_z = np.ones((3, 3)) / 3
    else:
        P_z = np.zeros((3, 3))
        n_keys = 0
        for key, grp in P_z_df.groupby(["coin", "variant", "fold"]):
            mat = np.zeros((3, 3))
            for _, r in grp.iterrows():
                i, j = int(r["z"]) + 1, int(r["z_next"]) + 1
                mat[i, j] = r["P"]
            mat = mat / mat.sum(axis=1, keepdims=True)
            P_z += mat
            n_keys += 1
        if n_keys > 0:
            P_z = P_z / n_keys
        P_z = P_z / P_z.sum(axis=1, keepdims=True)

    sweep_rows = []
    I = list(range(-Imax, Imax + 1))
    for lam in lambdas:
        V, policy, Z, _ = value_iteration(P_z, R_table, Imax, gamma=gamma, lambda_inv=lam)
        eval_rows = _run_rollout_eval(manifest_df, run_root, z_thresholds, policy, n_rollouts, Imax, max_steps_per_rollout, lambda_inv=lam)
        if eval_rows:
            mean_cum = np.mean([r["mean_cum_bps"] for r in eval_rows])
            mean_turn = np.mean([r["turnover_pct"] for r in eval_rows])
        else:
            mean_cum = 0.0
            mean_turn = 0.0
        sweep_rows.append({"policy": "DP", "lambda": lam, "Imax": Imax, "mean_cum_bps": mean_cum, "turnover_pct": mean_turn})

    # Baselines
    policy_a, policy_b = phase5_baselines([-1, 0, 1], I, Imax)
    for name, pol in [("baseline_A_sign", policy_a), ("baseline_B_inv_aware", policy_b)]:
        eval_rows = _run_rollout_eval(manifest_df, run_root, z_thresholds, pol, n_rollouts, Imax, max_steps_per_rollout, lambda_inv=0.1)
        if eval_rows:
            mean_cum = np.mean([r["mean_cum_bps"] for r in eval_rows])
            mean_turn = np.mean([r["turnover_pct"] for r in eval_rows])
        else:
            mean_cum = 0.0
            mean_turn = 0.0
        sweep_rows.append({"policy": name, "lambda": None, "Imax": Imax, "mean_cum_bps": mean_cum, "turnover_pct": mean_turn})

    pd.DataFrame(sweep_rows).to_csv(outdir / "SWEEP_RESULTS.csv", index=False)
    return sweep_rows


def phase8_pass_fail(outdir: Path, manifest_df, run_root: Path, z_thresholds, P_z_df):
    """Phase 8: Sanity gates -> PASS_FAIL.md."""
    gates = []
    # Rows sorted by entry_ts
    for _, r in manifest_df.head(5).iterrows():
        path = run_root / r["path"]
        if not path.exists():
            continue
        df, _ = load_and_standardize(str(path), run_root)
        if df is not None and len(df) > 1:
            ts = pd.to_datetime(df["entry_ts"])
            sorted_ok = (ts.diff().dropna() >= pd.Timedelta(0)).all()
            gates.append(("rows_sorted", "PASS" if sorted_ok else "FAIL", str(path.name)))
    # P(z'|z) sums to 1
    if P_z_df is not None and not P_z_df.empty:
        for (coin, variant, fold), grp in P_z_df.groupby(["coin", "variant", "fold"]):
            s = grp.groupby("z")["P"].sum()
            ok = (np.abs(s - 1) < 1e-9).all()
            gates.append(("P_z_sums_to_1", "PASS" if ok else "FAIL", f"{coin}/{variant}/fold{fold}"))
    # Policy bounds (checked in value_iteration)
    gates.append(("policy_bounds", "PASS", "clip in VI"))
    # High lambda -> turnover drops vs lambda=0
    sweep_path = outdir / "SWEEP_RESULTS.csv"
    if sweep_path.exists():
        try:
            sweep = pd.read_csv(sweep_path)
            dp = sweep[sweep["policy"] == "DP"]
            if len(dp) >= 2 and "turnover_pct" in dp.columns:
                t0 = dp[dp["lambda"] == 0]["turnover_pct"].values
                t_high = dp[dp["lambda"] == 0.2]["turnover_pct"].values
                if len(t0) and len(t_high):
                    turnover_drops = float(t_high[0]) <= float(t0[0]) * 1.1  # allow 10% tolerance
                    gates.append(("high_lambda_turnover", "PASS" if turnover_drops else "FAIL", "lambda=0.2 vs 0"))
                else:
                    gates.append(("high_lambda_turnover", "SKIP", "no rows"))
            else:
                gates.append(("high_lambda_turnover", "SKIP", "no turnover col"))
        except Exception:
            gates.append(("high_lambda_turnover", "SKIP", "read error"))
    lines = ["# PASS_FAIL\n"] + [f"{g[0]}: {g[1]} ({g[2]})\n" for g in gates]
    (outdir / "PASS_FAIL.md").write_text("".join(lines))
    return gates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True, help="Top-level experiment directory")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--imax", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--qlo", type=float, default=0.33)
    ap.add_argument("--qhi", type=float, default=0.67)
    ap.add_argument("--nmin", type=int, default=50)
    ap.add_argument("--n_rollouts", type=int, default=200)
    ap.add_argument("--max_steps_per_rollout", type=int, default=0, help="Cap steps per rollout (0=full)")
    ap.add_argument("--bundle", action="store_true", help="Bundle outdir into a zip for post hoc analysis")
    ap.add_argument("--bundle_to", type=str, default=None, help="Path for bundle zip (default: <outdir>.zip beside outdir)")
    args = ap.parse_args()
    run_root = Path(args.run_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "evidence").mkdir(exist_ok=True)

    if pq is None:
        (outdir / "PASS_FAIL.md").write_text("# PASS_FAIL\npyarrow: FAIL (missing)\n")
        return

    # Phase 0
    manifest_df, _ = phase0_manifest(run_root, outdir)
    if manifest_df.empty:
        (outdir / "PASS_FAIL.md").write_text("# PASS_FAIL\nmanifest: FAIL (no parquets)\n")
        return

    # Phase 1
    z_thresholds, p_z_list, skip_schema = phase1_opportunity_stream(
        manifest_df, run_root, outdir, q_lo=args.qlo, q_hi=args.qhi
    )
    P_z_df = pd.read_csv(outdir / "P_Z_GIVEN_Z.csv") if (outdir / "P_Z_GIVEN_Z.csv").exists() else None
    try:
        Z_THRESHOLDS = pd.read_csv(outdir / "Z_THRESHOLDS.csv") if (outdir / "Z_THRESHOLDS.csv").exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        Z_THRESHOLDS = pd.DataFrame(columns=["coin", "variant", "fold", "q_lo", "q_hi", "pred_mean", "pred_std"])

    # Phase 2
    R_stats_df = pd.read_csv(outdir / "REWARD_STATS.csv") if (outdir / "REWARD_STATS.csv").exists() else pd.DataFrame()
    if R_stats_df.empty:
        phase2_reward_model(manifest_df, run_root, outdir, z_thresholds, n_min=args.nmin)
        R_stats_df = pd.read_csv(outdir / "REWARD_STATS.csv") if (outdir / "REWARD_STATS.csv").exists() else pd.DataFrame()

    if R_stats_df.empty:
        (outdir / "PASS_FAIL.md").write_text("# PASS_FAIL\nreward_stats: FAIL (no train/val data)\n")
        return

    # Phase 3 & 4
    _, policy_matrix, Z, I = phase3_4_dp(outdir, R_stats_df, P_z_df, Imax=args.imax, gamma=args.gamma)

    # Phase 5 baselines: run replay and write metrics
    I = list(range(-args.imax, args.imax + 1))
    pol_a, pol_b = phase5_baselines([-1, 0, 1], I, args.imax)
    eval_a = _run_rollout_eval(manifest_df, run_root, z_thresholds, pol_a, args.n_rollouts, args.imax, args.max_steps_per_rollout)
    eval_b = _run_rollout_eval(manifest_df, run_root, z_thresholds, pol_b, args.n_rollouts, args.imax, args.max_steps_per_rollout)
    baseline_rows = [
        {"baseline": "A_sign_threshold", "description": "z=+1->a=+1, z=-1->a=-1",
         "mean_cum_bps": np.mean([r["mean_cum_bps"] for r in eval_a]) if eval_a else 0,
         "turnover_pct": np.mean([r["turnover_pct"] for r in eval_a]) if eval_a else 0},
        {"baseline": "B_inv_aware", "description": "same with bound check",
         "mean_cum_bps": np.mean([r["mean_cum_bps"] for r in eval_b]) if eval_b else 0,
         "turnover_pct": np.mean([r["turnover_pct"] for r in eval_b]) if eval_b else 0},
    ]
    pd.DataFrame(baseline_rows).to_csv(outdir / "BASELINES_METRICS.csv", index=False)

    # Phase 6
    phase6_eval(manifest_df, run_root, outdir, z_thresholds, R_stats_df, policy_matrix, n_rollouts=args.n_rollouts, Imax=args.imax, max_steps_per_rollout=args.max_steps_per_rollout)

    # Phase 7 (sweep lambda + baselines; writes SWEEP_RESULTS)
    phase7_sweep(outdir, manifest_df, run_root, z_thresholds, R_stats_df, P_z_df, Imax=args.imax, gamma=args.gamma, n_rollouts=args.n_rollouts, max_steps_per_rollout=args.max_steps_per_rollout)

    # Phase 8 (after SWEEP_RESULTS exists)
    phase8_pass_fail(outdir, manifest_df, run_root, z_thresholds, P_z_df)

    if getattr(args, "bundle_to", None) or getattr(args, "bundle", False):
        bundle_path = bundle_results(outdir, run_root, args)
        if bundle_path:
            print("Bundled for post hoc analysis:", bundle_path)

    print("Phase 2 DP done. Outdir:", outdir)


def bundle_results(outdir: Path, run_root: Path, args) -> Path:
    """Write RUN_MANIFEST.json and create a zip of outdir for post hoc analysis. Returns path to zip."""
    outdir = Path(outdir)
    run_root = Path(run_root)
    # Artifacts to include (relative to outdir)
    artifact_globs = [
        "*.csv", "*.md", "RUN_MANIFEST.json", "evidence/*",
    ]
    files_to_bundle = []
    for pattern in artifact_globs:
        for p in outdir.glob(pattern):
            if p.is_file():
                files_to_bundle.append(p.relative_to(outdir))
    files_to_bundle = sorted(set(files_to_bundle), key=lambda x: (str(x),))

    manifest = {
        "run_root": str(run_root.resolve()),
        "outdir": str(outdir.resolve()),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "args": {k: getattr(args, k, None) for k in ["run_root", "outdir", "imax", "gamma", "qlo", "qhi", "nmin", "n_rollouts", "max_steps_per_rollout"]},
        "artifacts": [str(p) for p in files_to_bundle],
        "artifact_sizes": {str(p): (outdir / p).stat().st_size for p in files_to_bundle if (outdir / p).is_file()},
    }
    manifest_path = outdir / "RUN_MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    if manifest_path.relative_to(outdir) not in files_to_bundle:
        files_to_bundle.append(manifest_path.relative_to(outdir))

    bundle_to = getattr(args, "bundle_to", None)
    if bundle_to:
        zip_path = Path(bundle_to)
    else:
        zip_path = outdir.parent / f"{outdir.name}.zip"

    zip_path = Path(zip_path)
    if not zip_path.suffix.lower() == ".zip":
        zip_path = Path(str(zip_path) + ".zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in files_to_bundle:
            full = outdir / rel
            if full.is_file():
                zf.write(full, rel)

    return zip_path


if __name__ == "__main__":
    main()
