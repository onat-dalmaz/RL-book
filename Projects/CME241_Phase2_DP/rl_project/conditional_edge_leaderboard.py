#!/usr/bin/env python3
"""
Conditional edge leaderboard for Step1 coin pool.
Screening pass only: fit z thresholds (no leakage), compute conditional edges and reliability; no DP.
Output: CONDITIONAL_EDGE_LEADERBOARD.csv, CONDITIONAL_EDGE_RANKING.md
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_io

COINS_ORDERED = [
    "SNX", "SUPER", "NEAR", "XAI", "RENDER", "WIF", "PUMP", "AXS",
    "TNSR", "LPT", "UNI", "ANKR", "LINK", "AAVE", "UMA", "MINA", "SAND",
]
Q_LO, Q_HI = 0.33, 0.67
MIN_BUCKET_N_OK = 500
JS_SHIFT_THRESH = 0.2
EPS = 1e-12


def _empty_row(coin: str, flags: str, score: float = np.nan) -> dict:
    """One row with all leaderboard columns for skipped/failed coins."""
    return {
        "coin": coin,
        "fit_split": "",
        "n_fit": np.nan,
        "pred_std_fit": np.nan,
        "val_n": np.nan, "val_mean_y": np.nan, "val_std_y": np.nan,
        "val_n_neg1": np.nan, "val_n_0": np.nan, "val_n_pos1": np.nan,
        "val_mean_y_neg1": np.nan, "val_mean_y_0": np.nan, "val_mean_y_pos1": np.nan,
        "val_long_edge": np.nan, "val_short_edge": np.nan, "val_breakeven_best": np.nan,
        "test_n": np.nan, "test_mean_y": np.nan, "test_std_y": np.nan,
        "test_n_neg1": np.nan, "test_n_0": np.nan, "test_n_pos1": np.nan,
        "test_mean_y_neg1": np.nan, "test_mean_y_0": np.nan, "test_mean_y_pos1": np.nan,
        "test_long_edge": np.nan, "test_short_edge": np.nan, "test_breakeven_best": np.nan,
        "js_val_test": np.nan, "min_bucket_n_val": np.nan, "min_bucket_n_test": np.nan,
        "passes_fee_1bps": False, "passes_fee_2bps": False, "passes_fee_4bps": False,
        "score": score,
        "flags": flags,
    }


def _fit_split_priority(dfs_by_split: dict) -> tuple:
    """Return (fit_split_name, fit_df). Priority: train > val > test70."""
    if dfs_by_split.get("train") is not None and len(dfs_by_split["train"]) > 0:
        return "train", dfs_by_split["train"]
    if dfs_by_split.get("val") is not None and len(dfs_by_split["val"]) > 0:
        return "val", dfs_by_split["val"]
    test_df = dfs_by_split.get("test")
    if test_df is not None and len(test_df) > 0:
        n70 = max(1, int(0.7 * len(test_df)))
        return "test70", test_df.iloc[:n70]
    return None, None


def _edge_metrics(df: pd.DataFrame, q_lo: float, q_hi: float) -> dict:
    """Compute mean_y, std_y, counts by z, means by z, long_edge, short_edge, breakeven_best, zero_frac_y."""
    pred = df["pred"].values
    y = df["y"].values
    z = np.zeros(len(pred), dtype=int)
    z[pred >= q_hi] = 1
    z[pred <= q_lo] = -1
    n = len(y)
    mean_y = float(np.nanmean(y))
    std_y = float(np.nanstd(y)) if n > 1 else 0.0
    zero_frac_y = float(np.sum(np.nan_to_num(y, nan=0.0) == 0) / n) if n else 0.0

    n_neg1 = int((z == -1).sum())
    n_0 = int((z == 0).sum())
    n_pos1 = int((z == 1).sum())
    mean_y_neg1 = float(np.nanmean(y[z == -1])) if n_neg1 else np.nan
    mean_y_0 = float(np.nanmean(y[z == 0])) if n_0 else np.nan
    mean_y_pos1 = float(np.nanmean(y[z == 1])) if n_pos1 else np.nan

    long_edge = mean_y_pos1 if not np.isnan(mean_y_pos1) else 0.0
    short_edge = -mean_y_neg1 if not np.isnan(mean_y_neg1) else 0.0
    breakeven_best = float(max(long_edge, short_edge))
    breakeven_long = float(long_edge)
    breakeven_short = float(short_edge)

    return {
        "n": n,
        "mean_y": mean_y,
        "std_y": std_y,
        "zero_frac_y": zero_frac_y,
        "n_neg1": n_neg1,
        "n_0": n_0,
        "n_pos1": n_pos1,
        "mean_y_neg1": mean_y_neg1,
        "mean_y_0": mean_y_0,
        "mean_y_pos1": mean_y_pos1,
        "long_edge": long_edge,
        "short_edge": short_edge,
        "breakeven_best": breakeven_best,
        "breakeven_long": breakeven_long,
        "breakeven_short": breakeven_short,
        "min_bucket_n": min(n_neg1, n_pos1),
    }


def _js_divergence(p_vals: np.ndarray, q_vals: np.ndarray, eps: float = EPS) -> float:
    """JS(p, q) with epsilon smoothing."""
    p = np.clip(np.asarray(p_vals, dtype=float) + eps, 0, 1)
    q = np.clip(np.asarray(q_vals, dtype=float) + eps, 0, 1)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    js = float(0.5 * (np.sum(p * (np.log(p) - np.log(m))) + np.sum(q * (np.log(q) - np.log(m)))))
    return js


def run_coin(coin: str, run_root: Path, fit_val_test_combined: bool = False) -> dict:
    """
    Load val/test for coin, fit thresholds, compute edges and reliability.
    If fit_val_test_combined: fit q_lo/q_hi on val+test combined (no holdout). Else: val only / train / test70 (no leakage).
    Returns one row dict for leaderboard or with flag SKIP_MISSING_SCHEMA.
    """
    run_root = Path(run_root)
    # Discover val/test paths (same pattern as Phase2: cv/COIN/fold_0/trades_val.parquet)
    val_path = run_root / "cv" / coin / "fold_0" / "trades_val.parquet"
    test_path = run_root / "cv" / coin / "fold_0" / "trades_test.parquet"
    if not val_path.exists():
        for p in run_root.rglob("trades_val.parquet"):
            val_path = p
            break
    if not test_path.exists():
        for p in run_root.rglob("trades_test.parquet"):
            test_path = p
            break
    if not val_path.exists() or not test_path.exists():
        return _empty_row(coin, "SKIP_MISSING_SCHEMA", np.nan)

    dfs_by_split = {}
    for label, path in [("val", val_path), ("test", test_path)]:
        df, err = data_io.load_and_canonicalize(path, run_root, "step1", allow_no_pred_s7=False)
        if err or df is None or len(df) == 0:
            return _empty_row(coin, "SKIP_MISSING_SCHEMA", np.nan)
        if "pred" not in df.columns or "y" not in df.columns:
            return _empty_row(coin, "SKIP_MISSING_SCHEMA", np.nan)
        dfs_by_split[label] = df

    if fit_val_test_combined:
        fit_split_name = "val_and_test"
        fit_df = pd.concat([dfs_by_split["val"], dfs_by_split["test"]], ignore_index=True)
    else:
        fit_split_name, fit_df = _fit_split_priority(dfs_by_split)
    if fit_df is None or len(fit_df) == 0:
        return _empty_row(coin, "SKIP_NO_FIT_SPLIT", np.nan)

    fit_pred = fit_df["pred"].values
    n_fit = len(fit_pred)
    pred_std_fit = float(np.nanstd(fit_pred))
    if pred_std_fit <= 0 or not np.isfinite(pred_std_fit):
        row = _empty_row(coin, "FAIL_PRED_STD", -np.inf)
        row["fit_split"] = fit_split_name
        row["n_fit"] = n_fit
        row["pred_std_fit"] = pred_std_fit
        return row

    q_lo_val = float(np.nanpercentile(fit_pred, Q_LO * 100))
    q_hi_val = float(np.nanpercentile(fit_pred, Q_HI * 100))
    if not np.isfinite(q_lo_val):
        q_lo_val = -0.01
    if not np.isfinite(q_hi_val):
        q_hi_val = 0.01
    if q_hi_val <= q_lo_val:
        q_hi_val = q_lo_val + 0.02

    val_met = _edge_metrics(dfs_by_split["val"], q_lo_val, q_hi_val)
    test_met = _edge_metrics(dfs_by_split["test"], q_lo_val, q_hi_val)

    # Z distribution for JS (val vs test)
    pred_val = dfs_by_split["val"]["pred"].values
    pred_test = dfs_by_split["test"]["pred"].values
    z_val = np.zeros(len(pred_val), dtype=int)
    z_val[pred_val >= q_hi_val] = 1
    z_val[pred_val <= q_lo_val] = -1
    z_test = np.zeros(len(pred_test), dtype=int)
    z_test[pred_test >= q_hi_val] = 1
    z_test[pred_test <= q_lo_val] = -1
    p_vals = np.array([(z_val == -1).sum(), (z_val == 0).sum(), (z_val == 1).sum()], dtype=float) / max(1, len(z_val))
    q_vals = np.array([(z_test == -1).sum(), (z_test == 0).sum(), (z_test == 1).sum()], dtype=float) / max(1, len(z_test))
    js_val_test = _js_divergence(p_vals, q_vals, EPS)

    min_bucket_n_val = val_met["min_bucket_n"]
    min_bucket_n_test = test_met["min_bucket_n"]
    breakeven_best_val = val_met["breakeven_best"]

    # Decision split for effective_sample_flag: val when fit on val, else test for OOS
    decision_min_n = min_bucket_n_val if fit_split_name == "val" else min_bucket_n_test
    effective_ok = decision_min_n >= MIN_BUCKET_N_OK
    shift_flag = "SHIFT" if js_val_test > JS_SHIFT_THRESH else "OK"

    score = float(breakeven_best_val)
    if min_bucket_n_val < MIN_BUCKET_N_OK:
        score *= 0.5
    if js_val_test > JS_SHIFT_THRESH:
        score *= 0.7
    if not np.isfinite(score):
        score = -np.inf

    passes_fee_1bps = breakeven_best_val > 1.0
    passes_fee_2bps = breakeven_best_val > 2.0
    passes_fee_4bps = breakeven_best_val > 4.0

    flags_list = []
    if not effective_ok:
        flags_list.append("LOW_N")
    if js_val_test > JS_SHIFT_THRESH:
        flags_list.append("SHIFT")
    flags = ",".join(flags_list) if flags_list else "OK"

    row = {
        "coin": coin,
        "fit_split": fit_split_name,
        "n_fit": n_fit,
        "pred_std_fit": pred_std_fit,
        "val_n": val_met["n"],
        "val_mean_y": val_met["mean_y"],
        "val_std_y": val_met["std_y"],
        "val_n_neg1": val_met["n_neg1"],
        "val_n_0": val_met["n_0"],
        "val_n_pos1": val_met["n_pos1"],
        "val_mean_y_neg1": val_met["mean_y_neg1"],
        "val_mean_y_0": val_met["mean_y_0"],
        "val_mean_y_pos1": val_met["mean_y_pos1"],
        "val_long_edge": val_met["long_edge"],
        "val_short_edge": val_met["short_edge"],
        "val_breakeven_best": val_met["breakeven_best"],
        "test_n": test_met["n"],
        "test_mean_y": test_met["mean_y"],
        "test_std_y": test_met["std_y"],
        "test_n_neg1": test_met["n_neg1"],
        "test_n_0": test_met["n_0"],
        "test_n_pos1": test_met["n_pos1"],
        "test_mean_y_neg1": test_met["mean_y_neg1"],
        "test_mean_y_0": test_met["mean_y_0"],
        "test_mean_y_pos1": test_met["mean_y_pos1"],
        "test_long_edge": test_met["long_edge"],
        "test_short_edge": test_met["short_edge"],
        "test_breakeven_best": test_met["breakeven_best"],
        "js_val_test": js_val_test,
        "min_bucket_n_val": min_bucket_n_val,
        "min_bucket_n_test": min_bucket_n_test,
        "passes_fee_1bps": passes_fee_1bps,
        "passes_fee_2bps": passes_fee_2bps,
        "passes_fee_4bps": passes_fee_4bps,
        "score": score,
        "flags": flags,
    }
    return row


def main():
    ap = argparse.ArgumentParser(description="Conditional edge leaderboard (Step1 coin pool)")
    ap.add_argument("--step1_root", required=True, help="Path to COINS dir (e.g. .../COINS)")
    ap.add_argument("--outdir", default=None, help="Output dir; default results/conditional_edge_scan_<timestamp>")
    ap.add_argument("--coins", default=None, help="Comma-separated coins; default use built-in list")
    ap.add_argument("--fit_val_test_combined", action="store_true", help="Fit z thresholds on val+test combined (no holdout)")
    args = ap.parse_args()

    step1_root = Path(args.step1_root)
    if not step1_root.exists():
        print("step1_root not found:", step1_root)
        sys.exit(1)

    coins = [c.strip() for c in args.coins.split(",")] if args.coins else COINS_ORDERED
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        outdir = Path("/home/ubuntu/onat/results") / f"conditional_edge_scan_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for coin in coins:
        run_root = step1_root / coin
        row = run_coin(coin, run_root, fit_val_test_combined=args.fit_val_test_combined)
        rows.append(row)

    df = pd.DataFrame(rows)
    # Sort by score descending; -inf and nan at end
    df["_sort_score"] = df["score"].replace(-np.inf, np.nan)
    df = df.sort_values("_sort_score", ascending=False, na_position="last").drop(columns=["_sort_score"])
    csv_path = outdir / "CONDITIONAL_EDGE_LEADERBOARD.csv"
    df.to_csv(csv_path, index=False)
    print("Wrote", csv_path)

    # Markdown summary
    md_lines = [
        "# Conditional Edge Ranking (Step1)",
        "",
        "Screening pass: conditional edge + reliability. No DP.",
        "",
        "## Top 5 by score",
        "",
    ]
    valid = df[df["score"].notna() & np.isfinite(df["score"]) & (df["score"] > -np.inf)]
    top5 = valid.head(5)
    if not top5.empty:
        md_lines.append("| coin | score | val_breakeven_best | passes_fee_1bps | passes_fee_2bps | passes_fee_4bps | flags |")
        md_lines.append("|------|-------|--------------------|----------------|----------------|-----------------|-------|")
        for _, r in top5.iterrows():
            md_lines.append(
                f"| {r['coin']} | {r['score']:.3f} | {r['val_breakeven_best']:.3f} | {r['passes_fee_1bps']} | {r['passes_fee_2bps']} | {r['passes_fee_4bps']} | {r['flags']} |"
            )
    else:
        md_lines.append("(No coins with finite score.)")
    md_lines.extend(["", "## Promising (passes 1 bps + OK sample)", ""])
    promising = valid[(valid["passes_fee_1bps"] == True) & (valid["flags"] == "OK")]
    if not promising.empty:
        for _, r in promising.iterrows():
            md_lines.append(f"- **{r['coin']}**: score={r['score']:.3f}, val_breakeven_best={r['val_breakeven_best']:.3f} bps")
    else:
        md_lines.append("(None.)")
    md_lines.extend(["", "## Requires ultra-low fees (breakeven 0–1 bps)", ""])
    ultra = valid[(valid["val_breakeven_best"] > 0) & (valid["val_breakeven_best"] <= 1.0)]
    if not ultra.empty:
        for _, r in ultra.iterrows():
            md_lines.append(f"- **{r['coin']}**: val_breakeven_best={r['val_breakeven_best']:.3f} bps, score={r['score']:.3f}")
    else:
        md_lines.append("(None.)")
    md_lines.extend(["", "## Unreliable (LOW_N or missing schema)", ""])
    unreliable = df[df["flags"].str.contains("LOW_N|SKIP_MISSING_SCHEMA|FAIL", na=False)]
    if not unreliable.empty:
        for _, r in unreliable.iterrows():
            md_lines.append(f"- **{r['coin']}**: {r['flags']}")
    else:
        md_lines.append("(None.)")
    md_lines.extend(["", "## Phase-2 candidate summary", ""])
    fee1 = valid[valid["passes_fee_1bps"] == True]
    fee2 = valid[valid["passes_fee_2bps"] == True]
    fee4 = valid[valid["passes_fee_4bps"] == True]
    md_lines.append(f"- **Fee 1 bps (maker-like):** {list(fee1['coin'])}")
    md_lines.append(f"- **Fee 2 bps (optimistic):** {list(fee2['coin'])}")
    md_lines.append(f"- **Fee 4 bps (taker):** {list(fee4['coin'])}")
    md_path = outdir / "CONDITIONAL_EDGE_RANKING.md"
    Path(md_path).write_text("\n".join(md_lines), encoding="utf-8")
    print("Wrote", md_path)

    n_finite = valid.shape[0]
    print(f"Coins with finite score: {n_finite} / {len(rows)}")
    print("Done. Outdir:", outdir)


if __name__ == "__main__":
    main()
