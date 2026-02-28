#!/usr/bin/env python3
"""
Run Phase 2 DP pipeline sequentially for multiple coins (fee 2 bps strong set, fee 1 bps broader set).
Uses --z_fit_mode val_and_test and --reward_fit_mode val_and_test (leakage screening mode).
Produces per-coin bundles and MULTICOIN_SUMMARY.csv + MULTICOIN_STATUS.csv.
"""

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Default coin sets from spec
COINS_FEE2 = ["NEAR", "RENDER", "AXS", "SNX"]
COINS_FEE1 = ["SUPER", "PUMP", "LPT", "WIF", "TNSR", "UMA"]
COINS_EXTRA = ["XAI"]  # optional

LAMBDA_01 = 0.1
STEP1_ROOT_DEFAULT = "experiments/step1_signal_feb13_feb24_train6_20260224_191001/COINS"


def _ts():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def run_one(
    coin: str,
    fee_bps: float,
    outdir: Path,
    step1_root: Path,
    extra_args: list,
) -> tuple:
    """Run phase2_dp_from_parquets for one coin. Returns (success: bool, log_lines: list)."""
    run_root = step1_root / coin
    if not run_root.exists():
        return False, [f"run_root not found: {run_root}"]
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "rl_project.phase2_dp_from_parquets",
        "--run_root",
        str(run_root),
        "--mode",
        "step1",
        "--outdir",
        str(outdir),
        "--fee_bps",
        str(fee_bps),
        "--z_fit_mode",
        "val_and_test",
        "--reward_fit_mode",
        "val_and_test",
        *extra_args,
    ]
    log_lines = [f"Command: {' '.join(cmd)}"]
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            timeout=3600 * 2,
        )
        log_lines.append(f"Return code: {result.returncode}")
        if result.stdout:
            log_lines.append("--- stdout ---")
            log_lines.append(result.stdout[-8000:] if len(result.stdout) > 8000 else result.stdout)
        if result.stderr:
            log_lines.append("--- stderr ---")
            log_lines.append(result.stderr[-8000:] if len(result.stderr) > 8000 else result.stderr)
        return result.returncode == 0, log_lines
    except subprocess.TimeoutExpired as e:
        log_lines.append(f"Timeout: {e}")
        return False, log_lines
    except Exception as e:
        log_lines.append(f"Exception: {e}")
        return False, log_lines


def _parse_reward_sanity_fit_row(outdir: Path) -> dict:
    """From REWARD_SANITY.csv take the fit-split row (val_and_test or first row). Return breakeven_best."""
    p = outdir / "REWARD_SANITY.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if df.empty:
        return {}
    # Prefer row with split val_and_test, else first row
    fit = df[df["split"] == "val_and_test"] if "split" in df.columns else df.iloc[:1]
    if fit.empty:
        fit = df.iloc[:1]
    row = fit.iloc[0]
    mean_pos1 = row.get("mean_y_z_pos1", float("nan"))
    mean_neg1 = row.get("mean_y_z_neg1", float("nan"))
    if pd.isna(mean_pos1):
        mean_pos1 = 0.0
    if pd.isna(mean_neg1):
        mean_neg1 = 0.0
    breakeven_best = max(float(mean_pos1), float(-mean_neg1)) if (mean_pos1 is not None and mean_neg1 is not None) else float("nan")
    return {"breakeven_best_fit": breakeven_best}


def _parse_eval_summary(outdir: Path) -> dict:
    """From EVAL_ROLLOUT_SUMMARY.csv get DP and baseline A/B at lambda=0.1."""
    p = outdir / "EVAL_ROLLOUT_SUMMARY.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    lam = LAMBDA_01
    out = {}
    for policy_label, policy_name in [
        ("dp", "DP"),
        ("baselineA", "baseline_A_sign"),
        ("baselineB", "baseline_B_inv_aware"),
    ]:
        sub = df[(df["lambda"] == lam) & (df["policy_name"] == policy_name)]
        if sub.empty:
            sub = df[(df["lambda"].astype(float) == lam) & (df["policy_name"] == policy_name)]
        if not sub.empty:
            r = sub.iloc[0]
            out[f"{policy_label}_mean_bps_per_step_l01"] = r.get("mean_bps_per_step", float("nan"))
            out[f"{policy_label}_turnover_l01"] = r.get("turnover_pct", float("nan"))
            out[f"{policy_label}_avg_abs_inv_l01"] = r.get("avg_abs_inv", float("nan"))
        else:
            out[f"{policy_label}_mean_bps_per_step_l01"] = float("nan")
            out[f"{policy_label}_turnover_l01"] = float("nan")
            out[f"{policy_label}_avg_abs_inv_l01"] = float("nan")
    return out


def _parse_z_shift(outdir: Path) -> dict:
    """From Z_SHIFT_METRICS.csv get js_val_test (first row)."""
    p = outdir / "Z_SHIFT_METRICS.csv"
    if not p.exists():
        return {"js_val_test": float("nan")}
    df = pd.read_csv(p)
    if df.empty:
        return {"js_val_test": float("nan")}
    return {"js_val_test": float(df.iloc[0].get("js_val_test", float("nan")))}


def _gates_status(outdir: Path) -> str:
    """FAIL if pred_std_fit==0 or thresholds invalid; WARN if min bucket n < 500 or js > 0.25."""
    statuses = []
    # Z_THRESHOLDS
    z_path = outdir / "Z_THRESHOLDS.csv"
    if z_path.exists():
        z_df = pd.read_csv(z_path)
        if not z_df.empty:
            pred_std = z_df.get("pred_std", pd.Series([float("nan")])).iloc[0]
            if pred_std == 0 or (pd.notna(pred_std) and not (pred_std > 0)):
                statuses.append("FAIL_pred_std")
    # REWARD_SANITY fit row bucket counts
    r_path = outdir / "REWARD_SANITY.csv"
    if r_path.exists():
        r_df = pd.read_csv(r_path)
        fit = r_df[r_df["split"] == "val_and_test"] if "split" in r_df.columns else r_df.iloc[:1]
        if fit.empty:
            fit = r_df.iloc[:1]
        if not fit.empty:
            n_neg1 = fit.iloc[0].get("n_z_neg1", 0)
            n_pos1 = fit.iloc[0].get("n_z_pos1", 0)
            try:
                n_neg1, n_pos1 = int(n_neg1), int(n_pos1)
                if min(n_neg1, n_pos1) < 500:
                    statuses.append("WARN_min_bucket_n_500")
            except (TypeError, ValueError):
                pass
    # Z_SHIFT_METRICS
    s_path = outdir / "Z_SHIFT_METRICS.csv"
    if s_path.exists():
        s_df = pd.read_csv(s_path)
        if not s_df.empty:
            js = s_df.iloc[0].get("js_val_test", 0)
            try:
                if float(js) > 0.25:
                    statuses.append("WARN_js_025")
            except (TypeError, ValueError):
                pass
    return ";".join(statuses) if statuses else "OK"


SUMMARY_COLS = [
    "coin", "fee_bps", "breakeven_best_fit",
    "dp_mean_bps_per_step_l01", "dp_turnover_l01", "dp_avg_abs_inv_l01",
    "baselineA_mean_bps_per_step_l01", "baselineA_turnover_l01", "baselineA_avg_abs_inv_l01",
    "baselineB_mean_bps_per_step_l01", "baselineB_turnover_l01", "baselineB_avg_abs_inv_l01",
    "js_val_test", "gates_status", "rank_score",
]


def aggregate_summary(results_root: Path, status_rows: list) -> list:
    """Build MULTICOIN_SUMMARY.csv rows from each coin outdir."""
    summary_rows = []
    for row in status_rows:
        coin = row.get("coin", "")
        fee_bps = row.get("fee_bps", "")
        outdir = results_root / row.get("outdir_name", "")
        if not outdir.exists():
            continue
        d = {"coin": coin, "fee_bps": fee_bps}
        d.update(_parse_reward_sanity_fit_row(outdir))
        d.update(_parse_eval_summary(outdir))
        d.update(_parse_z_shift(outdir))
        d["gates_status"] = _gates_status(outdir)
        dp_bps = d.get("dp_mean_bps_per_step_l01", float("nan"))
        dp_turn = d.get("dp_turnover_l01", float("nan"))
        js = d.get("js_val_test", float("nan"))
        try:
            rank_score = float(dp_bps) - 0.05 * float(dp_turn) - 0.1 * float(js)
        except (TypeError, ValueError):
            rank_score = float("nan")
        d["rank_score"] = rank_score
        for col in SUMMARY_COLS:
            if col not in d:
                d[col] = float("nan") if col != "gates_status" else ""
        summary_rows.append(d)
    return summary_rows


def main():
    ap = argparse.ArgumentParser(description="Phase 2 DP multicoin sequential runs (val_and_test fit)")
    ap.add_argument("--step1_root", default=STEP1_ROOT_DEFAULT, help="Path to COINS dir")
    ap.add_argument("--outdir", default=None, help="Output parent dir; default results/phase2_multicoin_<ts>")
    ap.add_argument("--coins_fee2", default=None, help="Comma-separated coins for fee 2 bps (default: NEAR,RENDER,AXS,SNX)")
    ap.add_argument("--coins_fee1", default=None, help="Comma-separated coins for fee 1 bps")
    ap.add_argument("--include_extra", action="store_true", help="Include XAI (fee 1 bps)")
    ap.add_argument("--imax", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--qlo", type=float, default=0.33)
    ap.add_argument("--qhi", type=float, default=0.67)
    ap.add_argument("--lambda_grid", default="0,0.01,0.05,0.1,0.2")
    ap.add_argument("--n_rollouts", type=int, default=200)
    ap.add_argument("--eval_num_windows", type=int, default=20)
    ap.add_argument("--eval_window_len", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    step1_root = Path(args.step1_root)
    if not step1_root.is_absolute():
        step1_root = Path.cwd() / step1_root
    results_root = Path(args.outdir) if args.outdir else Path.cwd() / "results" / f"phase2_multicoin_{_ts()}"
    results_root.mkdir(parents=True, exist_ok=True)

    coins_fee2 = [c.strip() for c in (args.coins_fee2 or ",".join(COINS_FEE2)).split(",") if c.strip()]
    coins_fee1 = [c.strip() for c in (args.coins_fee1 or ",".join(COINS_FEE1)).split(",") if c.strip()]
    if args.include_extra:
        coins_fee1 = list(coins_fee1) + list(COINS_EXTRA)

    extra_args = [
        "--imax", str(args.imax),
        "--gamma", str(args.gamma),
        "--qlo", str(args.qlo),
        "--qhi", str(args.qhi),
        "--lambda_grid", args.lambda_grid,
        "--baseline_eval_lambdas", "subset",
        "--eval_mode", "windows",
        "--eval_window_len", str(args.eval_window_len),
        "--eval_num_windows", str(args.eval_num_windows),
        "--n_rollouts", str(args.n_rollouts),
        "--seed", str(args.seed),
        "--bundle",
    ]

    status_rows = []
    log_lines_global = []

    # Fee 2 bps
    for coin in coins_fee2:
        outdir_name = f"{coin}_fee2"
        outdir = results_root / outdir_name
        ok, log_lines = run_one(coin, 2.0, outdir, step1_root, extra_args)
        status_rows.append({
            "coin": coin,
            "fee_bps": 2.0,
            "outdir_name": outdir_name,
            "status": "PASS" if ok else "FAIL",
        })
        log_lines_global.append(f"--- {coin} fee2 ---")
        log_lines_global.extend(log_lines)
        if not ok:
            log_lines_global.append("(continuing)")

    # Fee 1 bps
    for coin in coins_fee1:
        outdir_name = f"{coin}_fee1"
        outdir = results_root / outdir_name
        ok, log_lines = run_one(coin, 1.0, outdir, step1_root, extra_args)
        status_rows.append({
            "coin": coin,
            "fee_bps": 1.0,
            "outdir_name": outdir_name,
            "status": "PASS" if ok else "FAIL",
        })
        log_lines_global.append(f"--- {coin} fee1 ---")
        log_lines_global.extend(log_lines)
        if not ok:
            log_lines_global.append("(continuing)")

    # MULTICOIN_STATUS.csv
    status_path = results_root / "MULTICOIN_STATUS.csv"
    with open(status_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["coin", "fee_bps", "outdir_name", "status"])
        w.writeheader()
        w.writerows(status_rows)

    # MULTICOIN_SUMMARY.csv
    summary_rows = aggregate_summary(results_root, status_rows)
    summary_path = results_root / "MULTICOIN_SUMMARY.csv"
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False, columns=SUMMARY_COLS)

    # MULTICOIN_RUN_LOG.md
    (results_root / "MULTICOIN_RUN_LOG.md").write_text("# Multicoin Phase 2 run log\n\n" + "\n".join(log_lines_global))

    # README disclaimer
    readme = """# Phase 2 Multicoin Run

**Leakage screening mode:** This run uses val+test for z-threshold fit and reward estimation to improve estimate stability for screening. Results are for triage, not strict out-of-sample validation.

- MULTICOIN_STATUS.csv: per-coin, per-fee status (PASS/FAIL).
- MULTICOIN_SUMMARY.csv: breakeven_best_fit, DP/baseline metrics @ lambda=0.1, js_val_test, gates_status, rank_score.
- rank_score = dp_mean_bps_per_step_l01 - 0.05*dp_turnover_l01 - 0.1*js_val_test
"""
    (results_root / "README.md").write_text(readme)

    print("Multicoin Phase 2 done. Results:", results_root)
    print("MULTICOIN_STATUS.csv:", status_path)
    print("MULTICOIN_SUMMARY.csv:", summary_path)


if __name__ == "__main__":
    main()
