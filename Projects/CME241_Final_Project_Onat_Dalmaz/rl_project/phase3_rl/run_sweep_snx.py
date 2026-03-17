#!/usr/bin/env python3
"""Sweep for SNX viability: c_maker × p0 × p1. Outputs SWEEP_RESULTS.csv with delta_fixed/oracle, turnover, maker_share, etc."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

RUN_ROOT = "/home/ubuntu/onat/experiments/step1_signal_feb13_feb24_train6_20260224_191001/COINS/SNX"
# Grid: c_maker ∈ {1.0, 0.5, 0.25}, p0 ∈ {0.35, 0.45}, p1 ∈ {0.15, 0.10}
GRID = [
    {"c_maker_bps": 1.0, "p0": 0.35, "p1": 0.15},
    {"c_maker_bps": 1.0, "p0": 0.35, "p1": 0.10},
    {"c_maker_bps": 1.0, "p0": 0.45, "p1": 0.15},
    {"c_maker_bps": 1.0, "p0": 0.45, "p1": 0.10},
    {"c_maker_bps": 0.5, "p0": 0.35, "p1": 0.15},
    {"c_maker_bps": 0.5, "p0": 0.35, "p1": 0.10},
    {"c_maker_bps": 0.5, "p0": 0.45, "p1": 0.15},
    {"c_maker_bps": 0.5, "p0": 0.45, "p1": 0.10},
    {"c_maker_bps": 0.25, "p0": 0.35, "p1": 0.15},
    {"c_maker_bps": 0.25, "p0": 0.35, "p1": 0.10},
    {"c_maker_bps": 0.25, "p0": 0.45, "p1": 0.15},
    {"c_maker_bps": 0.25, "p0": 0.45, "p1": 0.10},
]


def parse_args():
    ap = argparse.ArgumentParser(description="Phase3 SNX sweep: c_maker × p0 × p1")
    ap.add_argument("--outdir", type=str, default="/home/ubuntu/onat/results/phase3_sweep_snx", help="Sweep output root")
    ap.add_argument("--run_root", type=str, default=RUN_ROOT, help="Run root (e.g. .../COINS/SNX)")
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--eval_windows", type=int, default=10)
    ap.add_argument("--fill_seeds", type=int, default=20)
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--seed", type=int, default=123)
    return ap.parse_args()


def main():
    import pandas as pd

    args = parse_args()
    out_base = Path(args.outdir)
    out_base.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, g in enumerate(GRID):
        outdir = out_base / f"sweep_{i}"
        outdir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "phase3_rl.cli_phase3",
            "--run_root", args.run_root,
            "--outdir", str(outdir),
            "--reward_mode", "inventory_mtm",
            "--eval_replay_mode", "deterministic",
            "--qlo", "0.33", "--qhi", "0.67", "--Imax", "3",
            "--vbin_method", "median_abs_y", "--vbin_quantile", "0.5",
            "--c_maker_bps", str(g["c_maker_bps"]),
            "--c_taker_bps", "4.0",
            "--lambda_inv", "0.1", "--eta_turnover", "0.0",
            "--p0", str(g["p0"]), "--p1", str(g["p1"]), "--dv", "0.05", "--d_age", "0.05",
            "--gamma", "0.99", "--alpha", "0.2", "--alpha_min", "0.05",
            "--eps", "0.2", "--eps_min", "0.02", "--q_init", "0.01",
            "--decay_episodes", str(args.episodes), "--n_train_episodes", str(args.episodes),
            "--n_train_windows", "50", "--train_window_len", "5000",
            "--eval_num_windows", str(args.eval_windows), "--eval_window_len", "5000",
            "--eval_fill_seeds", str(args.fill_seeds), "--bootstrap_iters", str(args.bootstrap),
            "--seed", str(args.seed), "--bundle", "0",
        ]
        r = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True, timeout=360000)
        row = {
            "config_id": i,
            "c_maker": g["c_maker_bps"],
            "p0": g["p0"],
            "p1": g["p1"],
            "ql_mean": None,
            "ql_ci_low": None,
            "ql_ci_high": None,
            "baselineA_mean": None,
            "baselineB_mean": None,
            "hold_mean": None,
            "best_fixed_baseline_mean": None,
            "best_fixed_baseline_name": None,
            "delta_fixed_mean": None,
            "delta_fixed_ci_low": None,
            "delta_fixed_ci_high": None,
            "delta_oracle_mean": None,
            "delta_oracle_ci_low": None,
            "delta_oracle_ci_high": None,
            "turnover": None,
            "avg_abs_inv": None,
            "maker_share": None,
            "taker_share": None,
            "hold_frac": None,
            "success": r.returncode == 0,
        }
        if (outdir / "EVAL_SUMMARY.csv").exists():
            df = pd.read_csv(outdir / "EVAL_SUMMARY.csv")
            ql = df[df["policy"] == "QL"]
            if not ql.empty:
                q = ql.iloc[0]
                row["ql_mean"] = q.get("mean_cum_bps")
                row["ql_ci_low"] = q.get("ci_low")
                row["ql_ci_high"] = q.get("ci_high")
                row["best_fixed_baseline_name"] = q.get("best_fixed_baseline", "")
                row["delta_fixed_mean"] = q.get("delta_fixed_mean")
                row["delta_fixed_ci_low"] = q.get("delta_fixed_ci_low")
                row["delta_fixed_ci_high"] = q.get("delta_fixed_ci_high")
                row["delta_oracle_mean"] = q.get("delta_oracle_mean")
                row["delta_oracle_ci_low"] = q.get("delta_oracle_ci_low")
                row["delta_oracle_ci_high"] = q.get("delta_oracle_ci_high")
                row["maker_share"] = q.get("maker_share")
                row["taker_share"] = q.get("taker_share")
                bf = str(row["best_fixed_baseline_name"]).strip()
                if bf:
                    bl = df[df["policy"] == bf]
                    if not bl.empty:
                        row["best_fixed_baseline_mean"] = bl["mean_cum_bps"].iloc[0]
            for name, key in [("A_sign_taker", "baselineA_mean"), ("B_sign_maker", "baselineB_mean"), ("C_hold", "hold_mean")]:
                bl = df[df["policy"] == name]
                if not bl.empty:
                    row[key] = bl["mean_cum_bps"].iloc[0]
        if (outdir / "EVAL_VISITATION.csv").exists():
            vis = pd.read_csv(outdir / "EVAL_VISITATION.csv")
            if not vis.empty:
                row["turnover"] = vis["turnover_pct"].mean()
                row["hold_frac"] = vis["action_frac_HOLD"].mean()
                if "maker_share" in vis.columns:
                    row["maker_share"] = row["maker_share"] if row["maker_share"] is not None else vis["maker_share"].mean()
                if "taker_share" in vis.columns:
                    row["taker_share"] = row["taker_share"] if row["taker_share"] is not None else vis["taker_share"].mean()
                if "avg_abs_inv" in vis.columns:
                    row["avg_abs_inv"] = vis["avg_abs_inv"].mean()
        rows.append(row)

    out_csv = out_base / "SWEEP_RESULTS.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Wrote", out_csv)

    # Top-K by highest delta_fixed_mean, tie-break by higher ql_mean (spec 2.3)
    successful = [r for r in rows if r.get("success") and r.get("delta_fixed_mean") is not None]
    top_configs = sorted(successful, key=lambda x: (x.get("delta_fixed_mean") or -1e9, x.get("ql_mean") or -1e9), reverse=True)[:2]
    top_configs_json = [
        {"rank": rank, "config_id": t["config_id"], "c_maker": t["c_maker"], "p0": t["p0"], "p1": t["p1"]}
        for rank, t in enumerate(top_configs, 1)
    ]
    (out_base / "TOP_CONFIGS.json").write_text(json.dumps(top_configs_json, indent=2))
    print("Wrote", out_base / "TOP_CONFIGS.json")

    # Optional: filter by turnover < 50%, maker_share > 30% for display
    valid = [r for r in rows if r.get("success") and r.get("turnover") is not None and r.get("maker_share") is not None]
    valid = [r for r in valid if (r.get("turnover") or 0) < 50 and (r.get("maker_share") or 0) > 0.30]
    if valid:
        top_valid = sorted(valid, key=lambda x: (x.get("delta_fixed_mean") or -1e9, x.get("ql_mean") or -1e9), reverse=True)[:2]
        print("Top-2 (turnover<50%, maker_share>30%):", [(t["config_id"], t["c_maker"], t["p0"], t["p1"]) for t in top_valid])
    print("TOP_CONFIGS (by delta_fixed_mean):", top_configs_json)


if __name__ == "__main__":
    main()
