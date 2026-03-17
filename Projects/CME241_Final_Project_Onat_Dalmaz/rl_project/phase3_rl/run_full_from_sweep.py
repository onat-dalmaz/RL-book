#!/usr/bin/env python3
"""
Launch full report runs for top-K configs from a completed sweep.
Reads TOP_CONFIGS.json from sweep outdir, runs cli_phase3 for each with matching c_maker,p0,p1.
Outdir: phase3_report_real_snx_cfg{config_id}_cm{c_maker}_p0{p0}_p1{p1}.
Post-run: verify EFFECTIVE_CONFIG matches sweep row and report settings; write INVALID_RUN.txt on mismatch.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description="Run full report for top-K sweep configs")
    ap.add_argument("--sweep_outdir", type=str, required=True, help="Sweep root (contains SWEEP_RESULTS.csv, TOP_CONFIGS.json)")
    ap.add_argument("--run_root", type=str, required=True, help="SNX run root (e.g. .../COINS/SNX)")
    ap.add_argument("--results_base", type=str, default="/home/ubuntu/onat/results", help="Base dir for report outdirs")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--eval_windows", type=int, default=20)
    ap.add_argument("--fill_seeds", type=int, default=50)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--bundle", type=int, default=1)
    ap.add_argument("--top_k", type=int, default=2)
    return ap.parse_args()


def main():
    args = parse_args()
    sweep_dir = Path(args.sweep_outdir)
    if not (sweep_dir / "TOP_CONFIGS.json").exists():
        print("Missing TOP_CONFIGS.json in", sweep_dir, file=sys.stderr)
        return 1
    top_configs = json.loads((sweep_dir / "TOP_CONFIGS.json").read_text())[: args.top_k]
    if not top_configs:
        print("No configs in TOP_CONFIGS.json", file=sys.stderr)
        return 1

    import pandas as pd
    sweep_df = pd.read_csv(sweep_dir / "SWEEP_RESULTS.csv") if (sweep_dir / "SWEEP_RESULTS.csv").exists() else None

    results_base = Path(args.results_base)
    for rec in top_configs:
        cid = rec["config_id"]
        c_maker = rec["c_maker"]
        p0 = rec["p0"]
        p1 = rec["p1"]
        outdir = results_base / f"phase3_report_real_snx_cfg{cid}_cm{c_maker}_p0{p0}_p1{p1}"
        outdir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "phase3_rl.cli_phase3",
            "--run_root", args.run_root,
            "--outdir", str(outdir),
            "--reward_mode", "inventory_mtm",
            "--eval_replay_mode", "deterministic",
            "--qlo", "0.33", "--qhi", "0.67", "--Imax", "3",
            "--vbin_method", "median_abs_y", "--vbin_quantile", "0.5",
            "--c_maker_bps", str(c_maker),
            "--c_taker_bps", "4.0",
            "--lambda_inv", "0.1", "--eta_turnover", "0.0",
            "--p0", str(p0), "--p1", str(p1), "--dv", "0.05", "--d_age", "0.05",
            "--gamma", "0.99", "--alpha", "0.2", "--alpha_min", "0.05",
            "--eps", "0.2", "--eps_min", "0.02", "--q_init", "0.01",
            "--decay_episodes", str(args.episodes), "--n_train_episodes", str(args.episodes),
            "--n_train_windows", "50", "--train_window_len", "5000",
            "--eval_num_windows", str(args.eval_windows), "--eval_window_len", "5000",
            "--eval_fill_seeds", str(args.fill_seeds), "--bootstrap_iters", str(args.bootstrap),
            "--seed", str(args.seed), "--bundle", str(args.bundle),
        ]
        r = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True, timeout=360000)
        if r.returncode != 0:
            print("Config", cid, "failed:", r.stderr[:500] if r.stderr else r.returncode, file=sys.stderr)

        # Mismatch check: EFFECTIVE_CONFIG vs sweep row and report settings
        invalid_lines = []
        if (outdir / "EFFECTIVE_CONFIG.json").exists() and sweep_df is not None:
            eff = json.loads((outdir / "EFFECTIVE_CONFIG.json").read_text())
            row = sweep_df[sweep_df["config_id"] == cid]
            if not row.empty:
                row = row.iloc[0]
                if abs(eff["execution"].get("c_maker_bps", 0) - row.get("c_maker", 0)) > 1e-6:
                    invalid_lines.append(f"c_maker_bps: effective={eff['execution'].get('c_maker_bps')} sweep={row.get('c_maker')}")
                if abs(eff["execution"].get("p0", 0) - row.get("p0", 0)) > 1e-6:
                    invalid_lines.append(f"p0: effective={eff['execution'].get('p0')} sweep={row.get('p0')}")
                if abs(eff["execution"].get("p1", 0) - row.get("p1", 0)) > 1e-6:
                    invalid_lines.append(f"p1: effective={eff['execution'].get('p1')} sweep={row.get('p1')}")
            if eff["training"].get("n_train_episodes") != args.episodes:
                invalid_lines.append(f"n_train_episodes: effective={eff['training'].get('n_train_episodes')} expected={args.episodes}")
            if eff["windowing"].get("eval_num_windows") != args.eval_windows:
                invalid_lines.append(f"eval_num_windows: effective={eff['windowing'].get('eval_num_windows')} expected={args.eval_windows}")
            if eff["windowing"].get("eval_fill_seeds") != args.fill_seeds:
                invalid_lines.append(f"eval_fill_seeds: effective={eff['windowing'].get('eval_fill_seeds')} expected={args.fill_seeds}")
            if eff["bootstrap"].get("bootstrap_iters") != args.bootstrap:
                invalid_lines.append(f"bootstrap_iters: effective={eff['bootstrap'].get('bootstrap_iters')} expected={args.bootstrap}")
        elif not (outdir / "EFFECTIVE_CONFIG.json").exists():
            invalid_lines.append("Missing EFFECTIVE_CONFIG.json")

        if invalid_lines:
            (outdir / "INVALID_RUN.txt").write_text("CONFIG_MISMATCH\n\n" + "\n".join(invalid_lines))
            if (outdir / "PASS_FAIL.json").exists():
                pf = json.loads((outdir / "PASS_FAIL.json").read_text())
                pf["overall"] = "FAIL_CONFIG_MISMATCH"
                (outdir / "PASS_FAIL.json").write_text(json.dumps(pf, indent=2))
            print("INVALID_RUN.txt written for config", cid, invalid_lines)
        else:
            print("Config", cid, "OK", outdir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
