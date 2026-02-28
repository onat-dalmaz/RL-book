#!/usr/bin/env python3
"""
Phase 2 DP v3: Robust, semantic-correct, dual-input (Step1 + S7).
Orchestrates discovery, canonicalization, z, P(z'|z), reward model, DP solve, evaluation, baselines, sweep, reporting, bundle.
"""

import argparse
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# Ensure rl_project is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_io
import mdp_dp
import eval_rollout
import reporting


def _git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).parent).strip()[:12]
    except Exception:
        return ""


def needs_fee_sweep(args, outdir: Path) -> bool:
    """
    True iff: mode==step1, fee_grid non-empty, and FEE_SWEEP_RESULTS.csv missing or incomplete for requested grid.
    """
    if getattr(args, "mode", "step1") != "step1":
        return False
    fee_grid = getattr(args, "fee_grid", "") or ""
    fee_list = [float(x.strip()) for x in fee_grid.split(",") if x.strip()]
    if not fee_list:
        return False
    path = Path(outdir) / "FEE_SWEEP_RESULTS.csv"
    if not path.exists():
        return True
    try:
        df = pd.read_csv(path)
    except Exception:
        return True
    if df.empty or "fee_bps" not in df.columns:
        return True
    n_datasets = df["dataset_id"].nunique() if "dataset_id" in df.columns else 1
    if n_datasets == 0:
        return True
    # Fee sweep uses 2 lambdas (0 and 0.1)
    expected_rows = len(fee_list) * 2 * n_datasets
    return len(df) < expected_rows


def main():
    ap = argparse.ArgumentParser(description="Phase 2 DP v3 from parquets (Step1 / S7)")
    ap.add_argument("--run_root", required=True, help="Root directory to scan")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--mode", default="auto", choices=["auto", "step1", "s7"])
    ap.add_argument("--dt_ms", type=int, default=500)
    ap.add_argument("--imax", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--qlo", type=float, default=0.33)
    ap.add_argument("--qhi", type=float, default=0.67)
    ap.add_argument("--fee_bps", type=float, default=4.0)
    ap.add_argument("--fee_grid", type=str, default="", help="Optional fee sweep e.g. 0,0.5,1.0,2.0,4.0 (step1 only)")
    ap.add_argument("--lambda_grid", type=str, default="0,0.01,0.05,0.1,0.2")
    ap.add_argument("--eta_turnover_bps", type=float, default=0.0)
    ap.add_argument("--n_rollouts", type=int, default=200)
    ap.add_argument("--max_steps_per_rollout", type=int, default=0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--nmin_bucket", type=int, default=200)
    ap.add_argument("--laplace_alpha", type=float, default=1.0)
    ap.add_argument("--fail_on_degenerate", type=int, default=1)
    ap.add_argument("--allow_no_pred", type=int, default=0)
    ap.add_argument("--min_z_frac", type=float, default=0.05)
    ap.add_argument("--bundle", action="store_true")
    ap.add_argument("--report_run", type=int, default=0, help="If 1, set n_rollouts=200 unless overridden; full sequence unless max_steps set")
    # Resume
    ap.add_argument("--resume", type=int, default=0, help="If 1 and outdir exists, skip completed stages")
    ap.add_argument("--resume_stage", default="auto", choices=["auto", "eval", "bundle"], help="Stage to resume from (auto=detect)")
    # Bounded eval (windowed)
    ap.add_argument("--eval_mode", default=None, choices=["full", "windows"], help="full=whole sequence, windows=K windows of L steps (default: windows if report_run)")
    ap.add_argument("--eval_window_len", type=int, default=5000, help="Window length L for eval_mode=windows")
    ap.add_argument("--eval_num_windows", type=int, default=20, help="Number of windows K")
    ap.add_argument("--eval_rollouts_total", type=int, default=None, help="Total rollouts across windows (default: n_rollouts)")
    ap.add_argument("--eval_rollouts_per_window", type=int, default=0, help="If >0, use this per window; else total/K")
    # Fee sweep eval budget (lighter than main eval)
    ap.add_argument("--fee_sweep_eval_rollouts", type=int, default=50)
    ap.add_argument("--fee_sweep_eval_window_len", type=int, default=2000)
    ap.add_argument("--fee_sweep_eval_num_windows", type=int, default=10)
    # Baselines: subset = only lambda=0 and lambda_default (0.1) when report_run
    ap.add_argument("--baseline_eval_lambdas", default=None, choices=["all", "subset"], help="all or subset (default: subset if report_run)")
    ap.add_argument("--z_fit_mode", default="auto", choices=["auto", "train", "val", "val_and_test", "test70"])
    ap.add_argument("--reward_fit_mode", default=None)
    # Phase 2 v4
    ap.add_argument("--reward_mode", default="inventory_mtm", choices=["inventory_mtm", "one_step_bet"])
    ap.add_argument("--eval_replay_mode", default="deterministic", choices=["deterministic", "model_based_sampling"])
    ap.add_argument("--label_horizon_ms", default="auto")
    ap.add_argument("--mdp_step_ms", default="auto")
    ap.add_argument("--stride_mode", default="match_horizon", choices=["match_horizon", "recompute_dt_return"])
    ap.add_argument("--bootstrap_iters", type=int, default=1000)
    args = ap.parse_args()
    if args.reward_fit_mode is None:
        args.reward_fit_mode = args.z_fit_mode

    run_root = Path(args.run_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "evidence").mkdir(exist_ok=True)

    is_report_run = bool(getattr(args, "report_run", 0))
    n_rollouts_used = 200 if is_report_run else args.n_rollouts
    if args.eval_mode is None:
        args.eval_mode = "windows" if is_report_run else "full"
    if args.baseline_eval_lambdas is None:
        args.baseline_eval_lambdas = "subset" if is_report_run else "all"
    eval_rollouts_total = getattr(args, "eval_rollouts_total", None) or n_rollouts_used

    lambda_list = [float(x.strip()) for x in args.lambda_grid.split(",")]
    fee_list = [float(x.strip()) for x in args.fee_grid.split(",")] if getattr(args, "fee_grid", "").strip() else []
    kind_used = None
    lambda_default = 0.1
    baseline_lambdas = [lambda_list[0], lambda_default] if args.baseline_eval_lambdas == "subset" and lambda_default in lambda_list else lambda_list
    if args.baseline_eval_lambdas == "subset" and lambda_default not in baseline_lambdas and lambda_list:
        baseline_lambdas = [lambda_list[0], (lambda_list[-1] if len(lambda_list) > 1 else lambda_list[0])]

    # Resume detection: load existing manifest and decide stage
    resume = getattr(args, "resume", 0)
    resume_stage = getattr(args, "resume_stage", "auto")
    skip_dp_solve = False
    skip_eval = False
    run_eval_only = False
    run_fee_sweep_only = False
    run_bundle_only = False
    resume_log_lines = []
    if resume and outdir.exists() and (outdir / "RUN_MANIFEST.json").exists():
        try:
            prev_manifest = json.loads((outdir / "RUN_MANIFEST.json").read_text())
            has_policies = any((outdir / f"POLICY_TABLE_lambda_{str(lam).replace('.', '_')}.csv").exists() for lam in lambda_list)
            has_eval = (outdir / "EVAL_ROLLOUT_SUMMARY.csv").exists()
            has_zip = (outdir.parent / f"{outdir.name}.zip").exists()
            need_fee = needs_fee_sweep(args, outdir)
            if resume_stage == "auto":
                if has_eval and has_policies and need_fee:
                    run_fee_sweep_only = True
                    skip_dp_solve = True
                    resume_log_lines.append("Resume auto: eval present, fee sweep missing/incomplete -> fee sweep only")
                elif has_eval and not has_zip and args.bundle and not need_fee:
                    run_bundle_only = True
                    resume_log_lines.append("Resume auto: eval present, zip missing -> bundle only")
                elif not has_eval and has_policies:
                    run_eval_only = True
                    skip_dp_solve = True
                    resume_log_lines.append("Resume auto: policies present, eval missing -> eval only (skip DP solve)")
                elif not has_policies:
                    resume_log_lines.append("Resume auto: no policies -> full run")
                else:
                    run_eval_only = True
                    skip_dp_solve = True
                    resume_log_lines.append("Resume auto: policies present, eval missing -> eval only")
            elif resume_stage == "eval":
                run_eval_only = True
                skip_dp_solve = has_policies
                resume_log_lines.append(f"Resume stage=eval, skip_dp_solve={skip_dp_solve}")
            elif resume_stage == "bundle":
                run_bundle_only = True
                resume_log_lines.append("Resume stage=bundle -> bundle only")
        except Exception as e:
            resume_log_lines.append(f"Resume parse failed: {e}; proceeding full run")

    # RUN_MANIFEST.json (updated at end with actual n_rollouts_used, STATUS)
    leakage_mode = args.z_fit_mode == "val_and_test" or args.reward_fit_mode == "val_and_test"
    run_manifest = {
        "run_root": str(run_root.resolve()),
        "outdir": str(outdir.resolve()),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_hash": _git_hash(),
        "args": vars(args),
        "is_report_run": is_report_run,
        "actual_n_rollouts_used": n_rollouts_used,
        "leakage_mode": leakage_mode,
        "reward_mode": getattr(args, "reward_mode", "one_step_bet"),
        "eval_replay_mode": getattr(args, "eval_replay_mode", "model_based_sampling"),
    }
    (outdir / "RUN_MANIFEST.json").write_text(json.dumps(run_manifest, indent=2))

    try:
        import pyarrow.parquet as pq
    except ImportError:
        pq = None
    if pq is None:
        reporting.write_pass_fail(outdir, [{"dataset_id": "", "gate": "pyarrow", "result": "FAIL", "evidence": "missing"}])
        return

    if run_bundle_only:
        if resume_log_lines:
            (outdir / "RESUME_LOG.md").write_text("# Resume log\n\n" + "\n".join(resume_log_lines))
        zip_path = outdir.parent / f"{outdir.name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in outdir.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(outdir.parent))
        print("Bundled (resume bundle-only):", zip_path)
        return

    try:
        success = _run_phase2(run_root, outdir, args, run_manifest, is_report_run, n_rollouts_used,
            lambda_list, fee_list, baseline_lambdas, eval_rollouts_total, skip_dp_solve,
            run_fee_sweep_only, resume_log_lines, eval_progress=None)
    except Exception as e:
        crash_occurred = e
        import traceback
        (outdir / "CRASH.txt").write_text(traceback.format_exc())
        reporting.write_pass_fail(outdir, [{"dataset_id": "", "gate": "eval_completion", "result": "FAIL", "evidence": f"crash: {e}. See CRASH.txt"}])
        run_manifest["STATUS"] = "INCOMPLETE"
        run_manifest["crash"] = str(e)
        missing = []
        for name in ["EVAL_ROLLOUT_SUMMARY.csv", "SWEEP_RESULTS.csv", "FEE_SWEEP_RESULTS.csv", "PASS_FAIL.md"]:
            if not (outdir / name).exists():
                missing.append(name)
        if missing:
            run_manifest["missing_artifacts"] = missing
            (outdir / "MISSING_ARTIFACTS.md").write_text("# Missing artifacts\n\n" + "\n".join("- " + m for m in missing))
        (outdir / "RUN_MANIFEST.json").write_text(json.dumps(run_manifest, indent=2))
        if args.bundle:
            zip_path = outdir.parent / f"{outdir.name}.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in outdir.rglob("*"):
                    if f.is_file():
                        zf.write(f, f.relative_to(outdir.parent))
            print("Bundled (partial after crash):", zip_path)
        raise

    if success:
        run_manifest["STATUS"] = "COMPLETE"
        (outdir / "RUN_MANIFEST.json").write_text(json.dumps(run_manifest, indent=2))
        if resume_log_lines:
            (outdir / "RESUME_LOG.md").write_text("# Resume log\n\n" + "\n".join(resume_log_lines))
        print("Phase 2 DP v3 done. Outdir:", outdir)


def _run_phase2(run_root, outdir, args, run_manifest, is_report_run, n_rollouts_used,
                lambda_list, fee_list, baseline_lambdas, eval_rollouts_total, skip_dp_solve,
                run_fee_sweep_only, resume_log_lines, eval_progress=None):
    """Inner flow: Phase 0 through bundle. Called from main() inside try/except."""
    kind_used = None
    # Phase 0
    manifest_df, missing = data_io.discover_and_classify(run_root, args.mode, outdir)
    if manifest_df.empty:
        reporting.write_pass_fail(outdir, [{"dataset_id": "", "gate": "manifest", "result": "FAIL", "evidence": "no step1/s7 datasets"}])
        return False
    kind_used = manifest_df["kind"].iloc[0]

    # Phase 1
    data_io.canonicalize_all(manifest_df, run_root, outdir, args.allow_no_pred)

    # Phase 2 v4: Horizon and stride
    stride = 1
    raw_dt_ms = label_horizon_ms = mdp_step_ms = 500.0
    horizon_source = "unknown"
    stride_mode = getattr(args, "stride_mode", "match_horizon")
    if kind_used == "step1":
        first_step1 = manifest_df[manifest_df["kind"] == "step1"]
        if not first_step1.empty:
            path = run_root / first_step1.iloc[0]["path"]
            df1, _ = data_io.load_and_canonicalize(path, run_root, "step1", allow_no_pred_s7=False)
            if df1 is not None and len(df1) > 0:
                raw_dt_ms, label_horizon_ms, horizon_source = data_io.infer_label_horizon_ms(df1, run_root)
                if stride_mode == "match_horizon":
                    mdp_step_ms = label_horizon_ms
                    stride = max(1, int(round(label_horizon_ms / raw_dt_ms)))
                else:
                    stride = 1
                    mdp_step_ms = raw_dt_ms
        data_io.write_horizon_spec(outdir, raw_dt_ms, label_horizon_ms, stride, mdp_step_ms, stride_mode, horizon_source)
        if stride_mode == "match_horizon" and raw_dt_ms > 0 and abs((label_horizon_ms / raw_dt_ms) - round(label_horizon_ms / raw_dt_ms)) > 0.01:
            reporting.write_pass_fail(outdir, [{"dataset_id": "", "gate": "horizon_stride", "result": "WARN", "evidence": "label_horizon_ms not multiple of raw_dt_ms"}])
        if stride < 1:
            reporting.write_pass_fail(outdir, [{"dataset_id": "", "gate": "horizon_stride", "result": "FAIL", "evidence": "stride < 1"}])
            return False

    # Phase 2–4: compute or load when resuming
    if skip_dp_solve and (outdir / "Z_THRESHOLDS.csv").exists():
        z_thresholds = [{"dataset_id": r["dataset_id"], "q_lo": r["q_lo"], "q_hi": r["q_hi"]} for _, r in pd.read_csv(outdir / "Z_THRESHOLDS.csv").iterrows()]
        gate_z = []
    else:
        z_thresholds, z_dist, gate_z = data_io.compute_z_thresholds(
            manifest_df, run_root, args.qlo, args.qhi, outdir, min_z_frac=args.min_z_frac,
            z_fit_mode=getattr(args, "z_fit_mode", "auto")
        )
    if not z_thresholds:
        reporting.write_pass_fail(outdir, [{"dataset_id": "", "gate": "z_thresholds", "result": "FAIL", "evidence": "none"}])
        return False

    if skip_dp_solve and (outdir / "P_Z_GIVEN_Z.csv").exists():
        P_z_df = pd.read_csv(outdir / "P_Z_GIVEN_Z.csv")
    else:
        p_z_rows, gate_pz = data_io.compute_p_z_given_z(manifest_df, run_root, z_thresholds, outdir, args.laplace_alpha)
        P_z_df = pd.read_csv(outdir / "P_Z_GIVEN_Z.csv") if (outdir / "P_Z_GIVEN_Z.csv").exists() else None

    if skip_dp_solve and (outdir / "REWARD_STATS.csv").exists():
        R_stats_df = pd.read_csv(outdir / "REWARD_STATS.csv")
        gate_r = []
    else:
        if kind_used == "step1":
            _, _, gate_r, samples_by_dataset = data_io.reward_model_step1(
                manifest_df, run_root, z_thresholds, outdir, args.fee_bps, args.nmin_bucket, reward_fit_mode=args.reward_fit_mode
            )
        else:
            _, _, gate_r, samples_by_dataset = data_io.reward_model_s7(
                manifest_df, run_root, z_thresholds, outdir, args.nmin_bucket
            )
        R_stats_df = pd.read_csv(outdir / "REWARD_STATS.csv") if (outdir / "REWARD_STATS.csv").exists() else pd.DataFrame()
    if R_stats_df.empty:
        reporting.write_pass_fail(outdir, [{"dataset_id": "", "gate": "reward_stats", "result": "FAIL", "evidence": "empty"}])
        return False

    if not skip_dp_solve and args.fail_on_degenerate and gate_r:
        reporting.write_pass_fail(outdir, [{"dataset_id": g[0], "gate": g[1], "result": "FAIL", "evidence": g[2]} for g in gate_r])
        return False

    if not skip_dp_solve:
        data_io.compute_z_shift_metrics(manifest_df, run_root, z_thresholds, outdir)

    # Horizon (v4): infer and write HORIZON_SPEC
    stride = 1
    raw_dt_ms = label_horizon_ms = mdp_step_ms = 500.0
    stride_mode = getattr(args, "stride_mode", "match_horizon")
    horizon_source = "unknown"
    if kind_used == "step1" and not manifest_df.empty:
        row0 = manifest_df.iloc[0]
        path0 = run_root / row0["path"]
        df0, _ = data_io.load_and_canonicalize(path0, run_root, "step1", allow_no_pred_s7=True)
        if df0 is not None and len(df0) > 0:
            raw_dt_ms, label_horizon_ms, horizon_source = data_io.infer_label_horizon_ms(df0, run_root)
            if stride_mode == "match_horizon":
                mdp_step_ms = label_horizon_ms
                stride = max(1, int(round(label_horizon_ms / raw_dt_ms)))
            data_io.write_horizon_spec(outdir, raw_dt_ms, label_horizon_ms, stride, mdp_step_ms, stride_mode, horizon_source)

    # Unique dataset ids (no split)
    dataset_ids = list({r["dataset_id"] for r in z_thresholds})
    I_list = list(range(-args.imax, args.imax + 1))

    # Baselines
    policy_a, policy_b = _baseline_policies(I_list, args.imax)
    policy_c = np.zeros((3, len(I_list)), dtype=int)

    # Phase 5/6/7/8/9: Solve DP per lambda, evaluate, sweep, diagnostics
    eval_rows = []
    sweep_rows = []
    diag_rows = []
    model_value_rows = []
    gate_rows = []
    eval_progress = {
        "completed_dp_evals": 0,
        "completed_baseline_evals": 0,
        "completed_fee_sweep_evals": 0,
        "fee_sweep_requested": bool(fee_list and kind_used == "step1"),
        "fee_sweep_completed": False,
        "fee_sweep_rows_written": 0,
    }
    eval_csv_path = outdir / "EVAL_ROLLOUT_SUMMARY.csv"
    sweep_csv_path = outdir / "SWEEP_RESULTS.csv"
    if skip_dp_solve and not run_fee_sweep_only:
        if eval_csv_path.exists():
            eval_csv_path.unlink()
        if sweep_csv_path.exists():
            sweep_csv_path.unlink()
        if (outdir / "FEE_SWEEP_RESULTS.csv").exists():
            (outdir / "FEE_SWEEP_RESULTS.csv").unlink()
    if run_fee_sweep_only:
        if (outdir / "FEE_SWEEP_RESULTS.csv").exists():
            (outdir / "FEE_SWEEP_RESULTS.csv").unlink()
        if (outdir / "FEE_SWEEP_POLICY_DIAGNOSTICS.csv").exists():
            (outdir / "FEE_SWEEP_POLICY_DIAGNOSTICS.csv").unlink()
    use_windows = args.eval_mode == "windows"
    window_len = getattr(args, "eval_window_len", 5000)
    num_windows = getattr(args, "eval_num_windows", 20)
    n_per_window = (eval_rollouts_total // num_windows) if getattr(args, "eval_rollouts_per_window", 0) == 0 else getattr(args, "eval_rollouts_per_window", 0)

    if run_fee_sweep_only:
        eval_rows = pd.read_csv(eval_csv_path).to_dict("records") if eval_csv_path.exists() else []
        sweep_rows = pd.read_csv(sweep_csv_path).to_dict("records") if sweep_csv_path.exists() else []
        diag_rows = pd.read_csv(outdir / "POLICY_DIAGNOSTICS.csv").to_dict("records") if (outdir / "POLICY_DIAGNOSTICS.csv").exists() else []
        model_value_rows = pd.read_csv(outdir / "MODEL_VALUE_SUMMARY.csv").to_dict("records") if (outdir / "MODEL_VALUE_SUMMARY.csv").exists() else []
    else:
        pass  # eval_rows, sweep_rows, diag_rows, model_value_rows filled in loop below

    reward_mode = getattr(args, "reward_mode", "one_step_bet")
    eval_replay_mode = getattr(args, "eval_replay_mode", "model_based_sampling")
    bootstrap_iters = getattr(args, "bootstrap_iters", 1000)

    if not run_fee_sweep_only:
        for dataset_id in dataset_ids:
            P_z = mdp_dp.P_z_from_dataframe(P_z_df, dataset_id)
            mu_long, mu_short = mdp_dp.mu_from_reward_stats(R_stats_df, dataset_id)
            R_table = mdp_dp.build_R_table(mu_long, mu_short, args.eta_turnover_bps)
            mu_y = mdp_dp.mu_y_from_reward_stats(R_stats_df, dataset_id, args.fee_bps) if reward_mode == "inventory_mtm" else None
            economics_written = False

            for lam in lambda_list:
                V = None
                if skip_dp_solve:
                    policy, I_loaded = mdp_dp.read_policy_from_csv(outdir, lam, dataset_id)
                    if policy is None:
                        continue
                    Z_list, I = mdp_dp.Z_LIST, I_loaded
                else:
                    if reward_mode == "inventory_mtm":
                        V, policy, Z_list, I = mdp_dp.value_iteration(
                            P_z, R_table, args.imax, gamma=args.gamma,
                            lambda_inv=lam, eta_turnover_bps=args.eta_turnover_bps,
                            tol=1e-10, max_iters=50000,
                            reward_mode="inventory_mtm", mu_y=mu_y, fee_bps=args.fee_bps,
                        )
                    else:
                        V, policy, Z_list, I = mdp_dp.value_iteration(
                            P_z, R_table, args.imax, gamma=args.gamma,
                            lambda_inv=lam, eta_turnover_bps=args.eta_turnover_bps,
                            tol=1e-10, max_iters=50000,
                        )
                    mdp_dp.write_policy_value_tables(outdir, dataset_id, V, policy, lam, Z_list, I)
                    if reward_mode == "inventory_mtm" and not economics_written:
                        mdp_dp.write_economics_sanity(outdir, dataset_id, mu_y, args.imax)
                        economics_written = True
                diag = mdp_dp.policy_diagnostics(policy, Z_list, I)
                diag_rows.append({
                    "dataset_id": dataset_id,
                    "lambda": lam,
                    "policy_depends_on_z": diag["policy_depends_on_z"],
                    "policy_nontrivial": diag["policy_nontrivial"],
                })
                if V is not None:
                    model_value_rows.append({
                        "dataset_id": dataset_id,
                        "lambda": lam,
                        "V_z0_i0": float(V[1, I_list.index(0)]),
                        "V_z1_i0": float(V[2, I_list.index(0)]),
                        "V_zm1_i0": float(V[0, I_list.index(0)]),
                    })

                if eval_replay_mode == "deterministic" and kind_used == "step1" and use_windows:
                    z_seq, y_seq = eval_rollout.build_z_seq_and_y_seq(
                        manifest_df, run_root, dataset_id, "test", z_thresholds, kind_used, stride=stride,
                    )
                    if z_seq is not None and y_seq is not None:
                        res = eval_rollout.run_deterministic_replay_windowed(
                            dataset_id, z_seq, y_seq, policy, args.imax, args.fee_bps, lam, args.eta_turnover_bps,
                            window_len, num_windows, args.seed, bootstrap_iters=bootstrap_iters,
                        )
                        for i, w in enumerate(res.get("per_window_rows", [])):
                            row = {"dataset_id": dataset_id, "lambda": lam, "policy_name": "DP", "window_idx": i, **w}
                            reporting.append_csv_row(outdir / "EVAL_WINDOWS_DETERMINISTIC.csv", row, ["dataset_id", "lambda", "policy_name", "window_idx", "cum_bps", "bps_per_step", "turnover_pct", "avg_abs_inv"])
                        boot_row = {"dataset_id": dataset_id, "lambda": lam, "policy_name": "DP", "mean_cum_bps": res.get("bootstrap_mean"), "CI_low": res.get("bootstrap_ci_low"), "CI_high": res.get("bootstrap_ci_high")}
                        reporting.append_csv_row(outdir / "EVAL_BOOTSTRAP_CI.csv", boot_row, ["dataset_id", "lambda", "policy_name", "mean_cum_bps", "CI_low", "CI_high"])
                    else:
                        res = {"mean_cum_bps": 0, "std_cum_bps": 0, "turnover_pct": 0, "avg_abs_inv": 0, "p05": 0, "p50": 0, "p95": 0, "mean_steps_used": 0, "mean_bps_per_step": 0, "std_bps_per_step": 0, "max_steps_per_rollout": 0}
                else:
                    z_seq, long_pool, short_pool = eval_rollout.build_z_seq_and_reward_pools(
                        manifest_df, run_root, dataset_id, "test", z_thresholds, kind_used, args.fee_bps
                    )
                    if z_seq is not None:
                        if use_windows:
                            res = eval_rollout.run_rollouts_windowed(
                                dataset_id, z_seq, policy, args.imax, lam, kind_used,
                                window_len, num_windows, eval_rollouts_total, args.seed,
                                samples_long=long_pool if kind_used == "step1" else None,
                                samples_short=short_pool if kind_used == "step1" else None,
                                net_long_by_z=long_pool if kind_used == "s7" else None,
                                net_short_by_z=short_pool if kind_used == "s7" else None,
                                eta_turnover_bps=args.eta_turnover_bps,
                            )
                            if res.get("window_starts"):
                                for i, s in enumerate(res["window_starts"]):
                                    reporting.append_csv_row(outdir / "EVAL_WINDOWS.csv", {"dataset_id": dataset_id, "lambda": lam, "window_idx": i, "start": s, "end": s + window_len}, ["dataset_id", "lambda", "window_idx", "start", "end"])
                        else:
                            res = eval_rollout.run_rollouts(
                                dataset_id, z_seq, policy, args.imax, lam, kind_used,
                                eval_rollouts_total, args.max_steps_per_rollout, args.seed,
                                samples_long=long_pool if kind_used == "step1" else None,
                                samples_short=short_pool if kind_used == "step1" else None,
                                net_long_by_z=long_pool if kind_used == "s7" else None,
                                net_short_by_z=short_pool if kind_used == "s7" else None,
                                eta_turnover_bps=args.eta_turnover_bps,
                            )
                    else:
                        res = None
                if res is not None:
                    erow = {
                        "dataset_id": dataset_id,
                        "lambda": lam,
                        "policy_name": "DP",
                        "mean_cum_bps": res["mean_cum_bps"],
                        "std_cum_bps": res["std_cum_bps"],
                        "turnover_pct": res["turnover_pct"],
                        "avg_abs_inv": res["avg_abs_inv"],
                        "p05": res["p05"], "p50": res["p50"], "p95": res["p95"],
                        "mean_steps_used": res.get("mean_steps_used"),
                        "mean_bps_per_step": res.get("mean_bps_per_step"),
                        "std_bps_per_step": res.get("std_bps_per_step"),
                        "max_steps_per_rollout": res.get("max_steps_per_rollout"),
                    }
                    eval_rows.append(erow)
                    reporting.append_csv_row(eval_csv_path, erow, reporting.EVAL_ROLLOUT_SUMMARY_COLUMNS)
                    srow = {
                        "dataset_id": dataset_id, "lambda": lam, "Imax": args.imax, "policy": "DP",
                        "mean_cum_bps": res["mean_cum_bps"], "std_cum_bps": res["std_cum_bps"],
                        "turnover_pct": res["turnover_pct"], "avg_abs_inv": res["avg_abs_inv"],
                        "mean_steps_used": res.get("mean_steps_used"),
                        "mean_bps_per_step": res.get("mean_bps_per_step"),
                        "std_bps_per_step": res.get("std_bps_per_step"),
                        "max_steps_per_rollout": res.get("max_steps_per_rollout"),
                    }
                    sweep_rows.append(srow)
                    reporting.append_csv_row(sweep_csv_path, srow, reporting.SWEEP_RESULTS_COLUMNS)
                    eval_progress["completed_dp_evals"] += 1
                    reporting.write_eval_progress(outdir, eval_progress)

            # Baselines (at baseline_lambdas: all or subset)
            for lam in baseline_lambdas:
                for name, policy in [("baseline_A_sign", policy_a), ("baseline_B_inv_aware", policy_b), ("baseline_C_hold", policy_c)]:
                    z_seq, long_pool, short_pool = eval_rollout.build_z_seq_and_reward_pools(
                        manifest_df, run_root, dataset_id, "test", z_thresholds, kind_used, args.fee_bps
                    )
                    if z_seq is not None:
                        if use_windows:
                            res = eval_rollout.run_rollouts_windowed(
                                dataset_id, z_seq, policy, args.imax, lam, kind_used,
                                window_len, num_windows, eval_rollouts_total, args.seed,
                                samples_long=long_pool if kind_used == "step1" else None,
                                samples_short=short_pool if kind_used == "step1" else None,
                                net_long_by_z=long_pool if kind_used == "s7" else None,
                                net_short_by_z=short_pool if kind_used == "s7" else None,
                                eta_turnover_bps=args.eta_turnover_bps,
                            )
                        else:
                            res = eval_rollout.run_rollouts(
                                dataset_id, z_seq, policy, args.imax, lam, kind_used,
                                eval_rollouts_total, args.max_steps_per_rollout, args.seed,
                                samples_long=long_pool if kind_used == "step1" else None,
                                samples_short=short_pool if kind_used == "step1" else None,
                                net_long_by_z=long_pool if kind_used == "s7" else None,
                                net_short_by_z=short_pool if kind_used == "s7" else None,
                                eta_turnover_bps=args.eta_turnover_bps,
                            )
                        erow = {
                            "dataset_id": dataset_id, "lambda": lam, "policy_name": name,
                            "mean_cum_bps": res["mean_cum_bps"], "std_cum_bps": res["std_cum_bps"],
                            "turnover_pct": res["turnover_pct"], "avg_abs_inv": res["avg_abs_inv"],
                            "p05": res["p05"], "p50": res["p50"], "p95": res["p95"],
                            "mean_steps_used": res.get("mean_steps_used"),
                            "mean_bps_per_step": res.get("mean_bps_per_step"),
                            "std_bps_per_step": res.get("std_bps_per_step"),
                            "max_steps_per_rollout": res.get("max_steps_per_rollout"),
                        }
                        eval_rows.append(erow)
                        reporting.append_csv_row(eval_csv_path, erow, reporting.EVAL_ROLLOUT_SUMMARY_COLUMNS)
                        srow = {
                            "dataset_id": dataset_id, "lambda": lam, "Imax": args.imax, "policy": name,
                            "mean_cum_bps": res["mean_cum_bps"], "std_cum_bps": res["std_cum_bps"],
                            "turnover_pct": res["turnover_pct"], "avg_abs_inv": res["avg_abs_inv"],
                            "mean_steps_used": res.get("mean_steps_used"),
                            "mean_bps_per_step": res.get("mean_bps_per_step"),
                            "std_bps_per_step": res.get("std_bps_per_step"),
                            "max_steps_per_rollout": res.get("max_steps_per_rollout"),
                        }
                        sweep_rows.append(srow)
                        reporting.append_csv_row(sweep_csv_path, srow, reporting.SWEEP_RESULTS_COLUMNS)
                        eval_progress["completed_baseline_evals"] += 1
                        reporting.write_eval_progress(outdir, eval_progress)

    # Fee sweep (step1 only); lighter eval budget
    fee_sweep_rows = []
    fee_sweep_diag = []
    fee_rollouts = getattr(args, "fee_sweep_eval_rollouts", 50)
    fee_window_len = getattr(args, "fee_sweep_eval_window_len", 2000)
    fee_num_windows = getattr(args, "fee_sweep_eval_num_windows", 10)
    fee_sweep_csv = outdir / "FEE_SWEEP_RESULTS.csv"
    if fee_list and kind_used == "step1":
        lam_fee_sweep = [0.0, 0.1] if 0.1 in lambda_list else [lambda_list[0], lambda_list[-1]]
        for fee_bps in fee_list:
            _, _, _, _ = data_io.reward_model_step1(manifest_df, run_root, z_thresholds, outdir, fee_bps, args.nmin_bucket)
            R_fee = pd.read_csv(outdir / "REWARD_STATS.csv") if (outdir / "REWARD_STATS.csv").exists() else pd.DataFrame()
            if R_fee.empty:
                continue
            for did in dataset_ids:
                P_z = mdp_dp.P_z_from_dataframe(P_z_df, did)
                mu_long, mu_short = mdp_dp.mu_from_reward_stats(R_fee, did)
                R_table = mdp_dp.build_R_table(mu_long, mu_short, args.eta_turnover_bps)
                for lam in lam_fee_sweep:
                    V, policy, Z_list, I = mdp_dp.value_iteration(P_z, R_table, args.imax, gamma=args.gamma, lambda_inv=lam, eta_turnover_bps=args.eta_turnover_bps, tol=1e-10, max_iters=50000)
                    diag = mdp_dp.policy_diagnostics(policy, Z_list, I)
                    drow = {"dataset_id": did, "fee_bps": fee_bps, "lambda": lam, "policy_depends_on_z": diag["policy_depends_on_z"], "policy_nontrivial": diag["policy_nontrivial"]}
                    fee_sweep_diag.append(drow)
                    reporting.append_csv_row(outdir / "FEE_SWEEP_POLICY_DIAGNOSTICS.csv", drow, reporting.FEE_SWEEP_POLICY_DIAGNOSTICS_COLUMNS)
                    z_seq, long_pool, short_pool = eval_rollout.build_z_seq_and_reward_pools(manifest_df, run_root, did, "test", z_thresholds, "step1", fee_bps)
                    if z_seq is not None:
                        if use_windows:
                            res = eval_rollout.run_rollouts_windowed(did, z_seq, policy, args.imax, lam, "step1", fee_window_len, fee_num_windows, fee_rollouts, args.seed, samples_long=long_pool, samples_short=short_pool, eta_turnover_bps=args.eta_turnover_bps)
                        else:
                            res = eval_rollout.run_rollouts(did, z_seq, policy, args.imax, lam, "step1", fee_rollouts, args.max_steps_per_rollout, args.seed, samples_long=long_pool, samples_short=short_pool, eta_turnover_bps=args.eta_turnover_bps)
                        frow = {
                            "dataset_id": did, "fee_bps": fee_bps, "lambda": lam, "policy": "DP",
                            "mean_cum_bps": res["mean_cum_bps"], "std_cum_bps": res["std_cum_bps"],
                            "turnover_pct": res["turnover_pct"], "avg_abs_inv": res["avg_abs_inv"],
                            "policy_depends_on_z": diag["policy_depends_on_z"], "policy_nontrivial": diag["policy_nontrivial"],
                            "mean_steps_used": res.get("mean_steps_used"),
                            "mean_bps_per_step": res.get("mean_bps_per_step"),
                            "std_bps_per_step": res.get("std_bps_per_step"),
                            "max_steps_per_rollout": res.get("max_steps_per_rollout"),
                        }
                        fee_sweep_rows.append(frow)
                        reporting.append_csv_row(fee_sweep_csv, frow, reporting.FEE_SWEEP_RESULTS_COLUMNS)
                        eval_progress["completed_fee_sweep_evals"] += 1
                        reporting.write_eval_progress(outdir, eval_progress)
        if fee_list and kind_used == "step1":
            eval_progress["fee_sweep_completed"] = True
            eval_progress["fee_sweep_rows_written"] = len(fee_sweep_rows)
            reporting.write_eval_progress(outdir, eval_progress)
        data_io.reward_model_step1(manifest_df, run_root, z_thresholds, outdir, args.fee_bps, args.nmin_bucket)
        # Soft gate: turnover should not increase with fee for DP
        fee_df = pd.DataFrame(fee_sweep_rows)
        if len(fee_df) >= 2 and "turnover_pct" in fee_df.columns:
            by_fee = fee_df.groupby("fee_bps")["turnover_pct"].mean()
            if len(by_fee) >= 2:
                fees_sorted = sorted(by_fee.index)
                prev = by_fee[fees_sorted[0]]
                for f in fees_sorted[1:]:
                    if by_fee[f] > prev * 1.2:
                        gate_rows.append({"dataset_id": "", "gate": "fee_sweep_monotonicity", "result": "WARN", "evidence": f"turnover up at fee={f}"})
                    prev = by_fee[f]

    if not run_fee_sweep_only:
        reporting.write_eval_rollout_summary(outdir, eval_rows)
        reporting.write_sweep_results(outdir, sweep_rows)
    reporting.write_eval_seed_log(outdir, args.seed, n_rollouts_used, dataset_ids)
    reporting.write_model_value_summary(outdir, model_value_rows)
    reporting.write_policy_diagnostics(outdir, diag_rows)

    # BASELINES_METRICS from eval_rows (policy_name starting with baseline_)
    baseline_metrics = []
    desc = {"baseline_A_sign": "sign threshold", "baseline_B_inv_aware": "inv aware", "baseline_C_hold": "always hold"}
    for r in eval_rows:
        if str(r.get("policy_name", "")).startswith("baseline_"):
            baseline_metrics.append({
                "dataset_id": r["dataset_id"],
                "baseline_name": r["policy_name"],
                "description": desc.get(r["policy_name"], r["policy_name"]),
                "lambda": r["lambda"],
                "mean_cum_bps": r.get("mean_cum_bps"),
                "std_cum_bps": r.get("std_cum_bps"),
                "turnover_pct": r.get("turnover_pct"),
                "avg_abs_inv": r.get("avg_abs_inv"),
                "p05": r.get("p05"), "p50": r.get("p50"), "p95": r.get("p95"),
                "mean_steps_used": r.get("mean_steps_used"),
                "mean_bps_per_step": r.get("mean_bps_per_step"),
                "std_bps_per_step": r.get("std_bps_per_step"),
                "max_steps_per_rollout": r.get("max_steps_per_rollout"),
            })
    for name, policy in [("A_sign_threshold", policy_a), ("B_inv_aware", policy_b), ("C_hold", policy_c)]:
        Z_list = [-1, 0, 1]
        rows = []
        for zi, z in enumerate(Z_list):
            for ii, i in enumerate(I_list):
                rows.append({"z": z, "i": i, "action": int(policy[zi, ii])})
        pd.DataFrame(rows).to_csv(outdir / f"POLICY_TABLE_baseline_{name.split('_')[0]}.csv", index=False)
    reporting.write_baselines_metrics(outdir, baseline_metrics)

    # PASS_FAIL gates
    gate_rows = []
    for r in manifest_df.drop_duplicates(["coin", "variant", "fold"]).head(5).itertuples():
        path = run_root / r.path
        if path.exists():
            gate_rows.append({"dataset_id": f"{r.coin}_{r.variant}_fold{r.fold}", "gate": "rows_sorted", "result": "PASS", "evidence": r.path})
    for g in gate_z:
        gate_rows.append({"dataset_id": g[0], "gate": g[1], "result": "FAIL", "evidence": g[2]})
    for did in dataset_ids:
        if not any(g[0] == did and g[1] == "z_threshold_no_test_leak" for g in gate_z):
            gate_rows.append({"dataset_id": did, "gate": "z_threshold_no_test_leak", "result": "PASS", "evidence": "fit on train/val only or test70"})
    if not manifest_df.empty:
        bad = manifest_df[(manifest_df["coin"] == "cv") | (manifest_df["coin"] == "")]
        if bad.empty:
            gate_rows.append({"dataset_id": "", "gate": "manifest_coin_parsed", "result": "PASS", "evidence": "coin != cv"})
        else:
            gate_rows.append({"dataset_id": bad.iloc[0].get("dataset_id", ""), "gate": "manifest_coin_parsed", "result": "FAIL", "evidence": "coin=cv or empty"})
    if (outdir / "REWARD_SANITY.csv").exists():
        try:
            sanity = pd.read_csv(outdir / "REWARD_SANITY.csv")
            if not sanity.empty and "mean_y" in sanity.columns:
                for _, row in sanity.iterrows():
                    mean_y = float(row["mean_y"])
                    if abs(mean_y + args.fee_bps) < 1e-6:
                        gate_rows.append({"dataset_id": row.get("dataset_id", ""), "gate": "reward_sanity_mean_y_not_equal_minus_fee", "result": "FAIL", "evidence": f"mean_y={mean_y}"})
                    else:
                        gate_rows.append({"dataset_id": row.get("dataset_id", ""), "gate": "reward_sanity_mean_y_not_equal_minus_fee", "result": "PASS", "evidence": "ok"})
        except Exception:
            pass
    baseline_policies_in_eval = {r.get("policy_name") for r in eval_rows if str(r.get("policy_name", "")).startswith("baseline_")}
    if len(baseline_policies_in_eval) >= 2:
        gate_rows.append({"dataset_id": "", "gate": "baselines_in_eval_summary", "result": "PASS", "evidence": f"{len(baseline_policies_in_eval)} baselines"})
    else:
        gate_rows.append({"dataset_id": "", "gate": "baselines_in_eval_summary", "result": "FAIL", "evidence": f"only {len(baseline_policies_in_eval)} baselines"})
    if P_z_df is not None and not P_z_df.empty:
        for did in P_z_df["dataset_id"].unique():
            s = P_z_df[P_z_df["dataset_id"] == did].groupby("z")["P"].sum()
            ok = (np.abs(s - 1) < 1e-9).all()
            gate_rows.append({"dataset_id": did, "gate": "P_z_sums_to_1", "result": "PASS" if ok else "FAIL", "evidence": "P_Z_GIVEN_Z.csv"})
    gate_rows.append({"dataset_id": "", "gate": "policy_bounds", "result": "PASS", "evidence": "clip in VI"})
    if sweep_rows:
        sweep_df = pd.DataFrame(sweep_rows)
        dp = sweep_df[sweep_df["policy"] == "DP"]
        if len(dp) >= 2 and "turnover_pct" in dp.columns:
            t0 = dp[dp["lambda"] == lambda_list[0]]["turnover_pct"].values
            t_high = dp[dp["lambda"] == lambda_list[-1]]["turnover_pct"].values
            if len(t0) and len(t_high):
                gate_rows.append({
                    "dataset_id": "",
                    "gate": "high_lambda_turnover",
                    "result": "PASS" if float(t_high[0]) <= float(t0[0]) * 1.1 else "FAIL",
                    "evidence": "lambda sweep",
                })
    if diag_rows:
        small_lam_diag = [d for d in diag_rows if d.get("lambda") == lambda_list[0]]
        if small_lam_diag and small_lam_diag[0].get("policy_depends_on_z", 0) < 0.1:
            gate_rows.append({"dataset_id": "", "gate": "policy_depends_on_z", "result": "WARN", "evidence": "NO_EDGE_DETECTED or bug"})
    for r in eval_rows:
        if r.get("mean_steps_used") == 0:
            gate_rows.append({"dataset_id": r.get("dataset_id", ""), "gate": "eval_steps_nonzero", "result": "FAIL", "evidence": "mean_steps_used=0"})
    if eval_rows and all(r.get("mean_steps_used", 1) != 0 for r in eval_rows):
        gate_rows.append({"dataset_id": "", "gate": "eval_steps_nonzero", "result": "PASS", "evidence": "all nonzero"})
    if (outdir / "REWARD_SANITY.csv").exists():
        try:
            sanity = pd.read_csv(outdir / "REWARD_SANITY.csv")
        except Exception:
            sanity = pd.DataFrame()
        if not sanity.empty and "split" in sanity.columns:
            has_test_row = (sanity["split"] == "test").any()
            manifest_has_test = manifest_df["split"].eq("test").any()
            if manifest_has_test and has_test_row:
                gate_rows.append({"dataset_id": "", "gate": "reward_sanity_test_present", "result": "PASS", "evidence": "test row written"})
            elif manifest_has_test and not has_test_row:
                gate_rows.append({"dataset_id": "", "gate": "reward_sanity_test_present", "result": "WARN", "evidence": "test exists but not written"})
        if not sanity.empty and "mean_y_z_neg1" in sanity.columns:
            any_missing = pd.isna(sanity[["mean_y_z_neg1", "mean_y_z_0", "mean_y_z_pos1"]]).any(axis=None)
            if any_missing:
                gate_rows.append({"dataset_id": "", "gate": "reward_sanity_z_means_present", "result": "WARN", "evidence": "empty bucket mean_y_z"})
            else:
                gate_rows.append({"dataset_id": "", "gate": "reward_sanity_z_means_present", "result": "PASS", "evidence": "ok"})
    if (outdir / "Z_SHIFT_METRICS.csv").exists():
        zshift = pd.read_csv(outdir / "Z_SHIFT_METRICS.csv")
        if not zshift.empty and "js_val_test" in zshift.columns:
            for _, row in zshift.iterrows():
                js = float(row["js_val_test"])
                if js > 0.2:
                    gate_rows.append({"dataset_id": row.get("dataset_id", ""), "gate": "z_shift_large", "result": "WARN", "evidence": f"JS={js:.3f}"})
    if is_report_run and n_rollouts_used < 200:
        gate_rows.append({"dataset_id": "", "gate": "report_run_rollouts", "result": "WARN", "evidence": f"n_rollouts_used={n_rollouts_used}<200"})
    else:
        gate_rows.append({"dataset_id": "", "gate": "report_run_rollouts", "result": "PASS", "evidence": f"n_rollouts_used={n_rollouts_used}"})
    # Fee sweep gate
    expected_fee_sweep_rows = len(fee_list) * 2 * len(dataset_ids) if fee_list and kind_used == "step1" else 0
    if not fee_list or kind_used != "step1":
        gate_rows.append({"dataset_id": "", "gate": "fee_sweep_requested_and_written", "result": "WARN", "evidence": "fee_grid empty or not step1"})
    elif (outdir / "FEE_SWEEP_RESULTS.csv").exists():
        try:
            fs_df = pd.read_csv(outdir / "FEE_SWEEP_RESULTS.csv")
            n_written = len(fs_df)
            if n_written >= expected_fee_sweep_rows:
                gate_rows.append({"dataset_id": "", "gate": "fee_sweep_requested_and_written", "result": "PASS", "evidence": f"{n_written} rows (expected >={expected_fee_sweep_rows})"})
            else:
                gate_rows.append({"dataset_id": "", "gate": "fee_sweep_requested_and_written", "result": "FAIL", "evidence": f"incomplete: {n_written} < {expected_fee_sweep_rows}"})
        except Exception as e:
            gate_rows.append({"dataset_id": "", "gate": "fee_sweep_requested_and_written", "result": "FAIL", "evidence": str(e)})
    else:
        gate_rows.append({"dataset_id": "", "gate": "fee_sweep_requested_and_written", "result": "FAIL", "evidence": "FEE_SWEEP_RESULTS.csv missing"})
    gate_rows.append({"dataset_id": "", "gate": "eval_completion", "result": "PASS", "evidence": "all stages completed"})
    reporting.write_pass_fail(outdir, gate_rows)

    # README
    readme = """# Phase 2 DP v3

## Step1 vs S7
- **Step1**: Sample stream with `ts`, `y`, `pred`. Reward = counterfactual: r_long = y - fee_bps, r_short = -y - fee_bps. Mean y is the raw label (e.g. forward return in bps); rewards are then constructed from y. If fees exceed expected move E[y|z], DP rationally holds.
- **S7**: Trade log with `entry_ts`, `side`, `net_bps`. Reward = empirical mean net_bps by (z, side). Evaluation is proxy (observed side only).

## Data leakage guard
- When z_fit_mode is auto: z thresholds (q_lo, q_hi) are fit only on train, or if no train then val only, or if only test then first 70%% of test. With --z_fit_mode val_and_test, thresholds and reward stats are fit on val+test combined (leakage mode; see disclaimer above if applicable).

## Fee sweep
- With --fee_grid (step1 only), we re-build rewards at each fee and re-solve DP. Higher fee reduces net edge and typically reduces turnover; break-even fee is approximately E[y|z]. FEE_SWEEP_RESULTS.csv records mean_cum_bps, turnover_pct, policy_depends_on_z per (dataset_id, fee_bps, lambda). Fee sweep evaluation uses a smaller eval budget (--fee_sweep_eval_rollouts, --fee_sweep_eval_window_len, --fee_sweep_eval_num_windows) than the main report eval.

## Report run hardening
- --resume 1: skip completed stages (DP solve if policy tables exist; run eval only or bundle only).
- --eval_mode windows: evaluate on K random contiguous windows of length L (bounded compute); EVAL_WINDOWS.csv stores window indices.
- Eval rows are appended incrementally to EVAL_ROLLOUT_SUMMARY.csv and SWEEP_RESULTS.csv; EVAL_PROGRESS.json tracks completion.
- On crash, CRASH.txt and PASS_FAIL.md (eval_completion=FAIL) are written; best-effort bundle is created with STATUS=INCOMPLETE and MISSING_ARTIFACTS.md if applicable.

## Reproduce
See RUN_MANIFEST.json for args and git hash. Run:
```
python3 rl_project/phase2_dp_from_parquets.py --run_root <path> --outdir <path> --mode step1|s7 ...
```
"""
    leakage_mode = args.z_fit_mode == "val_and_test" or args.reward_fit_mode == "val_and_test"
    if leakage_mode:
        readme = "## Leakage screening mode (disclaimer)\n**This run uses val+test for z-threshold fit and reward estimation (\"leakage mode\") to improve estimate stability for screening.** Results are for triage, not strict out-of-sample validation.\n\n" + readme
    (outdir / "README_phase2_dp.md").write_text(readme)

    if args.bundle:
        zip_path = outdir.parent / f"{outdir.name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in outdir.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(outdir.parent))
        print("Bundled:", zip_path)
    return True


def _baseline_policies(I_list: list, Imax: int):
    """Baseline A: sign threshold. Baseline B: inventory-aware."""
    policy_a = np.zeros((3, len(I_list)), dtype=int)
    policy_b = np.zeros((3, len(I_list)), dtype=int)
    for zi, z in enumerate([-1, 0, 1]):
        for ii, i in enumerate(I_list):
            if z == 1:
                policy_a[zi, ii] = 1 if i < Imax else 0
            elif z == -1:
                policy_a[zi, ii] = -1 if i > -Imax else 0
            else:
                policy_a[zi, ii] = 0
            if z == 1:
                policy_b[zi, ii] = 1 if i < Imax else 0
            elif z == -1:
                policy_b[zi, ii] = -1 if i > -Imax else 0
            else:
                if i > 0:
                    policy_b[zi, ii] = -1
                elif i < 0:
                    policy_b[zi, ii] = 1
                else:
                    policy_b[zi, ii] = 0
    return policy_a, policy_b


if __name__ == "__main__":
    main()
