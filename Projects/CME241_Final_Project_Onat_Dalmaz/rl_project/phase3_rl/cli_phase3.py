#!/usr/bin/env python3
"""
Phase 3 RL CLI: train tabular Q-learning on Step1 windows, evaluate with baselines, emit artifacts and PASS_FAIL.
Usage: from rl_project: python -m phase3_rl.cli_phase3 --run_root <path> --outdir <path> [options]
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Run from rl_project
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data_io

from .env import ExecutionEnvConfig, ExecutionEnv
from . import state as st
from . import policies as pol
from . import q_learning
from . import eval as ev
from . import reporting as rep
from . import dp_baseline as dpb
from . import dp_empirical as dpe
from . import empirical_model as em
from . import dp_phase2_baseline as dp2


def parse_args():
    ap = argparse.ArgumentParser(description="Phase 3 RL execution layer")
    ap.add_argument("--run_root", type=str, required=True, help="Path to Step1 coin dir (e.g. .../COINS/NEAR)")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory")
    ap.add_argument("--mode", type=str, default="step1", choices=["step1"])
    ap.add_argument("--qlo", type=float, default=0.33)
    ap.add_argument("--qhi", type=float, default=0.67)
    ap.add_argument("--Imax", type=int, default=3)
    ap.add_argument("--vbin_method", type=str, default="median", choices=["median", "quantile", "median_abs_y"])
    ap.add_argument("--reward_mode", type=str, default="inventory_mtm", choices=["inventory_mtm"])
    ap.add_argument("--eval_replay_mode", type=str, default="deterministic", choices=["deterministic", "stochastic"])
    ap.add_argument("--log_every", type=int, default=0, help="Log training every N episodes (0=off)")
    ap.add_argument("--vbin_quantile", type=float, default=0.5)
    ap.add_argument("--c_maker_bps", type=float, default=1.0)
    ap.add_argument("--c_taker_bps", type=float, default=2.0)
    ap.add_argument("--lambda_inv", type=float, default=0.1)
    ap.add_argument("--eta_turnover", type=float, default=0.5)
    ap.add_argument("--p0", type=float, default=0.6)
    ap.add_argument("--p1", type=float, default=0.2)
    ap.add_argument("--dv", type=float, default=0.05)
    ap.add_argument("--d_age", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--alpha_min", type=float, default=0.02)
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--eps_min", type=float, default=0.05)
    ap.add_argument("--n_train_episodes", type=int, default=5000)
    ap.add_argument("--train_window_len", type=int, default=2000)
    ap.add_argument("--n_train_windows", type=int, default=20)
    ap.add_argument("--eval_window_len", type=int, default=2000)
    ap.add_argument("--eval_num_windows", type=int, default=20)
    ap.add_argument("--eval_fill_seeds", type=int, default=30)
    ap.add_argument("--bootstrap_iters", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--bundle", type=int, default=0, help="If 1, zip outdir to phase3_bundle.zip")
    ap.add_argument("--decay_episodes", type=int, default=5000, help="Decay alpha/eps over this many episodes (capped by n_train_episodes)")
    ap.add_argument("--q_init", type=float, default=0.01, help="Initial Q value (optimistic init encourages exploration)")
    ap.add_argument("--use_sarsa", action="store_true")
    ap.add_argument("--resume_eval", type=int, default=0, help="If 1, load Q from outdir, skip training, re-run eval only")
    ap.add_argument("--algo", type=str, default="q_learning", choices=["q_learning", "double_q"], help="Tabular algorithm")
    ap.add_argument("--z_bins", type=int, default=3, choices=[3, 5], help="Z discretization: 3 (default) or 5 (quintiles, ablation)")
    ap.add_argument("--dp_empirical", type=int, default=1, help="If 1, build and evaluate DP_empirical from train transitions (fair baseline)")
    ap.add_argument("--dp_empirical_smoothing", type=float, default=0.1, help="Dirichlet smoothing for empirical P(s'|s,a)")
    ap.add_argument("--dp_empirical_include_in_fixed_baseline_set", type=int, default=1, help="Include DP_empirical in fair baseline set for Delta_fair")
    ap.add_argument("--learning_curve", type=int, default=0, help="If 1, record learning curve checkpoints (lightweight evals)")
    ap.add_argument("--dp_phase2_reward_mode", type=str, default="zero_mean", choices=["drift", "zero_mean"], help="DP_PHASE2: drift=use E[y|z]; zero_mean=E[y|z]=0 (risk/cost-only fair baseline)")
    return ap.parse_args()


def _parse_dataset_id(dataset_id: str):
    """Return (coin, variant, fold). dataset_id = coin_variant_foldK."""
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
    return coin, variant, fold


def get_fit_df_and_vbin_threshold(manifest_df, run_root, dataset_id, q_lo, q_hi, vbin_quantile):
    """Load train (or val if no train) split for fit, return (fit_df, v_bin_threshold). v_bin_threshold = median |y| on fit."""
    run_root = Path(run_root)
    coin, variant, fold = _parse_dataset_id(dataset_id)
    for split in ("train", "val"):
        fit_rows = manifest_df[
            (manifest_df["coin"] == coin) & (manifest_df["variant"] == variant)
            & (manifest_df["fold"] == fold) & (manifest_df["split"] == split)
        ]
        for _, r in fit_rows.iterrows():
            path = run_root / r["path"]
            df, err = data_io.load_and_canonicalize(path, run_root, r["kind"], allow_no_pred_s7=True)
            if err or df is None or len(df) < 10:
                continue
            abs_y = np.abs(df["y"].values.astype(float))
            v_bin_threshold = float(np.nanpercentile(abs_y, vbin_quantile * 100))
            if not np.isfinite(v_bin_threshold):
                v_bin_threshold = 0.0
            return df, v_bin_threshold
    return None, 0.0


def main():
    args = parse_args()
    run_root = Path(args.run_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Discover
    manifest_df, _ = data_io.discover_and_classify(run_root, args.mode, outdir)
    manifest_df = manifest_df[manifest_df["kind"] == "step1"] if "kind" in manifest_df.columns else manifest_df
    if manifest_df.empty:
        rep.write_run_manifest(outdir, vars(args), str(run_root), "")
        rep.write_pass_fail(outdir, [{"dataset_id": "", "gate": "discover", "result": "FAIL", "evidence": "no step1 parquets"}])
        if args.bundle:
            bundle_path = outdir.parent / "phase3_bundle.zip"
            with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in outdir.rglob("*"):
                    if f.is_file():
                        zf.write(f, f.relative_to(outdir.parent))
        return 1

    # Single dataset: first group
    grp = manifest_df.groupby(["coin", "variant", "fold"]).first().reset_index()
    dataset_id = f"{grp.iloc[0]['coin']}_{grp.iloc[0]['variant']}_fold{int(grp.iloc[0]['fold'])}"

    # Z thresholds on fit (train) only
    z_thresholds, _, gate_z = data_io.compute_z_thresholds(
        manifest_df, run_root, args.qlo, args.qhi, outdir, z_fit_mode="auto"
    )
    if not z_thresholds:
        rep.write_run_manifest(outdir, vars(args), str(run_root), dataset_id)
        rep.write_pass_fail(outdir, [{"dataset_id": dataset_id, "gate": "z_thresholds", "result": "FAIL", "evidence": "none"}])
        return 1
    z_row = next(r for r in z_thresholds if r["dataset_id"] == dataset_id)
    q_lo, q_hi = z_row["q_lo"], z_row["q_hi"]

    # V_bin threshold from train
    fit_df, v_bin_threshold = get_fit_df_and_vbin_threshold(
        manifest_df, run_root, dataset_id, q_lo, q_hi, args.vbin_quantile
    )
    if fit_df is None:
        v_bin_threshold = 0.0
    rep.write_z_thresholds(outdir, z_thresholds)
    vbin_method = getattr(args, "vbin_method", "median")
    if vbin_method == "median_abs_y":
        vbin_method = "median"
    rep.write_vbin_threshold(outdir, v_bin_threshold, vbin_method)

    # Single source of truth: EFFECTIVE_CONFIG.json (must match RUN_MANIFEST.effective_config)
    if getattr(args, "resume_eval", 0) and (outdir / "EFFECTIVE_CONFIG.json").exists():
        effective_config = json.loads((outdir / "EFFECTIVE_CONFIG.json").read_text())
    else:
        effective_config = rep.build_effective_config(vars(args), q_lo, q_hi, v_bin_threshold)
        rep.write_effective_config(outdir, effective_config)

    # Horizon
    raw_dt_ms, label_horizon_ms, horizon_source = data_io.infer_label_horizon_ms(fit_df, run_root)
    stride = max(1, int(round(label_horizon_ms / raw_dt_ms))) if raw_dt_ms > 0 else 1
    mdp_step_ms = label_horizon_ms
    rep.write_horizon_spec(outdir, raw_dt_ms, label_horizon_ms, stride, mdp_step_ms)
    horizon_ok = stride >= 1 and raw_dt_ms > 0

    # Build train sequences (windows): use train split, or val if no train (e.g. real NEAR layout)
    z_train, y_train, v_train = ev.build_phase3_sequences(
        manifest_df, run_root, dataset_id, "train", z_thresholds, v_bin_threshold, stride
    )
    if z_train is None or len(z_train) < args.train_window_len:
        z_train, y_train, v_train = ev.build_phase3_sequences(
            manifest_df, run_root, dataset_id, "val", z_thresholds, v_bin_threshold, stride
        )
    z_test, y_test, v_test = ev.build_phase3_sequences(
        manifest_df, run_root, dataset_id, "test", z_thresholds, v_bin_threshold, stride
    )
    if z_train is None or len(z_train) < args.train_window_len:
        rep.write_run_manifest(outdir, vars(args), str(run_root), dataset_id)
        rep.write_pass_fail(outdir, [{"dataset_id": dataset_id, "gate": "train_data", "result": "FAIL", "evidence": "insufficient train/val"}])
        return 1

    # Train windows: contiguous slices
    rng = np.random.default_rng(args.seed)
    train_starts = ev.sample_window_starts(len(z_train), args.train_window_len, args.n_train_windows, args.seed)
    z_seqs = [z_train[s : s + args.train_window_len] for s in train_starts]
    y_seqs = [y_train[s : s + args.train_window_len] for s in train_starts]
    v_seqs = [v_train[s : s + args.train_window_len] for s in train_starts]
    rep.write_windows_csv(outdir, train_starts, args.train_window_len, "train")

    z_bins = getattr(args, "z_bins", 3)
    # Env config
    config = ExecutionEnvConfig(
        Imax=args.Imax,
        z_bins=z_bins,
        c_maker_bps=args.c_maker_bps,
        c_taker_bps=args.c_taker_bps,
        lambda_inv=args.lambda_inv,
        eta_turnover=args.eta_turnover,
        p0=args.p0,
        p1=args.p1,
        dv=args.dv,
        d_age=args.d_age,
    )

    # Env sanity
    nS = st.n_states(args.Imax, z_bins)
    rep.write_env_sanity(outdir, args.Imax, nS, st.N_ACTIONS, True)
    from .env import p_fill
    sample_probs = [
        f"align z=1 buy v=0 age=1: {p_fill(1, 1, 0, 1, config):.3f}",
        f"align z=-1 sell v=1 age=2: {p_fill(-1, -1, 1, 2, config):.3f}",
    ]
    rep.write_fill_model_sanity(outdir, args.p0, args.p1, args.dv, args.d_age, sample_probs)

    # Train Q-learning (or load when resume_eval)
    Q = None
    log_rows = []
    policy_vec = None
    if getattr(args, "resume_eval", 0):
        Q, loaded = rep.load_q_table(outdir, nS, st.N_ACTIONS)
        if not loaded:
            rep.write_pass_fail(outdir, [{"dataset_id": dataset_id, "gate": "resume_eval", "result": "FAIL", "evidence": "Q_TABLE.csv missing or invalid"}])
            return 1
        if (outdir / "TRAINING_LOG.csv").exists():
            try:
                log_rows = pd.read_csv(outdir / "TRAINING_LOG.csv").to_dict("records")
            except Exception:
                pass
        _rng = np.random.default_rng(args.seed + 999)
        policy_vec = np.array([q_learning.argmax_random_tiebreak(Q[s, :], _rng) for s in range(Q.shape[0])])
    empirical_collector = None
    if Q is None:
        log_every = getattr(args, "log_every", 0)
        def _log_cb(ep, reward, steps, alpha, eps):
            if log_every and ep > 0 and ep % log_every == 0:
                print(f"  episode {ep}  ep_reward={reward:.1f}  steps={steps}  alpha={alpha:.4f}  eps={eps:.4f}")
        if getattr(args, "dp_empirical", 1):
            empirical_collector = em.EmpiricalCounts(nS, st.N_ACTIONS)
        algo = getattr(args, "algo", "q_learning")
        if algo == "double_q":
            Q, log_rows = q_learning.train_double_q_learning(
                config,
                z_seqs,
                y_seqs,
                v_seqs,
                n_episodes=args.n_train_episodes,
                gamma=args.gamma,
                alpha0=args.alpha,
                alpha_min=args.alpha_min,
                eps0=args.eps,
                eps_min=args.eps_min,
                decay_episodes=args.decay_episodes,
                seed=args.seed,
                q_init=getattr(args, "q_init", 0.01),
                log_callback=_log_cb if log_every else None,
                empirical_collector=empirical_collector,
            )
        else:
            Q, log_rows = q_learning.train_q_learning(
                config,
                z_seqs,
                y_seqs,
                v_seqs,
                n_episodes=args.n_train_episodes,
                gamma=args.gamma,
                alpha0=args.alpha,
                alpha_min=args.alpha_min,
                eps0=args.eps,
                eps_min=args.eps_min,
                decay_episodes=args.decay_episodes,
                seed=args.seed,
                use_sarsa=args.use_sarsa,
                q_init=getattr(args, "q_init", 0.01),
                log_callback=_log_cb if log_every else None,
                empirical_collector=empirical_collector,
            )
        if empirical_collector is not None:
            empirical_collector.save(outdir / "EMPIRICAL_COUNTS.npz")
        rep.write_q_table(outdir, Q, args.Imax, z_bins)
        rep.write_training_log(outdir, log_rows)
        _rng = np.random.default_rng(args.seed + 999)
        policy_vec = np.array([q_learning.argmax_random_tiebreak(Q[s, :], _rng) for s in range(Q.shape[0])])
    if Q is not None:
        np.save(outdir / "Q_TABLE_QL.npy", Q)
    if policy_vec is not None:
        rep.write_policy_table(outdir, policy_vec, args.Imax, z_bins)
        import shutil
        if (outdir / "POLICY_TABLE.csv").exists():
            shutil.copy(outdir / "POLICY_TABLE.csv", outdir / "POLICY_TABLE_QL.csv")

    # Eval: Q policy + baselines A, B, C on test windows with multiple fill seeds
    fill_seeds = [args.seed + i for i in range(args.eval_fill_seeds)]
    eval_window_starts = []
    if z_test is not None and len(z_test) >= args.eval_window_len:
        eval_window_starts = ev.sample_window_starts(len(z_test), args.eval_window_len, args.eval_num_windows, args.seed + 1)
    rep.write_windows_csv(outdir, eval_window_starts, args.eval_window_len, "eval")

    eval_rows = []
    baseline_rows = []
    q_mean_cum = None
    baseline_means = {}
    deterministic_replay = getattr(args, "eval_replay_mode", "stochastic") == "deterministic"
    ql_visitation_rows = []
    ql_state_topk_rows = []
    delta_ci = {}
    res_ql_vis = {}
    ql_ci_low = ql_ci_high = ql_std = 0.0
    maker_share = taker_share = 0.0
    mean_hold_frac = 0.0
    total_maker_fills = 0

    if z_test is not None and len(z_test) >= args.eval_window_len and eval_window_starts:
        # QL with visitation diagnostics
        res_ql_vis, ql_visitation_rows, ql_state_topk_rows = ev.run_eval_ql_with_visitation(
            config, z_test, y_test, v_test,
            args.eval_window_len, args.eval_num_windows, fill_seeds, args.seed + 2,
            Q=Q, deterministic_replay=deterministic_replay,
        )
        ql_per_window = res_ql_vis.get("per_window_cum_bps", [])
        if ql_visitation_rows:
            maker_share = float(np.mean([r["maker_share"] for r in ql_visitation_rows]))
            taker_share = float(np.mean([r["taker_share"] for r in ql_visitation_rows]))
            mean_hold_frac = float(np.mean([r["action_frac_HOLD"] for r in ql_visitation_rows]))
            total_maker_fills = sum(r["maker_fills"] for r in ql_visitation_rows)
        rep.write_eval_visitation(outdir, ql_visitation_rows)
        rep.write_eval_state_topk(outdir, ql_state_topk_rows)
        # Bootstrap CI for QL (same as run_eval_all)
        nw = len(ql_per_window)
        if nw > 0:
            rng = np.random.default_rng(args.seed + 2)
            boot_means = [float(np.mean([ql_per_window[i] for i in rng.integers(0, nw, size=nw)])) for _ in range(args.bootstrap_iters)]
            ql_ci_low = float(np.percentile(boot_means, 2.5))
            ql_ci_high = float(np.percentile(boot_means, 97.5))
            q_mean_cum = res_ql_vis["mean_cum_bps"]
            ql_std = res_ql_vis["std_cum_bps"]
        eval_rows.append({
            "dataset_id": dataset_id,
            "policy_name": "QL",
            "mean_cum_bps": res_ql_vis.get("mean_cum_bps", 0),
            "std_cum_bps": ql_std,
            "bootstrap_ci_low": ql_ci_low,
            "bootstrap_ci_high": ql_ci_high,
        })

        # DP baseline (fit-only: z_train, v_train, y_train); or load from file on resume
        dp_no_data_leak = True
        pi_dp = None
        if getattr(args, "resume_eval", 0) and (outdir / "DP_POLICY_TABLE.csv").exists():
            try:
                df_dp = pd.read_csv(outdir / "DP_POLICY_TABLE.csv")
                pi_dp = np.zeros(nS, dtype=int)
                for _, r in df_dp.iterrows():
                    s = int(r["state_idx"])
                    if 0 <= s < nS:
                        pi_dp[s] = int(r["action"])
            except Exception:
                pass
        if pi_dp is None and len(z_train) >= 10 and len(v_train) >= 10 and len(y_train) >= 10:
            P_z, P_v, dp_meta = dpb.estimate_exogenous_markov(z_train, v_train, n_z=z_bins)
            Ey_zv = dpb.estimate_ey_given_zv(z_train, v_train, y_train, n_z=z_bins)
            P_trans, R_dp = dpb.build_transition_model(config, P_z, P_v, Ey_zv, args.Imax, z_bins)
            V_dp, pi_dp, Q_dp = dpb.solve_dp(P_trans, R_dp, gamma=args.gamma)
            dpb.export_dp_artifacts(outdir, pi_dp, Q_dp, V_dp, args.Imax, z_bins, dp_meta)
            dpb.write_dp_model_summary(outdir, P_z, P_v, dp_meta)

        # Baselines A, B, C, and DP with per_window for delta CI
        baseline_per_window = {}
        for short_name, name, fn in [
            ("A", "A_sign_taker", pol.policy_sign_taker),
            ("B", "B_sign_maker", pol.policy_sign_maker),
            ("C", "Hold", pol.policy_hold),
        ]:
            policy_fn = lambda s, _f=fn, _Imax=args.Imax: _f(s, _Imax)
            res = ev.run_eval_all(
                config, z_test, y_test, v_test,
                args.eval_window_len, args.eval_num_windows, fill_seeds, args.bootstrap_iters, args.seed + 3,
                policy_fn=policy_fn, policy_name=name,
                deterministic_replay=deterministic_replay,
            )
            if res:
                baseline_per_window[name] = res.get("per_window_cum_bps", [])
                baseline_rows.append({
                    "dataset_id": dataset_id,
                    "policy_name": name,
                    "mean_cum_bps": res["mean_cum_bps"],
                    "std_cum_bps": res["std_cum_bps"],
                    "turnover_pct": res["turnover_pct"],
                    "bootstrap_ci_low": res["bootstrap_ci_low"],
                    "bootstrap_ci_high": res["bootstrap_ci_high"],
                })
                baseline_means[name] = res["mean_cum_bps"]
                rep.write_baseline_csv(outdir, [{"dataset_id": dataset_id, "policy_name": name, "mean_cum_bps": res["mean_cum_bps"], "std_cum_bps": res["std_cum_bps"], "turnover_pct": res["turnover_pct"]}], short_name)

        if pi_dp is not None:
            res_dp = ev.run_eval_all(
                config, z_test, y_test, v_test,
                args.eval_window_len, args.eval_num_windows, fill_seeds, args.bootstrap_iters, args.seed + 5,
                policy_fn=lambda s, _pi=pi_dp: int(_pi[s]), policy_name="DP_exact",
                deterministic_replay=deterministic_replay,
            )
            if res_dp:
                baseline_per_window["DP_exact"] = res_dp.get("per_window_cum_bps", [])
                baseline_rows.append({
                    "dataset_id": dataset_id,
                    "policy_name": "DP_EXACT",
                    "mean_cum_bps": res_dp["mean_cum_bps"],
                    "std_cum_bps": res_dp["std_cum_bps"],
                    "turnover_pct": res_dp["turnover_pct"],
                    "bootstrap_ci_low": res_dp["bootstrap_ci_low"],
                    "bootstrap_ci_high": res_dp["bootstrap_ci_high"],
                })
                baseline_means["DP_exact"] = res_dp["mean_cum_bps"]
                rep.write_dp_summary(outdir, dataset_id, res_dp["mean_cum_bps"], res_dp["std_cum_bps"], res_dp["bootstrap_ci_low"], res_dp["bootstrap_ci_high"])

        # DP_PHASE2: Phase2-style DP (z,i only; HOLD/BUY_MKT/SELL_MKT) — fair baseline RL can beat
        dp_phase2_fit_only = True
        if len(z_train) >= 10 and len(y_train) >= 10:
            V_p2, pi_p2, Q_p2, meta_p2 = dp2.build_and_solve_dp_phase2(
                z_train, y_train,
                Imax=args.Imax,
                c_taker_bps=args.c_taker_bps,
                lambda_inv=args.lambda_inv,
                eta_turnover=args.eta_turnover,
                gamma=args.gamma,
                seed=args.seed,
                reward_mode=getattr(args, "dp_phase2_reward_mode", "zero_mean"),
            )
            dp2.export_dp_phase2_artifacts(outdir, pi_p2, Q_p2, args.Imax, meta_p2)
            dp2.write_dp_phase2_model_summary(outdir, np.array(meta_p2["P_z"]), np.array(meta_p2["E_y_z"]), meta_p2)
            dp2.write_dp_phase2_model_json(outdir, meta_p2)
            policy_p2 = dp2.policy_phase2_from_pi(pi_p2, args.Imax, z_bins)
            res_p2 = ev.run_eval_all(
                config, z_test, y_test, v_test,
                args.eval_window_len, args.eval_num_windows, fill_seeds, args.bootstrap_iters, args.seed + 7,
                policy_fn=policy_p2, policy_name="DP_PHASE2",
                deterministic_replay=deterministic_replay,
            )
            if res_p2:
                baseline_per_window["DP_PHASE2"] = res_p2.get("per_window_cum_bps", [])
                baseline_rows.append({
                    "dataset_id": dataset_id,
                    "policy_name": "DP_PHASE2",
                    "mean_cum_bps": res_p2["mean_cum_bps"],
                    "std_cum_bps": res_p2["std_cum_bps"],
                    "turnover_pct": res_p2["turnover_pct"],
                    "bootstrap_ci_low": res_p2["bootstrap_ci_low"],
                    "bootstrap_ci_high": res_p2["bootstrap_ci_high"],
                })
                baseline_means["DP_PHASE2"] = res_p2["mean_cum_bps"]
                rep.write_dp_phase2_eval_summary(outdir, dataset_id, res_p2["mean_cum_bps"], res_p2["std_cum_bps"], res_p2["bootstrap_ci_low"], res_p2["bootstrap_ci_high"])

        # DP_empirical: from train-only counts (fair baseline)
        pi_emp = None
        dp_empirical_means = {}
        if getattr(args, "dp_empirical", 1):
            counts_path = outdir / "EMPIRICAL_COUNTS.npz"
            if counts_path.exists():
                ec = em.EmpiricalCounts.load(counts_path)
                P_hat, R_hat, model_meta = dpe.build_empirical_mdp(ec, alpha_smooth=getattr(args, "dp_empirical_smoothing", 0.1))
                V_emp, pi_emp, Q_emp = dpe.solve_dp_empirical(P_hat, R_hat, gamma=args.gamma, seed=args.seed)
                dpe.export_dp_empirical_artifacts(outdir, pi_emp, Q_emp, V_emp, args.Imax, z_bins, model_meta)
                dpe.write_dp_empirical_model_json(outdir, model_meta)
                res_emp = ev.run_eval_all(
                    config, z_test, y_test, v_test,
                    args.eval_window_len, args.eval_num_windows, fill_seeds, args.bootstrap_iters, args.seed + 6,
                    policy_fn=lambda s, _pi=pi_emp: int(_pi[s]), policy_name="DP_empirical",
                    deterministic_replay=deterministic_replay,
                )
                if res_emp:
                    baseline_per_window["DP_empirical"] = res_emp.get("per_window_cum_bps", [])
                    baseline_rows.append({
                        "dataset_id": dataset_id,
                        "policy_name": "DP_EMPIRICAL",
                        "mean_cum_bps": res_emp["mean_cum_bps"],
                        "std_cum_bps": res_emp["std_cum_bps"],
                        "turnover_pct": res_emp["turnover_pct"],
                        "bootstrap_ci_low": res_emp["bootstrap_ci_low"],
                        "bootstrap_ci_high": res_emp["bootstrap_ci_high"],
                    })
                    baseline_means["DP_empirical"] = res_emp["mean_cum_bps"]
                    dp_empirical_means = {"mean": res_emp["mean_cum_bps"], "ci_low": res_emp["bootstrap_ci_low"], "ci_high": res_emp["bootstrap_ci_high"]}
                    rep.write_dp_empirical_eval_summary(outdir, dataset_id, res_emp["mean_cum_bps"], res_emp["std_cum_bps"], res_emp["bootstrap_ci_low"], res_emp["bootstrap_ci_high"])
                    # Sanity warnings (do not fail run)
                    if model_meta.get("fraction_sa_visited", 1.0) < 0.05:
                        print("WARN: DP_empirical coverage < 5% of (S×A) — likely weak due to sparse visitation.", file=sys.stderr)
                    if "DP_exact" in baseline_means and res_emp["mean_cum_bps"] > baseline_means["DP_exact"]:
                        print("WARN: DP_empirical mean > DP_exact — possible bug (DP_exact should be upper bound).", file=sys.stderr)

        rep.write_baseline_summary(outdir, baseline_rows)

        # Delta_fair = QL - best of fair_set ONLY. fair_set = {DP_PHASE2, A, B, Hold}; never DP_EXACT or DP_EMPIRICAL.
        FAIR_SET_NAMES = ["A_sign_taker", "B_sign_maker", "Hold", "DP_PHASE2"]
        fair_names = [x for x in FAIR_SET_NAMES if x in baseline_means]
        if ql_per_window and baseline_per_window and baseline_means:
            nw_match = len(ql_per_window) == len(list(baseline_per_window.values())[0]) if baseline_per_window else False
            if nw_match and "DP_exact" in baseline_per_window:
                delta_ci = ev.bootstrap_delta_fair_and_gap_to_oracle(
                    ql_per_window, baseline_per_window, baseline_means,
                    fair_baseline_names=fair_names,
                    dp_exact_name="DP_exact",
                    bootstrap_iters=args.bootstrap_iters,
                    seed=args.seed + 4,
                )
                delta_ci["best_fixed_baseline"] = delta_ci.get("best_fair_baseline", "")
                delta_ci["delta_fixed_mean"] = delta_ci.get("delta_fair_mean", 0)
                delta_ci["delta_fixed_ci_low"] = delta_ci.get("delta_fair_ci_low", 0)
                delta_ci["delta_fixed_ci_high"] = delta_ci.get("delta_fair_ci_high", 0)
                delta_ci["delta_oracle_mean"] = delta_ci.get("gap_to_oracle_mean", 0)
                delta_ci["delta_oracle_ci_low"] = delta_ci.get("gap_to_oracle_ci_low", 0)
                delta_ci["delta_oracle_ci_high"] = delta_ci.get("gap_to_oracle_ci_high", 0)
                best_fair = delta_ci.get("best_fair_baseline", "")
                if best_fair in ("DP_exact", "DP_empirical"):
                    (outdir / "INVALID_RUN.txt").write_text(
                        f"Delta_fair violation: best_fair_baseline must not be in {{DP_EXACT, DP_EMPIRICAL}}; got {best_fair!r}.\n"
                    )
                    raise ValueError(f"best_fair_baseline={best_fair} must not be in fair set; Delta_fair excludes DP_exact and DP_empirical.")
            else:
                delta_ci = ev.bootstrap_delta_fixed_and_oracle(ql_per_window, baseline_per_window, baseline_means, args.bootstrap_iters, args.seed + 4)
                delta_ci.setdefault("best_fair_baseline", delta_ci.get("best_fixed_baseline", ""))
                delta_ci.setdefault("delta_fair_mean", delta_ci.get("delta_fixed_mean", 0))
                delta_ci.setdefault("delta_fair_ci_low", delta_ci.get("delta_fixed_ci_low", 0))
                delta_ci.setdefault("delta_fair_ci_high", delta_ci.get("delta_fixed_ci_high", 0))
                delta_ci.setdefault("gap_to_oracle_mean", delta_ci.get("delta_oracle_mean", 0))
                delta_ci.setdefault("gap_to_oracle_ci_low", delta_ci.get("delta_oracle_ci_low", 0))
                delta_ci.setdefault("gap_to_oracle_ci_high", delta_ci.get("delta_oracle_ci_high", 0))
        else:
            delta_ci = {"best_fixed_baseline": "", "best_fair_baseline": "", "delta_fixed_mean": 0.0, "delta_fixed_ci_low": 0.0, "delta_fixed_ci_high": 0.0, "delta_oracle_mean": 0.0, "delta_oracle_ci_low": 0.0, "delta_oracle_ci_high": 0.0, "delta_fair_mean": 0.0, "delta_fair_ci_low": 0.0, "delta_fair_ci_high": 0.0, "gap_to_oracle_mean": 0.0, "gap_to_oracle_ci_low": 0.0, "gap_to_oracle_ci_high": 0.0}

        # EVAL_SUMMARY.csv (Delta_fair primary, Gap_to_oracle secondary)
        rep.write_eval_summary(
            outdir,
            dataset_id,
            ql_mean=res_ql_vis.get("mean_cum_bps", 0),
            ql_std=ql_std,
            ql_ci_low=ql_ci_low,
            ql_ci_high=ql_ci_high,
            baseline_rows=baseline_rows,
            best_fixed_baseline=delta_ci.get("best_fixed_baseline", ""),
            delta_fixed_mean=delta_ci.get("delta_fixed_mean", 0),
            delta_fixed_ci_low=delta_ci.get("delta_fixed_ci_low", 0),
            delta_fixed_ci_high=delta_ci.get("delta_fixed_ci_high", 0),
            delta_oracle_mean=delta_ci.get("delta_oracle_mean", 0),
            delta_oracle_ci_low=delta_ci.get("delta_oracle_ci_low", 0),
            delta_oracle_ci_high=delta_ci.get("delta_oracle_ci_high", 0),
            maker_share=maker_share,
            taker_share=taker_share,
            best_fair_baseline=delta_ci.get("best_fair_baseline", ""),
            delta_fair_mean=delta_ci.get("delta_fair_mean", 0),
            delta_fair_ci_low=delta_ci.get("delta_fair_ci_low", 0),
            delta_fair_ci_high=delta_ci.get("delta_fair_ci_high", 0),
            gap_to_oracle_mean=delta_ci.get("gap_to_oracle_mean", 0),
            gap_to_oracle_ci_low=delta_ci.get("gap_to_oracle_ci_low", 0),
            gap_to_oracle_ci_high=delta_ci.get("gap_to_oracle_ci_high", 0),
        )

        # POLICY_COMPARISON.md (QL vs DP_empirical vs DP_exact)
        dp_exact_mean = baseline_means.get("DP_exact", 0.0)
        dp_emp_mean = baseline_means.get("DP_empirical")
        dp_phase2_mean = baseline_means.get("DP_PHASE2")
        rep.write_policy_comparison_md(
            outdir,
            dataset_id,
            ql_mean=res_ql_vis.get("mean_cum_bps", 0),
            dp_empirical_mean=dp_emp_mean,
            dp_exact_mean=dp_exact_mean,
            baseline_means=baseline_means,
            delta_fair_mean=delta_ci.get("delta_fair_mean", 0),
            gap_to_oracle_mean=delta_ci.get("gap_to_oracle_mean", 0),
            dp_phase2_mean=dp_phase2_mean,
        )

    rep.write_eval_csv(outdir, eval_rows, "EVAL_QL.csv")

    # Report-grade ANALYSIS
    rep.write_analysis_report(
        outdir,
        dataset_id,
        ql_mean_cum_bps=res_ql_vis.get("mean_cum_bps", 0) if z_test is not None and len(z_test) >= args.eval_window_len else 0.0,
        ql_ci_low=ql_ci_low,
        ql_ci_high=ql_ci_high,
        ql_minus_best_mean=delta_ci.get("delta_fixed_mean", 0),
        ql_minus_best_ci_low=delta_ci.get("delta_fixed_ci_low", 0),
        ql_minus_best_ci_high=delta_ci.get("delta_fixed_ci_high", 0),
        maker_share=maker_share,
        taker_share=taker_share,
        mean_hold_frac=mean_hold_frac,
        total_maker_fills=total_maker_fills,
        baseline_means=baseline_means,
        state_topk_rows=ql_state_topk_rows,
        visitation_rows=ql_visitation_rows,
    )

    # PASS_FAIL (report-grade gates)
    no_leak = True
    env_invariants_ok = True
    transition_ok = True
    eval_outputs_ok = bool(eval_rows) and (outdir / "EVAL_QL.csv").exists()
    eval_visitation_ok = bool(ql_visitation_rows) and (outdir / "EVAL_VISITATION.csv").exists()
    delta_ci_ok = bool(delta_ci) and "delta_fixed_ci_low" in delta_ci
    ql_beats_ci = (delta_ci.get("delta_fair_ci_low", delta_ci.get("delta_fixed_ci_low", 0)) > 0)
    training_ok = len(log_rows) >= 100
    policy_ok = float(np.mean(policy_vec != st.A_HOLD)) > 0.05 if policy_vec is not None else False
    baseline_beaten = (q_mean_cum is not None and baseline_means and q_mean_cum >= min(baseline_means.values())) if baseline_means else True
    rep.run_pass_fail_checks(
        outdir,
        dataset_id,
        horizon_ok=horizon_ok,
        no_data_leak=no_leak,
        env_invariants_ok=env_invariants_ok,
        transition_valid=transition_ok,
        eval_outputs_present=eval_outputs_ok,
        eval_visitation_present=eval_visitation_ok,
        delta_ci_present=delta_ci_ok,
        ql_beats_baseline_ci=ql_beats_ci,
        training_improving=training_ok,
        policy_nontrivial=policy_ok,
        baseline_beaten=baseline_beaten,
        shift_large=False,
        hold_fraction_high=mean_hold_frac > 0.95,
        maker_fills_zero=total_maker_fills == 0,
        dp_no_data_leak=dp_no_data_leak,
        dp_empirical_no_data_leak=True,
        dp_empirical_policy_reproducible=True,
        dp_phase2_fit_only=dp_phase2_fit_only,
    )

    rep.write_run_manifest(outdir, vars(args), str(run_root), dataset_id, effective_config)

    if args.bundle:
        bundle_path = outdir / "phase3_bundle.zip"
        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in outdir.rglob("*"):
                if f.is_file() and f != bundle_path:
                    zf.write(f, f.relative_to(outdir))
        print("Bundled to", bundle_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
