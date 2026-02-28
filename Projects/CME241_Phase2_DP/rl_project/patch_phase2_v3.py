"""Apply PATCH SPEC: baseline metrics from eval, fee sweep, gates, README."""
from pathlib import Path
import re

path = Path(__file__).parent / "phase2_dp_from_parquets.py"
text = path.read_text()

# 1) Baselines at each lambda: replace lam_base = 0.1 and single loop with loop over lambda_list
old_baseline = """        # Baselines (evaluate at lambda=0.1 as representative)
        lam_base = 0.1
        for name, policy in [("baseline_A_sign", policy_a), ("baseline_B_inv_aware", policy_b), ("baseline_C_hold", policy_c)]:
            z_seq, long_pool, short_pool = eval_rollout.build_z_seq_and_reward_pools(
                manifest_df, run_root, dataset_id, "test", z_thresholds, kind_used, args.fee_bps
            )
            if z_seq is not None:
                res = eval_rollout.run_rollouts(
                    dataset_id, z_seq, policy, args.imax, lam_base, kind_used,"""

new_baseline = """        # Baselines: evaluate at each lambda (same grid as DP)
        for lam_bl in lambda_list:
            for name, policy in [("baseline_A_sign", policy_a), ("baseline_B_inv_aware", policy_b), ("baseline_C_hold", policy_c)]:
                z_seq, long_pool, short_pool = eval_rollout.build_z_seq_and_reward_pools(
                    manifest_df, run_root, dataset_id, "test", z_thresholds, kind_used, args.fee_bps
                )
                if z_seq is not None:
                    res = eval_rollout.run_rollouts(
                        dataset_id, z_seq, policy, args.imax, lam_bl, kind_used,"""

if old_baseline in text:
    text = text.replace(old_baseline, new_baseline)
    text = text.replace('"lambda": lam_base,', '"lambda": lam_bl,')
    text = text.replace("'lambda': lam_base,", "'lambda': lam_bl,")
else:
    print("WARN: baseline block not found")

# 2) BASELINES_METRICS from eval_rows
old_bm = """    baseline_metrics = []
    for name, policy in [("A_sign_threshold", policy_a), ("B_inv_aware", policy_b), ("C_hold", policy_c)]:
        baseline_metrics.append({
            "baseline": name,
            "description": "sign threshold" if name == "A_sign_threshold" else ("inv aware" if name == "B_inv_aware" else "always hold"),
        })
        # Write POLICY_TABLE_baseline_X.csv"""

new_bm = """    baseline_metrics = []
    for name, policy in [("A_sign_threshold", policy_a), ("B_inv_aware", policy_b), ("C_hold", policy_c)]:
        # Write POLICY_TABLE_baseline_X.csv"""

if new_bm not in text:
    text = text.replace(
        """    baseline_metrics = []
    for name, policy in [("A_sign_threshold", policy_a), ("B_inv_aware", policy_b), ("C_hold", policy_c)]:
        baseline_metrics.append({
            "baseline": name,
            "description": "sign threshold" if name == "A_sign_threshold" else ("inv aware" if name == "B_inv_aware" else "always hold"),
        })
        # Write POLICY_TABLE_baseline_X.csv""",
        new_bm,
    )

# Add baseline_metrics from eval_rows before reporting.write_baselines_metrics
old_write = "    reporting.write_baselines_metrics(outdir, baseline_metrics)"
add_before_write = """
    for er in eval_rows:
        if str(er.get("policy_name", "")).startswith("baseline_"):
            baseline_metrics.append({
                "dataset_id": er["dataset_id"],
                "baseline_name": er["policy_name"],
                "lambda": er["lambda"],
                "mean_cum_bps": er["mean_cum_bps"],
                "std_cum_bps": er["std_cum_bps"],
                "turnover_pct": er["turnover_pct"],
                "avg_abs_inv": er["avg_abs_inv"],
                "p05": er.get("p05"), "p50": er.get("p50"), "p95": er.get("p95"),
            })
    if not baseline_metrics:
        baseline_metrics = [{"dataset_id": "", "baseline_name": "none", "lambda": None, "mean_cum_bps": None, "std_cum_bps": None, "turnover_pct": None, "avg_abs_inv": None, "p05": None, "p50": None, "p95": None}]
"""
if add_before_write.strip() not in text and "for er in eval_rows:" not in text:
    text = text.replace(
        "    reporting.write_baselines_metrics(outdir, baseline_metrics)",
        add_before_write + "    reporting.write_baselines_metrics(outdir, baseline_metrics)",
    )

path.write_text(text)
print("Patched phase2_dp_from_parquets.py")
