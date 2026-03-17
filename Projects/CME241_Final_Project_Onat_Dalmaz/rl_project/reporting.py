"""
Phase 2 DP v3: CSV writers and PASS_FAIL gates.
"""

from pathlib import Path
import csv
import json
from datetime import datetime, timezone

import pandas as pd


def append_csv_row(filepath: Path, row: dict, header_columns: list):
    """
    Append one row to a CSV. If file missing, write header then row.
    row keys should match header_columns; extra keys dropped, missing keys get empty string.
    """
    filepath = Path(filepath)
    values = [row.get(k, "") for k in header_columns]
    file_exists = filepath.exists()
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header_columns)
        w.writerow(values)


def write_eval_progress(outdir: Path, progress: dict):
    """Write EVAL_PROGRESS.json with completed_* counters and last_success_ts."""
    outdir = Path(outdir)
    progress["last_success_ts"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    (outdir / "EVAL_PROGRESS.json").write_text(json.dumps(progress, indent=2))


# Column order for incremental append (must match row keys used in phase2_dp_from_parquets)
EVAL_ROLLOUT_SUMMARY_COLUMNS = [
    "dataset_id", "lambda", "policy_name", "mean_cum_bps", "std_cum_bps", "turnover_pct", "avg_abs_inv",
    "p05", "p50", "p95", "mean_steps_used", "mean_bps_per_step", "std_bps_per_step", "max_steps_per_rollout",
]
SWEEP_RESULTS_COLUMNS = [
    "dataset_id", "lambda", "Imax", "policy", "mean_cum_bps", "std_cum_bps", "turnover_pct", "avg_abs_inv",
    "mean_steps_used", "mean_bps_per_step", "std_bps_per_step", "max_steps_per_rollout",
]
FEE_SWEEP_RESULTS_COLUMNS = [
    "dataset_id", "fee_bps", "lambda", "policy", "mean_cum_bps", "std_cum_bps", "turnover_pct", "avg_abs_inv",
    "policy_depends_on_z", "policy_nontrivial",
    "mean_steps_used", "mean_bps_per_step", "std_bps_per_step", "max_steps_per_rollout",
]
FEE_SWEEP_POLICY_DIAGNOSTICS_COLUMNS = ["dataset_id", "fee_bps", "lambda", "policy_depends_on_z", "policy_nontrivial"]


def write_eval_rollout_summary(outdir: Path, rows: list):
    """EVAL_ROLLOUT_SUMMARY.csv: dataset_id, lambda, policy_name, mean_cum_bps, std_cum_bps, turnover_pct, avg_abs_inv, p05, p50, p95."""
    outdir = Path(outdir)
    if rows:
        pd.DataFrame(rows).to_csv(outdir / "EVAL_ROLLOUT_SUMMARY.csv", index=False)


def write_eval_seed_log(outdir: Path, seed: int, n_rollouts: int, dataset_ids: list):
    """EVAL_ROLLOUT_SEED_LOG.csv for reproducibility."""
    outdir = Path(outdir)
    pd.DataFrame([{"seed": seed, "n_rollouts": n_rollouts, "dataset_ids": ",".join(dataset_ids)}]).to_csv(
        outdir / "EVAL_ROLLOUT_SEED_LOG.csv", index=False
    )


def write_model_value_summary(outdir: Path, rows: list):
    """MODEL_VALUE_SUMMARY.csv: dataset_id, lambda, V_z0_i0, V_z1_i0, etc."""
    outdir = Path(outdir)
    if rows:
        pd.DataFrame(rows).to_csv(outdir / "MODEL_VALUE_SUMMARY.csv", index=False)


def write_baselines_metrics(outdir: Path, rows: list):
    """BASELINES_METRICS.csv and POLICY_TABLE_baseline_A.csv etc."""
    outdir = Path(outdir)
    if rows:
        pd.DataFrame(rows).to_csv(outdir / "BASELINES_METRICS.csv", index=False)


def write_sweep_results(outdir: Path, rows: list):
    """SWEEP_RESULTS.csv: dataset_id, lambda, Imax, policy, mean_cum_bps, std_cum_bps, turnover_pct, avg_abs_inv."""
    outdir = Path(outdir)
    if rows:
        pd.DataFrame(rows).to_csv(outdir / "SWEEP_RESULTS.csv", index=False)


def write_policy_diagnostics(outdir: Path, rows: list):
    """POLICY_DIAGNOSTICS.csv: dataset_id, lambda, policy_depends_on_z, policy_nontrivial."""
    outdir = Path(outdir)
    if rows:
        pd.DataFrame(rows).to_csv(outdir / "POLICY_DIAGNOSTICS.csv", index=False)


def write_pass_fail(outdir: Path, gate_rows: list):
    """
    PASS_FAIL.md: table dataset_id x gate x PASS/FAIL x evidence file.
    gate_rows: list of dicts with dataset_id, gate, result (PASS/FAIL/SKIP), evidence.
    """
    outdir = Path(outdir)
    lines = ["# PASS_FAIL\n\n", "| dataset_id | gate | result | evidence |\n", "|------------|------|--------|----------|\n"]
    for r in gate_rows:
        lines.append(f"| {r.get('dataset_id', '')} | {r.get('gate', '')} | {r.get('result', '')} | {r.get('evidence', '')} |\n")
    (outdir / "PASS_FAIL.md").write_text("".join(lines))
    return gate_rows
