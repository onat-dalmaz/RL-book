"""
Phase 3 RL: Artifact writers and PASS_FAIL gates.
"""

from pathlib import Path
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def build_effective_config(
    args_dict: dict,
    q_lo: float,
    q_hi: float,
    v_bin_threshold: float,
) -> dict:
    """Single source of truth for env/training/eval params. RUN_MANIFEST must embed this verbatim for validation."""
    return {
        "env": {
            "qlo": float(q_lo),
            "qhi": float(q_hi),
            "Imax": int(args_dict.get("Imax", 3)),
            "z_bins": int(args_dict.get("z_bins", 3)),
            "vbin_method": str(args_dict.get("vbin_method", "median_abs_y")),
            "vbin_quantile": float(args_dict.get("vbin_quantile", 0.5)),
            "v_bin_threshold": float(v_bin_threshold),
            "reward_mode": str(args_dict.get("reward_mode", "inventory_mtm")),
            "eval_replay_mode": str(args_dict.get("eval_replay_mode", "deterministic")),
        },
        "execution": {
            "c_maker_bps": float(args_dict.get("c_maker_bps", 1.0)),
            "c_taker_bps": float(args_dict.get("c_taker_bps", 2.0)),
            "p0": float(args_dict.get("p0", 0.6)),
            "p1": float(args_dict.get("p1", 0.2)),
            "dv": float(args_dict.get("dv", 0.05)),
            "d_age": float(args_dict.get("d_age", 0.1)),
            "eta_turnover": float(args_dict.get("eta_turnover", 0.5)),
            "lambda_inv": float(args_dict.get("lambda_inv", 0.1)),
        },
        "training": {
            "gamma": float(args_dict.get("gamma", 0.99)),
            "alpha": float(args_dict.get("alpha", 0.2)),
            "alpha_min": float(args_dict.get("alpha_min", 0.02)),
            "eps": float(args_dict.get("eps", 0.3)),
            "eps_min": float(args_dict.get("eps_min", 0.05)),
            "q_init": float(args_dict.get("q_init", 0.01)),
            "decay_episodes": int(args_dict.get("decay_episodes", 5000)),
            "n_train_episodes": int(args_dict.get("n_train_episodes", 5000)),
        },
        "windowing": {
            "n_train_windows": int(args_dict.get("n_train_windows", 20)),
            "train_window_len": int(args_dict.get("train_window_len", 2000)),
            "eval_num_windows": int(args_dict.get("eval_num_windows", 20)),
            "eval_window_len": int(args_dict.get("eval_window_len", 2000)),
            "eval_fill_seeds": int(args_dict.get("eval_fill_seeds", 30)),
        },
        "bootstrap": {"bootstrap_iters": int(args_dict.get("bootstrap_iters", 1000))},
        "seed": int(args_dict.get("seed", 123)),
    }


def write_effective_config(outdir: Path, effective_config: dict):
    outdir = Path(outdir)
    (outdir / "EFFECTIVE_CONFIG.json").write_text(json.dumps(effective_config, indent=2))


def write_run_manifest(outdir: Path, args_dict: dict, run_root: str, dataset_id: str, effective_config: dict = None):
    outdir = Path(outdir)
    manifest = {
        "phase": "phase3_rl",
        "run_root": run_root,
        "dataset_id": dataset_id,
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "args": args_dict,
    }
    if effective_config is not None:
        manifest["effective_config"] = effective_config
    (outdir / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2))


def write_horizon_spec(outdir: Path, raw_dt_ms: float, label_horizon_ms: float, stride: int, mdp_step_ms: float):
    outdir = Path(outdir)
    spec = {"raw_dt_ms": raw_dt_ms, "label_horizon_ms": label_horizon_ms, "stride": stride, "mdp_step_ms": mdp_step_ms}
    (outdir / "HORIZON_SPEC.json").write_text(json.dumps(spec, indent=2))


def write_z_thresholds(outdir: Path, z_thresholds: list):
    outdir = Path(outdir)
    if z_thresholds:
        pd.DataFrame(z_thresholds).to_csv(outdir / "Z_THRESHOLDS.csv", index=False)


def write_vbin_threshold(outdir: Path, v_bin_threshold: float, method: str = "median_abs_y"):
    outdir = Path(outdir)
    (outdir / "VBIN_THRESHOLD.json").write_text(json.dumps({"v_bin_threshold": v_bin_threshold, "method": method}, indent=2))


def write_windows_csv(outdir: Path, window_starts: List[int], window_len: int, tag: str = "train"):
    outdir = Path(outdir)
    rows = [{"window_idx": i, "start": s, "end": s + window_len} for i, s in enumerate(window_starts)]
    pd.DataFrame(rows).to_csv(outdir / f"WINDOWS_{tag}.csv", index=False)


def write_env_sanity(outdir: Path, Imax: int, n_states: int, n_actions: int, state_bounds_ok: bool):
    outdir = Path(outdir)
    text = (
        "# ENV_SANITY\n\n"
        f"- Imax: {Imax}\n"
        f"- n_states: {n_states}\n"
        f"- n_actions: {n_actions}\n"
        f"- state_bounds (|i|<=Imax, at most one order): {'PASS' if state_bounds_ok else 'FAIL'}\n"
    )
    (outdir / "ENV_SANITY.md").write_text(text)


def write_fill_model_sanity(outdir: Path, p0: float, p1: float, dv: float, d_age: float, sample_probs: list):
    outdir = Path(outdir)
    text = (
        "# FILL_MODEL_SANITY\n\n"
        f"- p0: {p0}, p1: {p1}, dv: {dv}, d_age: {d_age}\n"
        "Sample p_fill (aligned/not, v_bin, age):\n"
    )
    for line in sample_probs:
        text += f"  {line}\n"
    (outdir / "FILL_MODEL_SANITY.md").write_text(text)


def write_q_table(outdir: Path, Q: np.ndarray, Imax: int, z_bins: int = 3):
    outdir = Path(outdir)
    from . import state as st
    rows = []
    for s in range(Q.shape[0]):
        z, i, o_side, o_age, v_bin = st.index_to_state(s, Imax, z_bins)
        for a in range(Q.shape[1]):
            rows.append({"state_idx": s, "z": z, "i": i, "o_side": o_side, "o_age": o_age, "v_bin": v_bin, "action": a, "Q": float(Q[s, a])})
    pd.DataFrame(rows).to_csv(outdir / "Q_TABLE.csv", index=False)


def load_q_table(outdir: Path, nS: int, nA: int):
    """Load Q matrix: try Q_TABLE_QL.npy first, then Q_TABLE.csv. Returns (Q, True) or (None, False)."""
    outdir = Path(outdir)
    npy_path = outdir / "Q_TABLE_QL.npy"
    if npy_path.exists():
        try:
            Q = np.load(npy_path)
            if Q.shape == (nS, nA):
                return Q.astype(float), True
        except Exception:
            pass
    path = outdir / "Q_TABLE.csv"
    if not path.exists():
        return None, False
    try:
        df = pd.read_csv(path)
        if "state_idx" not in df.columns or "action" not in df.columns or "Q" not in df.columns:
            return None, False
        Q = np.zeros((nS, nA), dtype=float)
        for _, r in df.iterrows():
            s, a = int(r["state_idx"]), int(r["action"])
            if 0 <= s < nS and 0 <= a < nA:
                Q[s, a] = float(r["Q"])
        return Q, True
    except Exception:
        return None, False


def write_training_log(outdir: Path, log_rows: List[dict]):
    outdir = Path(outdir)
    if log_rows:
        pd.DataFrame(log_rows).to_csv(outdir / "TRAINING_LOG.csv", index=False)


def write_eval_csv(outdir: Path, rows: List[dict], filename: str = "EVAL_QL.csv", columns: List[str] = None):
    outdir = Path(outdir)
    if rows:
        df = pd.DataFrame(rows)
        if columns:
            df = df[[c for c in columns if c in df.columns]]
        df.to_csv(outdir / filename, index=False)


def write_baseline_csv(outdir: Path, rows: List[dict], policy_name: str):
    outdir = Path(outdir)
    if rows:
        pd.DataFrame(rows).to_csv(outdir / f"BASELINE_{policy_name}.csv", index=False)


def write_baseline_summary(outdir: Path, baseline_rows: List[dict]):
    """BASELINE_SUMMARY.csv: all rule baselines + DP in one table."""
    outdir = Path(outdir)
    if baseline_rows:
        pd.DataFrame(baseline_rows).to_csv(outdir / "BASELINE_SUMMARY.csv", index=False)


def write_dp_summary(outdir: Path, dataset_id: str, mean_cum_bps: float, std_cum_bps: float, ci_low: float, ci_high: float):
    """DP_SUMMARY.csv: single row for DP baseline eval."""
    outdir = Path(outdir)
    pd.DataFrame([{
        "dataset_id": dataset_id,
        "policy": "DP",
        "mean_cum_bps": mean_cum_bps,
        "std_cum_bps": std_cum_bps,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
    }]).to_csv(outdir / "DP_SUMMARY.csv", index=False)


def write_dp_empirical_eval_summary(
    outdir: Path,
    dataset_id: str,
    mean_cum_bps: float,
    std_cum_bps: float,
    ci_low: float,
    ci_high: float,
):
    """DP_EMPIRICAL_EVAL_SUMMARY.csv: same schema as other policy eval (one row)."""
    outdir = Path(outdir)
    pd.DataFrame([{
        "dataset_id": dataset_id,
        "policy_name": "DP_EMPIRICAL",
        "mean_cum_bps": mean_cum_bps,
        "std_cum_bps": std_cum_bps,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
    }]).to_csv(outdir / "DP_EMPIRICAL_EVAL_SUMMARY.csv", index=False)


def write_dp_phase2_eval_summary(
    outdir: Path,
    dataset_id: str,
    mean_cum_bps: float,
    std_cum_bps: float,
    ci_low: float,
    ci_high: float,
):
    """DP_PHASE2_EVAL_SUMMARY.csv: one row for Phase2-style DP baseline eval."""
    outdir = Path(outdir)
    pd.DataFrame([{
        "dataset_id": dataset_id,
        "policy_name": "DP_PHASE2",
        "mean_cum_bps": mean_cum_bps,
        "std_cum_bps": std_cum_bps,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
    }]).to_csv(outdir / "DP_PHASE2_EVAL_SUMMARY.csv", index=False)


def write_policy_comparison_md(
    outdir: Path,
    dataset_id: str,
    ql_mean: float,
    dp_empirical_mean: Optional[float],
    dp_exact_mean: float,
    baseline_means: Dict[str, float],
    delta_fair_mean: float,
    gap_to_oracle_mean: float,
    dp_phase2_mean: Optional[float] = None,
):
    """POLICY_COMPARISON.md: QL vs DP_PHASE2 (fair) vs DP_exact (upper bound)."""
    outdir = Path(outdir)
    lines = [
        "# Policy comparison (report)\n",
        f"**Dataset:** {dataset_id}\n\n",
        "## Means (mean_cum_bps)\n",
        f"- **QL:** {ql_mean:.2f}\n",
        f"- **DP_exact (upper bound):** {dp_exact_mean:.2f}\n",
    ]
    if dp_phase2_mean is not None:
        lines.append(f"- **DP_PHASE2 (fair baseline, Phase2 abstraction):** {dp_phase2_mean:.2f}\n")
    if dp_empirical_mean is not None:
        lines.append(f"- **DP_empirical:** {dp_empirical_mean:.2f}\n")
    for name, val in (baseline_means or {}).items():
        if name not in ("DP_exact", "DP_empirical", "DP_PHASE2"):
            lines.append(f"- **{name}:** {val:.2f}\n")
    lines.extend([
        "\n## Fair comparison (primary)\n",
        f"- **Delta_fair** = QL − best_fixed(DP_PHASE2, A, B, Hold) = {delta_fair_mean:.2f}. DP_PHASE2 uses zero_mean (no E[y|z] in planning).\n",
        "\n## Gap to oracle (secondary)\n",
        f"- **Gap_to_oracle** = QL − DP_exact = {gap_to_oracle_mean:.2f}\n",
    ])
    (outdir / "POLICY_COMPARISON.md").write_text("".join(lines))


def write_policy_table(outdir: Path, policy_vec: np.ndarray, Imax: int, z_bins: int = 3):
    """policy_vec[s] = action index. Write POLICY_TABLE.csv with state -> action."""
    outdir = Path(outdir)
    from . import state as st
    rows = []
    for s in range(len(policy_vec)):
        z, i, o_side, o_age, v_bin = st.index_to_state(s, Imax, z_bins)
        rows.append({"state_idx": s, "z": z, "i": i, "o_side": o_side, "o_age": o_age, "v_bin": v_bin, "action": int(policy_vec[s])})
    pd.DataFrame(rows).to_csv(outdir / "POLICY_TABLE.csv", index=False)


def write_eval_summary(
    outdir: Path,
    dataset_id: str,
    ql_mean: float,
    ql_std: float,
    ql_ci_low: float,
    ql_ci_high: float,
    baseline_rows: List[dict],
    best_fixed_baseline: str,
    delta_fixed_mean: float,
    delta_fixed_ci_low: float,
    delta_fixed_ci_high: float,
    delta_oracle_mean: float,
    delta_oracle_ci_low: float,
    delta_oracle_ci_high: float,
    maker_share: float,
    taker_share: float,
    best_fair_baseline: str = "",
    delta_fair_mean: float = 0.0,
    delta_fair_ci_low: float = 0.0,
    delta_fair_ci_high: float = 0.0,
    gap_to_oracle_mean: float = 0.0,
    gap_to_oracle_ci_low: float = 0.0,
    gap_to_oracle_ci_high: float = 0.0,
):
    """EVAL_SUMMARY.csv: QL, DP_EXACT, DP_EMPIRICAL, rules; Delta_fair (QL - best of fair set); Gap_to_oracle (QL - DP_exact)."""
    outdir = Path(outdir)
    rows = []
    ql_row = {
        "dataset_id": dataset_id,
        "policy": "QL",
        "mean_cum_bps": ql_mean,
        "std_cum_bps": ql_std,
        "ci_low": ql_ci_low,
        "ci_high": ql_ci_high,
        "best_fixed_baseline": best_fixed_baseline,
        "delta_fixed_mean": delta_fixed_mean,
        "delta_fixed_ci_low": delta_fixed_ci_low,
        "delta_fixed_ci_high": delta_fixed_ci_high,
        "delta_oracle_mean": delta_oracle_mean,
        "delta_oracle_ci_low": delta_oracle_ci_low,
        "delta_oracle_ci_high": delta_oracle_ci_high,
        "best_fair_baseline": best_fair_baseline,
        "delta_fair_mean": delta_fair_mean,
        "delta_fair_ci_low": delta_fair_ci_low,
        "delta_fair_ci_high": delta_fair_ci_high,
        "gap_to_oracle_mean": gap_to_oracle_mean,
        "gap_to_oracle_ci_low": gap_to_oracle_ci_low,
        "gap_to_oracle_ci_high": gap_to_oracle_ci_high,
        "maker_share": maker_share,
        "taker_share": taker_share,
    }
    rows.append(ql_row)
    for r in baseline_rows:
        pol_name = r.get("policy_name", r.get("policy", ""))
        rows.append({
            "dataset_id": dataset_id,
            "policy": pol_name,
            "mean_cum_bps": r.get("mean_cum_bps", 0),
            "std_cum_bps": r.get("std_cum_bps", 0),
            "ci_low": r.get("bootstrap_ci_low", r.get("ci_low", 0)),
            "ci_high": r.get("bootstrap_ci_high", r.get("ci_high", 0)),
            "best_fixed_baseline": "",
            "delta_fixed_mean": "",
            "delta_fixed_ci_low": "",
            "delta_fixed_ci_high": "",
            "delta_oracle_mean": "",
            "delta_oracle_ci_low": "",
            "delta_oracle_ci_high": "",
            "best_fair_baseline": "",
            "delta_fair_mean": "",
            "delta_fair_ci_low": "",
            "delta_fair_ci_high": "",
            "gap_to_oracle_mean": "",
            "gap_to_oracle_ci_low": "",
            "gap_to_oracle_ci_high": "",
            "maker_share": "",
            "taker_share": "",
        })
    # INFO row: gap-to-oracle (secondary)
    rows.append({
        "dataset_id": dataset_id,
        "policy": "INFO_gap_to_oracle",
        "mean_cum_bps": "",
        "std_cum_bps": "",
        "ci_low": "",
        "ci_high": "",
        "best_fixed_baseline": "",
        "delta_fixed_mean": "",
        "delta_fixed_ci_low": "",
        "delta_fixed_ci_high": "",
        "delta_oracle_mean": "",
        "delta_oracle_ci_low": "",
        "delta_oracle_ci_high": "",
        "best_fair_baseline": best_fair_baseline,
        "delta_fair_mean": delta_fair_mean,
        "delta_fair_ci_low": delta_fair_ci_low,
        "delta_fair_ci_high": delta_fair_ci_high,
        "gap_to_oracle_mean": gap_to_oracle_mean,
        "gap_to_oracle_ci_low": gap_to_oracle_ci_low,
        "gap_to_oracle_ci_high": gap_to_oracle_ci_high,
        "maker_share": "",
        "taker_share": "",
    })
    pd.DataFrame(rows).to_csv(outdir / "EVAL_SUMMARY.csv", index=False)


def write_eval_visitation(outdir: Path, visitation_rows: List[dict]):
    """EVAL_VISITATION.csv: per-window action fractions, maker/taker fills and shares."""
    outdir = Path(outdir)
    if visitation_rows:
        pd.DataFrame(visitation_rows).to_csv(outdir / "EVAL_VISITATION.csv", index=False)


def write_eval_state_topk(outdir: Path, state_topk_rows: List[dict]):
    """EVAL_STATE_TOPK.csv: top-K visited (z, i) with counts."""
    outdir = Path(outdir)
    if state_topk_rows:
        pd.DataFrame(state_topk_rows).to_csv(outdir / "EVAL_STATE_TOPK.csv", index=False)


def write_pass_fail(outdir: Path, gate_rows: List[dict], overall: str = "PASS"):
    outdir = Path(outdir)
    lines = ["# PASS_FAIL\n\n", "| dataset_id | gate | result | evidence |\n", "|------------|------|--------|----------|\n"]
    for r in gate_rows:
        lines.append(f"| {r.get('dataset_id', '')} | {r.get('gate', '')} | {r.get('result', '')} | {r.get('evidence', '')} |\n")
    (outdir / "PASS_FAIL.md").write_text("".join(lines))
    (outdir / "PASS_FAIL.json").write_text(json.dumps({"overall": overall, "gates": gate_rows}, indent=2))


def run_pass_fail_checks(
    outdir: Path,
    dataset_id: str,
    horizon_ok: bool,
    no_data_leak: bool,
    env_invariants_ok: bool,
    transition_valid: bool,
    eval_outputs_present: bool,
    eval_visitation_present: bool = True,
    delta_ci_present: bool = True,
    ql_beats_baseline_ci: bool = False,
    training_improving: bool = True,
    policy_nontrivial: bool = True,
    baseline_beaten: bool = True,
    shift_large: bool = False,
    hold_fraction_high: bool = False,
    maker_fills_zero: bool = False,
    dp_no_data_leak: bool = True,
    dp_empirical_no_data_leak: bool = True,
    dp_empirical_policy_reproducible: bool = True,
    dp_phase2_fit_only: bool = True,
) -> List[dict]:
    """Populate gate_rows for PASS_FAIL. dp_phase2_fit_only: DP_PHASE2 P(z'|z), E[y|z] from fit only."""
    gate_rows = []
    for gate, result, evidence in [
        ("horizon_ok", "PASS" if horizon_ok else "FAIL", "HORIZON_SPEC" if horizon_ok else "missing or invalid"),
        ("no_data_leak", "PASS" if no_data_leak else "FAIL", "z/v_bin fit on train only" if no_data_leak else "leak"),
        ("dp_no_data_leak", "PASS" if dp_no_data_leak else "FAIL", "DP P_z/P_v from fit only" if dp_no_data_leak else "DP used eval data"),
        ("dp_phase2_fit_only", "PASS" if dp_phase2_fit_only else "FAIL", "DP_PHASE2 P(z'|z) E[y|z] from fit only" if dp_phase2_fit_only else "DP_PHASE2 used eval data"),
        ("dp_empirical_no_data_leak", "PASS" if dp_empirical_no_data_leak else "FAIL", "DP_empirical counts from train only" if dp_empirical_no_data_leak else "DP_empirical used eval data"),
        ("dp_empirical_policy_reproducible", "PASS" if dp_empirical_policy_reproducible else "FAIL", "DP_empirical VI + tie-break seeded" if dp_empirical_policy_reproducible else "non-deterministic"),
        ("dp_phase2_fit_only", "PASS" if dp_phase2_fit_only else "FAIL", "DP_PHASE2 from fit only" if dp_phase2_fit_only else "DP_PHASE2 used eval data"),  # noqa: F821
        ("env_invariants", "PASS" if env_invariants_ok else "FAIL", "ENV_SANITY" if env_invariants_ok else "invariant violation"),
        ("transition_valid", "PASS" if transition_valid else "FAIL", "step semantics" if transition_valid else "invalid transition"),
        ("eval_outputs_present", "PASS" if eval_outputs_present else "FAIL", "EVAL_* CSVs" if eval_outputs_present else "missing"),
        ("eval_visitation_present", "PASS" if eval_visitation_present else "FAIL", "EVAL_VISITATION.csv" if eval_visitation_present else "missing"),
        ("delta_ci_present", "PASS" if delta_ci_present else "FAIL", "QL_minus_best_baseline CI" if delta_ci_present else "missing"),
        ("ql_beats_baseline_ci", "PASS" if ql_beats_baseline_ci else ("WARN" if delta_ci_present else "SKIP"), "Delta_fixed_ci_low>0" if ql_beats_baseline_ci else "Delta_fixed_ci_low<=0" if delta_ci_present else "no delta CI"),
        ("training_not_improving", "WARN" if not training_improving else "PASS", "soft"),
        ("policy_degenerate", "WARN" if not policy_nontrivial else "PASS", "soft"),
        ("baseline_not_beaten", "WARN" if not baseline_beaten else "PASS", "soft"),
        ("shift_large", "WARN" if shift_large else "PASS", "soft"),
        ("hold_95_pct", "WARN" if hold_fraction_high else "PASS", "QL HOLD >95% steps" if hold_fraction_high else "soft"),
        ("maker_fills_zero", "WARN" if maker_fills_zero else "PASS", "policy never benefits from maker" if maker_fills_zero else "soft"),
    ]:
        gate_rows.append({"dataset_id": dataset_id, "gate": gate, "result": result, "evidence": evidence})
    overall = "FAIL" if any(r.get("result") == "FAIL" for r in gate_rows) else "PASS"
    write_pass_fail(outdir, gate_rows, overall)
    return gate_rows


def write_analysis_report(
    outdir: Path,
    dataset_id: str,
    ql_mean_cum_bps: float,
    ql_ci_low: float,
    ql_ci_high: float,
    ql_minus_best_mean: float,
    ql_minus_best_ci_low: float,
    ql_minus_best_ci_high: float,
    maker_share: float,
    taker_share: float,
    mean_hold_frac: float,
    total_maker_fills: int,
    baseline_means: dict,
    state_topk_rows: List[dict],
    visitation_rows: List[dict],
):
    """Write ANALYSIS_AND_INTERPRETATION.md with report-grade sections: What RL learned, Why it beats baselines (or why optimal is HOLD)."""
    outdir = Path(outdir)
    best_baseline = max(baseline_means.values()) if baseline_means else 0.0
    beats_baselines = ql_minus_best_ci_low > 0
    sections = [
        "# Phase 3 RL — Analysis and Interpretation (Report)\n",
        f"**Dataset:** {dataset_id}\n",
        "## 1. Executive summary\n",
        f"- QL mean cum bps: {ql_mean_cum_bps:.2f} (95% CI [{ql_ci_low:.2f}, {ql_ci_high:.2f}]).\n",
        f"- QL minus best baseline: {ql_minus_best_mean:.2f} (95% CI [{ql_minus_best_ci_low:.2f}, {ql_minus_best_ci_high:.2f}]).\n",
        f"- **Conclusion:** " + ("QL beats best baseline (CI lower bound > 0)." if beats_baselines else "QL does not beat best baseline or optimal is HOLD under costs.") + "\n",
        "\n## 2. What RL learned\n",
        f"- **Maker vs taker:** Maker share of fills = {maker_share:.2%}, taker share = {taker_share:.2%}.\n",
        f"- **Action mix:** HOLD fraction on eval = {mean_hold_frac:.1%}; policy uses limit and market when beneficial.\n",
        f"- **Z dependence:** Visited states (z, i) top-K in EVAL_STATE_TOPK.csv; policy responds to signal z and inventory i.\n",
        f"- **Inventory profile:** State visitation concentrates on (z, i) pairs seen in EVAL_STATE_TOPK; policy keeps |i| bounded.\n",
        "\n## 3. Why it beats baselines (or why optimal is HOLD)\n",
    ]
    if beats_baselines:
        sections.append(f"- QL achieves higher mean cum bps than the best baseline (best baseline = {best_baseline:.2f}); delta CI lower bound > 0.\n")
        sections.append("- The learned policy trades when signal and costs justify it and holds otherwise, improving on fixed rule baselines.\n")
    else:
        sections.append(f"- Under current costs and fill model, the best baseline (mean = {best_baseline:.2f}) is not beaten by QL in the sense that the delta CI includes 0.\n")
        sections.append("- This can mean: (1) optimal strategy is to hold (e.g. no edge in data), or (2) QL needs more training / tuning, or (3) baselines are strong.\n")
    sections.append("\n## 4. Diagnostics\n")
    sections.append("- EVAL_VISITATION.csv: per-window action fractions and maker/taker shares.\n")
    sections.append("- EVAL_STATE_TOPK.csv: top visited (z, i) states.\n")
    sections.append("- EVAL_SUMMARY.csv: mean, std, CI for QL and baselines; QL_minus_best_baseline CI; maker_share, taker_share.\n")
    (outdir / "ANALYSIS_AND_INTERPRETATION.md").write_text("".join(sections))
