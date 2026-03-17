#!/usr/bin/env bash
# Run cfg2 headline (RL beat DP_PHASE2) + optional fee ablation, then bundle.
# Usage: from anywhere, ./results/run_cfg2_headline_and_bundle.sh [--ablation]
# Or: bash /home/ubuntu/onat/results/run_cfg2_headline_and_bundle.sh

set -e
RESULTS_DIR="${RESULTS_DIR:-/home/ubuntu/onat/results}"
RL_PROJECT="${RL_PROJECT:-/home/ubuntu/onat/rl_project}"
RUN_ROOT="${RUN_ROOT:-/home/ubuntu/onat/experiments/step1_signal_feb13_feb24_train6_20260224_191001/COINS/SNX}"
BUNDLES_DIR="${BUNDLES_DIR:-/home/ubuntu/bundles}"
DO_ABLATION="${DO_ABLATION:-0}"

HEADLINE_OUTDIR="${RESULTS_DIR}/phase3_report_real_snx_cfg2_HEADLINE_v1"
ABLATION_OUTDIR="${RESULTS_DIR}/phase3_report_real_snx_cfg2_FEE_ABLATION_v1"

# --- Acceptance check: after a run, verify artifacts and optionally headline gate ---
check_run() {
  local outdir="$1"
  local label="$2"
  local require_delta_fair_positive="${3:-0}"   # 1 = headline gate: delta_fair_mean > 0

  if [[ ! -d "$outdir" ]]; then
    echo "check_run: $outdir not found"
    return 1
  fi

  local failed=0
  local gate_fail=""

  # DP_PHASE2 exists
  for f in DP_PHASE2_MODEL.json DP_PHASE2_POLICY_TABLE.csv DP_PHASE2_EVAL_SUMMARY.csv; do
    if [[ ! -f "$outdir/$f" ]]; then
      echo "FAIL [$label]: missing $outdir/$f"
      failed=1
    fi
  done

  # EVAL_SUMMARY: QL row, best_fair_baseline in allowed set
  if [[ -f "$outdir/EVAL_SUMMARY.csv" ]]; then
    local ql_line
    ql_line=$(grep '^SNX_SNX_fold0,QL,' "$outdir/EVAL_SUMMARY.csv" || true)
    if [[ -z "$ql_line" ]]; then
      echo "FAIL [$label]: no QL row in EVAL_SUMMARY.csv"
      failed=1
    else
      local best_fair
      best_fair=$(echo "$ql_line" | awk -F',' '{print $14}')
      case "$best_fair" in
        DP_PHASE2|A_sign_taker|B_sign_maker|Hold) ;;
        *) echo "FAIL [$label]: best_fair_baseline=$best_fair (illegal)"; failed=1 ;;
      esac
      if [[ "$require_delta_fair_positive" -eq 1 ]]; then
        local delta_fair_mean
        delta_fair_mean=$(echo "$ql_line" | awk -F',' '{print $15}')
        if [[ -z "$delta_fair_mean" ]] || ! awk -v m="$delta_fair_mean" 'BEGIN { exit (m+0 > 0 ? 0 : 1) }'; then
          gate_fail="delta_fair_mean=$delta_fair_mean (required > 0)"
        fi
      fi
    fi
  else
    echo "FAIL [$label]: missing EVAL_SUMMARY.csv"
    failed=1
  fi

  if [[ -n "$gate_fail" ]]; then
    echo "HEADLINE_GATE_FAIL [$label]: $gate_fail"
    echo -e "HEADLINE_GATE_FAIL\n\n$gate_fail\n\n$(date -Iseconds)" > "$outdir/HEADLINE_GATE_FAIL.txt"
    if [[ -f "$outdir/EVAL_SUMMARY.csv" ]]; then
      ql_line=$(grep '^SNX_SNX_fold0,QL,' "$outdir/EVAL_SUMMARY.csv" || true)
      echo "QL row: $ql_line" >> "$outdir/HEADLINE_GATE_FAIL.txt"
    fi
  fi

  if [[ $failed -eq 0 && -z "$gate_fail" ]]; then
    echo "PASS [$label]: artifacts and gate OK"
    return 0
  fi
  [[ $failed -eq 0 ]] && return 0
  return 1
}

# --- 1. Headline run ---
echo "=== Headline cfg2 (eta_turnover=0.05, stable RL hyperparams) ==="
mkdir -p "$HEADLINE_OUTDIR"
cd "$RL_PROJECT"
python3 -m phase3_rl.cli_phase3 \
  --run_root "$RUN_ROOT" \
  --outdir "$HEADLINE_OUTDIR" \
  --reward_mode inventory_mtm --eval_replay_mode deterministic \
  --qlo 0.33 --qhi 0.67 --Imax 3 \
  --vbin_method median_abs_y --vbin_quantile 0.5 \
  --c_maker_bps 1.0 --c_taker_bps 4.0 \
  --lambda_inv 0.1 --eta_turnover 0.05 \
  --p0 0.45 --p1 0.15 --dv 0.05 --d_age 0.05 \
  --gamma 0.99 --alpha 0.10 --alpha_min 0.02 --eps 0.15 --eps_min 0.01 \
  --q_init 0.01 --decay_episodes 2500 --n_train_episodes 5000 \
  --n_train_windows 50 --train_window_len 5000 \
  --eval_num_windows 20 --eval_window_len 5000 \
  --eval_fill_seeds 50 --bootstrap_iters 1000 \
  --log_every 50 --seed 123 --bundle 1 \
  --z_bins 3 --dp_empirical 1

check_run "$HEADLINE_OUTDIR" "HEADLINE_v1" 1 || true

# --- 2. Optional fee ablation ---
if [[ "$DO_ABLATION" -eq 1 ]]; then
  echo "=== Fee ablation cfg2 (c_taker_bps=6.0) ==="
  mkdir -p "$ABLATION_OUTDIR"
  python3 -m phase3_rl.cli_phase3 \
    --run_root "$RUN_ROOT" \
    --outdir "$ABLATION_OUTDIR" \
    --reward_mode inventory_mtm --eval_replay_mode deterministic \
    --qlo 0.33 --qhi 0.67 --Imax 3 \
    --vbin_method median_abs_y --vbin_quantile 0.5 \
    --c_maker_bps 1.0 --c_taker_bps 6.0 \
    --lambda_inv 0.1 --eta_turnover 0.05 \
    --p0 0.45 --p1 0.15 --dv 0.05 --d_age 0.05 \
    --gamma 0.99 --alpha 0.10 --alpha_min 0.02 --eps 0.15 --eps_min 0.01 \
    --q_init 0.01 --decay_episodes 2500 --n_train_episodes 5000 \
    --n_train_windows 50 --train_window_len 5000 \
    --eval_num_windows 20 --eval_window_len 5000 \
    --eval_fill_seeds 50 --bootstrap_iters 1000 \
    --log_every 50 --seed 123 --bundle 1 \
    --z_bins 3 --dp_empirical 1

  check_run "$ABLATION_OUTDIR" "FEE_ABLATION_v1" 0 || true
fi

# --- 3. Bundle (sweep + top_k cfg + HEADLINE + optional FEE_ABLATION) ---
echo "=== Bundle ==="
EXTRA_ARR=()
[[ -d "$HEADLINE_OUTDIR" ]] && EXTRA_ARR+=( "phase3_report_real_snx_cfg2_HEADLINE_v1" )
[[ "$DO_ABLATION" -eq 1 && -d "$ABLATION_OUTDIR" ]] && EXTRA_ARR+=( "phase3_report_real_snx_cfg2_FEE_ABLATION_v1" )

if [[ ${#EXTRA_ARR[@]} -gt 0 ]]; then
  python3 -m phase3_rl.bundle_final \
    --results_dir "$RESULTS_DIR" \
    --bundles_dir "$BUNDLES_DIR" \
    --top_k 2 \
    --extra_include "${EXTRA_ARR[@]}"
else
  python3 -m phase3_rl.bundle_final \
    --results_dir "$RESULTS_DIR" \
    --bundles_dir "$BUNDLES_DIR" \
    --top_k 2
fi

echo "Done. Bundle: $BUNDLES_DIR/phase3_snx_results_FINAL_HEADLINE_*.tar.gz (or FINAL_*.tar.gz)"
