# Phase 2 DP/MDP from Trade Parquets

Pre-RL DP/ADP execution: state (z_t, i_t), actions {-1,0,+1}, reward from empirical trade edge minus inventory penalty. Solved via value iteration; evaluated by replay simulation and baselines.

## Notes / Risks

- **Opportunity stream**: Trade parquets are used as a *sequence of opportunity epochs*, not a full market simulator. Each row is one candidate step.
- **Reward model**: Rewards are empirical (mean net_bps by z and side from train/val). They are action-conditional but not fully counterfactual. Suitable for Phase 2 DP demonstration; Phase 3 RL will use a minimal environment.
- **Discovery**: Script discovers `**/test_trades.parquet`, `**/val_trades.parquet`, `**/train_trades.parquet` and also `trades_test.parquet`, `trades_val.parquet`, `trades_train.parquet` under `--run_root`.

## Acceptance Criteria (done)

A run is **complete** if:

1. **PASS_FAIL.md** has all PASS for at least one (coin, variant, fold).
2. **POLICY_TABLE.csv**, **VALUE_TABLE.csv**, **P_Z_GIVEN_Z.csv**, **REWARD_STATS.csv** exist.
3. **SWEEP_RESULTS.csv** shows DP policy comparable or better than at least one baseline under some lambda.
4. Replay-sim outputs **EVAL_ROLLOUT_SUMMARY.csv** with mean/std cum reward, turnover %, and inventory metrics.

## Usage

```bash
python3 /home/ubuntu/onat/rl_project/phase2_dp_from_trades.py \
  --run_root /path/to/experiment_outputs \
  --outdir /path/to/phase2_results_$(date +%Y%m%d_%H%M%S) \
  --imax 3 --gamma 0.99 --qlo 0.33 --qhi 0.67 \
  --nmin 50 --n_rollouts 200
```

Optional: `--max_steps_per_rollout 2000` to cap evaluation length on long test sequences.

Output folder is self-contained and shareable (manifest, thresholds, P(z'|z), reward stats, policy, value table, sweep, baselines, PASS_FAIL).
