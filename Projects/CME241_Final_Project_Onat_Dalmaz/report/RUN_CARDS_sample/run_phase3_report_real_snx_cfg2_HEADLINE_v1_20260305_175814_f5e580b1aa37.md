# Run card: phase3_report_real_snx_cfg2_HEADLINE_v1

- **run_id:** phase3_report_real_snx_cfg2_HEADLINE_v1_20260305_175814_f5e580b1aa37
- **path:** /home/ubuntu/onat/results/phase3_report_real_snx_cfg2_HEADLINE_v1
- **source:** active_results
- **created_ts:** 2026-03-05T17:58:14.849131Z

## Key settings
- c_maker=1.0, c_taker=4.0, p0=0.45, p1=0.15
- eta_turnover=0.05, lambda_inv=0.1
- eval_windows=20, fill_seeds=50, bootstrap_iters=1000
- dp_phase2_reward_mode=zero_mean

## Metrics
- QL mean: 715.9256112796797 CI [443.0819003737464, 1007.2637820664708]
- best_fair_baseline: Hold
- delta_fair_mean: 715.9256112796797 CI [442.7301257842556, 997.6751954575934]
- gap_to_oracle_mean: -4979.045994457442

## File pointers
- EVAL_SUMMARY.csv: `/home/ubuntu/onat/results/phase3_report_real_snx_cfg2_HEADLINE_v1/EVAL_SUMMARY.csv`
- DP_PHASE2_MODEL.json: `/home/ubuntu/onat/results/phase3_report_real_snx_cfg2_HEADLINE_v1/DP_PHASE2_MODEL.json`
- POLICY_TABLE_QL.csv: `/home/ubuntu/onat/results/phase3_report_real_snx_cfg2_HEADLINE_v1/POLICY_TABLE_QL.csv`
- phase3_bundle.zip: `/home/ubuntu/onat/results/phase3_report_real_snx_cfg2_HEADLINE_v1/phase3_bundle.zip`

## OK to claim / Don't claim
- Delta_fair mean > 0 and CI lower > 0: can claim 'statistically significant improvement over fair baseline'.
- Delta_fair mean > 0, CI crosses 0: report as 'positive point estimate'.