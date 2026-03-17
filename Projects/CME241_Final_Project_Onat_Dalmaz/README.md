# CME241 Final Project — RL for Crypto Execution (Onat Dalmaz)

## 1. Project overview

This project implements a two-phase pipeline for execution and viability analysis on crypto (SNX) data:

- **Phase 2 DP baseline:** A compact tabular DP/MDP baseline with state roughly (z, i) — signal bucket and inventory. Exact planning via value iteration; evaluation by deterministic replay and comparison to simple rule baselines. Implemented in `rl_project/` (phase2_dp_from_parquets.py, phase2_dp_from_trades.py, mdp_dp.py, eval_rollout.py).

- **Phase 3 RL execution layer:** An execution-aware MDP with extended state (z, i, outstanding order, age, volume bin) and tabular Q-learning. Fair baselines include DP_PHASE2 (zero_mean), A_sign_taker, B_sign_maker, Hold; DP_EXACT is the model-based upper bound. A sweep over (c_maker, p0, p1) defines an execution “viability frontier.” Implemented in `rl_project/phase3_rl/`.

**Main question:** Can RL beat a fair Phase-2-style baseline (risk/cost-only DP) by learning execution and position management, and under which execution assumptions is RL viable?

## 2. Repository contents

| Path | Description |
|------|-------------|
| **rl_project/** | Phase 2 and Phase 3 source code. Root contains data_io, Phase 2 DP scripts, and shared reporting. |
| **rl_project/phase3_rl/** | Phase 3 RL package: env, state, policies, q_learning, eval, dp_baseline, dp_phase2_baseline, dp_empirical, CLI, sweep, bundling, harvest. |
| **scripts/** | Shell script to run headline cfg2 and optional fee ablation, then bundle (run_cfg2_headline_and_bundle.sh). |
| **report/** | Report-ready text and tables: Phase 3 report section, key findings, appendix (why DP_PHASE2 drift is not fair), index, TABLE_*.md, DUPLICATES_AND_CANONICAL.md, one sample run card. FIGURES/ is for delta_fair_bar.png if generated. |

## 3. Main entrypoints

- **Phase 2:** From `rl_project/` (parent of phase3_rl):
  ```bash
  python3 phase2_dp_from_trades.py --run_root <path_to_experiment> --outdir <outdir> --imax 3 --gamma 0.99 --qlo 0.33 --qhi 0.67 --nmin 50 --n_rollouts 200
  ```
  Or use phase2_dp_from_parquets.py for parquet-based discovery. See README_phase2_dp.md in rl_project/.

- **Phase 3 (single run):** From `rl_project/`:
  ```bash
  python3 -m phase3_rl.cli_phase3 --run_root <path_to_Step1_coin_dir> --outdir <outdir> [--c_maker_bps 1.0 --p0 0.45 --p1 0.15 ...]
  ```
  Use `--dp_phase2_reward_mode zero_mean` for the fair baseline. Use `--resume_eval 1` for eval-only (no training).

- **Phase 3 sweep:** From `rl_project/`:
  ```bash
  python3 -m phase3_rl.run_sweep_snx --run_root <SNX_dir> --outdir <sweep_outdir>
  ```

- **Headline + bundle:** Use scripts/run_cfg2_headline_and_bundle.sh (set RESULTS_DIR, RL_PROJECT, RUN_ROOT, BUNDLES_DIR as needed).

- **Harvest (tables + run cards):** From `rl_project/`:
  ```bash
  python3 -m phase3_rl.harvest_results --results_root <results> --bundles_root <bundles> --outdir <REPORT_HARVEST> --include_figures 1
  ```
  Reproducing the final report/slides: use the markdown in report/ (PHASE3_REPORT_SECTION.md, PHASE3_KEY_FINDINGS.md); numbers come from RUN_REGISTRY and TABLE_* produced by harvest (or from the full REPORT_HARVEST snapshot if you have it).

## 4. Notes on excluded data

Large experiment artifacts and raw result bundles were intentionally **excluded** from this GitHub submission to keep the repo clean and code-first. Excluded items include:

- Full REPORT_HARVEST/bundles_extracted/ (extracted tarballs from many runs)
- RUN_REGISTRY.csv/.json and full RUN_CARDS/ (77 run cards)
- Raw result directories (phase3_report_*, phase3_sweep_snx outputs)
- Any .tar.gz / .zip of prior bundles
- Checkpoints, caches, __pycache__, large CSVs/parquets

To reproduce tables and figures, run the harvest script locally against your results and bundles directories; the code in this package is the same as used for the submitted report.

## 5. Final headline result

- **Phase 2 DP** strongly beats simple heuristics under the same replay evaluation.
- **Phase 3 RL** beats the **fair** baseline set (DP_PHASE2 zero_mean, A_sign_taker, B_sign_maker, Hold) in the headline config (cfg2_HEADLINE_v1): QL mean +716 bps, Delta_fair = +716 bps (best fair = Hold), with bootstrap CI above zero.
- **Execution assumptions** create a **viability frontier**: config 2 (c_maker=1.0, p0=0.45, p1=0.15) is viable; config 6 (c_maker=0.5) is harder (QL negative, wide CI). DP_EXACT remains the upper bound; RL does not beat it.
