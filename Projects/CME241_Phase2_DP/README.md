# Phase 2 DP baseline code (CME241 project)

## What this is

Phase 2 DP baseline code for the CME241 project: MDP formulation (z, inventory), value iteration, position PnL reward (inventory_mtm), horizon-aligned data, and deterministic-window replay with bootstrap CI.

## Where to place in repo

Recommended: `Projects/CME241_Phase2_DP/rl_project/`  
Place the contents of `rl_project/` there so that `phase2_dp_from_parquets.py` is at `Projects/CME241_Phase2_DP/rl_project/phase2_dp_from_parquets.py`.

## How to run

Canonical command (edit paths as needed):

```bash
python3 rl_project/phase2_dp_from_parquets.py \
  --run_root <PATH_TO_STEP1_COIN_DIR> \
  --mode step1 \
  --outdir <OUTPUT_DIR> \
  --reward_mode inventory_mtm \
  --eval_replay_mode deterministic \
  --fee_bps 2.0 \
  --imax 3 --gamma 0.99 \
  --lambda_grid 0,0.01,0.05,0.1,0.2 \
  --eval_mode windows --eval_window_len 5000 --eval_num_windows 20 \
  --bootstrap_iters 1000 \
  --seed 123 --bundle
```

Example: `--run_root /data/COINS/NEAR` and `--outdir ./results/phase2_NEAR_fee2`.

## Dependencies

- Python 3.x
- numpy, pandas
- pyarrow (for parquet)
- matplotlib (optional, for some reporting)

## Notes

- No results are included; results are written to `--outdir`.
- Step1 coin dir must contain parquets with columns: `ts` (or `timestamp`), `y`, `pred` (e.g. `trades_val.parquet`, `trades_test.parquet` under `cv/<COIN>/fold_0/`).
