# Phase 3 Report — Index (CME241 SNX)

**Source of truth:** this directory (`REPORT_HARVEST`). Portable snapshot: `phase3_REPORT_HARVEST_*.tar.gz` in `/home/ubuntu/bundles/`.

## Report deliverables (drop-in for CME241 writeup)

| Document | Description |
|---------|-------------|
| **PHASE3_REPORT_SECTION.md** | Full Phase 3 results section: setup, baselines, evaluation protocol, results table, interpretation, limitations, citation pointers |
| **PHASE3_KEY_FINDINGS.md** | One-page executive summary (headline result, fair set, viability frontier, DP_EXACT as ceiling, mechanistic evidence) |
| **PHASE3_APPENDIX_DP_DRIFT.md** | Why DP_PHASE2 “drift” is not a fair baseline; zero_mean definition |

## Data and tables

| File | Description |
|------|-------------|
| **RUN_REGISTRY.csv** / **RUN_REGISTRY.json** | All runs, normalized schema |
| **TABLE_PHASE3_MAIN.md** | Report-ready main metrics (QL, Delta_fair, Gap_to_oracle, maker/taker, dp_phase2_reward_mode) |
| **TABLE_SWEEP_SUMMARY.md** | Sweep summary; TOP_CONFIGS (cfg2, cfg6) |
| **TABLE_POLICY_BEHAVIOR.md** | Maker/taker share, hold_frac, turnover, avg_abs_inv by run |
| **DUPLICATES_AND_CANONICAL.md** | Canonical selection; headline = cfg2_HEADLINE_v1 (zero_mean) |

## Run-level evidence

- **RUN_CARDS/** — one `run_<id>.md` per run: settings, metrics, **exact paths** to EVAL_SUMMARY.csv, DP_PHASE2_MODEL.json, POLICY_TABLE_QL.csv, phase3_bundle.zip, and “OK to claim” notes.

## Figure

- **FIGURES/delta_fair_bar.png** — Delta_fair mean and bootstrap CI across runs.

## Canonical headline run

- **cfg2_HEADLINE_v1** with **DP_PHASE2 reward_mode = zero_mean**. Path (active): `/home/ubuntu/onat/results/phase3_report_real_snx_cfg2_HEADLINE_v1`. Canonical for FIXDP bundle lineage per DUPLICATES_AND_CANONICAL.md.
