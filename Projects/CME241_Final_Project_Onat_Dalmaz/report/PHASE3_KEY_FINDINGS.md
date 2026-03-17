# Phase 3 (SNX) — Key Findings (One Page)

**CME241 Crypto Execution Project. Source: REPORT_HARVEST; canonical headline: cfg2_HEADLINE_v1 (DP_PHASE2 zero_mean).**

---

## Headline result

- **RL is positive and beats the fair baseline set** in the headline config (cfg2_HEADLINE_v1): QL mean **+716 bps** (95% CI [443, 1007]); best fair baseline is **Hold** (0 bps), so **Delta_fair = +716 bps** with CI above zero. RL learns execution and position management; the fair baseline DP_PHASE2 (zero_mean) is risk/cost-only and does not use alpha.

## Fair baseline set and metrics

- **FAIR_SET** = {DP_PHASE2 (zero_mean), A_sign_taker, B_sign_maker, Hold}. **Delta_fair** = QL − best of FAIR_SET. **Gap_to_oracle** = QL − DP_EXACT (DP_EXACT is the upper bound only, not in the fair set).

## Execution viability frontier (sweep)

- Sweep over (c_maker, p0, p1) identifies **config 2** (c_maker=1.0, p0=0.45, p1=0.15) and **config 6** (c_maker=0.5, p0=0.45, p1=0.15) as top configs. cfg2 is the “RL viable” regime; cfg6 is harder (QL negative, wide CI), motivating sensitivity to execution assumptions.

## DP_EXACT as ceiling

- **DP_EXACT** is the model-based optimum under the simplified MDP. We do **not** claim RL beats DP_EXACT; Gap_to_oracle is negative (e.g. −4979 bps in the headline run). The gap is expected and shows room under the simplified model.

## Mechanistic evidence (policy behavior)

- From **TABLE_POLICY_BEHAVIOR** and run cards: headline run has **maker_share ≈ 60.5%**, **taker_share ≈ 39.5%**, **hold fraction ≈ 85%**, **turnover ≈ 4.7%**, **avg |inventory| ≈ 0.04**. RL uses limit orders and holds when appropriate rather than churning. cfg2 original (drift DP_PHASE2) is more taker-heavy (maker ≈ 27%); cfg6 is highly maker (≈82%) with higher turnover (≈34%), consistent with a harder regime.

## Where to verify

- **RUN_REGISTRY.csv / RUN_REGISTRY.json** — all runs. **TABLE_PHASE3_MAIN.md** — main metrics. **RUN_CARDS/** — one card per run with paths to EVAL_SUMMARY.csv, DP_PHASE2_MODEL.json, POLICY_TABLE_QL.csv, phase3_bundle.zip. **DUPLICATES_AND_CANONICAL.md** — canonical selection. **FIGURES/delta_fair_bar.png** — Delta_fair across runs.
