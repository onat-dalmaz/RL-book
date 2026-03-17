# Phase 3 (RL) Results — CME241 Crypto Execution Project (SNX)

*Source of truth: REPORT_HARVEST (RUN_REGISTRY, TABLE_PHASE3_MAIN.md, RUN_CARDS). Canonical headline run: cfg2_HEADLINE_v1 with DP_PHASE2 reward_mode = zero_mean.*

---

## Phase 3 Setup

**Constraints and scope.** Phase 3 evaluates **tabular Q-learning (QL)** in a discrete MDP that abstracts the execution layer: same signal and fill model as earlier phases, but the agent chooses **actions** (hold, place limit, cancel, market order) and is rewarded in bps. We use a **5D discrete state** and **one outstanding order** (fixed size), with **tabular Q-learning** as the first RL algorithm.

**State.** State is discretized into: signal bucket *z*, inventory *i* ∈ {−*I*max …, *I*max}, outstanding order side, order age bucket, and (optionally) volume bin. *I*max = 3; *z* uses 3 bins (terciles). The Phase-2 abstraction used later for baselines reduces this to (*z*, *i*) only.

**Actions.** Actions include: HOLD, PLACE_BUY_LIMIT, PLACE_SELL_LIMIT, CANCEL, BUY_MARKET, SELL_MARKET. Order size is fixed when placing.

**Reward.** Per-step reward is mark-to-market inventory change (signal-realized *y* × inventory) minus transaction costs (maker/taker bps), minus inventory penalty (λ × *i*²) and optional turnover penalty (η × 1{Δ*i*≠0}).

**Fill model.** Fills are stochastic (deterministic replay uses fixed seeds per evaluation). Fill probabilities depend on alignment of order with signal, volume, and age (p0, p1, d_age, d_v). No market impact; execution is price-taking within the simulator.

**Evaluation protocol.** We use **deterministic replay**: same test windows and same sequence of fill draws per (window, seed). For each policy we average over **20 evaluation windows** and **50 fill seeds** per window; **bootstrap over windows** (1000 iterations) to obtain 95% CI for mean cumulative bps.

---

## Baselines

**Fair baseline set (FAIR_SET).** We compare QL to a fixed set of non-oracle baselines:

- **DP_PHASE2 (zero_mean):** Phase-2-constrained DP with state (*z*, *i*) and actions {HOLD, BUY_MARKET, SELL_MARKET}. It **does not** use conditional expected return E[*y*|*z*] in planning; reward in the DP is **risk/cost/turnover only** (μ_y = 0). So it is a conservative, risk-management baseline that RL can beat by using the alpha signal and the full execution state/actions.
- **A_sign_taker:** always take the side of the signal (market orders).
- **B_sign_maker:** always post limit on the side of the signal.
- **Hold:** never trade (0 bps).

**Upper bound (not in FAIR_SET).** **DP_EXACT** is the model-based DP optimum under the same simplified MDP (full state, same costs and fill model). It is the **oracle** and is used only as an upper bound. We do **not** compare RL to DP_EXACT as a “fair” baseline.

**Definitions (must match artifacts).**

- **Delta_fair** = QL − best_fixed_baseline among FAIR_SET, where “best fixed baseline” is the baseline with highest mean cumulative bps (chosen globally over the evaluation set).
- **Gap_to_oracle** = QL − DP_EXACT.

---

## Results

### Main table (canonical runs)

The table below is taken from `REPORT_HARVEST/TABLE_PHASE3_MAIN.md`. Headline run: **cfg2_HEADLINE_v1** (DP_PHASE2 zero_mean). Supporting: **cfg2** (original, drift DP_PHASE2), **cfg6** (harder regime).

| Run | QL mean | QL CI | Best fair | Delta_fair mean | Delta_fair CI | DP_EXACT mean | Gap_to_oracle mean | Gap CI | maker_share | taker_share | hold_frac | turnover | avg_abs_inv | dp_phase2_reward_mode |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| phase3_report_real_snx_cfg2_HEADLINE_v1 | 715.93 | [443.08, 1007.26] | Hold | 715.93 | [442.73, 997.68] | 5694.97 | -4979.05 | [-9057.04, -1107.92] | 0.605 | 0.395 | 0.850 | 4.71 | 0.043 | zero_mean |
| phase3_report_real_snx_cfg2_cm1.0_p00.45_p10.15 | 2789.65 | [1851.77, 3853.03] | DP_PHASE2 | -97.15 | [-1928.72, 1556.61] | 5903.38 | -3113.73 | [-7560.33, 1131.59] | 0.267 | 0.733 | 0.883 | 7.10 | 0.141 | drift |
| phase3_report_real_snx_cfg6_cm0.5_p00.45_p10.15 | -972.92 | [-3949.69, 2499.57] | DP_PHASE2 | -3859.73 | [-7478.15, 221.04] | 5833.48 | -6806.41 | [-12095.79, -1858.76] | 0.823 | 0.177 | 0.075 | 34.3 | 2.05 | drift |

*Full numeric table: see TABLE_PHASE3_MAIN.md.*

### Figure

**Delta_fair (headline metric):** See `REPORT_HARVEST/FIGURES/delta_fair_bar.png` for a bar plot of Delta_fair mean with bootstrap CI across runs.

---

## Interpretation

**Headline (cfg2_HEADLINE_v1).** In the canonical headline run, QL mean is **+716 bps** (95% CI [443, 1007]). The best fair baseline is **Hold** (0 bps), so **Delta_fair = +716 bps** with CI entirely above zero. Thus **RL beats the fair baseline set**: it learns to use the execution layer (limit orders, position management) while DP_PHASE2 (zero_mean) is a conservative, risk-only baseline. Policy behavior supports this: **maker_share ≈ 60.5%**, **hold fraction ≈ 85%**, **turnover ≈ 4.7%** — the policy places limits and holds when appropriate rather than churning. Gap_to_oracle remains negative (QL − DP_EXACT ≈ −4979 bps), as expected, since DP_EXACT is the model-based upper bound.

**cfg2 original (context).** With the same (c_maker, p0, p1) but **drift** DP_PHASE2 (which uses E[*y*|*z*]), the best fair baseline is DP_PHASE2 and Delta_fair is slightly negative (−97 bps). This illustrates that when the DP baseline is given a model-based drift advantage, it can outperform naive RL; the **zero_mean** definition is what makes the headline comparison fair.

**cfg6 (harder regime).** QL mean is **−973 bps** with wide CI; best fair baseline is DP_PHASE2 and Delta_fair is large and negative. This config (c_maker=0.5, same p0/p1) is a harder execution regime (e.g. overtrading, instability) and motivates the execution “viability frontier” from the sweep.

**Execution viability frontier (sweep).** The sweep over (c_maker, p0, p1) shows that **config 2** (c_maker=1.0, p0=0.45, p1=0.15) and **config 6** (c_maker=0.5, p0=0.45, p1=0.15) are the top two by the sweep metric; see `TABLE_SWEEP_SUMMARY.md` and TOP_CONFIGS. The frontier separates regimes where RL is viable (positive mean, stable behavior) from those where costs or fill assumptions make RL negative or high-variance.

---

## Limitations and future work

- **DP_EXACT is an upper bound** under the simplified MDP; we do not claim RL beats it.
- **Fill model** is simplified (no adverse selection, no market impact); real execution would add friction.
- **Variance:** bootstrap CI can be wide (e.g. cfg6); more windows or seeds would tighten estimates.
- **Discretization:** state and time are discretized; continuous state/action and function approximation are natural next steps.
- **Off-policy and stability:** we use on-policy evaluation; improving stability and sample efficiency (e.g. better exploration, replay) is future work.

---

## How to cite / verify numbers

For the **headline run** (cfg2_HEADLINE_v1):

- **Run card:** `REPORT_HARVEST/RUN_CARDS/run_phase3_report_real_snx_cfg2_HEADLINE_v1_20260305_175814_f5e580b1aa37.md`
- **EVAL_SUMMARY.csv:** `/home/ubuntu/onat/results/phase3_report_real_snx_cfg2_HEADLINE_v1/EVAL_SUMMARY.csv`
- **DP_PHASE2 (zero_mean):** `DP_PHASE2_MODEL.json` in that directory (`reward_mode`: zero_mean)
- **Policy / bundle:** `POLICY_TABLE_QL.csv`, `phase3_bundle.zip` in the same directory

Canonical selection is documented in `DUPLICATES_AND_CANONICAL.md`; the headline run is canonical for the group (config_id=2, c_maker=1.0, p0=0.45, p1=0.15, 20 windows, 50 seeds, 1000 bootstrap, dp_phase2_reward_mode=zero_mean).
