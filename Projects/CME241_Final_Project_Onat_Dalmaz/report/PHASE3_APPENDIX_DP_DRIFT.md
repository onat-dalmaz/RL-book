# Appendix: Why DP_PHASE2 “Drift” Mode Is Not a Fair Baseline

**Short version:** For the main report we use **DP_PHASE2 (zero_mean)** as the fair Phase-2 baseline. A variant that uses **E[*y*|*z*]** in the DP planner (“drift” mode) is **not** used in the fair set. Here is why.

---

## What “drift” DP_PHASE2 does

In **drift** mode, DP_PHASE2 is fitted on train data to estimate:

- **P(*z*′|*z*)** — transition of the discretized signal.
- **E[*y*|*z*]** — conditional expected one-step return (or drift) given *z*.

The DP then plans using reward terms that include **i*′* × E[*y*|*z*′]** (and costs/penalties). So the baseline **optimizes using an estimate of conditional expected return** — i.e. it gets a **model-based alpha** from the same data used to fit the MDP.

---

## Why that is not fair

- **RL** gets no direct access to E[*y*|*z*]; it has to learn from rewards.
- **Drift DP_PHASE2** gets E[*y*|*z*] from the fit and uses it in the Bellman equation. So it has a **privileged, model-based advantage**: it is effectively using “free” alpha in the planning step.
- Comparing RL to such a baseline would mix two things: (1) learning vs not learning, and (2) having vs not having a model of conditional expected return. The **pedagogical point** we want is: “RL can beat a **Phase-2-constrained** baseline that has the same information as Phase 2 (no alpha in the reward).”

---

## What we use instead: zero_mean

- In **zero_mean** mode, we set **E[*y*|*z*] = 0** in the DP planner. So DP_PHASE2 optimizes only:
  - inventory risk (λ *i*²),
  - turnover penalty (η),
  - transaction costs (taker/maker).
- That makes it a **risk/cost/turnover** baseline: “hold near zero inventory, trade sparingly.” It does **not** assume knowledge of expected return conditional on *z*.
- Under that definition, **RL can fairly beat** DP_PHASE2 by using the full state (including outstanding order) and learning to exploit the alpha signal and execution structure. That is what the headline run (cfg2_HEADLINE_v1) demonstrates: best fair baseline = Hold, Delta_fair > 0.

---

## Summary

- **DP_PHASE2 (drift)** = Phase-2 DP **with** E[*y*|*z*] in the reward → **not** a fair baseline (model-based alpha).
- **DP_PHASE2 (zero_mean)** = Phase-2 DP **without** E[*y*|*z*] (risk/cost only) → **fair** baseline for “RL beats Phase-2-style execution.”

All reported **Delta_fair** and “RL beats fair baseline” claims use the **zero_mean** definition. Drift-mode results appear only as context (e.g. cfg2 original) to show how a stronger, model-based DP can sit above QL.
