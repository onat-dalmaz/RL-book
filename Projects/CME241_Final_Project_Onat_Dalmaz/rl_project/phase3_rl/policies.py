"""
Phase 3 RL: Baselines and policy runner.
A = sign-taker (market in signal direction), B = sign-maker (limit with one-step patience), C = hold.
"""

from typing import Callable, Optional
import numpy as np

from . import state as st


def policy_hold(state_idx: int, Imax: int = 3) -> int:
    """Baseline C: always HOLD."""
    return st.A_HOLD


def policy_sign_taker(state_idx: int, Imax: int = 3) -> int:
    """Baseline A: market in signal direction (z). If z>0 buy market, z<0 sell market, else hold."""
    z, i, o_side, o_age, v_bin = st.index_to_state(state_idx, Imax)
    if z > 0 and i < Imax:
        return st.A_BUY_MARKET
    if z < 0 and i > -Imax:
        return st.A_SELL_MARKET
    return st.A_HOLD


def policy_sign_maker(state_idx: int, Imax: int = 3) -> int:
    """Baseline B: limit in signal direction with one-step patience. Place limit; if already have limit, hold one step then act."""
    z, i, o_side, o_age, v_bin = st.index_to_state(state_idx, Imax)
    if z > 0 and i < Imax:
        if o_side == 1:
            return st.A_HOLD  # already have buy limit
        return st.A_PLACE_BUY
    if z < 0 and i > -Imax:
        if o_side == -1:
            return st.A_HOLD
        return st.A_PLACE_SELL
    return st.A_HOLD


def policy_from_q(Q: np.ndarray, epsilon: float, rng: np.random.Generator, Imax: int = 3) -> Callable[[int], int]:
    """Return a callable state_idx -> action that uses ε-greedy on Q."""

    def pi(s: int) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(0, st.N_ACTIONS))
        return int(np.argmax(Q[s, :]))

    return pi


def policy_greedy_from_q(
    Q: np.ndarray,
    Imax: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> Callable[[int], int]:
    """Return a callable state_idx -> action that is greedy w.r.t. Q.
    If rng is provided, break ties randomly (avoids HOLD-first bias when Q values tie)."""
    if rng is not None:
        from .q_learning import argmax_random_tiebreak
        def pi(s: int) -> int:
            return argmax_random_tiebreak(Q[s, :], rng)
    else:
        def pi(s: int) -> int:
            return int(np.argmax(Q[s, :]))
    return pi


def run_policy_on_env(
    env,
    policy: Callable[[int], int],
    start_idx: int = 0,
    seed: Optional[int] = None,
) -> dict:
    """
    Reset env at start_idx with seed, run policy to end. policy(state_idx) -> action.
    Returns dict with total_reward (cum_bps), steps, turnovers, turnover_pct, bps_per_step.
    """
    total_r = 0.0
    steps = 0
    turnovers = 0
    s, _ = env.reset(start_idx=start_idx, seed=seed)
    done = False
    while not done:
        a = policy(s)
        s, r, done, info = env.step(a)
        total_r += r
        steps += 1
        if info.get("di", 0) != 0:
            turnovers += 1
    return {
        "total_reward": total_r,
        "cum_bps": total_r,
        "steps": steps,
        "turnovers": turnovers,
        "turnover_pct": 100.0 * turnovers / max(1, steps),
        "bps_per_step": total_r / max(1, steps),
    }


def run_policy_on_env_with_trajectory(
    env,
    policy: Callable[[int], int],
    start_idx: int = 0,
    seed: Optional[int] = None,
) -> dict:
    """Same as run_policy_on_env but also returns trajectory for visitation diagnostics."""
    total_r = 0.0
    steps = 0
    turnovers = 0
    actions = []
    fill_types = []
    states_z = []
    states_i = []
    state_idxs = []
    s, info = env.reset(start_idx=start_idx, seed=seed)
    done = False
    while not done:
        a = policy(s)
        z, i = info.get("z", 0), info.get("i", 0)
        state_idxs.append(s)
        states_z.append(z)
        states_i.append(i)
        actions.append(a)
        s, r, done, info = env.step(a)
        fill_types.append(info.get("fill_type", "none"))
        total_r += r
        steps += 1
        if info.get("di", 0) != 0:
            turnovers += 1
    return {
        "total_reward": total_r,
        "cum_bps": total_r,
        "steps": steps,
        "turnovers": turnovers,
        "turnover_pct": 100.0 * turnovers / max(1, steps),
        "bps_per_step": total_r / max(1, steps),
        "actions": actions,
        "fill_types": fill_types,
        "states_z": states_z,
        "states_i": states_i,
        "state_idxs": state_idxs,
    }
