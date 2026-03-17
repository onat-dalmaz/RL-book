"""
Phase 3 RL: Execution MDP environment.
State (z, i, o_side, o_age, v_bin). Actions: HOLD, PLACE_BUY_LIMIT, PLACE_SELL_LIMIT, BUY_MARKET, SELL_MARKET.
One outstanding order, fixed size 1. Stochastic fill for limits; market fills with prob 1.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from . import state as st


@dataclass
class ExecutionEnvConfig:
    Imax: int = 3
    z_bins: int = 3
    c_maker_bps: float = 1.0
    c_taker_bps: float = 2.0
    lambda_inv: float = 0.1
    eta_turnover: float = 0.5
    p0: float = 0.6
    p1: float = 0.2
    dv: float = 0.05
    d_age: float = 0.1


def p_fill(
    side: int,
    z: int,
    v_bin: int,
    o_age: int,
    cfg: ExecutionEnvConfig,
) -> float:
    """Fill probability for limit order. side in {-1, 1}, z in {-1,0,1}, v_bin in {0,1}, o_age in {0,1,2}."""
    aligned = (side == 1 and z == 1) or (side == -1 and z == -1)
    p = cfg.p0 if aligned else cfg.p1
    p += cfg.dv * v_bin
    if o_age >= 2:
        p -= cfg.d_age
    return float(np.clip(p, 0.0, 1.0))


class ExecutionEnv:
    """
    Step-by-step execution env over a window of (z_seq, y_seq, v_bin_seq).
    At each step t we have z_t, y_t, v_bin_t from data; state is (z_t, i, o_side, o_age, v_bin_t).
    """

    def __init__(self, config: ExecutionEnvConfig, z_seq: np.ndarray, y_seq: np.ndarray, v_bin_seq: np.ndarray):
        self.config = config
        self.z_seq = np.asarray(z_seq, dtype=int)
        self.y_seq = np.asarray(y_seq, dtype=float)
        self.v_bin_seq = np.asarray(v_bin_seq, dtype=int)
        self.T = min(len(self.z_seq), len(self.y_seq), len(self.v_bin_seq))
        self.Imax = config.Imax
        self.z_bins = getattr(config, "z_bins", 3)
        self.nS = st.n_states(self.Imax, self.z_bins)
        self.nA = st.N_ACTIONS

    def reset(self, start_idx: int = 0, seed: Optional[int] = None) -> Tuple[int, dict]:
        """Reset to start_idx. Returns (state_index, info)."""
        self._rng = np.random.default_rng(seed)
        self._t = start_idx
        self._i = 0
        self._o_side = 0
        self._o_age = 0
        if self._t < self.T:
            z = int(self.z_seq[self._t])
            v_bin = int(np.clip(self.v_bin_seq[self._t], 0, 1))
        else:
            z = 0
            v_bin = 0
        s = st.state_to_index(z, self._i, self._o_side, self._o_age, v_bin, self.Imax, self.z_bins)
        return s, {"t": self._t, "z": z, "i": self._i, "o_side": self._o_side, "o_age": self._o_age, "v_bin": v_bin}

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        Execute action. Returns (next_state_index, reward, done, info).
        Reward: r = i_next * y_t - c(fill)*|Δi| - λ*i_next² - η*1{|Δi|>0}.
        """
        cfg = self.config
        t = self._t
        if t >= self.T:
            return st.state_to_index(0, self._i, 0, 0, 0, self.Imax, self.z_bins), 0.0, True, {"t": t}

        z = int(self.z_seq[t])
        y = float(self.y_seq[t])
        v_bin = int(np.clip(self.v_bin_seq[t], 0, 1))
        i_prev = self._i
        o_side = self._o_side
        o_age = self._o_age

        # Cancel existing order if we place or market
        if action in (st.A_PLACE_BUY, st.A_PLACE_SELL, st.A_BUY_MARKET, st.A_SELL_MARKET) and o_side != 0:
            o_side = 0
            o_age = 0

        di = 0
        cost_bps = 0.0
        fill_type = "none"  # "maker" | "taker" | "none"

        if action == st.A_HOLD:
            # Resolve limit fill if we have an order
            if o_side != 0:
                p = p_fill(o_side, z, v_bin, o_age, cfg)
                if self._rng.random() < p:
                    di = o_side  # buy +1, sell -1
                    cost_bps = cfg.c_maker_bps
                    fill_type = "maker"
                    o_side = 0
                    o_age = 0
            # Next step: if order still there, it becomes stale
            if o_side != 0:
                o_age = 2

        elif action == st.A_PLACE_BUY:
            o_side = 1
            o_age = 1
            p = p_fill(1, z, v_bin, 1, cfg)
            if self._rng.random() < p:
                di = 1
                cost_bps = cfg.c_maker_bps
                fill_type = "maker"
                o_side = 0
                o_age = 0

        elif action == st.A_PLACE_SELL:
            o_side = -1
            o_age = 1
            p = p_fill(-1, z, v_bin, 1, cfg)
            if self._rng.random() < p:
                di = -1
                cost_bps = cfg.c_maker_bps
                fill_type = "maker"
                o_side = 0
                o_age = 0

        elif action == st.A_BUY_MARKET:
            di = 1
            cost_bps = cfg.c_taker_bps
            fill_type = "taker"

        elif action == st.A_SELL_MARKET:
            di = -1
            cost_bps = cfg.c_taker_bps
            fill_type = "taker"

        i_next = int(np.clip(i_prev + di, -self.Imax, self.Imax))
        # If we would exceed bounds, no fill (e.g. limit that would push over)
        if abs(i_next) > self.Imax:
            i_next = i_prev
            di = 0
            cost_bps = 0.0
            if action in (st.A_PLACE_BUY, st.A_PLACE_SELL):
                o_side = 1 if action == st.A_PLACE_BUY else -1
                o_age = 1

        # Reward: r = i_next * y - c*|Δi| - λ*i_next² - η*1{|Δi|>0}
        r = i_next * y - cost_bps * abs(di) - cfg.lambda_inv * (i_next ** 2) - cfg.eta_turnover * (1.0 if di != 0 else 0.0)

        self._i = i_next
        self._o_side = o_side
        # Outstanding order becomes stale next step
        self._o_age = 2 if o_side != 0 else 0
        self._t = t + 1

        if self._t < self.T:
            z_next = int(self.z_seq[self._t])
            v_bin_next = int(np.clip(self.v_bin_seq[self._t], 0, 1))
        else:
            z_next = 0
            v_bin_next = 0
        next_s = st.state_to_index(z_next, self._i, self._o_side, self._o_age, v_bin_next, self.Imax, self.z_bins)
        done = self._t >= self.T
        info = {"t": self._t, "di": di, "i": self._i, "y": y, "fill_type": fill_type, "z": z}
        return next_s, r, done, info
