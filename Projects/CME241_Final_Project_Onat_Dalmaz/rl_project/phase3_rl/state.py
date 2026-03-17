"""
Phase 3 RL: State encoding/decoding for execution env.
State = (z, i, o_side, o_age, v_bin).
z in {-1,0,+1}, i in {-Imax..+Imax}, o_side in {-1,0,+1}, o_age in {0,1,2}, v_bin in {0,1}.
"""

import numpy as np


def state_dimensions(Imax: int = 3, z_bins: int = 3):
    """Return (n_z, n_i, n_o_side, n_o_age, n_v_bin). z_bins=3 (default) or 5 for quintiles."""
    n_z = int(z_bins)
    n_i = 2 * Imax + 1
    n_o_side = 3
    n_o_age = 3
    n_v_bin = 2
    return n_z, n_i, n_o_side, n_o_age, n_v_bin


def state_to_index(z: int, i: int, o_side: int, o_age: int, v_bin: int, Imax: int = 3, z_bins: int = 3) -> int:
    """Map (z, i, o_side, o_age, v_bin) to integer in [0, n_states). z_bins=3: z in {-1,0,1}; z_bins=5: z in {0..4}."""
    n_z, n_i, n_o_side, n_o_age, n_v_bin = state_dimensions(Imax, z_bins)
    zx = (int(z) + 1) if n_z == 3 else int(z)
    zx = np.clip(zx, 0, n_z - 1)
    ix = int(i) + Imax
    ox = int(o_side) + 1
    ax = int(o_age)
    vx = int(v_bin)
    ix = np.clip(ix, 0, n_i - 1)
    ax = np.clip(ax, 0, n_o_age - 1)
    vx = np.clip(vx, 0, n_v_bin - 1)
    return zx * (n_i * n_o_side * n_o_age * n_v_bin) + ix * (n_o_side * n_o_age * n_v_bin) + ox * (n_o_age * n_v_bin) + ax * n_v_bin + vx


def index_to_state(idx: int, Imax: int = 3, z_bins: int = 3) -> tuple:
    """Map index to (z, i, o_side, o_age, v_bin)."""
    n_z, n_i, n_o_side, n_o_age, n_v_bin = state_dimensions(Imax, z_bins)
    rem = idx
    vx = rem % n_v_bin
    rem //= n_v_bin
    ax = rem % n_o_age
    rem //= n_o_age
    ox = rem % n_o_side
    rem //= n_o_side
    ix = rem % n_i
    rem //= n_i
    zx = rem % n_z
    z = (zx - 1) if n_z == 3 else zx
    i = ix - Imax
    o_side = ox - 1
    o_age = ax
    v_bin = vx
    return (z, i, o_side, o_age, v_bin)


def n_states(Imax: int = 3, z_bins: int = 3) -> int:
    """Total number of states."""
    n_z, n_i, n_o_side, n_o_age, n_v_bin = state_dimensions(Imax, z_bins)
    return n_z * n_i * n_o_side * n_o_age * n_v_bin


# Action constants (match env)
A_HOLD = 0
A_PLACE_BUY = 1
A_PLACE_SELL = 2
A_BUY_MARKET = 3
A_SELL_MARKET = 4
N_ACTIONS = 5

ACTION_NAMES = ["HOLD", "PLACE_BUY_LIMIT", "PLACE_SELL_LIMIT", "BUY_MARKET", "SELL_MARKET"]
