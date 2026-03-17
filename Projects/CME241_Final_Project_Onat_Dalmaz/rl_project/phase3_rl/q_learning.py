"""
Phase 3 RL: Tabular Q-learning (and optional SARSA) with ε-greedy and α/ε schedules.
"""

from typing import List, Optional, Tuple, Any
import numpy as np

from . import state as st
from .env import ExecutionEnv, ExecutionEnvConfig


def argmax_random_tiebreak(q_values: np.ndarray, rng: np.random.Generator) -> int:
    """Return one of the actions that maximize q_values, chosen uniformly (breaks HOLD-first bias when Q ties)."""
    max_q = np.max(q_values)
    (candidates,) = np.where(q_values >= max_q - 1e-9)
    return int(rng.choice(candidates))


def alpha_schedule(episode: int, alpha0: float, alpha_min: float, decay_episodes: int) -> float:
    """Linear decay to alpha_min over decay_episodes."""
    if decay_episodes <= 0:
        return alpha0
    frac = min(1.0, episode / max(1, decay_episodes))
    return max(alpha_min, alpha0 - frac * (alpha0 - alpha_min))


def epsilon_schedule(episode: int, eps0: float, eps_min: float, decay_episodes: int) -> float:
    """Linear decay to eps_min over decay_episodes."""
    if decay_episodes <= 0:
        return eps0
    frac = min(1.0, episode / max(1, decay_episodes))
    return max(eps_min, eps0 - frac * (eps0 - eps_min))


def train_q_learning(
    config: ExecutionEnvConfig,
    z_seqs: List[np.ndarray],
    y_seqs: List[np.ndarray],
    v_bin_seqs: List[np.ndarray],
    n_episodes: int,
    gamma: float = 0.99,
    alpha0: float = 0.2,
    alpha_min: float = 0.02,
    eps0: float = 0.3,
    eps_min: float = 0.05,
    decay_episodes: int = 5000,
    seed: Optional[int] = None,
    use_sarsa: bool = False,
    log_callback: Optional[callable] = None,
    q_init: float = 0.0,
    empirical_collector: Optional[Any] = None,
) -> Tuple[np.ndarray, List[dict]]:
    """
    Train tabular Q-learning on windows (z_seqs[k], y_seqs[k], v_bin_seqs[k]).
    Each episode picks a random window and random start in that window, runs to end.
    empirical_collector: if set, must have .add(s, a, r, s_next, done) for train-only transition logging.
    Returns (Q, log_rows).
    """
    rng = np.random.default_rng(seed)
    z_bins = getattr(config, "z_bins", 3)
    nS = st.n_states(config.Imax, z_bins)
    nA = st.N_ACTIONS
    Q = np.full((nS, nA), q_init, dtype=float)

    log_rows = []
    n_windows = len(z_seqs)
    if n_windows == 0:
        return Q, log_rows

    # Effective decay over training: decay over min(decay_episodes, n_episodes) so short runs still decay
    eff_decay = min(decay_episodes, max(1, n_episodes))

    for episode in range(n_episodes):
        alpha = alpha_schedule(episode, alpha0, alpha_min, eff_decay)
        eps = epsilon_schedule(episode, eps0, eps_min, eff_decay)
        w = int(rng.integers(0, n_windows))
        z_seq, y_seq, v_seq = z_seqs[w], y_seqs[w], v_bin_seqs[w]
        T = min(len(z_seq), len(y_seq), len(v_seq))
        if T < 2:
            continue
        start = int(rng.integers(0, max(1, T - 1)))
        env = ExecutionEnv(config, z_seq, y_seq, v_seq)
        s, _ = env.reset(start_idx=start, seed=int(rng.integers(0, 2**31)))
        ep_reward = 0.0
        steps = 0
        done = False
        a = int(rng.integers(0, nA)) if rng.random() < eps else argmax_random_tiebreak(Q[s, :], rng)

        while not done:
            s_next, r, done, info = env.step(a)
            if empirical_collector is not None:
                empirical_collector.add(s, a, float(r), s_next, done)
            ep_reward += r
            steps += 1
            a_next = int(rng.integers(0, nA)) if rng.random() < eps else argmax_random_tiebreak(Q[s_next, :], rng)
            if use_sarsa:
                target = r + gamma * Q[s_next, a_next]
            else:
                target = r + gamma * np.max(Q[s_next, :])
            Q[s, a] += alpha * (target - Q[s, a])
            s, a = s_next, a_next

        if log_callback:
            log_callback(episode, ep_reward, steps, alpha, eps)
        log_rows.append({
            "episode": episode,
            "ep_reward": ep_reward,
            "steps": steps,
            "alpha": alpha,
            "eps": eps,
        })

    return Q, log_rows


def train_double_q_learning(
    config: ExecutionEnvConfig,
    z_seqs: List[np.ndarray],
    y_seqs: List[np.ndarray],
    v_bin_seqs: List[np.ndarray],
    n_episodes: int,
    gamma: float = 0.99,
    alpha0: float = 0.2,
    alpha_min: float = 0.02,
    eps0: float = 0.3,
    eps_min: float = 0.05,
    decay_episodes: int = 5000,
    seed: Optional[int] = None,
    log_callback: Optional[callable] = None,
    q_init: float = 0.0,
    empirical_collector: Optional[Any] = None,
) -> Tuple[np.ndarray, List[dict]]:
    """Double Q-learning. empirical_collector: optional .add(s,a,r,s_next,done) for DP_empirical."""
    rng = np.random.default_rng(seed)
    z_bins = getattr(config, "z_bins", 3)
    nS = st.n_states(config.Imax, z_bins)
    nA = st.N_ACTIONS
    Q1 = np.full((nS, nA), q_init, dtype=float)
    Q2 = np.full((nS, nA), q_init, dtype=float)

    log_rows = []
    n_windows = len(z_seqs)
    if n_windows == 0:
        return Q1 + Q2, log_rows

    eff_decay = min(decay_episodes, max(1, n_episodes))

    for episode in range(n_episodes):
        alpha = alpha_schedule(episode, alpha0, alpha_min, eff_decay)
        eps = epsilon_schedule(episode, eps0, eps_min, eff_decay)
        w = int(rng.integers(0, n_windows))
        z_seq, y_seq, v_seq = z_seqs[w], y_seqs[w], v_bin_seqs[w]
        T = min(len(z_seq), len(y_seq), len(v_seq))
        if T < 2:
            continue
        start = int(rng.integers(0, max(1, T - 1)))
        env = ExecutionEnv(config, z_seq, y_seq, v_seq)
        s, _ = env.reset(start_idx=start, seed=int(rng.integers(0, 2**31)))
        ep_reward = 0.0
        steps = 0
        done = False
        Q = Q1 + Q2
        a = int(rng.integers(0, nA)) if rng.random() < eps else argmax_random_tiebreak(Q[s, :], rng)

        while not done:
            s_next, r, done, info = env.step(a)
            if empirical_collector is not None:
                empirical_collector.add(s, a, float(r), s_next, done)
            ep_reward += r
            steps += 1
            Q = Q1 + Q2
            a_next = int(rng.integers(0, nA)) if rng.random() < eps else argmax_random_tiebreak(Q[s_next, :], rng)
            if rng.random() < 0.5:
                target = r + gamma * Q2[s_next, argmax_random_tiebreak(Q1[s_next, :], rng)]
                Q1[s, a] += alpha * (target - Q1[s, a])
            else:
                target = r + gamma * Q1[s_next, argmax_random_tiebreak(Q2[s_next, :], rng)]
                Q2[s, a] += alpha * (target - Q2[s, a])
            s, a = s_next, a_next

        if log_callback:
            log_callback(episode, ep_reward, steps, alpha, eps)
        log_rows.append({
            "episode": episode,
            "ep_reward": ep_reward,
            "steps": steps,
            "alpha": alpha,
            "eps": eps,
        })

    return Q1 + Q2, log_rows
