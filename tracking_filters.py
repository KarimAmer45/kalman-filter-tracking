import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


"""Exercise 07 (MA-INF 2201 Computer Vision WS25/26) - solution.py

This file implements the *coding* parts of Sheet07:
  - Task 2: Linear Kalman Filter for 2D position measurements with velocity+acceleration state
  - Task 3: Fixed-lag smoothing (lag = 5) built on top of the Kalman filter
  - Task 4: Unicycle motion model simulation + Extended Kalman Filter (EKF)
"""


# ================================================================
# Utilities
# ================================================================

def _ensure_output_dir() -> Path:
    out_dir = Path(__file__).resolve().parent / "outputs_sheet07"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _find_observations_file() -> Path:
    """Try to locate observations.npy in the typical locations.

    The sheet mentions: data/observations.npy
    The user note mentions: observations.npy exists within data folder

    In the provided environment, the file may also be next to this script.
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here / "data" / "observations.npy",
        here / "observations.npy",
        Path("data") / "observations.npy",
        Path("observations.npy"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find observations.npy. Tried: " + ", ".join(str(c) for c in candidates)
    )


def _plot_xy(trajs, labels, title, save_path: Path):
    """Plot x-y trajectories for multiple sequences.

    trajs: list of arrays with shape (N,2)
    """
    plt.figure()
    for arr, lab in zip(trajs, labels):
        plt.plot(arr[:, 0], arr[:, 1], label=lab)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _plot_components(time_s, obs_xy, est_xy, title, save_path: Path):
    """Plot x(t) and y(t) for observations and estimates."""
    plt.figure()
    plt.plot(time_s, obs_xy[:, 0], label="obs x")
    plt.plot(time_s, est_xy[:, 0], label="est x")
    plt.title(title + " - x(t)")
    plt.xlabel("t [s]")
    plt.ylabel("x")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path.with_name(save_path.stem + "_x.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(time_s, obs_xy[:, 1], label="obs y")
    plt.plot(time_s, est_xy[:, 1], label="est y")
    plt.title(title + " - y(t)")
    plt.xlabel("t [s]")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path.with_name(save_path.stem + "_y.png"), dpi=200)
    plt.close()


# ================================================================
# Task 2: Kalman Filter (Linear)
# ================================================================

def make_kf_model(dt: float = 0.1, sp: float = 0.001, sm: float = 0.05):
    """Create the matrices for the linear KF in Task 2.

    State: [x, y, vx, vy, ax, ay]^T

    Time evolution (given in sheet):
        x_t = Psi * x_{t-1} + w_t
    Measurement:
        m_t = Phi * x_t + v_t

    with diagonal Q = Sigma_p, R = Sigma_m.
    """
    # State transition Psi (6x6)
    Psi = np.array(
        [
            [1, 0, dt, 0, (dt ** 2) / 2.0, 0],
            [0, 1, 0, dt, 0, (dt ** 2) / 2.0],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=float,
    )

    # Measurement matrix Phi (2x6): observe only x,y
    Phi = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ],
        dtype=float,
    )

    # Covariances (diagonal as required)
    Q = sp * np.eye(6, dtype=float)
    R = sm * np.eye(2, dtype=float)

    return Psi, Phi, Q, R


def kalman_filter(observations_xy: np.ndarray,
                  dt: float = 0.1,
                  sp: float = 0.001,
                  sm: float = 0.05,
                  x0: np.ndarray | None = None,
                  P0: np.ndarray | None = None):
    """Run a standard linear Kalman filter.

    observations_xy: (N,2)

    Returns:
        mu_filt: (N,6) filtered means (after measurement update)
        P_filt: (N,6,6) filtered covariances
        mu_pred: (N,6) predicted means (before measurement update)
        P_pred: (N,6,6) predicted covariances
        Psi, Phi, Q, R: model matrices
    """
    Psi, Phi, Q, R = make_kf_model(dt=dt, sp=sp, sm=sm)

    N = int(observations_xy.shape[0])

    if x0 is None:
        x0 = np.array([-10.0, -150.0, 1.0, -2.0, 0.0, 0.0], dtype=float)
    if P0 is None:
        # Not specified in the sheet; choose a reasonable initial uncertainty.
        # (Large enough to adapt quickly, but not too large to explode numerically.)
        P0 = np.diag([10.0, 10.0, 10.0, 10.0, 1.0, 1.0]).astype(float)

    mu = x0.reshape(6,)
    P = P0.copy()

    mu_filt = np.zeros((N, 6), dtype=float)
    P_filt = np.zeros((N, 6, 6), dtype=float)
    mu_pred = np.zeros((N, 6), dtype=float)
    P_pred = np.zeros((N, 6, 6), dtype=float)

    I = np.eye(6, dtype=float)

    for t in range(N):
        z = observations_xy[t].reshape(2,)

        # -------------------------
        # Prediction (time update)
        # -------------------------
        mu_bar = Psi @ mu
        P_bar = Psi @ P @ Psi.T + Q

        mu_pred[t] = mu_bar
        P_pred[t] = P_bar

        # -------------------------
        # Correction (measurement update)
        # -------------------------
        # Innovation covariance
        S = Phi @ P_bar @ Phi.T + R
        # Kalman gain
        K = P_bar @ Phi.T @ np.linalg.inv(S)
        # Innovation (measurement residual)
        y = z - (Phi @ mu_bar)
        # Updated mean
        mu = mu_bar + K @ y

        # Updated covariance (Joseph form for numerical stability)
        KH = K @ Phi
        P = (I - KH) @ P_bar @ (I - KH).T + K @ R @ K.T

        mu_filt[t] = mu
        P_filt[t] = P

    return mu_filt, P_filt, mu_pred, P_pred, Psi, Phi, Q, R


# ================================================================
# Task 3: Fixed-Lag Smoothing (lag=5)
# ================================================================

def fixed_lag_smoother_from_kf(mu_filt: np.ndarray,
                               P_filt: np.ndarray,
                               mu_pred: np.ndarray,
                               P_pred: np.ndarray,
                               Psi: np.ndarray,
                               lag: int = 5):
    """Compute fixed-lag smoothed estimates using a short RTS recursion.

    Fixed-lag smoothing with lag L provides an estimate for time t using measurements
    up to time t+L. Since L is small (5), we can compute it efficiently by smoothing
    only over the window [t, t+L] for each t.

    Inputs are the results of the forward KF pass.

    Returns:
        mu_s_lag: (N,6) smoothed means with lag L
        P_s_lag:  (N,6,6) smoothed covariances with lag L
    """
    N = mu_filt.shape[0]
    mu_s_lag = np.zeros_like(mu_filt)
    P_s_lag = np.zeros_like(P_filt)

    for t in range(N):
        k_end = min(N - 1, t + lag)

        # Start from the filtered estimate at k_end
        mu_s = mu_filt[k_end].copy()
        P_s = P_filt[k_end].copy()

        # Backward RTS recursion down to time t
        for k in range(k_end - 1, t - 1, -1):
            # Smoother gain
            C = P_filt[k] @ Psi.T @ np.linalg.inv(P_pred[k + 1])

            mu_s_prev = mu_filt[k] + C @ (mu_s - mu_pred[k + 1])
            P_s_prev = P_filt[k] + C @ (P_s - P_pred[k + 1]) @ C.T

            mu_s, P_s = mu_s_prev, P_s_prev

        mu_s_lag[t] = mu_s
        P_s_lag[t] = P_s

    return mu_s_lag, P_s_lag


# ================================================================
# Task 4: Unicycle model + Extended Kalman Filter (EKF)
# ================================================================

def unicycle_g(x: np.ndarray, t: int, dt: float) -> np.ndarray:
    """Nonlinear time evolution g(x_t) for the unicycle model.

    State x = [x, y, theta, v]^T

    From the sheet:
        x_{t+1} = [ x + dt*v*cos(theta),
                    y + dt*v*sin(theta),
                    theta,
                    v ]^T

    For data generation, theta is driven by:
        theta_t = theta_{t-1} + 0.6*sin(0.2*t*dt)*dt

    We incorporate this as a known deterministic term in g(x,t).
    """
    px, py, th, v = x
    th_next = th + 0.6 * np.sin(0.2 * t * dt) * dt
    return np.array(
        [
            px + dt * v * np.cos(th),
            py + dt * v * np.sin(th),
            th_next,
            v,
        ],
        dtype=float,
    )


def unicycle_G_jacobian(x: np.ndarray, dt: float) -> np.ndarray:
    """Jacobian of g(x) w.r.t state x.

    g1 = x + dt*v*cos(theta)
    g2 = y + dt*v*sin(theta)
    g3 = theta + (known term in t)  -> derivative w.r.t theta is 1
    g4 = v
    """
    _, _, th, v = x
    G = np.array(
        [
            [1.0, 0.0, -dt * v * np.sin(th), dt * np.cos(th)],
            [0.0, 1.0,  dt * v * np.cos(th), dt * np.sin(th)],
            [0.0, 0.0,  1.0,                 0.0],
            [0.0, 0.0,  0.0,                 1.0],
        ],
        dtype=float,
    )
    return G


def unicycle_h(x: np.ndarray) -> np.ndarray:
    """Measurement model h(x): observe only position (x,y)."""
    return np.array([x[0], x[1]], dtype=float)


def unicycle_H_jacobian() -> np.ndarray:
    """Jacobian of h(x) w.r.t x is constant: [[1,0,0,0],[0,1,0,0]]."""
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=float,
    )


def simulate_unicycle(T: int = 200,
                     dt: float = 0.1,
                     x0: np.ndarray | None = None,
                     eps_std: np.ndarray | None = None,
                     delta_std: np.ndarray | None = None,
                     seed: int = 0):
    """Generate ground-truth states and noisy observations for the unicycle model."""
    if x0 is None:
        x0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    if eps_std is None:
        eps_std = np.array([0.001, 0.001, 0.001, 0.001], dtype=float)
    if delta_std is None:
        delta_std = np.array([0.005, 0.005], dtype=float)

    rng = np.random.default_rng(seed)

    states = np.zeros((T, 4), dtype=float)
    obs = np.zeros((T, 2), dtype=float)

    states[0] = x0
    obs[0] = unicycle_h(states[0]) + rng.normal(0.0, delta_std, size=(2,))

    for t in range(0, T - 1):
        # Deterministic evolution
        x_det = unicycle_g(states[t], t=t, dt=dt)
        # Add process noise epsilon_t
        eps = rng.normal(0.0, eps_std, size=(4,))
        states[t + 1] = x_det + eps
        # Measurement + measurement noise
        obs[t + 1] = unicycle_h(states[t + 1]) + rng.normal(0.0, delta_std, size=(2,))

    return states, obs


def extended_kalman_filter_unicycle(observations_xy: np.ndarray,
                                    dt: float = 0.1,
                                    x0: np.ndarray | None = None,
                                    P0: np.ndarray | None = None,
                                    eps_std: np.ndarray | None = None,
                                    delta_std: np.ndarray | None = None):
    """Extended Kalman Filter for the unicycle model."""
    if x0 is None:
        x0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    if P0 is None:
        P0 = np.diag([0.1, 0.1, 0.1, 0.1]).astype(float)
    if eps_std is None:
        eps_std = np.array([0.001, 0.001, 0.001, 0.001], dtype=float)
    if delta_std is None:
        delta_std = np.array([0.005, 0.005], dtype=float)

    Q = np.diag(eps_std ** 2).astype(float)
    R = np.diag(delta_std ** 2).astype(float)

    H = unicycle_H_jacobian()

    T = int(observations_xy.shape[0])
    mu = x0.copy()
    P = P0.copy()

    mu_filt = np.zeros((T, 4), dtype=float)
    P_filt = np.zeros((T, 4, 4), dtype=float)

    I = np.eye(4, dtype=float)

    for t in range(T):
        z = observations_xy[t].reshape(2,)

        # -------------------------
        # Prediction
        # -------------------------
        mu_bar = unicycle_g(mu, t=t, dt=dt)
        G = unicycle_G_jacobian(mu, dt=dt)
        P_bar = G @ P @ G.T + Q

        # -------------------------
        # Update
        # -------------------------
        z_hat = unicycle_h(mu_bar)
        y = z - z_hat
        S = H @ P_bar @ H.T + R
        K = P_bar @ H.T @ np.linalg.inv(S)

        mu = mu_bar + K @ y

        # Joseph form
        KH = K @ H
        P = (I - KH) @ P_bar @ (I - KH).T + K @ R @ K.T

        mu_filt[t] = mu
        P_filt[t] = P

    return mu_filt, P_filt


# ================================================================
# Main
# ================================================================

def main():
    out_dir = _ensure_output_dir()

    # -----------------------------
    # Load observations for Task 2/3
    # -----------------------------
    obs_path = _find_observations_file()
    observations = np.load(obs_path)

    if observations.ndim != 2 or observations.shape[1] != 2:
        raise ValueError(f"observations must have shape (N,2). Got {observations.shape}")

    dt = 0.1

    # ============================================================
    # TASK 2: Kalman Filter
    # ============================================================
    mu_filt, P_filt, mu_pred, P_pred, Psi, Phi, Q, R = kalman_filter(
        observations_xy=observations,
        dt=dt,
        sp=0.001,
        sm=0.05,
        x0=np.array([-10.0, -150.0, 1.0, -2.0, 0.0, 0.0], dtype=float),
        P0=None,
    )

    est_kf_xy = mu_filt[:, 0:2]

    # Visualization: observations vs filtered estimates
    _plot_xy(
        trajs=[observations, est_kf_xy],
        labels=["observations", "KF estimate"],
        title="Task 2 - Kalman Filter (x-y)",
        save_path=out_dir / "task2_kf_xy.png",
    )

    time_s = np.arange(observations.shape[0]) * dt
    _plot_components(
        time_s=time_s,
        obs_xy=observations,
        est_xy=est_kf_xy,
        title="Task 2 - Kalman Filter",
        save_path=out_dir / "task2_kf_components.png",
    )

    # ============================================================
    # TASK 3: Fixed-lag smoothing (lag=5)
    # ============================================================
    lag = 5
    mu_s_lag, P_s_lag = fixed_lag_smoother_from_kf(
        mu_filt=mu_filt,
        P_filt=P_filt,
        mu_pred=mu_pred,
        P_pred=P_pred,
        Psi=Psi,
        lag=lag,
    )

    est_smooth_xy = mu_s_lag[:, 0:2]

    _plot_xy(
        trajs=[observations, est_kf_xy, est_smooth_xy],
        labels=["observations", "KF", f"fixed-lag smoother (L={lag})"],
        title=f"Task 3 - Fixed-lag smoothing (x-y, L={lag})",
        save_path=out_dir / "task3_fixed_lag_xy.png",
    )

    _plot_components(
        time_s=time_s,
        obs_xy=observations,
        est_xy=est_smooth_xy,
        title=f"Task 3 - Fixed-lag smoothing (L={lag})",
        save_path=out_dir / "task3_fixed_lag_components.png",
    )

    # ============================================================
    # TASK 4: Unicycle model + EKF
    # ============================================================
    T = 200

    # (a) Low measurement noise: delta = [0.005, 0.005]
    true_states, obs_unicycle_low = simulate_unicycle(
        T=T,
        dt=dt,
        x0=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
        eps_std=np.array([0.001, 0.001, 0.001, 0.001], dtype=float),
        delta_std=np.array([0.005, 0.005], dtype=float),
        seed=0,
    )
    ekf_est_low, _ = extended_kalman_filter_unicycle(
        observations_xy=obs_unicycle_low,
        dt=dt,
        x0=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
        P0=None,
        eps_std=np.array([0.001, 0.001, 0.001, 0.001], dtype=float),
        delta_std=np.array([0.005, 0.005], dtype=float),
    )

    _plot_xy(
        trajs=[true_states[:, 0:2], obs_unicycle_low, ekf_est_low[:, 0:2]],
        labels=["true", "observations", "EKF"],
        title="Task 4 - Unicycle EKF (delta=[0.005,0.005])",
        save_path=out_dir / "task4_ekf_low_noise_xy.png",
    )

    # (b) Higher measurement noise: delta = [0.05, 0.05]
    true_states2, obs_unicycle_high = simulate_unicycle(
        T=T,
        dt=dt,
        x0=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
        eps_std=np.array([0.001, 0.001, 0.001, 0.001], dtype=float),
        delta_std=np.array([0.05, 0.05], dtype=float),
        seed=0,
    )
    ekf_est_high, _ = extended_kalman_filter_unicycle(
        observations_xy=obs_unicycle_high,
        dt=dt,
        x0=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
        P0=None,
        eps_std=np.array([0.001, 0.001, 0.001, 0.001], dtype=float),
        delta_std=np.array([0.05, 0.05], dtype=float),
    )

    _plot_xy(
        trajs=[true_states2[:, 0:2], obs_unicycle_high, ekf_est_high[:, 0:2]],
        labels=["true", "observations", "EKF"],
        title="Task 4 - Unicycle EKF (delta=[0.05,0.05])",
        save_path=out_dir / "task4_ekf_high_noise_xy.png",
    )

    # Optional: plot theta(t) and v(t) for EKF (helpful diagnostics)
    t_axis = np.arange(T) * dt

    plt.figure()
    plt.plot(t_axis, true_states[:, 2], label="true theta")
    plt.plot(t_axis, ekf_est_low[:, 2], label="EKF theta")
    plt.title("Task 4 - theta(t) with low measurement noise")
    plt.xlabel("t [s]")
    plt.ylabel("theta")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "task4_theta_low_noise.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(t_axis, true_states[:, 3], label="true v")
    plt.plot(t_axis, ekf_est_low[:, 3], label="EKF v")
    plt.title("Task 4 - v(t) with low measurement noise")
    plt.xlabel("t [s]")
    plt.ylabel("v")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "task4_v_low_noise.png", dpi=200)
    plt.close()

    # Print where outputs are saved (useful when running from terminal)
    print(f"Loaded Task2/3 observations from: {obs_path}")
    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()