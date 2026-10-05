import numpy as np
from scipy.linalg import inv


STATE_NAMES = ["Low Risk", "Medium Risk", "High Risk"]


def load_transient_generator(q_path="data/processed/q_matrix.npy"):
    """Load Q and return the transient sub-generator T."""
    Q = np.asarray(np.load(q_path), dtype=float)

    if Q.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 Q matrix, got {Q.shape}.")
    if not np.allclose(Q.sum(axis=1), 0.0, atol=1e-10):
        raise ValueError("Q must have row sums equal to zero.")
    if not np.allclose(Q[3], 0.0, atol=1e-10):
        raise ValueError("State 3 (Critical) must be absorbing.")

    T = Q[:3, :3]
    try:
        inv(T)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "The transient generator T is singular; absorption time is not "
            "finite for all transient starting states."
        ) from exc

    return Q, T


def phase_type_mgf(s, initial_state, T):
    """Moment-generating function of the Phase-Type absorption time.

    For initial row vector alpha:
        M_tau(s) = alpha [-(T + sI)^(-1)] t
    where t = -T 1 is the absorption-rate vector.
    """
    if not 0 <= initial_state < T.shape[0]:
        raise ValueError("initial_state must be 0, 1, or 2.")

    alpha = np.zeros(T.shape[0])
    alpha[initial_state] = 1.0
    ones = np.ones(T.shape[0])
    absorption = -T @ ones
    system = -(T + float(s) * np.eye(T.shape[0]))

    return float(alpha @ inv(system) @ absorption)


def phase_type_moments(T):
    """Return mean, variance and standard deviation by starting state."""
    ones = np.ones(T.shape[0])
    T_inv = inv(T)

    first_moment = -T_inv @ ones
    second_moment = 2.0 * (T_inv @ T_inv) @ ones
    variance = second_moment - np.square(first_moment)

    # Small negative values can arise from floating-point round-off.
    variance = np.where(variance < 0, np.maximum(variance, -1e-10), variance)
    if np.any(variance < 0) or np.any(first_moment <= 0):
        raise ValueError("Invalid Phase-Type moments; check the estimated Q matrix.")

    return first_moment, variance, np.sqrt(variance)


def calculate_expected_time(q_path="data/processed/q_matrix.npy"):
    Q, T = load_transient_generator(q_path)
    means, variances, stds = phase_type_moments(T)

    print("--- Phase-Type Prognosis ---")
    print("Transient generator T:")
    print(np.round(T, 6))

    for i, state in enumerate(STATE_NAMES):
        print(
            f"{state}: E[tau]={means[i]:.2f} h, "
            f"SD={stds[i]:.2f} h, Var={variances[i]:.2f} h^2"
        )

    for i in range(3):
        value_at_zero = phase_type_mgf(0.0, i, T)
        if not np.isclose(value_at_zero, 1.0, atol=1e-8):
            raise RuntimeError("Phase-Type MGF sanity check failed.")

    return means, variances, stds


if __name__ == "__main__":
    calculate_expected_time()
