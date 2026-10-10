"""Numerically guarded Phase-Type / MGF calculations for the CTMC prognosis model."""
from pathlib import Path

import numpy as np
from scipy.linalg import solve

STATE_NAMES = ["Low Risk", "Medium Risk", "High Risk"]
NUM_STATES = 4
CRITICAL_STATE = 3
MAX_CONDITION_NUMBER = 1e12
TOL = 1e-10


def validate_transient_generator(T):
    """Validate a transient CTMC sub-generator and eventual absorption."""
    T = np.asarray(T, dtype=float)
    if T.ndim != 2 or T.shape[0] != T.shape[1] or T.shape[0] == 0:
        raise ValueError("Transient generator T must be a non-empty square matrix.")
    if not np.isfinite(T).all():
        raise ValueError("Transient generator T contains non-finite values.")

    off_diagonal = T.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    if np.any(off_diagonal < -TOL):
        raise ValueError("Transient generator off-diagonal rates must be non-negative.")
    if np.any(T.sum(axis=1) > TOL):
        raise ValueError("Transient generator row sums must be non-positive.")

    absorption_rates = -T @ np.ones(T.shape[0])
    if np.any(absorption_rates < -TOL):
        raise ValueError("Transient generator implies negative absorption rates.")

    eigenvalues = np.linalg.eigvals(T)
    if not np.isfinite(eigenvalues).all() or np.any(np.real(eigenvalues) >= -TOL):
        raise ValueError(
            "Absorption is not guaranteed from every transient state; "
            "all eigenvalues of T must have strictly negative real parts."
        )

    condition = float(np.linalg.cond(-T))
    if not np.isfinite(condition) or condition > MAX_CONDITION_NUMBER:
        raise ValueError(
            f"Transient generator is singular or ill-conditioned (cond={condition:.3g})."
        )
    return T


def load_transient_generator(q_path="data/processed/q_matrix.npy"):
    """Load a 4-state Q and validate its transient sub-generator T."""
    path = Path(q_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run CTMC estimation first.")
    Q = np.asarray(np.load(path), dtype=float)

    if Q.shape != (NUM_STATES, NUM_STATES):
        raise ValueError(f"Expected a 4x4 Q matrix, got {Q.shape}.")
    if not np.isfinite(Q).all():
        raise ValueError("Q contains non-finite values.")
    if not np.allclose(Q.sum(axis=1), 0.0, atol=TOL):
        raise ValueError("Q must have row sums equal to zero.")

    off_diagonal = Q.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    if np.any(off_diagonal < -TOL):
        raise ValueError("Q off-diagonal rates must be non-negative.")
    if not np.allclose(Q[CRITICAL_STATE], 0.0, atol=TOL):
        raise ValueError("State 3 (Critical) must be absorbing.")

    T = validate_transient_generator(Q[:CRITICAL_STATE, :CRITICAL_STATE])
    return Q, T


def phase_type_mgf(s, initial_state, T):
    """Evaluate the Phase-Type MGF M(s)=alpha[-(T+sI)]^-1 t.

    MGF values are only valid where the system matrix is nonsingular and the
    mathematical moment-generating function exists.
    """
    T = validate_transient_generator(T)
    if not 0 <= initial_state < T.shape[0]:
        raise ValueError(f"initial_state must be in [0, {T.shape[0] - 1}].")
    s = float(s)
    if not np.isfinite(s):
        raise ValueError("s must be finite.")
    spectral_abscissa = float(np.max(np.real(np.linalg.eigvals(T))))
    if spectral_abscissa + s >= -TOL:
        raise ValueError(
            "MGF does not exist at this s: T + sI is not strictly stable."
        )

    alpha = np.zeros(T.shape[0])
    alpha[initial_state] = 1.0
    absorption = -T @ np.ones(T.shape[0])
    system = -(T + s * np.eye(T.shape[0]))
    condition = float(np.linalg.cond(system))
    if not np.isfinite(condition) or condition > MAX_CONDITION_NUMBER:
        raise ValueError("MGF system is singular or numerically ill-conditioned.")
    try:
        solution = solve(system, absorption, assume_a="gen")
    except (np.linalg.LinAlgError, ValueError) as exc:
        raise ValueError("Could not solve the Phase-Type MGF system.") from exc
    value = float(alpha @ solution)
    if not np.isfinite(value):
        raise ValueError("Phase-Type MGF produced a non-finite value.")
    return value


def phase_type_moments(T):
    """Return finite mean, variance and standard deviation by starting state."""
    T = validate_transient_generator(T)
    system = -T
    ones = np.ones(T.shape[0])
    try:
        first_moment = solve(system, ones, assume_a="gen")
        second_moment = 2.0 * solve(system, first_moment, assume_a="gen")
    except (np.linalg.LinAlgError, ValueError) as exc:
        raise ValueError("Could not solve the Phase-Type moment equations.") from exc

    variance = second_moment - np.square(first_moment)
    if (
        not np.isfinite(first_moment).all()
        or not np.isfinite(second_moment).all()
        or not np.isfinite(variance).all()
        or np.any(first_moment <= 0)
        or np.any(variance < -1e-8)
    ):
        raise ValueError("Invalid Phase-Type moments; check the estimated Q matrix.")

    # Clamp only tiny negative round-off to zero; materially negative variance fails above.
    variance = np.maximum(variance, 0.0)
    stds = np.sqrt(variance)
    return first_moment, variance, stds


def calculate_expected_time(q_path="data/processed/q_matrix.npy"):
    _, T = load_transient_generator(q_path)
    means, variances, stds = phase_type_moments(T)

    print("--- Phase-Type prognosis: expected time under model assumptions ---")
    print("Transient generator T:")
    print(np.round(T, 6))

    for i, state in enumerate(STATE_NAMES):
        print(
            f"{state}: expected absorption time={means[i]:.2f} h, "
            f"SD={stds[i]:.2f} h, Var={variances[i]:.2f} h^2"
        )

    for i in range(T.shape[0]):
        value_at_zero = phase_type_mgf(0.0, i, T)
        if not np.isclose(value_at_zero, 1.0, atol=1e-8):
            raise RuntimeError("Phase-Type MGF sanity check failed.")
    print("Interpretation: these are model-based population expectations, not individual countdowns.")
    return means, variances, stds


if __name__ == "__main__":
    calculate_expected_time()
