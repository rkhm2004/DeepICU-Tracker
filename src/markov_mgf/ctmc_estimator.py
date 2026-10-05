from pathlib import Path

import numpy as np
import pandas as pd
import torch


NUM_STATES = 4
CRITICAL_STATE = 3
DEFAULT_DT_HOURS = 1.0


def estimate_ctmc(
    latent_path="data/processed/latent_states.csv",
    output_path="data/processed/q_matrix.npy",
    dt_hours=DEFAULT_DT_HOURS,
):
    """Estimate a CTMC generator from patient-aware state trajectories.

    For fully observed continuous-time paths, the MLE is
        q_ij = N_ij / T_i,  i != j
        q_ii = -sum_{j != i} q_ij
    where N_ij is the number of observed transitions i -> j and T_i is
    the total observed time spent in state i.

    The data are hourly snapshots, so exposure is approximated by the
    one-hour interval between consecutive observations. Transitions are
    counted only within the same ICU stay and only across consecutive hours.

    Critical is treated as absorbing, matching the proposed Phase-Type model.
    Once a trajectory first reaches Critical, later observations from that
    stay are not used to estimate transient-state rates.
    """
    path = Path(latent_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src/vae/train.py first."
        )

    df = pd.read_csv(path)
    required = {"stay_id", "hour", "state"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    df["state"] = df["state"].astype(int)

    if not df["state"].between(0, NUM_STATES - 1).all():
        raise ValueError("State values must be integers in [0, 3].")

    transition_counts = np.zeros((NUM_STATES, NUM_STATES), dtype=float)
    exposure_hours = np.zeros(NUM_STATES, dtype=float)

    print("Estimating CTMC from patient-specific trajectories...")

    for _, stay in df.groupby("stay_id", sort=False):
        stay = stay.sort_values("hour")
        rows = stay[["hour", "state"]].to_numpy()

        for current, nxt in zip(rows[:-1], rows[1:]):
            current_hour, current_state = int(current[0]), int(current[1])
            next_hour, next_state = int(nxt[0]), int(nxt[1])
            gap = next_hour - current_hour

            # The parser creates a complete hourly grid. This guard prevents
            # accidental transitions across missing/non-consecutive records.
            if gap != 1:
                continue

            if current_state == CRITICAL_STATE:
                # Critical is absorbing in the mathematical model.
                break

            exposure_hours[current_state] += dt_hours

            if next_state != current_state:
                transition_counts[current_state, next_state] += 1.0

            if next_state == CRITICAL_STATE:
                # Stop at first absorption.
                break

    Q = np.zeros((NUM_STATES, NUM_STATES), dtype=float)

    for i in range(CRITICAL_STATE):
        if exposure_hours[i] > 0:
            for j in range(NUM_STATES):
                if i != j and transition_counts[i, j] > 0:
                    Q[i, j] = transition_counts[i, j] / exposure_hours[i]
        Q[i, i] = -np.sum(Q[i, :])

    # Critical is absorbing.
    Q[CRITICAL_STATE, :] = 0.0

    if not np.allclose(Q.sum(axis=1), 0.0, atol=1e-10):
        raise RuntimeError("Estimated Q is not a valid row-sum-zero generator.")
    if np.any(Q[:CRITICAL_STATE, :CRITICAL_STATE][~np.eye(CRITICAL_STATE, dtype=bool)] < -1e-12):
        raise RuntimeError("CTMC generator has a negative off-diagonal rate.")

    np.save(output_path, Q)
    np.save("data/processed/ctmc_transition_counts.npy", transition_counts)
    np.save("data/processed/ctmc_exposure_hours.npy", exposure_hours)

    print("\n--- Estimated CTMC Q Matrix (rates/hour) ---")
    print(np.round(Q, 6))
    print("\nTransition counts:")
    print(transition_counts.astype(int))
    print("\nExposure hours by state:")
    print(np.round(exposure_hours, 2))
    print(f"\nSaved Q matrix to {output_path}")

    return Q


def transition_matrix(Q, t_hours):
    """Compute P(t) = exp(Q t) for the CTMC."""
    from scipy.linalg import expm

    return expm(Q * float(t_hours))


if __name__ == "__main__":
    estimate_ctmc()
