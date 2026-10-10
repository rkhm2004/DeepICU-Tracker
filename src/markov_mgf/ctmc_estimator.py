from pathlib import Path

import numpy as np
import pandas as pd


NUM_STATES = 4
CRITICAL_STATE = 3
DEFAULT_DT_HOURS = 1.0


def estimate_generator(df, dt_hours=DEFAULT_DT_HOURS):
    """Estimate a 4-state absorbing CTMC generator from patient trajectories.

    For fully observed continuous-time paths, the MLE is
        q_ij = N_ij / T_i,  i != j
        q_ii = -sum(q_ij)
    where N_ij is the number of observed transitions and T_i is total
    observed exposure time in state i.

    The supplied data are hourly snapshots, so exposure is approximated by the
    one-hour interval between consecutive observations. Transitions are counted
    only within the same ICU stay and only across consecutive hours.
    Critical (state 3) is absorbing, matching the Phase-Type model.
    """
    required = {"stay_id", "hour", "state"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if not np.isfinite(float(dt_hours)) or float(dt_hours) <= 0:
        raise ValueError("dt_hours must be a finite positive number.")

    data = df.sort_values(["stay_id", "hour"]).reset_index(drop=True).copy()
    if data.duplicated(["stay_id", "hour"]).any():
        raise ValueError("Duplicate stay_id/hour observations cannot define a CTMC trajectory.")
    numeric_states = pd.to_numeric(data["state"], errors="coerce")
    if numeric_states.isna().any() or not np.equal(numeric_states, np.floor(numeric_states)).all():
        raise ValueError("State values must be finite integers in [0, 3].")
    data["state"] = numeric_states.astype(int)

    if not data["state"].between(0, NUM_STATES - 1).all():
        raise ValueError("State values must be integers in [0, 3].")
    numeric_hours = pd.to_numeric(data["hour"], errors="coerce")
    if (
        not np.isfinite(numeric_hours).all()
        or not np.equal(numeric_hours, np.floor(numeric_hours)).all()
        or (numeric_hours < 0).any()
    ):
        raise ValueError("Hour values must be non-negative finite integers.")
    data["hour"] = numeric_hours.astype(int)

    transition_counts = np.zeros((NUM_STATES, NUM_STATES), dtype=float)
    exposure_hours = np.zeros(NUM_STATES, dtype=float)

    for _, stay in data.groupby("stay_id", sort=False):
        rows = stay[["hour", "state"]].to_numpy()

        for current, nxt in zip(rows[:-1], rows[1:]):
            current_hour, current_state = int(current[0]), int(current[1])
            next_hour, next_state = int(nxt[0]), int(nxt[1])

            if next_hour - current_hour != 1:
                continue

            if current_state == CRITICAL_STATE:
                break

            exposure_hours[current_state] += dt_hours

            if next_state != current_state:
                transition_counts[current_state, next_state] += 1.0

            if next_state == CRITICAL_STATE:
                break

    Q = np.zeros((NUM_STATES, NUM_STATES), dtype=float)
    for i in range(CRITICAL_STATE):
        if exposure_hours[i] > 0:
            for j in range(NUM_STATES):
                if i != j and transition_counts[i, j] > 0:
                    Q[i, j] = transition_counts[i, j] / exposure_hours[i]
        Q[i, i] = -np.sum(Q[i, :])

    Q[CRITICAL_STATE, :] = 0.0

    if not np.allclose(Q.sum(axis=1), 0.0, atol=1e-10):
        raise RuntimeError("Estimated Q is not a valid row-sum-zero generator.")
    for i in range(CRITICAL_STATE):
        for j in range(NUM_STATES):
            if i != j and Q[i, j] < -1e-12:
                raise RuntimeError("CTMC off-diagonal rates must be non-negative.")

    return Q, transition_counts, exposure_hours


def estimate_ctmc(
    latent_path="data/processed/latent_states.csv",
    output_path="data/processed/q_matrix.npy",
    dt_hours=DEFAULT_DT_HOURS,
    training_only=True,
):
    path = Path(latent_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src/vae/train.py first."
        )

    df = pd.read_csv(path)
    if training_only:
        if "split" not in df.columns:
            raise ValueError(
                "latent_states.csv must contain a split column when training_only=True."
            )
        df = df[df["split"] == "train"].copy()
        if df.empty:
            raise ValueError("The training split contains no trajectory records.")
        print(
            f"Estimating CTMC using training split only: "
            f"{len(df):,} records / {df['stay_id'].nunique():,} ICU stays."
        )

    Q, transition_counts, exposure_hours = estimate_generator(df, dt_hours)

    np.save(output_path, Q)
    np.save("data/processed/ctmc_transition_counts.npy", transition_counts)
    np.save("data/processed/ctmc_exposure_hours.npy", exposure_hours)

    print("--- Estimated CTMC Q Matrix (rates/hour) ---")
    print(np.round(Q, 6))
    print("\nTransition counts:")
    print(transition_counts.astype(int))
    print("\nExposure hours by state:")
    print(np.round(exposure_hours, 2))
    print(f"\nSaved Q matrix to {output_path}")

    return Q


def transition_matrix(Q, t_hours):
    """Compute P(t) = exp(Q t), the CTMC transition probability matrix."""
    from scipy.linalg import expm

    Q = np.asarray(Q, dtype=float)
    t_hours = float(t_hours)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1] or Q.shape[0] == 0:
        raise ValueError("Q must be a non-empty square matrix.")
    if not np.isfinite(Q).all() or not np.isfinite(t_hours) or t_hours < 0:
        raise ValueError("Q and t_hours must be finite, with t_hours non-negative.")
    if not np.allclose(Q.sum(axis=1), 0.0, atol=1e-8):
        raise ValueError("Q must have row sums equal to zero.")
    off_diagonal = Q.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    if np.any(off_diagonal < -1e-10):
        raise ValueError("Q off-diagonal rates must be non-negative.")
    return expm(Q * t_hours)


if __name__ == "__main__":
    estimate_ctmc()
