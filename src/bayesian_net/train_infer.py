"""Discrete Bayesian Network for ICU risk-state explanation.

Network structure:
    HeartRateLevel  ─┐
    SBPLevel        ─┼──> RiskState
    WBCLevel        ─┘

Feature discretization cut points and CPT parameters are learned from TRAIN
records only. The network uses the physiological variables currently available
in this repository; static demographic/admission priors can be added when those
features are included in the data pipeline.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["heart_rate", "sbp", "wbc"]
FEATURE_LABELS = {
    "heart_rate": "HeartRateLevel",
    "sbp": "SBPLevel",
    "wbc": "WBCLevel",
}
STATE_NAMES = ["Low Risk", "Medium Risk", "High Risk", "Critical"]
LEVEL_NAMES = ["Low", "Moderate", "High"]
NUM_LEVELS = 3
NUM_STATES = 4
ALPHA = 1.0  # Laplace smoothing


def fit_discretization(train_df, features=FEATURES):
    """Fit tertile cut points on training records only."""
    cutoffs = {}
    for feature in features:
        values = pd.to_numeric(train_df[feature], errors="coerce").dropna().to_numpy()
        if values.size == 0:
            raise ValueError(f"No training values available for {feature}.")
        q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
        # Duplicate cut points are permitted; digitize still returns valid bins.
        cutoffs[feature] = [float(q1), float(q2)]
    return cutoffs


def apply_discretization(df, cutoffs, features=FEATURES):
    """Map continuous features to 0=Low, 1=Moderate, 2=High."""
    result = pd.DataFrame(index=df.index)
    for feature in features:
        if feature not in df:
            raise ValueError(f"Missing feature column: {feature}")
        values = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)
        if np.isnan(values).any():
            raise ValueError(f"Missing values found in feature {feature}.")
        result[feature] = np.digitize(values, np.asarray(cutoffs[feature]), right=False)
    return result


def fit_bayesian_network(train_df, cutoffs, alpha=ALPHA):
    """Fit a smoothed discrete BN CPT: P(RiskState | HR, SBP, WBC)."""
    required = set(FEATURES + ["state"])
    missing = required - set(train_df.columns)
    if missing:
        raise ValueError(f"Missing columns required to fit BN: {sorted(missing)}")

    bins = apply_discretization(train_df, cutoffs)
    states = train_df["state"].astype(int).to_numpy()
    if not np.isin(states, np.arange(NUM_STATES)).all():
        raise ValueError("Risk states must be integers from 0 to 3.")

    # Shape: HR bin x SBP bin x WBC bin x risk state.
    counts = np.full((NUM_LEVELS, NUM_LEVELS, NUM_LEVELS, NUM_STATES), float(alpha))
    for row, state in zip(bins[FEATURES].to_numpy(dtype=int), states):
        counts[row[0], row[1], row[2], state] += 1.0
    cpt = counts / counts.sum(axis=3, keepdims=True)
    return cpt, counts


def infer_risk_probabilities(feature_df, cutoffs, cpt):
    """Return P(RiskState | discretized physiological features) for each row."""
    bins = apply_discretization(feature_df, cutoffs)
    probabilities = np.vstack([
        cpt[row[0], row[1], row[2], :]
        for row in bins[FEATURES].to_numpy(dtype=int)
    ])
    return probabilities, bins


def explain_row(row, bins_row, probabilities):
    """Create a transparent, simple explanation of the strongest BN result."""
    predicted = int(np.argmax(probabilities))
    feature_levels = []
    for feature in FEATURES:
        level = LEVEL_NAMES[int(bins_row[feature])]
        feature_levels.append(f"{FEATURE_LABELS[feature]}={level}")
    return {
        "bn_predicted_state": STATE_NAMES[predicted],
        "bn_confidence": float(probabilities[predicted]),
        "bn_feature_profile": "; ".join(feature_levels),
        "bn_state_probabilities": json.dumps(
            {STATE_NAMES[i]: float(probabilities[i]) for i in range(NUM_STATES)}
        ),
    }


def run_bn(
    latent_path="data/processed/latent_states.csv",
    index_path="data/processed/mimic_index.csv",
    output_path="results/bn_predictions.csv",
    model_path="data/processed/bn_model.json",
):
    latent_path = Path(latent_path)
    index_path = Path(index_path)
    for path in (latent_path, index_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run the data parser and VAE training first."
            )

    latent = pd.read_csv(latent_path)
    index = pd.read_csv(index_path)
    if len(latent) != len(index):
        raise ValueError("latent_states.csv and mimic_index.csv row counts differ.")
    if "split" not in latent or "state" not in latent:
        raise ValueError("latent_states.csv must contain split and state columns.")

    # Use raw-unit feature values from mimic_index.csv for interpretable bins.
    merged = latent[["subject_id", "stay_id", "hour", "split", "state", "state_name"]].copy()
    for feature in FEATURES:
        if feature not in index:
            raise ValueError(f"{feature} missing from mimic_index.csv.")
        merged[feature] = index[feature].to_numpy()

    train = merged[merged["split"] == "train"].copy()
    if train.empty:
        raise ValueError("No training rows available to fit the Bayesian Network.")

    cutoffs = fit_discretization(train)
    cpt, counts = fit_bayesian_network(train, cutoffs)

    # Fit and report on the held-out test split; no test records enter CPT fitting.
    test = merged[merged["split"] == "test"].copy()
    if test.empty:
        raise ValueError("No test rows available for BN evaluation.")
    probs, bins = infer_risk_probabilities(test, cutoffs, cpt)

    outputs = test[["subject_id", "stay_id", "hour", "split", "state", "state_name"]].copy()
    outputs["bn_predicted_state_id"] = np.argmax(probs, axis=1)
    outputs["bn_predicted_state"] = [STATE_NAMES[i] for i in outputs["bn_predicted_state_id"]]
    for i, state_name in enumerate(STATE_NAMES):
        outputs[f"prob_{state_name.lower().replace(' ', '_')}"] = probs[:, i]
    explanations = [
        explain_row(test.iloc[i], bins.iloc[i], probs[i])
        for i in range(len(test))
    ]
    explanation_df = pd.DataFrame(explanations, index=outputs.index)
    outputs = pd.concat([outputs, explanation_df], axis=1)

    accuracy = float((outputs["bn_predicted_state_id"].to_numpy() == test["state"].to_numpy()).mean())
    report = {
        "network": "HeartRateLevel, SBPLevel, WBCLevel -> RiskState",
        "features": FEATURES,
        "training_records": int(len(train)),
        "test_records": int(len(test)),
        "training_patients": int(train["subject_id"].nunique()),
        "test_patients": int(test["subject_id"].nunique()),
        "discretization_cutoffs_fit_on": "train",
        "cpt_fit_on": "train",
        "laplace_alpha": ALPHA,
        "cutoffs": cutoffs,
        "cpt_shape": list(cpt.shape),
        "test_accuracy_against_vae_states": accuracy,
        "interpretation_warning": (
            "This is a proof-of-concept BN trained to explain/reproduce VAE-derived "
            "risk states from the same physiological features, not an independent "
            "clinical outcome predictor or causal explanation. Test accuracy is "
            "agreement with VAE state labels, not clinical accuracy."
        ),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    outputs.to_csv(output_path, index=False)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **report,
                "state_names": STATE_NAMES,
                "feature_level_names": LEVEL_NAMES,
                "cpt": cpt.tolist(),
                "counts_with_smoothing": counts.tolist(),
            },
            handle,
            indent=2,
        )
    report_path = output_path.with_name("bn_evaluation.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("--- Bayesian Network ---")
    print("Structure: HeartRateLevel + SBPLevel + WBCLevel -> RiskState")
    print(f"Training records: {len(train):,}; test records: {len(test):,}")
    print(f"Test agreement with VAE states: {accuracy:.4f}")
    print(f"Predictions: {output_path}")
    print(f"Model and CPT: {model_path}")
    print(f"Evaluation report: {report_path}")
    print("Note: agreement with VAE-derived labels is not clinical accuracy.")
    return report


if __name__ == "__main__":
    run_bn()
