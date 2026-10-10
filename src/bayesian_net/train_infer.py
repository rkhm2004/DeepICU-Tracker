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
        values = pd.to_numeric(train_df[feature], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
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
        if not np.isfinite(values).all():
            raise ValueError(f"Missing or non-finite values found in feature {feature}.")
        result[feature] = np.digitize(values, np.asarray(cutoffs[feature]), right=False)
    return result


def fit_bayesian_network(train_df, cutoffs, alpha=ALPHA):
    """Fit a smoothed discrete BN CPT: P(RiskState | HR, SBP, WBC)."""
    required = set(FEATURES + ["state"])
    missing = required - set(train_df.columns)
    if missing:
        raise ValueError(f"Missing columns required to fit BN: {sorted(missing)}")

    if not np.isfinite(float(alpha)) or float(alpha) <= 0:
        raise ValueError("Laplace smoothing alpha must be finite and positive.")
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


def compute_classification_metrics(y_true, y_pred):
    """Return confusion matrix and class-wise metrics against surrogate labels.

    These metrics quantify agreement with VAE-derived states, not clinical accuracy.
    Confusion-matrix rows are true VAE states; columns are BN-predicted states.
    """
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must have the same non-empty shape.")
    if not np.isin(y_true, np.arange(NUM_STATES)).all():
        raise ValueError("y_true contains invalid risk-state IDs.")
    if not np.isin(y_pred, np.arange(NUM_STATES)).all():
        raise ValueError("y_pred contains invalid risk-state IDs.")

    matrix = np.zeros((NUM_STATES, NUM_STATES), dtype=int)
    np.add.at(matrix, (y_true, y_pred), 1)
    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    true_positive = np.diag(matrix).astype(float)
    precision = np.divide(
        true_positive, predicted_count,
        out=np.zeros(NUM_STATES, dtype=float), where=predicted_count > 0
    )
    recall = np.divide(
        true_positive, support,
        out=np.zeros(NUM_STATES, dtype=float), where=support > 0
    )
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros(NUM_STATES, dtype=float), where=(precision + recall) > 0
    )
    present = support > 0
    accuracy = float(true_positive.sum() / y_true.size)
    return {
        "labels_order": STATE_NAMES,
        "confusion_matrix": matrix.tolist(),
        "support_by_class": {STATE_NAMES[i]: int(support[i]) for i in range(NUM_STATES)},
        "predicted_count_by_class": {STATE_NAMES[i]: int(predicted_count[i]) for i in range(NUM_STATES)},
        "per_class_precision": {STATE_NAMES[i]: float(precision[i]) for i in range(NUM_STATES)},
        "per_class_recall": {STATE_NAMES[i]: float(recall[i]) for i in range(NUM_STATES)},
        "per_class_f1": {STATE_NAMES[i]: float(f1[i]) for i in range(NUM_STATES)},
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        # Standard balanced accuracy averages recall over classes present in y_true.
        "balanced_accuracy": float(np.mean(recall[present])),
        "balanced_accuracy_classes_present": [STATE_NAMES[i] for i in range(NUM_STATES) if present[i]],
        "metric_target": "agreement with VAE-derived risk-state labels; not clinical outcome accuracy",
    }


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


def merge_bn_inputs(latent, index):
    """Align latent labels and raw features by subject/stay/hour keys."""
    keys = ["subject_id", "stay_id", "hour"]
    required_latent = set(keys + ["split", "state", "state_name"])
    required_index = set(keys + ["split"] + FEATURES)
    if not required_latent.issubset(latent.columns):
        raise ValueError(f"latent data missing columns: {sorted(required_latent - set(latent.columns))}")
    if not required_index.issubset(index.columns):
        raise ValueError(f"index data missing columns: {sorted(required_index - set(index.columns))}")
    if latent.duplicated(keys).any() or index.duplicated(keys).any():
        raise ValueError("Duplicate subject_id/stay_id/hour keys found in latent or index data.")

    merged = latent[keys + ["split", "state", "state_name"]].merge(
        index[keys + ["split"] + FEATURES],
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("_latent", "_index"),
        indicator=True,
        sort=False,
    )
    if len(merged) != len(latent) or not (merged["_merge"] == "both").all():
        raise ValueError("Not every latent row has exactly one matching index row by keys.")
    if not (merged["split_latent"] == merged["split_index"]).all():
        raise ValueError("Latent and index split assignments differ for matching keys.")
    return merged.drop(columns=["_merge", "split_index"]).rename(
        columns={"split_latent": "split"}
    ).reset_index(drop=True)


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
    # A key-based one-to-one join prevents silent feature/label misalignment.
    merged = merge_bn_inputs(latent, index)

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

    metrics = compute_classification_metrics(
        test["state"].to_numpy(),
        outputs["bn_predicted_state_id"].to_numpy(),
    )
    accuracy = metrics["accuracy"]
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
        **metrics,
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
    print(f"Macro-F1 against VAE states: {metrics['macro_f1']:.4f}")
    print(f"Balanced accuracy against VAE states: {metrics['balanced_accuracy']:.4f}")
    print("Confusion matrix (rows=true VAE state, columns=BN prediction):")
    print(np.asarray(metrics["confusion_matrix"], dtype=int))
    print(f"Predictions: {output_path}")
    print(f"Model and CPT: {model_path}")
    print(f"Evaluation report: {report_path}")
    print("Note: agreement with VAE-derived labels is not clinical accuracy.")
    return report


if __name__ == "__main__":
    run_bn()
