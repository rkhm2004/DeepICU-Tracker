"""Create an honest, stay-level holdout prognosis comparison plot.

Run after: python src/evaluation.py

This is a descriptive diagnostic, not a censoring-aware survival metric or
clinical validation. Predictions are Phase-Type means by initial model-defined
state. Observed means include only stays that reached the model-defined Critical
state during the observed window; censored stays are counted but not treated as
if their event time were observed.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT_PATH = RESULTS / "evaluation_report.json"
OUTPUT_PATH = RESULTS / "evaluation_prognosis_holdout.png"
STATE_ORDER = ["Low Risk", "Medium Risk", "High Risk"]


def main():
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"{REPORT_PATH} not found. Run the pipeline through src/evaluation.py first."
        )

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    prognosis = report.get("test_prognosis_comparison", {})
    summary = prognosis.get("comparison_by_initial_state_stay_level", {})
    if not summary:
        raise ValueError(
            "No stay-level holdout prognosis summary found in evaluation_report.json. "
            "Re-run src/evaluation.py using the current repository version."
        )

    states = [
        state for state in STATE_ORDER
        if summary.get(state, {}).get("stays", 0) > 0
        and summary.get(state, {}).get("events_observed", 0) > 0
        and summary.get(state, {}).get("predicted_mean_hours_mean") is not None
        and summary.get(state, {}).get("observed_event_time_mean_hours") is not None
    ]
    if not states:
        raise ValueError(
            "No initial-state group has observed Critical events in the test holdout; "
            "a predicted-versus-observed event-time bar chart cannot be computed."
        )

    x = np.arange(len(states))
    width = 0.36
    predicted = [summary[s]["predicted_mean_hours_mean"] for s in states]
    observed = [summary[s]["observed_event_time_mean_hours"] for s in states]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars_pred = ax.bar(
        x - width / 2, predicted, width,
        label="Phase-Type predicted mean",
        color="#2878B5", edgecolor="white",
    )
    bars_obs = ax.bar(
        x + width / 2, observed, width,
        label="Observed first-Critical mean (events only)",
        color="#E87500", edgecolor="white",
    )

    ax.set_xticks(x, states)
    ax.set_ylabel("Hours from start of ICU stay")
    ax.set_xlabel("Initial model-defined state at hour 0")
    ax.set_title("Holdout: Predicted vs Observed Time to Model-Defined Critical")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)

    for bars in (bars_pred, bars_obs):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f} h",
                (bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=9,
            )

    details = []
    for state in states:
        item = summary[state]
        mae = item.get("event_only_mae_hours")
        mae_text = f"{mae:.1f} h" if mae is not None else "n/a"
        details.append(
            f"{state}: {item['events_observed']}/{item['stays']} events, "
            f"{item['right_censored']} censored, event-only MAE={mae_text}"
        )

    note = (
        "One first ICU stay per subject. Observed means use only stays that reached "
        "Critical; censored stays are reported, not assigned a fictitious event time.\n"
        "Critical is VAE-defined. Descriptive comparison only; not survival-analysis "
        "validation or a clinical prediction claim.\n"
        + " | ".join(details)
    )
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {OUTPUT_PATH}")
    print("This is an exploratory holdout diagnostic; inspect event/censoring counts before presenting.")
    for line in details:
        print(line)


if __name__ == "__main__":
    main()
