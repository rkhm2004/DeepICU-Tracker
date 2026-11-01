"""Read-only diagnostic audit for the DeepICU-Tracker pipeline.

Run from the repository root:
    python src/diagnose_pipeline.py

The script does not retrain models, modify processed data, or overwrite results.
It writes one report to results/diagnostic_report.json.
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

ROOT = Path(".")
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FEATURES = ["heart_rate", "sbp", "wbc"]
SPLITS = ["train", "validation", "test"]
STATE_NAMES = ["Low", "Medium", "High", "Critical"]


def load_json(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_float(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {
        "title": "DeepICU-Tracker read-only diagnostic audit",
        "scope": "Data, VAE-state, CTMC and Phase-Type diagnostics; not clinical validation.",
        "checks": [],
        "warnings": [],
        "errors": [],
    }

    def check(name, status, detail):
        report["checks"].append({"name": name, "status": status, "detail": detail})
        print(f"[{status}] {name}: {detail}")

    index_path = PROCESSED / "mimic_index.csv"
    latent_path = PROCESSED / "latent_states.csv"
    quality_path = PROCESSED / "data_quality_report.json"
    prep_path = PROCESSED / "preprocessing.json"
    meta_path = PROCESSED / "vae_metadata.json"
    metrics_path = PROCESSED / "vae_training_metrics.json"
    q_path = PROCESSED / "q_matrix.npy"
    counts_path = PROCESSED / "ctmc_transition_counts.npy"
    exposure_path = PROCESSED / "ctmc_exposure_hours.npy"

    index = pd.read_csv(index_path) if index_path.exists() else None
    latent = pd.read_csv(latent_path) if latent_path.exists() else None
    quality = load_json(quality_path)
    prep = load_json(prep_path)
    metadata = load_json(meta_path)
    vae_metrics = load_json(metrics_path)

    # 1. Dataset and split integrity.
    if index is None:
        report["errors"].append(f"Missing {index_path}")
        check("Processed index", "ERROR", f"Missing {index_path}; run the parser first.")
    else:
        required = {"subject_id", "stay_id", "hour", "split", *FEATURES}
        missing = sorted(required - set(index.columns))
        if missing:
            report["errors"].append(f"mimic_index.csv missing columns: {missing}")
            check("Index schema", "ERROR", f"Missing columns: {missing}")
        else:
            split_summary = {}
            for split in SPLITS:
                part = index[index["split"] == split]
                split_summary[split] = {
                    "hourly_records": int(len(part)),
                    "patients": int(part["subject_id"].nunique()),
                    "icu_stays": int(part["stay_id"].nunique()),
                }
            subjects = {
                split: set(index.loc[index["split"] == split, "subject_id"].dropna())
                for split in SPLITS
            }
            overlaps = {}
            for i, a in enumerate(SPLITS):
                for b in SPLITS[i + 1:]:
                    overlaps[f"{a}_vs_{b}"] = sorted(subjects[a] & subjects[b])
            leakage = any(overlaps.values())
            report["dataset"] = {
                "hourly_records": int(len(index)),
                "unique_patients": int(index["subject_id"].nunique()),
                "icu_stays": int(index["stay_id"].nunique()),
                "split_summary": split_summary,
                "patient_overlap_between_splits": overlaps,
            }
            check("Patient-level split", "FAIL" if leakage else "PASS",
                  "Patient IDs overlap across splits." if leakage else
                  "No patient overlap across train/validation/test.")
            if leakage:
                report["errors"].append("Patient leakage detected between data splits.")

            duplicate = int(index.duplicated(["stay_id", "hour"]).sum())
            check("Unique stay-hour rows", "FAIL" if duplicate else "PASS",
                  f"{duplicate} duplicate stay_id/hour rows.")
            if duplicate:
                report["errors"].append(f"{duplicate} duplicate stay-hour rows.")

            gaps = 0
            for _, stay in index.sort_values(["stay_id", "hour"]).groupby("stay_id"):
                hours = pd.to_numeric(stay["hour"], errors="coerce").dropna().to_numpy()
                if len(hours) > 1:
                    gaps += int(np.sum(np.diff(hours) != 1))
            check("Hourly continuity", "WARN" if gaps else "PASS",
                  f"{gaps} non-consecutive hour intervals; CTMC estimator skips these intervals.")
            if gaps:
                report["warnings"].append(
                    "Non-consecutive hourly intervals are skipped by the current CTMC estimator."
                )

    # 2. Missingness and outlier screening.
    if index is not None:
        missingness = {}
        feature_summary = {}
        bounds = {
            "heart_rate": (20.0, 250.0),
            "sbp": (30.0, 300.0),
            "wbc": (0.1, 100.0),
        }
        for feature in FEATURES:
            if feature not in index:
                continue
            values = pd.to_numeric(index[feature], errors="coerce")
            missing_col = f"{feature}_was_missing"
            if missing_col in index:
                original_missing = index[missing_col].astype(bool)
            else:
                original_missing = values.isna()
            observed = values[~original_missing].dropna()
            low, high = bounds[feature]
            outside = observed[(observed < low) | (observed > high)]
            missingness[feature] = {
                "originally_missing_records": int(original_missing.sum()),
                "original_missing_fraction": float(original_missing.mean()),
                "originally_observed_records": int((~original_missing).sum()),
            }
            feature_summary[feature] = {
                "min": safe_float(observed.min()) if len(observed) else None,
                "median": safe_float(observed.median()) if len(observed) else None,
                "max": safe_float(observed.max()) if len(observed) else None,
                "outside_review_bounds_count": int(len(outside)),
                "review_bounds": [low, high],
                "outside_review_bounds_examples": [safe_float(v) for v in outside.head(10)],
            }
        report["feature_quality"] = {
            "missingness": missingness,
            "observed_value_summary": feature_summary,
            "note": "Review bounds flag values for inspection; they are not clinical exclusion rules.",
        }
        wbc_n = missingness.get("wbc", {}).get("originally_observed_records", 0)
        wbc_total = len(index)
        check("WBC observation coverage", "WARN" if wbc_total and wbc_n / wbc_total < 0.25 else "PASS",
              f"{wbc_n}/{wbc_total} hourly rows originally had WBC ({wbc_n / wbc_total:.1%} if rows exist)."
              if wbc_total else "No rows available.")
        if wbc_total and wbc_n / wbc_total < 0.25:
            report["warnings"].append(
                "WBC is sparsely observed. Forward-filled and median-imputed values are not new measurements."
            )
        outlier_total = sum(v["outside_review_bounds_count"] for v in feature_summary.values())
        check("Broad-range screening", "WARN" if outlier_total else "PASS",
              f"{outlier_total} originally observed values fall outside configured review bounds.")
        if outlier_total:
            report["warnings"].append(
                "Values outside broad review bounds need source/unit verification; do not delete automatically."
            )

    if quality is not None:
        report["parser_data_quality_report"] = quality
    if prep is not None:
        report["preprocessing_metadata"] = {
            key: prep.get(key) for key in [
                "features", "imputation_method", "max_forward_fill_hours",
                "fit_on_split", "split_strategy", "feature_units_review_notes",
                "feature_review_bounds",
            ]
        }
        check("Train-only preprocessing", "PASS" if prep.get("fit_on_split") == "train" else "WARN",
              f"preprocessing fit_on_split={prep.get('fit_on_split')!r}.")
        if prep.get("fit_on_split") != "train":
            report["warnings"].append("Preprocessing metadata does not confirm train-only fitting.")

    # 3. VAE state assignment and clinical direction caveat.
    if metadata is not None:
        cutoffs = np.asarray(metadata.get("cutoffs", []), dtype=float)
        valid_cutoffs = len(cutoffs) == 3 and np.isfinite(cutoffs).all() and np.all(np.diff(cutoffs) > 0)
        report["vae_metadata"] = metadata
        check("VAE cutoffs", "PASS" if valid_cutoffs else "FAIL",
              f"Cutoffs={cutoffs.tolist()}; expected three finite ascending thresholds.")
        if not valid_cutoffs:
            report["errors"].append("VAE cutoffs are missing, invalid, or not strictly ascending.")
        report["warnings"].append(
            "Latent-bin names Low/Medium/High/Critical are model-defined. The direction and clinical meaning of mu require independent outcome validation."
        )
        check("Clinical meaning of latent states", "WARN",
              "Cannot be established from reconstruction loss or quantile cutoffs alone.")
    else:
        check("VAE metadata", "WARN", f"Missing {meta_path}; cannot verify state thresholds.")

    if vae_metrics is not None:
        report["vae_training_metrics"] = {
            key: vae_metrics.get(key) for key in [
                "observed_target_counts_by_split", "per_feature_reconstruction_mse_by_split",
                "per_feature_generalization_gap", "best_validation_loss",
                "test_loss_at_best_checkpoint", "epochs_trained",
            ]
        }
        check("VAE masked-loss diagnostics", "PASS",
              "Training metrics file found; review observed-target counts and per-feature gaps in JSON.")
    else:
        check("VAE training metrics", "WARN", f"Missing {metrics_path}.")

    if latent is None:
        check("Latent state trajectory", "WARN", f"Missing {latent_path}.")
    else:
        needed = {"subject_id", "stay_id", "hour", "split", "state", "risk_score_mu"}
        absent = sorted(needed - set(latent.columns))
        if absent:
            check("Latent state schema", "FAIL", f"Missing columns: {absent}")
            report["errors"].append(f"latent_states.csv missing columns: {absent}")
        else:
            state_summary = {}
            for split in SPLITS:
                part = latent[latent["split"] == split]
                counts = part["state"].value_counts().reindex(range(4), fill_value=0)
                state_summary[split] = {
                    STATE_NAMES[s]: int(counts[s]) for s in range(4)
                }
            report["state_distribution"] = state_summary
            train_counts = state_summary["train"]
            check("Training state coverage", "WARN" if any(train_counts[n] == 0 for n in STATE_NAMES) else "PASS",
                  f"Training state counts: {train_counts}.")
            if any(train_counts[n] == 0 for n in STATE_NAMES):
                report["warnings"].append("At least one state has no training records.")

    # 4. CTMC count/exposure/Q audit.
    q = np.load(q_path) if q_path.exists() else None
    counts = np.load(counts_path) if counts_path.exists() else None
    exposure = np.load(exposure_path) if exposure_path.exists() else None
    if counts is not None and exposure is not None:
        report["ctmc_observations"] = {
            "transition_counts": counts.astype(int).tolist(),
            "exposure_hours": exposure.astype(float).tolist(),
            "total_transitions": int(counts.sum()),
            "total_exposure_hours": float(exposure.sum()),
        }
        check("Critical exposure", "INFO" if exposure[3] == 0 else "PASS",
              f"Critical exposure={exposure[3]:.2f} h. Zero can be expected for an absorbing state; inspect entries into Critical.")
        entry_count = int(counts[:3, 3].sum())
        check("Observed entries into Critical", "WARN" if entry_count < 5 else "PASS",
              f"{entry_count} training transitions into Critical.")
        if entry_count < 5:
            report["warnings"].append(
                "Very few observed transitions into Critical; absorption-time estimates may be unstable."
            )
        if counts.sum() == 0:
            report["errors"].append("No state transitions were observed in the saved CTMC counts.")
    else:
        check("CTMC count/exposure artifacts", "WARN",
              "Missing transition-count or exposure artifact; run the CTMC estimator.")

    if q is not None:
        q_checks = {
            "shape": list(q.shape),
            "row_sums": q.sum(axis=1).tolist() if q.ndim == 2 else None,
            "finite": bool(np.isfinite(q).all()),
            "off_diagonal_nonnegative": bool(
                q.ndim == 2 and np.all((q - np.diag(np.diag(q))) >= -1e-10)
            ),
            "critical_absorbing_row": q[3].tolist() if q.shape == (4, 4) else None,
        }
        valid_q = (
            q.shape == (4, 4) and np.isfinite(q).all()
            and np.allclose(q.sum(axis=1), 0, atol=1e-8)
            and np.all((q - np.diag(np.diag(q))) >= -1e-10)
            and np.allclose(q[3], 0, atol=1e-10)
        )
        report["ctmc_generator_checks"] = q_checks
        check("CTMC generator validity", "PASS" if valid_q else "FAIL",
              "Q must be finite 4x4, have nonnegative off-diagonal rates, zero row sums and absorbing Critical row.")
        if not valid_q:
            report["errors"].append("Saved Q matrix fails generator-matrix validity checks.")

        if counts is not None and exposure is not None and valid_q:
            reconstructed = np.zeros((4, 4), dtype=float)
            for i in range(3):
                if exposure[i] > 0:
                    for j in range(4):
                        if i != j:
                            reconstructed[i, j] = counts[i, j] / exposure[i]
                reconstructed[i, i] = -reconstructed[i].sum()
            difference = float(np.max(np.abs(reconstructed - q)))
            report["ctmc_generator_checks"]["max_abs_difference_from_counts_over_exposure"] = difference
            check("Q versus counts/exposure", "PASS" if difference < 1e-7 else "FAIL",
                  f"Maximum absolute difference={difference:.3g}.")
            if difference >= 1e-7:
                report["errors"].append("Saved Q does not match the saved transition counts/exposure.")

        try:
            from src.markov_mgf.mgf_calculator import phase_type_moments
            means, variances, stds = phase_type_moments(q[:3, :3])
            report["phase_type"] = {
                "mean_hours_by_start_state": dict(zip(STATE_NAMES[:3], means.tolist())),
                "sd_hours_by_start_state": dict(zip(STATE_NAMES[:3], stds.tolist())),
                "variance_hours_squared": variances.tolist(),
                "transient_subgenerator_condition_number": float(np.linalg.cond(-q[:3, :3])),
            }
            check("Phase-Type moments", "PASS", f"Expected absorption means (h): {np.round(means, 2).tolist()}.")
        except Exception as exc:
            report["phase_type"] = {"error": str(exc)}
            check("Phase-Type moments", "FAIL", str(exc))
            report["errors"].append(f"Phase-Type moment calculation failed: {exc}")
    else:
        check("CTMC Q matrix", "WARN", f"Missing {q_path}; run the CTMC estimator.")

    # 5. Cross-check latent CSV state sequences against saved CTMC counts.
    if latent is not None and {"stay_id", "hour", "state", "split"}.issubset(latent.columns):
        recomputed = np.zeros((4, 4), dtype=int)
        recomputed_exposure = np.zeros(4, dtype=float)
        train_latent = latent[latent["split"] == "train"].sort_values(["stay_id", "hour"])
        for _, stay in train_latent.groupby("stay_id", sort=False):
            rows = stay[["hour", "state"]].to_numpy()
            for current, nxt in zip(rows[:-1], rows[1:]):
                h0, s0 = int(current[0]), int(current[1])
                h1, s1 = int(nxt[0]), int(nxt[1])
                if h1 - h0 != 1 or s0 == 3:
                    continue
                recomputed_exposure[s0] += 1.0
                if s1 != s0:
                    recomputed[s0, s1] += 1
                if s1 == 3:
                    break
        report["ctmc_recomputed_from_latent_csv"] = {
            "transition_counts": recomputed.tolist(),
            "exposure_hours": recomputed_exposure.tolist(),
        }
        if counts is not None and exposure is not None:
            same = np.array_equal(recomputed, counts.astype(int)) and np.allclose(recomputed_exposure, exposure)
            check("CTMC artifacts vs latent trajectories", "PASS" if same else "FAIL",
                  "Recomputed counts/exposure match saved artifacts." if same else
                  "Recomputed counts/exposure do not match saved artifacts; check stale files or pipeline order.")
            if not same:
                report["errors"].append("CTMC artifacts disagree with current latent_states.csv.")
    # 6. Evaluation report and methodological cautions.
    evaluation = load_json(RESULTS / "evaluation_report.json")
    if evaluation is not None:
        prognosis = evaluation.get("test_prognosis_comparison", {})
        stay_summary = prognosis.get("comparison_by_initial_state_stay_level", {})
        report["existing_evaluation_summary"] = {
            "test_prognosis_comparison": prognosis,
            "test_transition_validation": evaluation.get("test_transition_validation"),
        }
        check("Existing prognosis evaluation", "INFO",
              f"Found evaluation report; review event counts and right-censored counts by starting state: {stay_summary}.")
    else:
        check("Existing evaluation report", "WARN", "No results/evaluation_report.json found; run evaluation.py after the pipeline.")

    report["warnings"].extend([
        "The four VAE states are latent-score bins, not validated clinical severity labels.",
        "The Phase-Type mean is a population expectation under a fitted homogeneous CTMC, not an individual countdown.",
        "Event-only observed means exclude right-censored stays and are not directly comparable to population expected absorption times.",
        "Hourly snapshots can miss intermediate state changes; the CTMC rates are approximations.",
        "This diagnostic cannot establish clinical validity or prove that outliers caused the prediction errors.",
    ])
    report["summary"] = {
        "error_count": len(report["errors"]),
        "warning_count": len(report["warnings"]),
        "check_count": len(report["checks"]),
        "overall": "ERRORS_FOUND" if report["errors"] else "REVIEW_WARNINGS" if report["warnings"] else "NO_AUTOMATED_ISSUES_FOUND",
    }

    output = RESULTS / "diagnostic_report.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print("\n================ DIAGNOSTIC SUMMARY ================")
    print(json.dumps(report["summary"], indent=2))
    print(f"Full report saved to: {output}")
    print("This is a read-only audit; no model or data files were modified.")


if __name__ == "__main__":
    main()
