#!/usr/bin/env python3
"""Non-destructive audit checks for the DeepICU-Tracker research pipeline.

Run from the repository root:
    python audit_pipeline.py

This script diagnoses issues; it does not modify source code or generated artifacts.
PASS = checked and passed, FAIL = concrete defect detected, WARN = limitation or
check that requires human/research judgement, SKIP = required artifact unavailable.
It is an engineering audit, not clinical validation.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

CHECKS: list[tuple[str, str, str]] = []
COUNTS = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0}


def report(status: str, area: str, detail: str) -> None:
    status = status.upper()
    COUNTS[status] += 1
    CHECKS.append((status, area, detail))
    print(f"[{status:4}] {area}: {detail}")


def read_text(relative: str) -> str | None:
    path = ROOT / relative
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def read_json(relative: str):
    path = ROOT / relative
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report("FAIL", f"Artifact {relative}", f"Cannot read valid JSON: {exc}")
        return None


def check_source_patterns() -> None:
    parser = read_text("src/data_pipeline/mimic_parser.py")
    if parser is None:
        report("SKIP", "1. Temporal leakage", "Parser source not found.")
        report("SKIP", "2. Observation window", "Parser source not found.")
    else:
        if re.search(r"\.bfill\s*\(", parser):
            report(
                "FAIL",
                "1. Temporal leakage",
                "Parser calls backward-fill (.bfill), which can use future observations for earlier hours.",
            )
        elif (
            "def causal_impute_features" in parser
            and ".ffill(limit=limit)" in parser
            and "_was_missing" in parser
            and "train_medians" in parser
        ):
            report(
                "PASS",
                "1. Temporal leakage",
                "Causal forward-fill has a per-feature limit, records missingness, and uses train-only median fallback; no backward-fill call found.",
            )
        else:
            report("WARN", "1. Temporal leakage", "Review imputation implementation; causal safeguards were not all detected.")

        has_valid_builder = (
            "def build_observation_index" in parser
            and "hour < row.duration_hours" in parser
            and "np.ceil(row.duration_hours)" in parser
        )
        has_event_bounds = (
            'all_events["charttime"] >= all_events["intime"]' in parser
            and 'all_events["charttime"] <= all_events["outtime"]' in parser
        )
        if has_valid_builder and has_event_bounds:
            report(
                "PASS",
                "2. Observation window",
                "Timeline builder caps the window at 48 hours and excludes bins/events at or after/outside actual ICU outtime.",
            )
        else:
            report(
                "FAIL",
                "2. Observation window",
                "Could not verify a duration-aware hourly index and event-time bounds in parser source.",
            )

    vae = read_text("src/vae/train.py")
    if vae and ("quantile" in vae.lower() or "cutoff" in vae.lower()):
        report(
            "WARN",
            "3. VAE state meaning",
            "Training-derived cutoffs appear in the VAE source, but latent states still need independent clinical-outcome validation.",
        )
    else:
        report("WARN", "3. VAE state meaning", "Could not verify cutoff fitting from source scan; inspect VAE state mapping.")

    readme_for_scope = read_text("README.md") or ""
    quality_source = read_text("src/data_pipeline/mimic_parser.py") or ""
    if (
        parser and all(name in parser for name in ("heart_rate", "sbp", "wbc"))
        and "not a comprehensive ICU severity assessment" in readme_for_scope.lower()
        and "FEATURE_UNITS" in quality_source
        and "data_quality_report.json" in quality_source
    ):
        report(
            "PASS",
            "4/10. Feature scope and data-quality documentation",
            "Three-feature scope is documented; unit review notes and a raw missingness/range report are generated.",
        )
    else:
        report("WARN", "4/10. Feature scope and data-quality documentation", "Document the narrow feature scope and generate unit/missingness/range review metadata.")

    ctmc = read_text("src/markov_mgf/ctmc_estimator.py")
    if ctmc and "next_hour - current_hour != 1" in ctmc and "Q[CRITICAL_STATE, :] = 0.0" in ctmc:
        report(
            "PASS",
            "6. CTMC source safeguards",
            "Source checks consecutive-hour transitions and explicitly makes Critical absorbing; artifact checks follow.",
        )
    else:
        report("WARN", "6. CTMC source safeguards", "Expected gap/absorbing-state safeguards were not both found.")

    mgf = read_text("src/markov_mgf/mgf_calculator.py")
    if mgf and "M(0)" in mgf and "np.isfinite" not in mgf:
        report(
            "WARN",
            "7. Phase-Type edge cases",
            "MGF-at-zero sanity check exists, but finite-value/conditioning checks are not evident in this source scan.",
        )
    elif mgf:
        report("PASS", "7. Phase-Type edge cases", "Some explicit numerical safeguards were detected; artifact checks follow.")
    else:
        report("SKIP", "7. Phase-Type edge cases", "MGF source not found.")

    bn = read_text("src/bayesian_net/train_infer.py")
    if (
        bn
        and "def merge_bn_inputs" in bn
        and 'keys = ["subject_id", "stay_id", "hour"]' in bn
        and 'validate="one_to_one"' in bn
        and 'merged["split_latent"] == merged["split_index"]' in bn
    ):
        report(
            "PASS",
            "13. BN row alignment",
            "BN aligns features and labels with unique subject/stay/hour keys and validates one-to-one matches and split agreement.",
        )
    elif bn:
        report(
            "FAIL",
            "13. BN row alignment",
            "BN source does not show complete key-based one-to-one alignment safeguards.",
        )
    else:
        report("SKIP", "13. BN row alignment", "BN source not found.")

    readme = read_text("README.md")
    if readme:
        stale = [
            phrase for phrase in (
                "Bayesian Network — Pending",
                "No BN results are reported yet",
                "PENDING",
            ) if phrase.casefold() in readme.casefold()
        ]
        if stale:
            report(
                "FAIL",
                "14. README consistency",
                "README still contains potentially stale BN status wording: " + ", ".join(stale),
            )
        else:
            report("PASS", "14. README consistency", "No known stale BN-pending phrases detected.")
    else:
        report("SKIP", "14. README consistency", "README.md not found.")

    inference = read_text("src/inference.py") or ""
    plot_source = read_text("src/plot.py") or ""
    readme_source = read_text("README.md") or ""
    if (
        "not an individual countdown" in inference.lower()
        and "RESEARCH PROTOTYPE" in inference
        and "Research Estimate" in plot_source
        and "not clinically validated" in readme_source.lower()
    ):
        report(
            "PASS",
            "8/15. User-facing interpretation",
            "Inference, plots and README label prognosis as a research estimate rather than a guaranteed individual countdown.",
        )
    else:
        report("WARN", "8/15. User-facing interpretation", "Review inference labels, plots and README for clinical overstatement.")


def check_data_artifacts() -> None:
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        report("SKIP", "Artifact audit", f"NumPy/pandas unavailable: {exc}")
        return

    index_path = ROOT / "data/processed/mimic_index.csv"
    latent_path = ROOT / "data/processed/latent_states.csv"
    if not index_path.exists():
        report("SKIP", "5. Patient-level split", "mimic_index.csv not found; run the parser first.")
        report("SKIP", "10. Data quality", "mimic_index.csv not found.")
        report("SKIP", "13. BN row alignment", "mimic_index.csv not found.")
        index = None
    else:
        try:
            index = pd.read_csv(index_path)
        except Exception as exc:
            report("FAIL", "mimic_index.csv", f"Cannot read CSV: {exc}")
            index = None

    if index is not None:
        needed = {"subject_id", "stay_id", "hour", "split"}
        if not needed.issubset(index.columns):
            report("FAIL", "5. Patient-level split", f"Missing index columns: {sorted(needed - set(index.columns))}")
        else:
            split_sets = {
                split: set(index.loc[index["split"] == split, "subject_id"].dropna().tolist())
                for split in ("train", "validation", "test")
            }
            overlaps = []
            names = list(split_sets)
            for i, left in enumerate(names):
                for right in names[i + 1:]:
                    common = split_sets[left] & split_sets[right]
                    if common:
                        overlaps.append(f"{left}/{right}: {len(common)} shared patients")
            mixed_subjects = int(index.groupby("subject_id")["split"].nunique().gt(1).sum())
            mixed_stays = int(index.groupby("stay_id")["split"].nunique().gt(1).sum())
            if overlaps or mixed_subjects or mixed_stays:
                report("FAIL", "5. Patient-level split", "; ".join(overlaps + [
                    f"patients assigned to multiple splits={mixed_subjects}",
                    f"stays assigned to multiple splits={mixed_stays}",
                ]))
            else:
                report(
                    "PASS",
                    "5. Patient-level split",
                    f"Disjoint patients verified: train={len(split_sets['train'])}, validation={len(split_sets['validation'])}, test={len(split_sets['test'])}.",
                )

            duplicate_keys = int(index.duplicated(["subject_id", "stay_id", "hour"]).sum())
            if duplicate_keys:
                report("FAIL", "2. Observation timeline", f"{duplicate_keys} duplicate subject/stay/hour rows found.")
            else:
                report("PASS", "2. Observation timeline", "No duplicate subject/stay/hour keys.")

        features = [c for c in ("heart_rate", "sbp", "wbc") if c in index.columns]
        if len(features) != 3:
            report("FAIL", "10. Data quality", f"Expected raw feature columns are incomplete: found {features}.")
        else:
            values = index[features].apply(pd.to_numeric, errors="coerce")
            nonfinite = int((~np.isfinite(values.to_numpy(dtype=float))).sum())
            nonpositive = {c: int((values[c] <= 0).sum()) for c in features}
            ranges = {c: [float(values[c].min()), float(values[c].max())] for c in features}
            if nonfinite:
                report("FAIL", "10. Data quality", f"{nonfinite} missing/non-finite feature values remain after preprocessing.")
            elif any(v > 0 for v in nonpositive.values()):
                report("WARN", "10. Data quality", f"Non-positive values need review: {nonpositive}; observed ranges: {ranges}.")
            else:
                report("PASS", "10. Data quality", f"No missing/non-finite/non-positive values; observed ranges: {ranges}. Plausibility/units still need domain review.")

            # Compare generated hours with actual stay duration if raw stay metadata is available.
            raw_stays = ROOT / "data/raw/mimic-iv-clinical-database-demo-2.2/icu/icustays.csv"
            if raw_stays.exists() and {"stay_id", "hour"}.issubset(index.columns):
                try:
                    stays = pd.read_csv(raw_stays, usecols=["stay_id", "intime", "outtime"])
                    stays["intime"] = pd.to_datetime(stays["intime"], errors="coerce")
                    stays["outtime"] = pd.to_datetime(stays["outtime"], errors="coerce")
                    stays["duration_hours"] = (stays["outtime"] - stays["intime"]).dt.total_seconds() / 3600.0
                    joined = index[["stay_id", "hour"]].merge(
                        stays[["stay_id", "duration_hours"]], on="stay_id", how="left", validate="many_to_one"
                    )
                    unknown = int(joined["duration_hours"].isna().sum())
                    invalid = joined["duration_hours"].notna() & (joined["hour"] >= joined["duration_hours"])
                    invalid_count = int(invalid.sum())
                    if invalid_count:
                        report("FAIL", "2. Observation window", f"{invalid_count} rows have hour >= actual ICU stay duration; {unknown} rows lack matching duration metadata.")
                    elif unknown:
                        report("WARN", "2. Observation window", f"No out-of-stay hours found among matched rows, but {unknown} rows lack duration metadata.")
                    else:
                        report("PASS", "2. Observation window", "Every generated hour begins before the recorded ICU outtime.")
                except Exception as exc:
                    report("WARN", "2. Observation window", f"Could not compare with raw stay durations: {exc}")
            else:
                report("SKIP", "2. Observation window", "Raw icustays.csv unavailable; cannot verify generated hours against outtime.")

    preprocessing = read_json("data/processed/preprocessing.json")
    if preprocessing is None:
        report("SKIP", "5. Train-only preprocessing", "preprocessing.json not found or invalid.")
    else:
        if preprocessing.get("fit_on_split") == "train":
            report("PASS", "5. Train-only preprocessing", "Metadata records preprocessing fit_on_split=train.")
        else:
            report("FAIL", "5. Train-only preprocessing", f"Unexpected fit_on_split={preprocessing.get('fit_on_split')!r}.")
        for key in ("imputation_median", "scaler_mean", "scaler_scale"):
            if key not in preprocessing:
                report("FAIL", "5. Train-only preprocessing", f"Missing preprocessing metadata: {key}.")

    # Check alignment by explicit keys, not merely row count.
    if index is not None and latent_path.exists():
        try:
            latent = pd.read_csv(latent_path)
            keys = ["subject_id", "stay_id", "hour"]
            if not set(keys).issubset(index.columns) or not set(keys).issubset(latent.columns):
                report("FAIL", "13. BN row alignment", "Index/latent artifacts do not both contain subject_id, stay_id and hour.")
            elif len(index) != len(latent):
                report("FAIL", "13. BN row alignment", f"Row counts differ: index={len(index)}, latent={len(latent)}.")
            elif not index[keys].reset_index(drop=True).equals(latent[keys].reset_index(drop=True)):
                report("FAIL", "13. BN row alignment", "Index and latent keys differ by row order; positional feature attachment would misalign records.")
            else:
                report("PASS", "13. BN row alignment", "Index and latent keys match row-by-row.")
        except Exception as exc:
            report("FAIL", "13. BN row alignment", f"Could not compare index/latent keys: {exc}")
    else:
        report("SKIP", "13. BN row alignment", "Index or latent_states.csv unavailable.")

    bn_path = ROOT / "results/bn_predictions.csv"
    bn_eval = read_json("results/bn_evaluation.json")
    if bn_path.exists():
        try:
            predictions = pd.read_csv(bn_path)
            keys = ["subject_id", "stay_id", "hour"]
            if not set(keys).issubset(predictions.columns):
                report("FAIL", "13. BN prediction keys", "BN predictions lack one or more patient/stay/hour keys.")
            elif predictions.duplicated(keys).any():
                report("FAIL", "13. BN prediction keys", f"{int(predictions.duplicated(keys).sum())} duplicate prediction keys.")
            else:
                report("PASS", "13. BN prediction keys", f"{len(predictions)} predictions have unique patient/stay/hour keys.")
        except Exception as exc:
            report("FAIL", "13. BN prediction keys", f"Could not read predictions: {exc}")
    else:
        report("SKIP", "13. BN prediction keys", "bn_predictions.csv not found.")

    if bn_eval:
        required_metrics = (
            "confusion_matrix",
            "per_class_precision",
            "per_class_recall",
            "per_class_f1",
            "macro_f1",
            "balanced_accuracy",
        )
        missing = [key for key in required_metrics if key not in bn_eval]
        if missing:
            report("WARN", "13. BN evaluation", f"Detailed metrics not present in bn_evaluation.json: {', '.join(missing)}.")
        else:
            report("PASS", "13. BN evaluation", "Confusion matrix and class-wise/aggregate metrics are present.")
        if "test_accuracy_against_vae_states" in bn_eval:
            report("PASS", "13. BN interpretation", "Report names agreement with VAE states; do not describe it as clinical accuracy.")
    else:
        report("SKIP", "13. BN evaluation", "bn_evaluation.json not found or invalid.")

    quality = read_json("data/processed/data_quality_report.json")
    if quality is None:
        report("SKIP", "10. Data quality report", "data_quality_report.json not found or invalid.")
    elif not {"feature_quality_before_imputation", "feature_quality_after_imputation", "missingness_indicators"}.issubset(quality):
        report("FAIL", "10. Data quality report", "Quality report is missing required raw-missingness or post-imputation summaries.")
    else:
        flagged = {
            feature: details.get("outside_broad_review_bounds_count", 0)
            for feature, details in quality["feature_quality_before_imputation"].items()
            if details.get("outside_broad_review_bounds_count", 0) > 0
        }
        if flagged:
            report(
                "WARN",
                "10. Data quality report",
                "Raw values outside broad review bounds require source/unit review (not automatic deletion): "
                + json.dumps(flagged),
            )
        else:
            report("PASS", "10. Data quality report", "Raw missingness, post-imputation ranges and missingness-indicator counts are present; no broad-bound flags were raised.")

    evaluation = read_json("results/evaluation_report.json")
    if evaluation is None:
        report("SKIP", "9. Prognosis sample size", "evaluation_report.json not found or invalid.")
    else:
        prognosis = evaluation.get("test_prognosis_comparison", {})
        required_prognosis = {
            "uncensored_records", "uncensored_stays", "total_test_stays",
            "uncensored_patients", "total_test_patients", "censored_stays",
            "censoring_note", "interpretation_note",
        }
        if not required_prognosis.issubset(prognosis):
            report("WARN", "9. Prognosis sample size", "Prognosis report does not yet contain complete event/censoring counts; re-run src/evaluation.py.")
        elif (
            int(prognosis.get("uncensored_records", 0)) < 30
            or int(prognosis.get("uncensored_stays", 0)) < 10
        ):
            report(
                "WARN",
                "9. Prognosis sample size",
                "Censoring counts are reported, but the uncensored subset is small "
                f"({prognosis.get('uncensored_records')} records, {prognosis.get('uncensored_stays')} stays, "
                f"{prognosis.get('uncensored_patients')} patients). This remains an exploratory diagnostic, not a code defect.",
            )
        else:
            report(
                "WARN",
                "9. Prognosis validation",
                "Sample counts and censoring are reported, but clinical time-to-event validity still requires a prespecified endpoint and independent censoring-aware validation.",
            )

    expected_outputs = [
        "data/processed/mimic_tensor.pt",
        "data/processed/mimic_index.csv",
        "data/processed/latent_states.csv",
        "data/processed/q_matrix.npy",
        "results/inference_results.csv",
        "results/evaluation_report.json",
        "results/bn_predictions.csv",
        "results/bn_evaluation.json",
        "data/processed/data_quality_report.json",
    ]
    missing_outputs = [p for p in expected_outputs if not (ROOT / p).exists()]
    if missing_outputs:
        report("WARN", "11. Pipeline artifacts", "Missing outputs: " + ", ".join(missing_outputs))
    else:
        report("PASS", "11. Pipeline artifacts", f"All {len(expected_outputs)} expected pipeline artifacts exist.")

    q_path = ROOT / "data/processed/q_matrix.npy"
    if q_path.exists():
        try:
            q = np.load(q_path)
            valid_shape = q.shape == (4, 4)
            finite = bool(np.isfinite(q).all())
            row_sums = bool(np.allclose(q.sum(axis=1), 0.0, atol=1e-8)) if valid_shape else False
            offdiag = q.copy()
            if valid_shape:
                np.fill_diagonal(offdiag, 0.0)
            nonnegative = bool((offdiag >= -1e-10).all()) if valid_shape else False
            absorbing = bool(np.allclose(q[3], 0.0, atol=1e-10)) if valid_shape else False
            if valid_shape and finite and row_sums and nonnegative and absorbing:
                report("PASS", "6. CTMC generator artifact", "Q is 4x4 finite, row-sum-zero, nonnegative off-diagonal, with absorbing Critical.")
            else:
                report("FAIL", "6. CTMC generator artifact", f"Invalid Q properties: shape={q.shape}, finite={finite}, row_sums={row_sums}, offdiag_nonnegative={nonnegative}, critical_absorbing={absorbing}.")
        except Exception as exc:
            report("FAIL", "6. CTMC generator artifact", f"Cannot validate Q matrix: {exc}")
    else:
        report("SKIP", "6. CTMC generator artifact", "q_matrix.npy not found.")

    counts_path = ROOT / "data/processed/ctmc_transition_counts.npy"
    exposure_path = ROOT / "data/processed/ctmc_exposure_hours.npy"
    if counts_path.exists() and exposure_path.exists():
        try:
            counts, exposure = np.load(counts_path), np.load(exposure_path)
            if counts.shape != (4, 4) or exposure.shape != (4,) or not np.isfinite(counts).all() or not np.isfinite(exposure).all() or (counts < 0).any() or (exposure < 0).any():
                report("FAIL", "6. CTMC counts/exposure", "Transition counts or exposure have invalid dimensions/values.")
            else:
                report("PASS", "6. CTMC counts/exposure", f"Counts and exposure arrays valid; counts={int(counts.sum())}, exposure_hours={exposure.tolist()}.")
        except Exception as exc:
            report("FAIL", "6. CTMC counts/exposure", f"Cannot validate arrays: {exc}")
    else:
        report("SKIP", "6. CTMC counts/exposure", "Transition-count/exposure artifacts not found.")

    # Mathematical checks use the project's own implementation on the saved Q.
    if q_path.exists():
        try:
            from src.markov_mgf.mgf_calculator import load_transient_generator, phase_type_mgf, phase_type_moments
            _, transient = load_transient_generator(q_path)
            means, variances, stds = phase_type_moments(transient)
            mgf_zero = [phase_type_mgf(0.0, i, transient) for i in range(3)]
            valid = (
                np.isfinite(means).all()
                and np.isfinite(variances).all()
                and np.isfinite(stds).all()
                and (means > 0).all()
                and (variances >= -1e-10).all()
                and np.allclose(mgf_zero, 1.0, atol=1e-8)
            )
            if valid:
                report("PASS", "7. Phase-Type numerical check", f"Finite moments and MGF(0)=1 verified; means={np.round(means, 3).tolist()} hours.")
            else:
                report("FAIL", "7. Phase-Type numerical check", f"Numerical sanity check failed: means={means}, variances={variances}, MGF(0)={mgf_zero}.")
        except Exception as exc:
            report("FAIL", "7. Phase-Type numerical check", f"Phase-Type calculation raised {type(exc).__name__}: {exc}")


def run_unit_tests() -> None:
    test_modules = [
        "tests/test_mathematical_pipeline.py",
        "tests/test_bayesian_network.py",
        "tests/test_clinical_validation.py",
    ]
    existing = [p for p in test_modules if (ROOT / p).exists()]
    missing = [p for p in test_modules if not (ROOT / p).exists()]
    if missing:
        report("WARN", "12. Test coverage", "Missing known test files: " + ", ".join(missing))
    if not existing:
        report("SKIP", "12. Unit tests", "No expected test modules found.")
        return

    for module in existing:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", module.replace("/", ".").removesuffix(".py")],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        if result.returncode == 0:
            summary = (result.stdout + "\n" + result.stderr).strip().splitlines()
            tail = " | ".join(summary[-3:]) if summary else "completed"
            report("PASS", f"12. Unit tests ({module})", tail)
        else:
            output = (result.stdout + "\n" + result.stderr).strip()
            report("FAIL", f"12. Unit tests ({module})", output[-1800:] or f"exit code {result.returncode}")


def main() -> int:
    print("=" * 76)
    print("DEEPICU-TRACKER — NON-DESTRUCTIVE PIPELINE FLAW AUDIT")
    print("Run from repository root. This script diagnoses; it does not fix files.")
    print("=" * 76)
    check_source_patterns()
    check_data_artifacts()
    run_unit_tests()
    print("\n" + "=" * 76)
    print("AUDIT SUMMARY")
    for status in ("PASS", "FAIL", "WARN", "SKIP"):
        print(f"{status:4}: {COUNTS[status]}")
    print("-" * 76)
    if COUNTS["FAIL"]:
        print("Priority: address FAIL findings first, then review WARN findings.")
    elif COUNTS["WARN"]:
        print("No concrete FAIL findings detected by available checks; review all WARN items.")
    else:
        print("No findings from the checks that were able to run; this is not clinical validation.")
    print("Clinical validity and suitability for patient care are outside this script's scope.")
    print("=" * 76)
    return 1 if COUNTS["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
