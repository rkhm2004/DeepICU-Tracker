"""Independent clinical-outcome and censoring-aware validation for DeepICU-Tracker.

Run from the repository root after the parser, VAE, and CTMC stages:
    python src/clinical_validation.py

This script uses raw MIMIC demo source tables when available. It never changes
or silently filters the source data. The clinical endpoint is in-hospital
mortality (hospital_expire_flag, with deathtime fallback); the prognosis event
is explicitly the first *model-defined* Critical state, not a clinical event.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
RAW = Path("data/raw/mimic-iv-clinical-database-demo-2.2")
ICU_DIR = RAW / "icu"
HOSP_DIR = RAW / "hosp"
INDEX_PATH = Path("data/processed/mimic_index.csv")
LATENT_PATH = Path("data/processed/latent_states.csv")
Q_PATH = Path("data/processed/q_matrix.npy")
OUT_DIR = Path("results")
FEATURES = ["heart_rate", "sbp", "wbc"]
STATE_NAMES = ["Low Risk", "Medium Risk", "High Risk", "Critical"]
WBC_ITEM_ID = 51301
WBC_REVIEW_BOUND = 100.0
MAX_WINDOW_HOURS = 48.0
GRID_HOURS = [0.0, 6.0, 12.0, 24.0, 36.0, 48.0]


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input not found: {path}. Ensure the MIMIC-IV demo files "
            "are present locally and run the parser/VAE/CTMC stages first."
        )
    return path


def _read_stays() -> pd.DataFrame:
    path = require(ICU_DIR / "icustays.csv")
    header = pd.read_csv(path, nrows=0).columns
    required = ["subject_id", "stay_id", "intime", "outtime"]
    if "hadm_id" in header:
        required.append("hadm_id")
    stays = pd.read_csv(path, usecols=required)
    stays["intime"] = pd.to_datetime(stays["intime"], errors="coerce")
    stays["outtime"] = pd.to_datetime(stays["outtime"], errors="coerce")
    return stays


def trace_wbc_outliers(stays: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Trace raw WBC results above the broad review bound into ICU-hour rows."""
    path = require(HOSP_DIR / "labevents.csv")
    available = list(pd.read_csv(path, nrows=0).columns)
    wanted = [
        "labevent_id", "subject_id", "hadm_id", "itemid", "charttime",
        "storetime", "value", "valuenum", "valueuom", "ref_range_lower",
        "ref_range_upper", "flag", "priority", "comments",
    ]
    usecols = [c for c in wanted if c in available]
    chunks = []
    # Chunking avoids loading the full lab table into memory.
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000, low_memory=False):
        if "itemid" not in chunk or "valuenum" not in chunk:
            raise ValueError("labevents.csv must include itemid and valuenum.")
        chunk = chunk.loc[
            (pd.to_numeric(chunk["itemid"], errors="coerce") == WBC_ITEM_ID)
            & (pd.to_numeric(chunk["valuenum"], errors="coerce") > WBC_REVIEW_BOUND)
        ].copy()
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        result = pd.DataFrame(columns=wanted + ["stay_id", "intime", "outtime", "hour", "processed_wbc"])
        result.to_csv(OUT_DIR / "wbc_outlier_trace.csv", index=False)
        return result

    labs = pd.concat(chunks, ignore_index=True)
    labs["charttime"] = pd.to_datetime(labs["charttime"], errors="coerce")
    join_cols = ["subject_id"]
    if "hadm_id" in labs.columns and "hadm_id" in stays.columns:
        # First match by subject, then use admission and timestamp to disambiguate.
        joined = labs.merge(stays, on="subject_id", how="left", suffixes=("", "_stay"))
        if "hadm_id_stay" in joined.columns:
            same_admission = (
                joined["hadm_id"].isna()
                | joined["hadm_id_stay"].isna()
                | (joined["hadm_id"] == joined["hadm_id_stay"])
            )
            joined = joined.loc[same_admission].copy()
    else:
        joined = labs.merge(stays, on="subject_id", how="left", suffixes=("", "_stay"))

    in_stay = (
        joined["charttime"].notna()
        & joined["intime"].notna()
        & joined["outtime"].notna()
        & (joined["charttime"] >= joined["intime"])
        & (joined["charttime"] <= joined["outtime"])
    )
    joined = joined.loc[in_stay].copy()
    joined["hour"] = (
        (joined["charttime"] - joined["intime"]).dt.total_seconds() // 3600
    ).astype(int)

    if {"stay_id", "hour", "wbc"}.issubset(index.columns):
        processed = index[["stay_id", "hour", "wbc"]].rename(
            columns={"wbc": "processed_wbc"}
        )
        joined = joined.merge(
            processed, on=["stay_id", "hour"], how="left", validate="many_to_one"
        )
    else:
        joined["processed_wbc"] = np.nan

    # Preserve source values, unit, reference range, flag, and timestamp for review.
    front = [
        c for c in [
            "labevent_id", "subject_id", "hadm_id", "stay_id", "itemid",
            "charttime", "storetime", "value", "valuenum", "valueuom",
            "ref_range_lower", "ref_range_upper", "flag", "priority",
            "comments", "intime", "outtime", "hour", "processed_wbc",
        ] if c in joined.columns
    ]
    joined = joined[front].sort_values(["valuenum", "charttime"], ascending=[False, True])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joined.to_csv(OUT_DIR / "wbc_outlier_trace.csv", index=False)
    return joined


def _safe_auc(y_true, score):
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2 or not np.isfinite(s).all():
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def _binary_metrics(y_true, probability):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if len(y) == 0:
        return {"n": 0, "events": 0, "non_events": 0, "roc_auc": None,
                "average_precision": None, "brier_score": None, "threshold_0_5": None}
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Outcome probabilities must be finite and within [0, 1].")
    auc = _safe_auc(y, p)
    average_precision = None
    if len(np.unique(y)) == 2:
        from sklearn.metrics import average_precision_score
        average_precision = float(average_precision_score(y, p))
    from sklearn.metrics import brier_score_loss, confusion_matrix, precision_score, recall_score, f1_score
    pred = (p >= 0.5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "non_events": int(len(y) - y.sum()),
        "roc_auc": auc,
        "average_precision": average_precision,
        "brier_score": float(brier_score_loss(y, p)),
        "threshold_0_5": {
            "confusion_matrix_rows_true_columns_predicted": cm,
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
        },
        "small_sample_note": (
            "Metrics are exploratory; report uncertainty and do not treat this demo cohort as clinical validation."
        ),
    }


def _fit_discrete_mortality_bn(train: pd.DataFrame, test: pd.DataFrame):
    """Fit P(in-hospital death | HR bin, SBP bin, WBC bin) on baseline patients."""
    cutoffs = {}
    for feature in FEATURES:
        x = pd.to_numeric(train[feature], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if not len(x):
            raise ValueError(f"No finite training values for {feature}.")
        cutoffs[feature] = np.quantile(x, [1 / 3, 2 / 3]).tolist()

    def bins_for(frame):
        binned = {}
        for feature in FEATURES:
            values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Missing/non-finite baseline values for {feature}.")
            binned[feature] = np.digitize(values, cutoffs[feature]).astype(int)
        return np.column_stack([binned[f] for f in FEATURES])

    train_bins = bins_for(train)
    test_bins = bins_for(test)
    y_train = train["hospital_expire_flag"].to_numpy(dtype=int)
    # Laplace smoothing for a two-state target; all cutoffs/CPT cells use train only.
    counts = np.ones((3, 3, 3, 2), dtype=float)
    for row, target in zip(train_bins, y_train):
        counts[row[0], row[1], row[2], target] += 1
    cpt = counts / counts.sum(axis=-1, keepdims=True)
    p_death = np.array([cpt[r[0], r[1], r[2], 1] for r in test_bins], dtype=float)
    predictions = test[[
        "subject_id", "hadm_id", "stay_id", "split", "hospital_expire_flag",
        "heart_rate", "sbp", "wbc",
    ]].copy()
    predictions["predicted_hospital_mortality_probability"] = p_death
    predictions["predicted_mortality_at_0_5"] = (p_death >= 0.5).astype(int)
    return {
        "network": "HeartRateLevel + SBPLevel + WBCLevel -> HospitalMortality",
        "target": "independent MIMIC hospital_expire_flag (or documented deathtime fallback)",
        "training_subjects": int(train["subject_id"].nunique()),
        "test_subjects": int(test["subject_id"].nunique()),
        "training_events": int(y_train.sum()),
        "cutoffs_fit_on": "training baseline patients only",
        "conditional_probability_table_fit_on": "training baseline patients only",
        "test_metrics": _binary_metrics(
            test["hospital_expire_flag"].to_numpy(dtype=int), p_death
        ),
        "warning": (
            "This is a new exploratory outcome-targeted BN, separate from the existing BN "
            "that reproduces VAE labels. Small demo-cohort metrics are not clinical validation."
        ),
    }, predictions


def _kaplan_meier(times, events, grid):
    """Kaplan-Meier survival estimate using event indicators and right censoring."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    grid = np.asarray(grid, dtype=float)
    if len(times) == 0 or len(times) != len(events):
        raise ValueError("KM requires non-empty, equal-length time/event arrays.")
    if not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError("Follow-up times must be finite and non-negative.")
    if not np.isin(events, [0, 1]).all():
        raise ValueError("Event indicators must be 0 (censored) or 1 (event).")

    survival = 1.0
    event_times = np.unique(times[(events == 1)])
    steps = []
    for t in sorted(event_times):
        at_risk = int(np.sum(times >= t))
        deaths = int(np.sum((times == t) & (events == 1)))
        if at_risk > 0:
            survival *= 1.0 - deaths / at_risk
        steps.append((float(t), float(survival)))
    values = []
    for t in grid:
        s = 1.0
        for event_time, after in steps:
            if event_time <= t:
                s = after
            else:
                break
        values.append(s)
    return np.asarray(values, dtype=float)


def _phase_type_survival(initial_state: int, T: np.ndarray, time_hours: float) -> float:
    if initial_state == 3:
        return 0.0
    if not 0 <= initial_state < 3:
        raise ValueError(f"Invalid initial state {initial_state}.")
    transition = expm(T * float(time_hours))
    return float(np.sum(transition[initial_state, :]))


def _prognosis_survival_analysis(latent, stays, q_matrix, selected_test_stays):
    """Compare KM for first model-defined Critical with fitted CTMC survival."""
    test_ids = set(selected_test_stays["stay_id"].tolist())
    rows = latent.loc[
        (latent["split"] == "test") & latent["stay_id"].isin(test_ids)
    ].copy()
    if rows.empty:
        return {"available": False, "reason": "No test trajectories available."}, pd.DataFrame()

    durations = stays[["stay_id", "intime", "outtime"]].copy()
    durations["duration_hours"] = (
        (durations["outtime"] - durations["intime"]).dt.total_seconds() / 3600.0
    )
    durations["followup_hours"] = durations["duration_hours"].clip(lower=0, upper=MAX_WINDOW_HOURS)
    rows = rows.merge(
        durations[["stay_id", "followup_hours"]],
        on="stay_id", how="left", validate="many_to_one"
    )
    times, events, initial_states = [], [], []
    for stay_id, stay in rows.groupby("stay_id", sort=False):
        stay = stay.sort_values("hour")
        if stay.empty or pd.isna(stay["followup_hours"].iloc[0]):
            continue
        followup = float(stay["followup_hours"].iloc[0])
        if followup <= 0:
            continue
        initial = int(stay.iloc[0]["state"])
        critical_hours = stay.loc[stay["state"] == 3, "hour"]
        if not critical_hours.empty and float(critical_hours.min()) < followup:
            event_time = float(critical_hours.min())
            times.append(event_time)
            events.append(1)
        else:
            times.append(followup)
            events.append(0)
        initial_states.append(initial)

    if not times:
        return {"available": False, "reason": "No valid test follow-up times."}, pd.DataFrame()

    T = np.asarray(q_matrix, dtype=float)[:3, :3]
    km = _kaplan_meier(times, events, GRID_HOURS)
    model = []
    for t in GRID_HOURS:
        predicted = [
            _phase_type_survival(int(state), T, t) for state in initial_states
        ]
        model.append(float(np.mean(predicted)))
    abs_error = np.abs(km - np.asarray(model))
    # Trapezoidal integral normalized by the evaluation-window width.
    integrated_abs_error = float(
        sum(
            (GRID_HOURS[i + 1] - GRID_HOURS[i])
            * (abs_error[i] + abs_error[i + 1]) / 2
            for i in range(len(GRID_HOURS) - 1)
        ) / (GRID_HOURS[-1] - GRID_HOURS[0])
    )
    curve = pd.DataFrame({
        "time_hours": GRID_HOURS,
        "kaplan_meier_probability_not_yet_critical": km,
        "phase_type_mean_predicted_survival": model,
        "absolute_difference": abs_error,
    })
    report = {
        "available": True,
        "event_definition": "first model-defined VAE state 3 (Critical), not an independent clinical event",
        "independent_unit": "first ICU stay per test subject",
        "subjects": int(len(times)),
        "events": int(sum(events)),
        "right_censored": int(len(events) - sum(events)),
        "followup_cap_hours": MAX_WINDOW_HOURS,
        "grid_hours": GRID_HOURS,
        "kaplan_meier_survival": km.tolist(),
        "phase_type_mean_predicted_survival": model,
        "integrated_absolute_survival_difference_0_to_48h": integrated_abs_error,
        "interpretation": (
            "Censoring-aware descriptive validation of time to the model-defined Critical label. "
            "It is not clinical prognosis validation; states are generated by the same VAE being assessed."
        ),
    }
    return report, curve


def run_clinical_validation():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in [INDEX_PATH, LATENT_PATH, Q_PATH]:
        require(path)
    stays = _read_stays()
    index = pd.read_csv(INDEX_PATH)
    latent = pd.read_csv(LATENT_PATH)
    q_matrix = np.load(Q_PATH)

    # 1. Trace the raw WBC result, including original units/reference metadata.
    wbc_trace = trace_wbc_outliers(stays, index)

    # Join source stay/admission metadata to model outputs; outcomes are not used
    # to fit VAE states or the original VAE-surrogate BN.
    stay_columns = ["subject_id", "stay_id", "intime", "outtime"]
    if "hadm_id" in stays.columns:
        stay_columns.append("hadm_id")
    stay_meta = stays[stay_columns].drop_duplicates("stay_id")
    frame = index.merge(
        stay_meta, on=["subject_id", "stay_id"], how="left", validate="many_to_one"
    )
    latent_cols = [
        c for c in ["subject_id", "stay_id", "hour", "risk_score_mu", "state", "state_name"]
        if c in latent.columns
    ]
    frame = frame.merge(
        latent[latent_cols], on=["subject_id", "stay_id", "hour"],
        how="left", validate="one_to_one", suffixes=("", "_latent")
    )
    if "state" not in frame or "risk_score_mu" not in frame:
        raise ValueError("latent_states.csv must contain state and risk_score_mu.")

    admissions_path = require(HOSP_DIR / "admissions.csv")
    admissions_header = pd.read_csv(admissions_path, nrows=0).columns
    admission_cols = [
        c for c in ["hadm_id", "subject_id", "hospital_expire_flag", "deathtime"]
        if c in admissions_header
    ]
    if "hadm_id" not in admission_cols:
        raise ValueError("admissions.csv must contain hadm_id to link clinical outcomes.")
    admissions = pd.read_csv(admissions_path, usecols=admission_cols).drop_duplicates("hadm_id")
    if "hospital_expire_flag" in admissions.columns:
        admissions["hospital_expire_flag"] = pd.to_numeric(
            admissions["hospital_expire_flag"], errors="coerce"
        )
    else:
        admissions["hospital_expire_flag"] = np.nan
    if "deathtime" in admissions.columns:
        admissions["deathtime"] = pd.to_datetime(admissions["deathtime"], errors="coerce")
        fallback = admissions["deathtime"].notna().astype(int)
        admissions["hospital_expire_flag"] = admissions["hospital_expire_flag"].fillna(fallback)
    if admissions["hospital_expire_flag"].isna().any():
        raise ValueError(
            "Some admissions lack hospital_expire_flag and deathtime; cannot assign an independent outcome."
        )
    admissions["hospital_expire_flag"] = admissions["hospital_expire_flag"].astype(int)
    if not admissions["hospital_expire_flag"].isin([0, 1]).all():
        raise ValueError("hospital_expire_flag must be binary (0/1).")

    if "hadm_id" not in frame.columns:
        raise ValueError("ICU stay table does not expose hadm_id; cannot link outcomes.")
    frame = frame.merge(
        admissions[["hadm_id", "hospital_expire_flag"]],
        on="hadm_id", how="left", validate="many_to_one"
    )
    frame = frame.loc[frame["hospital_expire_flag"].notna()].copy()

    # One baseline observation and one first ICU stay per subject prevents
    # repeated hourly rows from inflating apparent sample size.
    baseline = frame.loc[frame["hour"] == 0].sort_values(
        ["subject_id", "intime", "stay_id"]
    ).drop_duplicates("subject_id", keep="first").copy()
    baseline["hospital_expire_flag"] = baseline["hospital_expire_flag"].astype(int)
    train = baseline.loc[baseline["split"] == "train"].copy()
    test = baseline.loc[baseline["split"] == "test"].copy()
    if train.empty or test.empty:
        raise ValueError("No baseline subjects available in train or test split.")

    y_test = test["hospital_expire_flag"].to_numpy(dtype=int)
    # Mapping convention is state=3-digitize(mu); lower mu therefore means
    # higher-numbered state. Report both score orientations rather than hiding it.
    risk_score = -test["risk_score_mu"].to_numpy(dtype=float)
    vae_state = test["state"].to_numpy(dtype=int)
    state_auc = _safe_auc(y_test, vae_state)
    score_auc = _safe_auc(y_test, risk_score)
    reverse_score_auc = _safe_auc(y_test, test["risk_score_mu"].to_numpy(dtype=float))
    mortality_by_state = {}
    for state_id, name in enumerate(STATE_NAMES):
        subset = test.loc[test["state"] == state_id, "hospital_expire_flag"]
        mortality_by_state[name] = {
            "n": int(len(subset)),
            "deaths": int(subset.sum()),
            "mortality_rate": float(subset.mean()) if len(subset) else None,
        }

    # 4. Fit a separate BN whose target is the independent hospital outcome.
    bn_report, bn_predictions = _fit_discrete_mortality_bn(train, test)
    bn_predictions.to_csv(OUT_DIR / "bn_hospital_mortality_predictions.csv", index=False)

    # 3. Use all right-censored test stays in a KM comparison against Phase-Type.
    # The cohort is one first ICU stay per test subject, selected above.
    prognosis_report, prognosis_curve = _prognosis_survival_analysis(
        latent, stays, q_matrix, test[["subject_id", "stay_id"]]
    )
    if not prognosis_curve.empty:
        prognosis_curve.to_csv(OUT_DIR / "prognosis_survival_curve.csv", index=False)

    outcome_report = {
        "clinical_endpoint": "in-hospital mortality (MIMIC admissions.hospital_expire_flag; deathtime used only when flag is missing)",
        "independent_unit": "one baseline hour-0 record from the first ICU stay per subject",
        "split_policy": "patient-level train/test split preserved; test outcomes were not used to fit cutoffs or CPT",
        "train_subjects": int(train["subject_id"].nunique()),
        "train_deaths": int(train["hospital_expire_flag"].sum()),
        "test_subjects": int(test["subject_id"].nunique()),
        "test_deaths": int(y_test.sum()),
        "test_survivors": int(len(y_test) - y_test.sum()),
        "event_count_warning": (
            "Only two or fewer test deaths: AUC and average precision are extremely unstable "
            "and must be reported as exploratory, not as evidence of clinical validation."
            if int(y_test.sum()) <= 2
            else (
                "Report confidence intervals and validate on a larger external cohort; "
                "this demo cohort alone is not clinical validation."
            )
        ),
        "vae_state_ordinal_roc_auc": state_auc,
        "vae_latent_score_auc_assuming_lower_mu_means_higher_risk": score_auc,
        "vae_latent_score_auc_reverse_orientation": reverse_score_auc,
        "test_mortality_by_model_defined_state": mortality_by_state,
        "interpretation": (
            "This is an independent-outcome check against in-hospital mortality, not proof that the four labels "
            "are clinically calibrated. The demo cohort is small; AUC is undefined if the test split contains "
            "only one outcome class. Report sample counts and avoid clinical claims."
        ),
    }
    report = {
        "wbc_source_trace": {
            "itemid": WBC_ITEM_ID,
            "review_bound": WBC_REVIEW_BOUND,
            "raw_high_value_rows_matched_to_icu_stays": int(len(wbc_trace)),
            "output_csv": "results/wbc_outlier_trace.csv",
            "unit_counts_among_matched_high_values": (
                wbc_trace["valueuom"].fillna("<missing>").value_counts().to_dict()
                if "valueuom" in wbc_trace.columns else {}
            ),
            "interpretation": (
                "Review raw value, unit, reference range, flag, timestamp and processed hourly value. "
                "This script flags records for review; it does not decide clinical plausibility or delete them."
            ),
        },
        "vae_independent_outcome_evaluation": outcome_report,
        "bayesian_network_independent_outcome_evaluation": bn_report,
        "censoring_aware_prognosis_evaluation": prognosis_report,
    }
    with (OUT_DIR / "clinical_validation_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print("--- WBC source trace ---")
    print(f"Raw WBC values > {WBC_REVIEW_BOUND}: {len(wbc_trace)} ICU-matched rows")
    print(f"Trace: {OUT_DIR / 'wbc_outlier_trace.csv'}")
    if len(wbc_trace):
        cols = [c for c in ["subject_id", "hadm_id", "stay_id", "charttime", "valuenum", "valueuom", "ref_range_lower", "ref_range_upper", "flag", "processed_wbc"] if c in wbc_trace]
        print(wbc_trace[cols].to_string(index=False))
    print("--- VAE states vs independent in-hospital mortality ---")
    print(json.dumps(outcome_report, indent=2))
    print("--- Separate outcome-targeted BN ---")
    print(json.dumps(bn_report, indent=2))
    print("--- Censoring-aware first-Critical prognosis comparison ---")
    print(json.dumps(prognosis_report, indent=2))
    print(f"Saved full report: {OUT_DIR / 'clinical_validation_report.json'}")
    return report


if __name__ == "__main__":
    run_clinical_validation()
