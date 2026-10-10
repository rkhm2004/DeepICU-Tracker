# DeepICU-Tracker — Mathematical ICU Early-Warning System

## 1. Project Overview

DeepICU-Tracker is an ICU early-warning and risk-tracking system that converts raw ICU physiological measurements into a compact latent risk representation, interpretable risk states, stochastic state-transition dynamics, and an estimated time-to-Critical state.

The proposed architecture is:

Raw ICU Vitals/Labs → Leakage-Safe Preprocessing → VAE → Risk States → CTMC → Phase-Type/MGF → Bayesian Network → Clinical Risk/Explanation Output

**Current status:** preprocessing, VAE, risk-state mapping, CTMC, Phase-Type/MGF, inference, visualization, mathematical evaluation, and a first discrete Bayesian Network proof of concept are implemented. The BN currently models risk-state probabilities from the three physiological features; static patient priors and clinical-outcome validation remain future work.

---

## 2. Problem Statement

ICU patients can move between different levels of physiological risk over time. ICU systems continuously generate measurements such as heart rate, blood pressure, and laboratory values, but raw measurements do not directly provide:

1. a compact representation of the patient's current physiological condition;
2. an interpretable risk state;
3. a mathematical model of how risk changes over time;
4. the probability of moving between risk states; or
5. an estimate of the expected time until a Critical state.

Another important challenge is that ICU observations are longitudinal. A patient's ICU stays must not be incorrectly joined together, and information from the same patient must not leak between training and evaluation sets.

The project therefore combines **representation learning with stochastic-process modelling** rather than treating the problem as only a conventional classification task.

---

## 3. Proposed Solution

### Stage 1 — Variational Autoencoder (VAE)

The VAE receives three currently used ICU features:

- Heart Rate
- Systolic Blood Pressure (SBP)
- White Blood Cell count (WBC)

The encoder compresses these measurements into a low-dimensional latent representation. The implementation uses a scalar latent risk score to represent the patient's physiological condition.

### Stage 2 — Risk-State Mapping

The continuous latent score is converted into four states:

| State | Meaning |
|---|---|
| 0 | Low Risk |
| 1 | Medium Risk |
| 2 | High Risk |
| 3 | Critical |

The state cutoffs are learned from **training latent scores only**, saved as metadata, and reused during inference.

### Stage 3 — Continuous-Time Markov Chain (CTMC)

The sequence of risk states within each ICU stay is modelled as a CTMC.

For transient states, the generator rates are estimated using:

q_ij = N_ij / T_i, for i ≠ j

where N_ij is the number of observed transitions from state i to j and T_i is the observed exposure time in state i.

The diagonal is:

q_ii = − Σ(j≠i) q_ij

Critical is explicitly treated as an absorbing state.

### Stage 4 — Phase-Type Distribution / MGF

The transient CTMC generator is used to model the time until absorption into Critical.

For transient generator T:

E[tau] = pi(-T^-1)1

The implementation also provides:

- Phase-Type MGF;
- expected time-to-Critical;
- variance;
- standard deviation.

Thus, the system reports a model-based expected absorption time and its standard deviation; this is not an individual countdown.

### Stage 5 — Bayesian Network — Proof of Concept Implemented

The current discrete BN estimates risk-state probabilities from heart rate, systolic blood pressure and WBC feature bins. Feature cutoffs and the conditional probability table are fitted on training records only. Its output is agreement with VAE-derived labels, not clinical outcome accuracy. Static patient/admission priors and causal explanations are not implemented yet.

---

## 4. Complete Architecture

    DEEPICU-TRACKER
    ICU EARLY-WARNING SYSTEM

    MIMIC-IV ICU Vitals + Labs
                |
                v
    +-----------------------------+
    | Data Preprocessing          |
    | Patient-level 70/15/15     |
    | Train/Validation/Test       |
    | Train-only imputation       |
    | Train-only StandardScaler   |
    | Hourly trajectories          |
    +--------------+--------------+
                   |
                   v
    +-----------------------------+
    | VAE                         |
    | Heart Rate + SBP + WBC     |
    |       -> latent score       |
    +--------------+--------------+
                   |
                   v
    +-----------------------------+
    | Risk-State Mapping          |
    | Low -> Medium -> High       |
    |             -> Critical     |
    +--------------+--------------+
                   |
                   v
    +-----------------------------+
    | CTMC                        |
    | Transition counts +         |
    | exposure -> generator Q    |
    | Critical = absorbing       |
    +--------------+--------------+
                   |
                   v
    +-----------------------------+
    | Phase-Type / MGF            |
    | transient generator T      |
    | -> mean + variance + SD     |
    | -> expected time-to-Critical|
    +--------------+--------------+
                   |
          +--------+---------+
          |                  |
          v                  v
    +-------------+   +----------------------+
    | Bayesian    |   | Clinical Risk Output |
    | Network     |   | Current state        |
    |             |   | Expected time        |
    | BN proof    |   | Uncertainty / SD     |
    | Explanation |   | Risk trajectory      |
    +-------------+   +----------------------+

The BN proof of concept is implemented. Broader integration with static patient/admission priors and a final dashboard remains future work.

---

## 5. Leakage-Safe Preprocessing and Data Quality

The parser creates no more than the first 48 hourly bins of each ICU stay, and only keeps a bin when its start hour is before the recorded ICU outtime. Events outside the documented intime/outtime interval are excluded.

The current feature set is heart rate, systolic blood pressure (SBP), and white blood cell count (WBC). This is a narrow prototype feature set and is **not a comprehensive ICU severity assessment**. Expected units are recorded as review notes; confirm them against the source item metadata before interpreting values. Broad review bounds flag values for inspection and do not silently clip or delete them.

Missing-value handling is causal:
- Forward-fill occurs only within a stay and has per-feature limits: heart rate 6 hours, SBP 6 hours, WBC 24 hours.
- Backward-fill is not used.
- Missingness indicators are preserved in the index for auditing.
- Remaining missing values use training-split medians only.
- StandardScaler is fitted on training records only.
- Patient IDs are disjoint across train, validation and test.
- VAE state cutoffs, BN discretization cutoffs and BN CPT parameters are fitted using training data only.

The parser generates data/processed/data_quality_report.json with raw missingness, observed ranges, non-positive counts, broad-range flags and unit-review notes. Re-run the parser and every downstream stage after any preprocessing change; do not reuse old tensors or model artifacts with a newly generated index.

## 6. Model and Evaluation Semantics

### VAE states

The VAE reconstructs the three standardized physiological features and produces a one-dimensional latent score. Three cutoffs derived from training latent scores map each hour to Low Risk, Medium Risk, High Risk or Critical. These are **model-defined labels**, not validated clinical severity categories. Reconstruction loss alone does not prove that the score predicts deterioration or mortality.

The VAE assigns each hourly state independently, so its label may move back from Critical to a non-Critical state. The CTMC/Phase-Type model instead treats Critical as an absorbing state and uses only pre-Critical trajectory intervals for its transition model. This is a modelling assumption, not proof that a real patient's condition is irreversible.

### CTMC

The generator is estimated from training trajectories only. Transition intervals are counted only between consecutive hours within the same stay; gaps are skipped. Critical is absorbing for CTMC prognosis. The implementation and audit check finite Q, non-negative off-diagonal rates, zero row sums and transition/exposure artifact validity.

### Phase-Type / MGF

The transient generator is checked for finite values, non-negative off-diagonal rates, non-positive row sums, eventual absorption and numerical conditioning. The code validates finite moments and M(0)=1.

Expected absorption time and its standard deviation are model-based population quantities under fitted CTMC assumptions. They are **not an individual countdown, a calibrated prediction interval or a clinical prediction**. Discharge, death, treatment, transfer, changing physiology and censoring are not fully represented by this simple model.

### Bayesian Network

The current network is:

    HeartRateLevel ─┐
    SBPLevel        ─┼──> RiskState
    WBCLevel        ─┘

It estimates P(RiskState | HeartRateLevel, SBPLevel, WBCLevel) with a smoothed conditional probability table. Latent labels and raw feature values are aligned using subject_id, stay_id and hour keys; duplicate keys, missing matches and inconsistent splits raise errors.

The BN report includes a confusion matrix, per-class precision/recall/F1, macro-F1 and balanced accuracy. All these metrics measure agreement with VAE-derived surrogate labels, not clinical outcome accuracy. The BN is not a causal explanation model. Static patient/admission priors are not yet included.

### Prognosis evaluation

The current prognosis comparison is descriptive and includes only observations before the first Critical label in stays where Critical is observed. Stays that end without an observed Critical label are right-censored; they must not be assigned an invented event time. A small uncensored subset is exploratory only. A stronger analysis requires a prespecified event, a prediction horizon, suitable censoring-aware methods, independent outcomes and uncertainty estimates.

## 7. Current Implementation Status

| Component | Status |
|---|---|
| Patient-level data split | Implemented and checked |
| Duration-aware hourly index | Implemented |
| Causal imputation and missingness indicators | Implemented |
| Train-only fallback medians and scaler | Implemented |
| Data-quality report | Implemented |
| VAE and training-derived latent cutoffs | Implemented; clinical meaning not validated |
| CTMC estimation and absorbing Critical model | Implemented |
| Phase-Type/MGF numerical guards | Implemented |
| Inference and research-labelled plots | Implemented |
| BN CPT, keyed alignment and class-wise metrics | Implemented |
| Static patient/admission priors | Not implemented |
| Independent clinical-outcome validation | Not implemented |
| Clinical dashboard | Not implemented |

Passing software tests does not establish clinical validity or suitability for bedside decision-making.

## 8. Run Commands

Run these from the repository root in the activated project environment. The parser must run first after the preprocessing changes, because all downstream artifacts depend on its row set and feature values.

\`\`\`powershell
# Regression tests
python -m unittest tests/test_mathematical_pipeline.py
python -m unittest tests/test_bayesian_network.py

# Rebuild all artifacts using the corrected parser
python src/data_pipeline/mimic_parser.py
python src/vae/train.py
python src/markov_mgf/ctmc_estimator.py
python src/markov_mgf/mgf_calculator.py
python src/inference.py
python src/plot.py
python src/evaluation.py
python src/bayesian_net/train_infer.py

# Run the non-destructive audit against the new artifacts
python audit_pipeline.py
\`\`\`

## 9. Generated Artifacts

- data/processed/mimic_tensor.pt — scaled three-feature model input.
- data/processed/mimic_index.csv — subject/stay/hour keys, split, imputed raw-unit features and missingness indicators.
- data/processed/preprocessing.json — training medians, scaler parameters and preprocessing configuration.
- data/processed/data_quality_report.json — raw missingness, feature ranges and review flags.
- data/processed/latent_states.csv — VAE latent scores and model-defined states.
- data/processed/q_matrix.npy — estimated CTMC generator.
- results/inference_results.csv — per-hour state, model-based expected absorption time and missing-input quality flags.
- results/evaluation_report.json — split integrity, VAE metrics, CTMC diagnostics and descriptive prognosis evaluation.
- results/bn_predictions.csv — held-out BN probabilities and explanations.
- results/bn_evaluation.json — confusion matrix and agreement metrics against VAE labels.

## 10. Final Note

DeepICU-Tracker is a mathematical/software research prototype. It is not certified and is not clinically validated or intended for bedside decisions. Its current contribution is a reproducible pipeline with leakage-aware preprocessing, stochastic-process modelling, numerical checks and transparent surrogate-label evaluation. Independent clinical outcome validation remains future work.

## 8. Independent Clinical Outcome and Prognosis Validation

After regenerating the parser, VAE, and CTMC artifacts, run:

```powershell
python src/clinical_validation.py
python -m unittest tests/test_clinical_validation.py
```

The script creates:

- `results/wbc_outlier_trace.csv`: raw WBC lab values above the broad review threshold, including source timestamps, units, reference ranges/flags when available, ICU stay linkage, and the matching processed hourly value. Inspect this file to determine whether the WBC value of 116.1 is supported by the original record and what unit was recorded. The code flags values; it does not automatically remove them or claim clinical plausibility.
- `results/clinical_validation_report.json`: evaluation of the first-hour VAE state against the independent MIMIC in-hospital mortality field, including outcome counts, mortality rates by model-defined state, and score AUCs in both latent-score orientations.
- `results/bn_hospital_mortality_predictions.csv`: predictions from a separate outcome-targeted discrete BN, `P(hospital mortality | HR bin, SBP bin, WBC bin)`, trained only on one baseline record per training subject and evaluated on baseline records from test subjects.
- `results/prognosis_survival_curve.csv`: Kaplan-Meier survival estimates for time to the first model-defined Critical state, compared with Phase-Type survival predictions, retaining right-censored test trajectories.

**Important interpretation limits:** the outcome endpoint is in-hospital mortality, not a comprehensive measure of deterioration. The demo dataset and held-out test cohort are small, so AUC may be undefined if the test split contains only one outcome class. The VAE state names/cutoffs are still model-defined; mortality-rate patterns can provide evidence about their direction but do not by themselves validate calibrated clinical severity categories. The survival analysis is censoring-aware for time to the model-defined Critical state, not a clinical endpoint, because the event is generated from the VAE itself. Do not use these results for patient-care decisions.

The original BN in `src/bayesian_net/train_infer.py` continues to predict VAE surrogate labels. Its agreement metrics must not be described as clinical accuracy; the separate mortality-targeted BN above is the independent-outcome experiment.

