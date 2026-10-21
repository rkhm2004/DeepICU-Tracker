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
| Patient-level data split | Implemented; verify locally with audit |
| Duration-aware hourly index | Implemented |
| Causal imputation and missingness indicators | Implemented |
| Train-only fallback medians and scaler | Implemented |
| Data-quality report | Implemented |
| VAE and training-derived latent cutoffs | Implemented; clinical meaning not validated |
| CTMC estimation and absorbing Critical model | Implemented |
| Phase-Type/MGF numerical guards | Implemented |
| Inference and research-labelled plots | Implemented |
| BN CPT, keyed alignment and surrogate-label metrics | Implemented |
| WBC source tracing workflow | Implemented; actual record still requires local data review |
| Independent-outcome validation script | Implemented; local metrics not yet verified |
| Static patient/admission priors in explanatory BN | Not implemented |
| Clinical dashboard | Not implemented |

Passing software tests does not establish clinical validity or suitability for bedside decision-making.

## 8. Environment and Dataset Setup

Run all commands from the repository root in an activated Python environment. The examples below use PowerShell on Windows.

\`\`\`powershell
# 1. Switch to the active branch and sync it
git fetch origin
git checkout codex
git pull origin codex

# 2. Create and activate a virtual environment (skip if already configured)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
\`\`\`

Download the MIMIC-IV demo dataset separately through its authorized source. Do not commit clinical data to Git. The parser expects these files at the paths below:

\`\`\`text
data/raw/mimic-iv-clinical-database-demo-2.2/
├── icu/
│   ├── icustays.csv
│   └── chartevents.csv
└── hosp/
    ├── labevents.csv
    └── admissions.csv       # required for independent mortality validation
\`\`\`

The parser needs the ICU stay and event files; src/clinical_validation.py additionally needs hosp/admissions.csv and raw WBC lab metadata. If your local dataset folder uses a different name, either place the files in the expected folder or update the constants in the scripts before running.

## 9. Run the Pipeline in Order

**Important:** after any parser/preprocessing change, rerun the pipeline from Step 1. Do not reuse a tensor, latent states, Q matrix or evaluation results produced from an older index.

### Step 1 — Run regression tests

\`\`\`powershell
python -m unittest tests/test_mathematical_pipeline.py
python -m unittest tests/test_bayesian_network.py
python -m unittest tests/test_clinical_validation.py
\`\`\`

### Step 2 — Parse and preprocess raw MIMIC data

\`\`\`powershell
python src/data_pipeline/mimic_parser.py
\`\`\`

Generates data/processed/mimic_tensor.pt, mimic_index.csv, preprocessing.json, split_summary.csv and data_quality_report.json.

### Step 3 — Train the VAE and create model-defined states

\`\`\`powershell
python src/vae/train.py
\`\`\`

Generates the VAE checkpoint, training metrics/history, metadata, and data/processed/latent_states.csv.

### Step 4 — Estimate the CTMC transition generator

\`\`\`powershell
python src/markov_mgf/ctmc_estimator.py
\`\`\`

Generates data/processed/q_matrix.npy and the transition-count/exposure artifacts.

### Step 5 — Calculate Phase-Type/MGF results

\`\`\`powershell
python src/markov_mgf/mgf_calculator.py
\`\`\`

Calculates expected absorption time, variance and standard deviation under the fitted CTMC assumptions. These are model-based population estimates, not guaranteed patient countdowns.

### Step 6 — Run inference

\`\`\`powershell
python src/inference.py
\`\`\`

Generates results/inference_results.csv.

### Step 7 — Generate visualizations

\`\`\`powershell
python src/plot.py
\`\`\`

Generates the project plots using the current artifacts.

### Step 8 — Evaluate the mathematical pipeline

\`\`\`powershell
python src/evaluation.py
\`\`\`

Generates results/evaluation_report.json, split metrics and evaluation plots. The time-to-Critical comparison is descriptive; the Critical event is a VAE-defined state rather than an independent clinical endpoint.

### Step 9 — Train and evaluate the VAE-surrogate Bayesian Network

\`\`\`powershell
python src/bayesian_net/train_infer.py
\`\`\`

Generates results/bn_predictions.csv, results/bn_evaluation.json and data/processed/bn_model.json. These metrics measure agreement with VAE-derived labels, **not clinical accuracy**.

### Step 10 — Run independent-outcome and WBC validation

\`\`\`powershell
python src/clinical_validation.py
\`\`\`

Generates:
- results/wbc_outlier_trace.csv — raw WBC values above the review threshold, with available unit/reference metadata and processed-hour linkage.
- results/clinical_validation_report.json — VAE state/latent-score association with independent in-hospital mortality, plus separate mortality-targeted BN metrics and censoring-aware prognosis diagnostics.
- results/bn_hospital_mortality_predictions.csv — test predictions from the outcome-targeted BN.
- results/prognosis_survival_curve.csv — Kaplan–Meier estimates compared with Phase-Type survival predictions for the model-defined Critical state.

Review the raw WBC source rows before deciding whether a high value is erroneous. The script flags values for inspection; it does not silently delete them or establish clinical plausibility automatically. The mortality and prognosis analyses are exploratory and do not constitute clinical validation.

### Step 11 — Run the full non-destructive audit

\`\`\`powershell
python audit_pipeline.py
\`\`\`

Run this **after** rebuilding the artifacts so that the audit checks the current outputs. Investigate all FAIL items and review WARN items; a clean audit is not evidence of clinical effectiveness.

## 10. Key Outputs

| Output | Purpose |
|---|---|
| data/processed/mimic_tensor.pt | Scaled three-feature VAE input |
| data/processed/mimic_index.csv | Subject/stay/hour keys, split, features and missingness indicators |
| data/processed/preprocessing.json | Train-only imputation/scaling metadata |
| data/processed/data_quality_report.json | Raw feature ranges, missingness and review flags |
| data/processed/latent_states.csv | VAE latent scores and model-defined states |
| data/processed/q_matrix.npy | Estimated CTMC generator |
| results/inference_results.csv | Per-hour state and expected absorption-time estimate |
| results/evaluation_report.json | Split integrity, VAE metrics, CTMC diagnostics and descriptive prognosis evaluation |
| results/bn_evaluation.json | BN agreement metrics against VAE-derived labels |
| results/clinical_validation_report.json | Independent-outcome exploratory analysis and validation summary |
| results/wbc_outlier_trace.csv | Raw WBC source records requiring review |
| results/bn_hospital_mortality_predictions.csv | Separate outcome-targeted BN predictions |
| results/prognosis_survival_curve.csv | Censoring-aware survival comparison for model-defined Critical |

## 11. Limitations and Interpretation

- The VAE's Low, Medium, High and Critical labels are **model-defined bins**, not validated clinical severity classes.
- The current feature set contains only heart rate, systolic blood pressure and WBC. It is not a comprehensive ICU severity assessment.
- The original Bayesian Network predicts VAE-derived labels. Its agreement score is not clinical accuracy and it is not a causal explanation model.
- The separate mortality-targeted BN and VAE-vs-mortality checks use an independent endpoint, but performance must be interpreted with test sample/event counts and uncertainty. Small demo cohorts may yield unstable or undefined metrics.
- Phase-Type expected absorption time and standard deviation are model-based population quantities under CTMC assumptions, not an individual countdown or calibrated clinical prediction interval.
- The survival comparison handles right-censoring for time to the first **model-defined Critical label**. It does not establish clinical time-to-deterioration performance.
- Static patient/admission priors, broader physiological features, external-cohort validation and the clinical dashboard remain future work.

**DeepICU-Tracker is a mathematical/software research prototype, not a clinically validated or certified system. It must not be used to guide patient-care decisions.**
