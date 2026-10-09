# DeepICU-Tracker — Mathematical ICU Early-Warning System

## 1. Project Overview

DeepICU-Tracker is an ICU early-warning and risk-tracking system that converts raw ICU physiological measurements into a compact latent risk representation, interpretable risk states, stochastic state-transition dynamics, and an estimated time-to-Critical state.

The proposed architecture is:

Raw ICU Vitals/Labs → Leakage-Safe Preprocessing → VAE → Risk States → CTMC → Phase-Type/MGF → Bayesian Network → Clinical Risk/Explanation Output

**Current status:** preprocessing, VAE, risk-state mapping, CTMC, Phase-Type/MGF, inference, visualization, and mathematical evaluation are implemented and tested. The **Bayesian Network (BN) is the remaining major architecture component and is intentionally pending.**

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

Thus, the system reports both an expected countdown and uncertainty around that countdown.

### Stage 5 — Bayesian Network — Pending

The planned BN provides the explanatory layer. It will connect static patient/admission factors with the real-time risk state to answer questions such as:

- Which patient factors are associated with the current risk?
- How do static factors affect risk probabilities?
- Which factors can provide a useful root-cause explanation?

**No BN results are reported yet because this component has not been implemented.**

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
    | PENDING     |   | Uncertainty / SD     |
    | Explanation |   | Risk trajectory      |
    +-------------+   +----------------------+

The BN and final explanatory integration are the only major architecture stages still pending.

---

## 5. Leakage-Safe Experimental Design

The current run contains:

| Split | Patients | ICU Stays | Hourly Records |
|---|---:|---:|---:|
| Train | 70 | 95 | 4,560 |
| Validation | 15 | 23 | 1,104 |
| Test | 15 | 22 | 1,056 |
| **Total** | **100** | **140** | **6,720** |

The split is **patient-level**, so all ICU stays belonging to a patient remain in the same split.

The pipeline also enforces:

- fallback imputation statistics fitted on training data only;
- StandardScaler fitted on training data only;
- VAE trained on training records;
- validation used for checkpoint selection and early stopping;
- test used only for final evaluation;
- VAE state cutoffs learned from training latent scores only;
- CTMC estimated from training trajectories only.

The evaluation confirms zero subject overlap between train, validation, and test.

---

# 6. Results Obtained So Far

## 6.1 Data Processing

The current run successfully extracted **6,720 hourly records across 140 ICU stays and 100 patients**.

- Train: 4,560 records
- Validation: 1,104 records
- Test: 1,056 records

No patient appears in more than one split.

---

## 6.2 VAE Results

The VAE was trained for up to 50 epochs with validation-based early stopping. Training stopped at epoch 40 after the validation loss stopped improving.

| Split | Loss / Record | Reconstruction MSE | KL |
|---|---:|---:|---:|
| Train | 2.5913 | 2.0584 | 0.5329 |
| Validation | 5.1269 | 2.4282 | 2.6987 |
| Test | 5.3843 | 4.2800 | 1.1043 |

Training-derived latent cutoffs:

    [-0.3169, 0.0126, 0.4792]

These produce the four reproducible states Low, Medium, High, and Critical.

---

## 6.3 Risk-State Distribution

Training is approximately balanced because the state cutoffs are derived from the training latent distribution.

### Validation

| State | Percentage |
|---|---:|
| Low Risk | 38.95% |
| Medium Risk | 36.32% |
| High Risk | 8.70% |
| Critical | 16.03% |

### Test

| State | Percentage |
|---|---:|
| Low Risk | 42.33% |
| Medium Risk | 17.71% |
| High Risk | 9.19% |
| Critical | 30.78% |

These are observed distributions and are not themselves clinical performance measures.

---

## 6.4 CTMC Results

The CTMC was fitted using the **training split only**.

Estimated generator Q, in rates/hour:

    [[-0.096529,  0.082430,  0.013015,  0.001085],
     [ 0.120930, -0.231008,  0.099225,  0.010853],
     [ 0.020349,  0.171512, -0.270349,  0.078488],
     [ 0.000000,  0.000000,  0.000000,  0.000000]]

Critical is absorbing.

Training transition counts:

    [[ 0, 76, 12,  1],
     [78,  0, 64,  7],
     [ 7, 59,  0, 27],
     [ 0,  0,  0,  0]]

Training exposure:

| State | Exposure |
|---|---:|
| Low Risk | 922 h |
| Medium Risk | 645 h |
| High Risk | 344 h |
| Critical | Absorbing |

The estimated Q satisfies the CTMC requirements: non-negative off-diagonal rates, zero row sums, and an absorbing Critical state.

---

## 6.5 Held-Out CTMC Validation

The trained CTMC is evaluated on the unseen test set using the one-hour transition matrix:

    P(1) = exp(Q)

### Observed vs modelled probabilities

**Low Risk**

Observed: [0.9061, 0.0856, 0.0083, 0.0000]

Model: [0.9125, 0.0713, 0.0143, 0.0020]

**Medium Risk**

Observed: [0.2832, 0.5841, 0.1327, 0.0000]

Model: [0.1040, 0.8046, 0.0782, 0.0131]

**High Risk**

Observed: [0.0526, 0.2982, 0.6140, 0.0351]

Model: [0.0256, 0.1348, 0.7699, 0.0698]

### Mean Absolute Error

| Current State | MAE |
|---|---:|
| Low Risk | 0.0072 |
| Medium Risk | 0.1168 |
| High Risk | 0.0953 |
| **Overall transient-state MAE** | **0.0731** |

This is a **transition-model diagnostic**, not a clinical accuracy metric.

The low-risk transitions match particularly closely. Medium- and high-risk transitions show larger discrepancies, which is useful information for future refinement.

---

## 6.6 Phase-Type / MGF Results

Expected remaining time until Critical:

| Current State | Expected Time | SD |
|---|---:|---:|
| Low Risk | 67.99 h | 62.48 h |
| Medium Risk | 60.08 h | 61.84 h |
| High Risk | 46.93 h | 59.18 h |
| Critical | 0 h | 0 h |

The model therefore gives the intended ordering:

    Low Risk       ≈ 67.99 h
    Medium Risk    ≈ 60.08 h
    High Risk      ≈ 46.93 h
    Critical       = 0 h

The standard deviation is large, so the output should be interpreted as an expected time with substantial uncertainty rather than a precise countdown.

---

## 6.7 Prognosis Diagnostic on Test Data

A descriptive comparison was performed between predicted Phase-Type time-to-Critical and observed remaining time to the first Critical state.

Only test stays that actually reached Critical within the 48-hour observation window were usable.

- Usable records: **26**
- Usable ICU stays: **11**

| State | Records | Predicted Mean | Observed Mean Remaining | MAE |
|---|---:|---:|---:|---:|
| Low Risk | 0 | 67.99 h | — | — |
| Medium Risk | 1 | 60.08 h | 4.00 h | 56.08 h |
| High Risk | 14 | 46.93 h | 5.14 h | 41.79 h |

These values **must not be presented as clinical prediction accuracy**. The usable sample is small, and stays that never reach Critical during the 48-hour observation window are right-censored.

Therefore this is currently a **descriptive diagnostic**, not a validated clinical performance metric.

---

## 6.8 Inference

The inference pipeline successfully generates:

    results/inference_results.csv

Each record includes the latent score, risk state, expected time-to-Critical, and Phase-Type standard deviation.

Example from the current run:

    -0.1278 -> High Risk -> 46.93 h
    -0.1717 -> High Risk -> 46.93 h
    -0.3601 -> Critical  -> 0 h

Inference uses the same saved training-derived state cutoffs used by the VAE stage.

---

## 7. Validation and Testing

The automated tests and evaluation pipeline currently check:

- patient/stay boundaries are respected;
- CTMC matrices are mathematically valid;
- P(t)=exp(Qt) behaves as a stochastic transition matrix;
- the Phase-Type MGF satisfies M(0)=1;
- Phase-Type moments are positive where expected;
- post-Critical records are excluded from remaining-time calculations;
- empirical transition validation includes self-transitions;
- split behaviour is reproducible;
- no patient appears in multiple train/validation/test splits;
- preprocessing metadata confirms train-only fitting;
- VAE cutoffs confirm train-only fitting;
- CTMC estimation uses training trajectories only.

Run the full evaluation with:

    python src/evaluation.py

It generates:

- results/evaluation_report.json
- results/vae_split_metrics.csv
- evaluation graphs

---

## 8. Current Project Status

| Component | Status |
|---|---|
| MIMIC-IV preprocessing | Complete |
| Patient-level data split | Complete |
| Leakage-safe imputation/scaling | Complete |
| VAE | Complete |
| Latent risk score | Complete |
| Low/Medium/High/Critical mapping | Complete |
| CTMC estimation | Complete |
| Absorbing Critical state | Complete |
| Phase-Type MGF | Complete |
| Expected time-to-Critical | Complete |
| Variance / Standard deviation | Complete |
| Inference pipeline | Complete |
| Visualizations | Complete |
| Mathematical regression tests | Complete |
| Held-out CTMC validation | Complete |
| Prognosis diagnostic | Complete |
| **Bayesian Network (discrete CPT model)** | **Implemented — proof of concept** |
| **Static demographic/admission priors** | **Not available in current feature set** |
| **Final explanatory/dashboard integration** | **BN prediction CSV available; dashboard integration pending** |

---

## 9. Bayesian Network — Current Implementation

A first discrete Bayesian Network implementation is available at `src/bayesian_net/train_infer.py`.

Current structure:

    HeartRateLevel ─┐
    SBPLevel       ─┼──> RiskState
    WBCLevel       ─┘

Each continuous physiological feature is discretized into Low / Moderate / High levels. The tertile cutoffs are fitted on training records only. A smoothed conditional probability table (CPT) estimates `P(RiskState | HeartRateLevel, SBPLevel, WBCLevel)`, using Laplace smoothing to avoid zero probabilities. The trained CPT is then evaluated on the held-out test split.

The script produces:

- `results/bn_predictions.csv` — per-record risk-state probabilities, predicted state, confidence, and a readable feature-bin profile;
- `results/bn_evaluation.json` — held-out agreement with VAE-derived risk labels and evaluation notes;
- `data/processed/bn_model.json` — training cutoffs and learned CPT.

**Important limitation:** this BN currently explains/reproduces the VAE-derived risk-state labels from the available physiological inputs. Its test score measures agreement with those labels, **not clinical outcome accuracy**. Static demographic/admission priors are not currently present in the processed feature set, so they are not claimed to be part of this first BN version. The explanatory/dashboard integration can be expanded once appropriate prior variables and validation targets are available.

---

## 10. Running the Current Pipeline

From the repository root:

    python -m unittest tests/test_mathematical_pipeline.py
    python src/data_pipeline/mimic_parser.py
    python src/vae/train.py
    python src/markov_mgf/ctmc_estimator.py
    python src/markov_mgf/mgf_calculator.py
    python src/inference.py
    python src/plot.py
    python src/evaluation.py
    python -m unittest tests/test_bayesian_network.py
    python src/bayesian_net/train_infer.py

Run the BN after parser + VAE training, because it consumes `mimic_index.csv` and `latent_states.csv`. Generated patient-derived data and model artifacts are ignored by Git through .gitignore.

---

## 11. Repository Structure

    DeepICU-Tracker/
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── src/
    │   ├── data_pipeline/
    │   │   └── mimic_parser.py
    │   ├── vae/
    │   │   ├── model.py
    │   │   └── train.py
    │   ├── markov_mgf/
    │   │   ├── ctmc_estimator.py
    │   │   └── mgf_calculator.py
    │   ├── bayesian_net/
    │   │   └── train_infer.py
    │   ├── evaluation.py
    │   ├── inference.py
    │   └── plot.py
    └── tests/
        ├── test_mathematical_pipeline.py
        └── test_bayesian_network.py

---

## 12. Final Summary

The mathematical core of the proposed ICU early-warning system is currently operational:

    ICU measurements
          ↓
    Leakage-safe preprocessing
          ↓
    VAE latent physiological representation
          ↓
    Four interpretable risk states
          ↓
    Training-only CTMC estimation
          ↓
    Phase-Type / MGF time-to-Critical model
          ↓
    Expected time + uncertainty
          ↓
    Held-out mathematical validation
          ↓
    Discrete Bayesian Network
    (physiological feature bins -> risk probabilities)

The current implementation has a working **VAE → Risk State → CTMC → Phase-Type/MGF** pipeline and a first **discrete Bayesian Network proof of concept**. Further work is still needed to incorporate static patient/admission priors and validate against independent clinical outcomes before claiming clinical explanations or a fully validated clinical early-warning system.
