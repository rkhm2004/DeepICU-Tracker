# Team Workflow Guide — ICU Early-Warning Project (4 Members, GitHub)

This guide splits the 4-stage pipeline (VAE → CTMC → Phase-Type/MGF → Bayesian Network)
across 4 team members, with a Git branching strategy, a commit-by-commit breakdown per
person (10–12 commits each), and exact steps to resolve merge conflicts.

---

## 1. Team Split (1 member = 1 stage = 1 owned module)

| Member | Owns | Branch name |
|---|---|---|
| Member A | Data extraction + preprocessing (MIMIC-IV) | `feature/data-pipeline` |
| Member B | Stage 1 — VAE (compressor) | `feature/vae-encoder` |
| Member C | Stage 2 + 3 — CTMC + Phase-Type/MGF (tracker + countdown) | `feature/markov-mgf` |
| Member D | Stage 4 — Bayesian Network (explainer) + final dashboard integration | `feature/bayesian-network` |

Why this split works: each module has a clean input/output contract (e.g., Member B's VAE
outputs a latent score that Member C's Markov model consumes), so members can work in
parallel without touching each other's files most of the time — this is what keeps merge
conflicts rare instead of constant.

---

## 2. Repository Structure (agree on this BEFORE anyone starts coding)

```
icu-early-warning/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/              # raw MIMIC-IV extracts (gitignored — don't commit patient data)
│   └── processed/        # cleaned, feature-engineered data
├── src/
│   ├── data_pipeline/    # Member A
│   ├── vae/              # Member B
│   ├── markov_mgf/       # Member C
│   ├── bayesian_net/     # Member D
│   └── dashboard/        # Member D (final integration)
├── notebooks/            # exploration notebooks, one per member, prefixed with name
├── tests/
└── docs/
    └── report/
```

**First commit of the whole project** (whoever creates the repo) should just be this empty
folder structure + `.gitignore` + `requirements.txt` + `README.md`. Everyone else branches
off this.

---

## 3. Branching Strategy

- `main` — always working, never broken. Nobody commits directly to `main`.
- `develop` (optional but recommended for 4 people) — integration branch where feature
  branches merge first, before eventually going to `main`.
- `feature/<name>` — one branch per person per stage (see table above).

Flow: `feature/xyz` → Pull Request → reviewed by 1 teammate → merge into `develop` →
periodically `develop` → `main` once stable.

### Setting up (whoever owns the repo does this once):
```bash
git init
git add .
git commit -m "Initial project structure"
git branch develop
git push -u origin main
git push -u origin develop
```

### Everyone else, to start working:
```bash
git clone <repo-url>
cd icu-early-warning
git checkout develop
git pull origin develop
git checkout -b feature/vae-encoder      # each person uses their own branch name
```

---

## 4. Commit Breakdown Per Person (10–12 commits each)

Keep commits **small and specific** — each commit should be one logical change, not
"finished everything." This is what naturally gets you to 10–12 commits instead of 1 giant
one, and it makes code review and conflict resolution much easier.

### Member A — Data Pipeline
1. `Add MIMIC-IV data download/setup instructions to README`
2. `Add script to extract chartevents (vitals) for cohort`
3. `Add script to extract labevents (labs) for cohort`
4. `Add script to extract admissions/patient baseline covariates`
5. `Add script to extract outcome labels (mortality, LOS)`
6. `Merge vitals+labs into unified per-patient timeseries table`
7. `Handle missing values and irregular timestamps`
8. `Add train/validation/test split logic`
9. `Add data validation checks (unit tests)`
10. `Add data summary/EDA notebook`
11. `Refactor extraction scripts into reusable functions`
12. `Document data schema in docs/`

### Member B — VAE (Stage 1)
1. `Add VAE model skeleton (encoder/decoder classes)`
2. `Implement reparameterization trick`
3. `Implement VAE loss (reconstruction + KL divergence)`
4. `Add training loop`
5. `Add normalization/preprocessing for vitals input`
6. `Train VAE on Member A's processed data, save checkpoint`
7. `Add latent score extraction function for a given patient`
8. `Add sigma-based confidence output handling`
9. `Add unit tests for VAE forward pass`
10. `Tune hidden layer size / latent dimension`
11. `Add visualization of latent space`
12. `Document VAE module usage in docs/`

### Member C — CTMC + Phase-Type/MGF (Stages 2 & 3)
1. `Add state discretization function (Low/Medium/High/Critical cutoffs)`
2. `Add Q matrix estimation from state sequences (MLE)`
3. `Add P(t) = expm(Qt) transition probability function`
4. `Add sub-generator matrix T extraction (drop absorbing state)`
5. `Implement expected time-to-critical formula (MGF-derived)`
6. `Add variance-of-time-to-critical calculation`
7. `Integrate with Member B's VAE output (latent score → state)`
8. `Add unit tests for Markov/MGF functions`
9. `Add transition diagram visualization`
10. `Validate Q estimates against known/simulated ground truth`
11. `Handle edge cases (patient already in Critical, sparse data)`
12. `Document Markov/MGF module usage in docs/`

### Member D — Bayesian Network + Final Integration (Stage 4)
1. `Add Bayesian Network structure definition (pgmpy)`
2. `Add CPD estimation from baseline covariates`
3. `Add inference/query function (VariableElimination)`
4. `Integrate baseline covariates from Member A's data pipeline`
5. `Add root-cause explanation output formatting`
6. `Build combined dashboard function (all 4 stages → 1 output)`
7. `Add end-to-end pipeline script (data → VAE → Markov → MGF → BN → output)`
8. `Add integration tests across all modules`
9. `Add example output for sample patients`
10. `Fix integration bugs found during merge testing`
11. `Add final results notebook`
12. `Write final report/README integration section`

**Note:** commits don't have to be perfectly evenly sized — the point is granularity.
Small, frequent commits with clear messages make review and conflict resolution far
easier than a few massive ones.

---

## 5. Day-to-Day Workflow (each person, each work session)

```bash
# 1. Always start by syncing with develop before you start new work
git checkout develop
git pull origin develop
git checkout feature/vae-encoder
git merge develop          # bring any new shared changes into your branch

# 2. Do your work, then commit in small chunks
git add src/vae/model.py
git commit -m "Implement reparameterization trick"

# 3. Push regularly (don't wait until everything is done)
git push origin feature/vae-encoder
```

Push at the end of every work session, even if the feature isn't finished — this is what
gives you your 10–12 commit history and also means you never lose a day's work.

---

## 6. Opening a Pull Request (PR) and Merging

1. Once your stage is working and tested, open a PR: `feature/vae-encoder` → `develop`.
2. Assign at least 1 teammate to review it — they check the code runs and makes sense,
   not just skim it.
3. Fix anything they flag, push again (updates the same PR automatically).
4. Once approved, merge using **"Squash and merge" or a regular merge** (team's choice —
   squash gives a cleaner `develop` history, regular merge keeps your 10–12 commits visible
   in `develop`'s history too, which is nicer for grading/proof-of-contribution).

---

## 7. Merge Conflicts — What They Are and Exactly How to Resolve Them

**Why they happen:** two people changed the *same lines* of the *same file* in different
ways, and Git can't automatically decide which version to keep. This is common on shared
files like `requirements.txt`, `README.md`, or the final integration script — less common
on files only one person touches (which is why the module split above minimizes this).

### Step-by-step resolution:

```bash
# You're on your feature branch, trying to merge in develop's latest changes
git checkout feature/bayesian-network
git pull origin develop        # or: git merge develop

# If there's a conflict, Git will tell you:
# CONFLICT (content): Merge conflict in requirements.txt
```

Open the conflicted file. Git marks the conflicting section like this:

```
torch==2.1.0
```

- Everything between `<<<<<<< HEAD` and `=======` is **your** version.
- Everything between `=======` and `>>>>>>> develop` is **their** version (from `develop`).
- You manually edit the file to keep whichever is correct (or both, if they're not
  actually conflicting in meaning — e.g., two different new dependencies added to the
  same line region). Delete the `<<<<<<<`, `=======`, `>>>>>>>` marker lines completely.

```bash
# After manually fixing the file:
git add requirements.txt
git commit -m "Resolve merge conflict in requirements.txt"
git push origin feature/bayesian-network
```

### A few practical rules that prevent most conflicts:
- Pull `develop` into your branch **daily**, not just once at the end — small frequent
  merges have small easy conflicts; merging once after 2 weeks has huge painful ones.
- Don't all edit `README.md` or `requirements.txt` at the same time without telling each
  other — these are the most common conflict points precisely because everyone touches
  them occasionally.
- If a conflict looks confusing, **talk to the teammate who wrote the other version**
  before guessing — this is faster and safer than silently picking one side.
- If you make a mistake mid-resolution and want to bail out entirely:
  ```bash
  git merge --abort
  ```
  This safely cancels the merge and returns you to before you started.

---

## 8. Final Integration (before submission)

1. Once all 4 feature branches are merged into `develop` and tested together, open one
   final PR: `develop` → `main`.
2. All 4 members review this final PR together.
3. Tag a release: `git tag v1.0` / `git push --tags` — useful to point to in your report as
   "final submitted version."

---

## 9. Proving Individual Contribution (for grading)

- GitHub's **Insights → Contributors** tab shows commit counts per person automatically —
  this is exactly why hitting 10–12 real, meaningful commits per person (not one dump)
  matters for fair grading.
- Make sure everyone commits using their **own GitHub account** (not one person typing for
  everyone) — check `git config user.name` / `git config user.email` on each person's
  machine before the first commit.

---

## 10. Codex Mathematical Alignment Fixes

The `codex` branch corrects the implementation details that must be consistent with
the proposed architecture in the project review:

1. **Patient-aware trajectories** — `mimic_parser.py` now preserves `stay_id` and
   hourly position in `mimic_index.csv`. Laboratory observations are assigned to an ICU
   stay only when their timestamp falls inside that stay.
2. **Reproducible VAE state mapping** — training saves the exact latent cutoffs in
   `data/processed/vae_metadata.json`. Positive/high latent values map toward Low Risk
   and lower values toward Critical, matching the interpretation used in the project
   review.
3. **Valid CTMC generator estimation** — `ctmc_estimator.py` estimates
   `q_ij = N_ij / T_i` from within-stay transitions and exposure time instead of using
   an unconstrained matrix logarithm of an empirical transition matrix. Critical is
   explicitly absorbing.
4. **Phase-Type/MGF consistency** — `mgf_calculator.py` now exposes the Phase-Type MGF,
   expected absorption time, variance, and standard deviation. This keeps the countdown
   calculation tied directly to the transient generator `T`.
5. **Inference consistency** — `inference.py` loads the saved VAE thresholds rather than
   hard-coding a second set of thresholds, and reports both mean time-to-Critical and its
   Phase-Type standard deviation.
6. **Plot consistency** — `plot.py` uses the actual computed countdown values and saved
   VAE thresholds; no hand-entered prognosis numbers remain.
7. **Regression tests** — `tests/test_mathematical_pipeline.py` checks that patient
   trajectories do not leak into one another, `P(t)=exp(Qt)` is stochastic, and the
   Phase-Type MGF/moments satisfy basic mathematical sanity checks.
8. **Leakage-safe ML evaluation** — the parser assigns subjects (and therefore all of
   their ICU stays) to train/validation/test before model fitting. Training-only medians
   and StandardScaler statistics are saved in `preprocessing.json`; the VAE trains on
   train records, selects its checkpoint using validation loss, reports untouched test
   loss, and learns latent-state cutoffs from the training split only. CTMC estimation
   also uses the training trajectories only, while inference can score all saved records.

### Current execution order

Run these commands from the repository root after activating the virtual environment:

```powershell
python -m unittest tests/test_mathematical_pipeline.py
python src/data_pipeline/mimic_parser.py
python src/vae/train.py
python src/markov_mgf/ctmc_estimator.py
python src/markov_mgf/mgf_calculator.py
python src/inference.py
python src/plot.py
python src/evaluation.py
```

The parser creates a deterministic patient-level 70/15/15 train/validation/test split.
It fits imputation fallback medians and StandardScaler parameters on the training
subjects only. The VAE uses the training split for optimization, validation loss for
checkpoint selection/early stopping, and the test split only for final evaluation.
The CTMC is estimated from training trajectories only. Re-run the parser whenever you
change the split/preprocessing code so all downstream artifacts are regenerated.

The Bayesian Network remains intentionally pending. Its planned role is still Stage 4:
connect static patient/admission priors with the real-time VAE/Markov risk state to provide
the causal/root-cause explanation described by the project architecture.


## 11. Evaluation and leakage verification

After the complete pipeline has run, execute:

```powershell
python src/evaluation.py
```

The evaluation module performs the following checks and produces report-ready outputs:

- verifies that no patient appears in more than one train/validation/test split;
- verifies that preprocessing metadata reports train-only fitting;
- verifies that VAE state cutoffs were fitted on the training split;
- reports VAE total, reconstruction, and KL loss separately for train/validation/test;
- reports Low/Medium/High/Critical state distributions for each split;
- recomputes the training-only CTMC transition counts and exposure;
- validates the training CTMC on held-out test one-hour transition probabilities using (P(1)=exp(Q)), including self-transitions and stopping trajectories at first Critical;
- compares Phase-Type expected time-to-Critical against observed remaining time on uncensored test trajectories (stays that actually reach Critical inside the 48-hour window);
- writes `results/evaluation_report.json` and `results/vae_split_metrics.csv`;
- generates evaluation graphs in `results/`.

The held-out transition validation is a model-diagnostic check, not a clinical accuracy metric. The observed prognosis comparison is also descriptive only: a test stay that never reaches Critical within the 48-hour observation window is right-censored and is not treated as a zero-time event, and observations after first Critical are excluded.

Generated artifacts are ignored by Git and should not be committed with patient data.
