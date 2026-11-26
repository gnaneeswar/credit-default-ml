# Credit Default Prediction — End-to-End ML + MLOps Pipeline

Predicts whether a credit-card client will **default on next month's payment**, using the
UCI *Default of Credit Card Clients* dataset (id 350, Taiwan 2005, 30,000 clients).

`Dockerfile.train` ingests the data and builds the EDA. It then tunes four model families
with leak-free 3-fold CV, picks a decision threshold from out-of-fold deciles, and registers
the winner as `models:/credit-default-pipeline@champion` in a SQLite MLflow registry. After
that it evaluates the champion on a 20% holdout, runs Evidently drift monitoring, and
generates a 5-slide interview deck (`artifacts/credit_default_summary.pptx`).
`Dockerfile.serve` loads that champion into **FastAPI**. On a 12-core laptop (WSL2) the
pipeline takes **34–45 s** and the training container about 40–50 s end to end.

```bash
git clone https://github.com/gnaneeswar/credit-default-ml.git && cd credit-default-ml
docker compose up --build                          # train -> evaluate -> register -> serve :8000
curl -s -X POST localhost:8000/invocations \
     -H 'Content-Type: application/json' -d @sample_request.json
```

---

## Contents
1. [Business context](#1-business-context)
2. [Results at a glance](#2-results-at-a-glance)
3. [Architecture](#3-architecture)
4. [Design Decisions & Trade-offs](#4-design-decisions--trade-offs)
5. [Prerequisites](#5-prerequisites)
6. [Run with Docker](#6-run-with-docker)
7. [Verify the API](#7-verify-the-api)
8. [Outputs](#8-outputs)
9. [Local development](#9-local-development)
10. [Cleanup](#10-cleanup)
11. [Limitations & next steps](#11-limitations--next-steps)

---

## 1. Business context

A card issuer wants to flag clients likely to default next month (October 2005), so it can
act early: reduce the limit, contact the client, or offer a payment plan. The inputs are
six months (April–September 2005) of repayment status, bill amounts and payments, plus the
credit limit and demographics.

The two kinds of error cost different things:

| Error | What it means | Cost |
|---|---|---|
| **False positive** (good client flagged) | credit friction | lost revenue, churn, complaints |
| **False negative** (defaulter missed) | missed default | charge-off loss |

About 22% of clients default (3.5 non-defaulters per defaulter). Accuracy would therefore
reward a model that flags nobody, so the pipeline **selects on F1**. It reports ROC-AUC,
PR-AUC, precision and recall alongside, and makes the cut-off an **explicit, registered
decision** instead of a hidden 0.5.

### Data dictionary summary

The full dictionary is [data/data_dictionary.md](data/data_dictionary.md). It maps the raw
`X1..X23` / `Y` columns to their domain names.

**Standardised naming.** UCI names September's repayment status `PAY_0`, while bills and
payments use `BILL_AMT1` / `PAY_AMT1`. On load, `PAY_0` is **renamed to `PAY_1`**, so all
three monthly families share 1–6 indexing. This is a **breaking API change** from earlier
versions: requests must send `PAY_1`. A payload still using `PAY_0` gets a 422 listing
`missing_columns: ["PAY_1"]` and `unknown_columns: ["PAY_0"]`.

| n | Month (2005) | Columns |
|---|---|---|
| 1 | September (most recent) | `PAY_1`, `BILL_AMT1`, `PAY_AMT1` |
| 2 | August | `PAY_2`, `BILL_AMT2`, `PAY_AMT2` |
| 3–5 | July – May | `PAY_3..5`, `BILL_AMT3..5`, `PAY_AMT3..5` |
| 6 | April (oldest) | `PAY_6`, `BILL_AMT6`, `PAY_AMT6` |

The target `default_payment_next_month` refers to **October 2005**.

**Billing-cycle mechanics** (the project's domain definition):
- Statements run on rolling cycles and are named after the month they **close** in.
  Example: a 16th→15th cycle, with payment due on the 5th of the following month.
- **`BILL_AMT_n`** is the balance on month *n*'s closing date: previous unpaid balance +
  purchases + interest/fees − payments posted before the close.
- **`PAY_AMT_n`** is the cash received in **calendar** month *n*. A Sept 4 payment is
  deducted from `BILL_AMT1`. A Sept 22 payment counts in `PAY_AMT1` but carries over to
  October's statement.
- **`PAY_n`** is a snapshot on month *n*'s close of how the *previous* statement was
  handled by its due date:
  - `-2` inactive cycle
  - `-1` paid in full
  - `0` revolving / minimum met
  - `1..9` the number of consecutive missed due dates

  It counts missed due dates, not billing cycles. Under this rule, `PAY_1` = 1 means the
  Sept 5 due date (for the August bill) was missed.

**Empirical check.** The EDA tests these rules against the data (`domain_rule_checks` in
[artifacts/eda_report.md](artifacts/eda_report.md)). Most hold: the risk ordering of every
behavioural segment matches the domain logic (§4.2). Three do not match exactly:
- **Timing.** A "paid in full" status in month *k* is confirmed by statement *k* being
  covered by the *following* month's payment in **98–99%** of rows. The §4 pairing
  ("previous statement paid in month *k*") explains only 73–80%. So the data aligns status
  codes about one month later than the rule.
- **Inactive codes.** "`-2` ⇒ `BILL_AMT_n ≤ 0`" holds for only 33–71% of `-2` rows.
- **Code `1` is month-specific.** It appears 3,688 times in `PAY_1` but only 28/4/2/0/0
  times in `PAY_2..PAY_6`, so the source coded the latest month differently.

None of these change the engineered features, which use the status sequence and the
per-month amounts directly. They are documented rather than silently "fixed"; see
[§4.1](#41-undocumented-category-codes-and-domain-rules).

## 2. Results at a glance

**Model selection** (3-fold stratified CV on the 24,000-row training split; best candidate per family):

| Model | CV F1 | ± std | ROC-AUC | PR-AUC | Precision | Recall |
|---|---|---|---|---|---|---|
| **Random Forest** ✅ champion | **0.549** | 0.004 | 0.787 | 0.558 | 0.524 | 0.577 |
| XGBoost | 0.545 | 0.008 | 0.789 | 0.561 | 0.468 | 0.653 |
| LightGBM | 0.543 | 0.007 | 0.785 | 0.559 | 0.481 | 0.625 |
| Logistic Regression (WoE/IV) | 0.535 | 0.009 | 0.780 | 0.540 | 0.465 | 0.631 |

**Decision threshold:** 0.478, which means flagging the riskiest 26% of clients. It was
picked from out-of-fold predictions; [§4.4](#44-model-selection-cv-and-threshold-tuning)
explains how.

**Holdout** (6,000 clients the model never saw, scored by the *registered* `@champion`):

| Cutoff | F1 | Precision | Recall | Accuracy | ROC-AUC | PR-AUC | Flagged |
|---|---|---|---|---|---|---|---|
| registered 0.478 | 0.547 | 0.505 | 0.596 | 0.782 | 0.780 | 0.558 | 26.1% |
| default 0.5 (reference) | 0.549 | 0.524 | 0.576 | 0.791 | 0.780 | 0.558 | 24.3% |

Confusion at 0.478: the model catches **791 of 1,327 defaults** and flags 775 good clients
(TN 3,898 · FP 775 · FN 536 · TP 791). The train (OOF) F1 was 0.549 and holdout F1 is
0.547, so the cutoff generalises.

**Lift:** the riskiest 10% of clients default at 3.2× the base rate. The top 30% contain
64% of all defaults.

**Monitoring** (Evidently, compared against the out-of-fold training baseline):

| Batch | Drifted features | F1 (Δ) | Alerts |
|---|---|---|---|
| holdout | 0 / 23 | 0.547 (−0.002) | none |
| simulated downturn (limits −40%, statuses `PAY_1..6` +1 month, bills +40%) | 13 / 23 | 0.450 (−0.098) | data drift + performance |
| concept drift (30% of labels flipped) | 0 / 23 | 0.397 (−0.152) | performance only |

## 3. Architecture

```
                         ┌──────────────────────── Dockerfile.train ───────────────────────┐
 data/raw/*.csv ───────▶ │ ingest → EDA → GridSearchCV ×4 (3-fold) → OOF deciles/threshold │
 (committed, offline)    │        → log + register @champion → holdout eval → Evidently    │
                         └──────────────┬───────────────────────────────┬──────────────────┘
                                        │ writes                        │ writes
                                        ▼                               ▼
                    ./mlruns  (sqlite:///mlruns/mlflow.db +     ./artifacts (EDA, IV, deciles,
                               model artifacts, skops)           metrics, monitoring, .pptx deck)
                                        │ reads @champion + threshold tag
                         ┌──────────────▼─────────── Dockerfile.serve ──────────┐
   client ─ POST ──────▶ │ FastAPI + Uvicorn :8000   GET /health                │
   sample_request.json   │   POST /invocations (dataframe_split → probability,  │
   batch_client.py       │   prediction at the registered threshold)            │
                         └──────────────────────────────────────────────────────┘
```

| Package | Responsibility | Shipped in |
|---|---|---|
| `src/common` | `Path`-based settings, stage-aware stdout logging, tables, MLflow registry loader | train + serve |
| `src/data` | schema/canonical names, offline ingest + data dictionary, stratified split, EDA | train |
| `src/features` | `DomainCleaner`, `DomainFeatureEngineer`, `WoEIVSelector` (sklearn transformers) | train + serve |
| `src/training` | model specs/grids, tuning, OOF threshold, registry, holdout eval, orchestrator | train |
| `src/inference` | FastAPI app, request schemas, batch client — **no imports from training/data** | train + serve |
| `src/monitoring` | Evidently scenarios, reports, alerts | train |
| `src/reporting` | interview deck built from `artifacts/` (python-pptx) | train |

The pipeline's stages are `ingest → eda → train → evaluate → monitor → report`. Running
`python -m src.training.run` runs them all by default; `--stages`/`--models` select a subset.
The training run prints a stage-timing table at the end.

## 4. Design Decisions & Trade-offs

### 4.1 Undocumented category codes and domain rules

The EDA ([artifacts/eda_report.md](artifacts/eda_report.md), generated) checks every code
that appears in the data but not in the UCI documentation:

| Column | Codes | Rows | Evidence | Decision |
|---|---|---|---|---|
| `EDUCATION` | 0, 5, 6 | 345 (1.2%) | default rates 0–16%, close to code 4 "others" (5.7%), far from codes 1–3 (19–25%) | **remap to 4**: supported by the data |
| `MARRIAGE` | 0 | 54 (0.2%) | default rate 9.3% vs 26% for code 3 | **remap to 3** on meaning (unknown ≈ "others"). The rates do *not* match; 54 rows are too few for a reliable level of their own |
| `PAY_n` | −2, 0 | most common codes | `0` revolving (12.8%) and `−2` no consumption (13.2%) default **less** than `−1` paid in full (16.8%) | **keep as distinct codes**: collapsing them into −1 would lose real signal |

**`PAY_n` codes are fed to the models as ordinal numbers** (−2 < −1 < 0 < 1 …), not one-hot
encoded. Their risk is **not monotone** at the low end (−1 is riskier than 0 and −2), so
each model family handles that its own way:
- the trees split on the code several times;
- LR's WoE gives every code its own bin, so it learns a non-monotone mapping;
- the roll-rate features (§4.2) read each code as its own state.

One-hot or LightGBM-native categorical encodings were not compared.

The `PAY_n` meanings follow the project's billing-cycle definition
(see [§1](#data-dictionary-summary)). The EDA checks them against the data:
- **`0` and `−1` match.** Revolving accounts carry large balances; paid-in-full accounts
  carry small ones.
- **`−2` does *not* imply a zero bill.** In 67% of `PAY_1 = −2` rows a balance, fees or a
  credit still appears. So "inactive by status" (`N_INACTIVE`) and "nothing billed"
  (`N_BILL_NONPOS`) are **separate features**, and neither is derived from the other.
- **Timing.** A "paid in full" status is confirmed by the *following* month's payment in
  98–99% of rows, against 73–80% for the rule's own pairing.
- **`PAY_1` caveat.** Under the rule, `PAY_1` (judged on Sept 15 against the Sept 5 due
  date) holds no October information. Under the data's alignment, it may partly reflect
  October payments. An ablation that neutralises `PAY_1` costs about 0.04 CV F1. The model
  keeps it (as in the original paper), but in production it should be used only if the
  September status is known before the October decision.

### 4.2 Domain feature engineering

`DomainFeatureEngineer` is **stateless and vectorised**: it learns nothing from the data,
so it cannot leak, and it takes about 50 ms on 24,000 rows. It adds **45 features** to the
23 raw ones (68 in total). Every ratio is divided by `LIMIT_BAL`, never by a bill, and all
68 columns are tested finite for zero or negative bills, a zero limit and huge payments.

**Core features (22):**

* **Utilisation** `UTIL_n = BILL_AMTn / LIMIT_BAL`, plus its mean and max: how stretched the client is.
* **Payment coverage** `PAY_COVER_n = PAY_AMTn / BILL_AMT(n+1)`: the payment made in month *n*
  against the *previous* statement. If nothing was owed the value is 1. It is clipped to [0, 5].
* **6-month trends**: least-squares slope of bills and of payments, normalised by the limit.
* **Behaviour counts**: months delinquent (`PAY_n > 0`), worst delay, months revolving,
  months paid in full, months with no consumption, months with a bill ≤ 0.

**Billing-cycle accounting (12).** These are 6-month aggregates plus a flag for the most
recent month. Per-month copies of every signal would have added about 70 sparse,
near-duplicate columns.

| Signal | Features | Logic | Default rate (any month) |
|---|---|---|---|
| Credit balance (low risk) | `HAS_CREDIT_BAL_1`, `N_CREDIT_BAL`, `MIN_CREDIT_BAL_RATIO` | `BILL_AMT_n < 0`; negative bills are never clipped | 16.5% |
| Over limit (distress) | `OVER_LIMIT_1`, `N_OVER_LIMIT`, `MAX_OVER_LIMIT_EXCESS` | `BILL_AMT_n > LIMIT_BAL`; excess = `max(BILL − LIMIT, 0) / LIMIT` | 30.1% |
| Intra-cycle paydown (liquidity) | `PAYDOWN_1`, `N_PAYDOWN`, `MAX_PAYDOWN_EXCESS` | `PAY_AMT_n > BILL_AMT_n`; excess = `max(PAY − BILL, 0) / LIMIT` | 15.7% |
| Implied spend | `SPEND_1`, `SPEND_MEAN_3`, `SPEND_MEAN_5` | `BILL_AMT_n − (BILL_AMT_(n+1) − PAY_AMT_n)`, divided by `LIMIT`; 3- and 5-month rolling means smooth calendar-month vs. statement-cutoff timing | — |

**Roll-rate trajectory (11).** These cover the 5 transitions `PAY_6 → … → PAY_1`.
`*_RECENT` means the latest transition, `PAY_2 → PAY_1`. Single-month snapshots treat a
never-late account (`0,0,0`) like a recently cured one (`2,2,0`). These features separate
them.

| Signal | Features | Logic | Default rate (any) |
|---|---|---|---|
| Cure | `CURED_RECENT`, `N_CURES` | prev ≥ 2 and cur ≤ 0 | 26.9% (cured, now current) vs **11.7%** never late |
| Roll forward (worsening) | `ROLL_FWD_RECENT`, `N_ROLL_FWD`, `MAX_ROLL_FWD_STEP` | cur > prev **and cur ≥ 1**; step = cur − max(prev, 0) | 42.0% |
| Dormant → revolver | `N_DORMANT_TO_REVOLVER` | prev = −2 and cur = 0 | 18.2% |
| Roll backward (improving, still late) | `ROLL_BACK_RECENT`, `N_ROLL_BACK` | cur < prev, both > 0 | 47.3% |
| Stagnation (chronic delay) | `STAGNANT_RECENT`, `N_STAGNANT`, `MAX_DELINQ_STREAK` | cur = prev ≥ 2; longest run of `PAY ≥ 2` | 57.0% |

**Two deliberate deviations from the literal roll-forward rule** (`PAY_n > PAY_(n+1)`):
1. **Roll-forward requires `cur ≥ 1`.** Moves among the non-delinquent codes are not
   "worsening into delinquency". `−2 → 0` (dormant → revolver) is its own signal instead.
   Its 18.2% default rate sits between never-late-and-never-dormant (11.1%) and roll-forward
   (42.0%), so it would have diluted both.
2. **The step is measured from `max(prev, 0)`.** So `−2 → 2` counts as 2 missed due dates,
   not 4.

There is no recent flag for dormant → revolver, because `PAY_2 = −2 → PAY_1 = 0` occurs in
**0** of 30,000 rows. The latest transition also behaves differently from earlier ones: it
has 1,922 roll-backs against 74–161 elsewhere, because code `1` effectively exists only in
`PAY_1`. The `*_RECENT` flags isolate it.

**What IV says.**
- **Raw bills vs engineered ratios.** Raw `BILL_AMT1–5` have IV 0.013–0.016 and are dropped.
  `UTIL_*` reach 0.08–0.12, and `PAY_COVER_1` (0.24) beats raw `PAY_AMT1` (0.18).
- **The new trajectory features are among the strongest.** Five of the top 12: 
  `MAX_DELINQ_STREAK` 0.85, `N_STAGNANT` 0.57, `N_ROLL_FWD` 0.48, `MAX_ROLL_FWD_STEP` 0.48,
  `STAGNANT_RECENT` 0.38.
- **Sparse flags fall below 0.02 for LR.** `OVER_LIMIT_1`, `N_CREDIT_BAL`, `CURED_RECENT`,
  `N_DORMANT_TO_REVOLVER`, `HAS_CREDIT_BAL_1` and `MIN_CREDIT_BAL_RATIO` are dropped there.
  The tree models still use them.

**Impact vs. the 45-feature baseline** (same splits, grids and seeds; `main` @ `bfc33f6`):

| | Baseline (45 features) | Billing-cycle features (68) | Δ |
|---|---|---|---|
| RF CV F1 (champion) | 0.5460 ± 0.0065 | **0.5493 ± 0.0042** | +0.003 |
| XGBoost / LightGBM / LR CV F1 | 0.5420 / 0.5438 / 0.5346 | 0.5447 / 0.5434 / 0.5354 | +0.003 / −0.000 / +0.001 |
| Holdout F1 @ registered cutoff | 0.5379 | **0.5468** | +0.009 |
| Holdout precision / recall | 0.485 / 0.604 | 0.505 / 0.596 | +0.021 / −0.008 |
| Holdout ROC-AUC / PR-AUC | 0.777 / 0.553 | 0.780 / 0.558 | +0.003 / +0.005 |
| Pipeline runtime | 41–45 s | 34–41 s | unchanged (noise) |

The gain is **consistent but modest**. Three of four families improve in CV, and the
holdout improves on every ranking metric. On its own, +0.009 F1 is still within the
holdout's sampling error (about ±0.015).

### 4.3 Leak-free WoE/IV for Logistic Regression; raw features for trees

* **LR pipeline:** `DomainCleaner → DomainFeatureEngineer → WoEIVSelector → LogisticRegression`.
  - **Binning:** `WoEIVSelector` gives each code column one bin per value and splits other
    columns into ≤10 quantile bins. Bins under 1% of rows are merged with a neighbour.
  - **Encoding and selection:** it computes Laplace-smoothed WoE and IV, and drops features
    with **IV < 0.02** (13 of 68: `BILL_AMT1–5`, `SEX`, `MARRIAGE` and six sparse flags).
  - **Why LR needs it:** WoE puts every feature on a log-odds scale and makes its effect
    roughly linear, which is what a linear model needs. It also caps the influence of
    outliers.
* **Leak-free by construction:** the selector sits *inside* the sklearn Pipeline, so
  `GridSearchCV` refits it on each training fold, and the held-out fold's labels never
  influence its bins or WoE values. Tests prove three things:
  - Each fold's fitted selector is identical to one fitted on that fold's training rows alone.
  - Flipping the held-out labels leaves their encoding unchanged.
  - A shuffled target yields IV ≈ 0.
* **Trees get unbinned features.** RF, LightGBM and XGBoost find their own split points and
  interactions; WoE binning would only throw resolution away. They see all 68 features.
* **`SEX` (fairness trade-off, deliberate):** IV removes `SEX` from LR (IV 0.011), but the
  tree models still receive it, and the champion is a tree model. This was a conscious
  choice for the assignment: its signal is weak and the model would barely change without
  it. In production I would drop protected attributes from all models, or run a per-group
  fairness audit (approval and error rates by `SEX`) before deployment.

### 4.4 Model selection, CV and threshold tuning

* **3-fold `StratifiedKFold`, no static validation split.** Every training row is scored
  exactly once by a model that did not see it, so out-of-fold (OOF) predictions cover 100%
  of the training data. The same fold object is shared by all models and the threshold
  step. The 20% test set is touched **once**, by the registered champion.
* **Compact grids** (8/4/4/4 candidates) with bounded depths, 100–200 trees and
  `tree_method="hist"`. Parallelism is one process per CV fit (`GridSearchCV(n_jobs=-1)`).
  Each estimator runs single-threaded (`n_jobs=1`, `OMP_NUM_THREADS=1`), which avoids
  nested OpenMP contention in WSL2 and Docker.
* **Imbalance and `scoring="f1"`.** F1 in the search is computed at the default 0.5
  cutoff, so each model is weighted towards the minority class (`class_weight`,
  `is_unbalance`, `scale_pos_weight`). This shifts its probabilities so that 0.5 is a
  sensible cutoff, which makes F1 a fair way to compare candidates. The LR grid shows the
  effect directly. Weighted and unweighted LR rank clients equally well (**ROC-AUC 0.780
  for both**). Without weights, F1 at 0.5 collapses to **0.47** (recall 0.36) instead of
  **0.535**. The trade-off is that weighted models are not calibrated probabilities. That
  is acceptable here because decisions use a tuned cutoff, not the raw scores.
* **Winner:** RF (`max_depth=12, min_samples_leaf=10`), by CV F1 (0.549). The top three are
  within 0.006 F1 of each other (about 1.4× the champion's fold std), so they are close to
  tied, and RF wins on the stated criterion. XGBoost has the best ROC-AUC and PR-AUC.
* **Threshold: coarse-to-fine, decile-based.**
  1. The OOF decile table ([artifacts/train_decile_analysis.csv](artifacts/train_decile_analysis.csv))
     shows bad rate, lift, cumulative capture, and the F1 you would get from flagging the
     top *k* deciles. The best is *k* = 3.
  2. Percentile cutoffs (1% steps) are then searched only inside deciles 2–4. Result:
     **flag the riskiest 26%, cutoff 0.478** (OOF F1 0.549).

  Every cutoff therefore reads as a credit policy ("review the riskiest 26%"). Deciles
  alone move 10% of clients at a time, too coarse to find the best F1. An unconstrained
  search could chase noise.
* **Honest caveat:** the F1 curve is **flat** between about 0.47 and 0.52. On the OOF data,
  the default 0.5 gives F1 0.5493 against 0.5488 for the chosen cutoff. On the holdout, 0.5
  scores 0.549 against 0.547 for the registered cutoff, and the order flipped between
  development runs. Differences of ±0.003 sit well inside the CV standard deviation (0.007)
  and the holdout's sampling error (about ±0.015 with 1,327 defaults). I did **not** retune
  on the test set, which would leak it into the model. On a flat plateau the cutoff mostly
  chooses a precision/recall mix. In production it should come from a cost matrix (loss
  per missed default vs. cost per false alarm), and this pipeline makes it a single
  registered, auditable number.

### 4.5 Decoupled architecture: FastAPI + MLflow Registry + Evidently

* **MLflow is used as a library only.** It provides tracking, artifacts and the model
  registry in `sqlite:///mlruns/mlflow.db` with artifacts in `./mlruns`. A parent run logs
  the pipeline, with one nested run per model family. The winner is registered as
  `credit-default-pipeline`, with alias `@champion` and version tags (`optimal_threshold`,
  `cv_f1`, `best_params` …). No `mlflow models serve` or `build-docker` is used.
* **Portable paths.** Both containers use `WORKDIR /app` and the same relative URIs, so the
  `file:///app/mlruns/...` locations written by the training container resolve unchanged in
  the serving container. If a database from another working directory is mounted (e.g.
  trained on the host), the pipeline **fails fast with a clear message** instead of loading
  broken paths.
* **Safe serialisation.** MLflow 3's sklearn flavour stores the pipeline with **skops**,
  not pickle. Only explicitly trusted types can load, so loading a model cannot run
  arbitrary code. The trusted list comes from the fitted pipeline and is checked against an
  allowlist (`src.features.`, `sklearn.`, `xgboost.`, `lightgbm.`). An unexpected type
  aborts registration.
* **One loader for evaluation, serving and monitoring.** All three call
  `src.common.mlflow_utils.load_champion`. So the holdout evaluation tests the exact code
  path the API uses: the batch client's F1 over HTTP matches the holdout evaluation
  exactly.
* **FastAPI service** (`src/inference`) has no imports from training or data code; AST and
  runtime tests enforce this.
  - **Request contract:** `dataframe_split` with exactly the 23 signature columns, in any
    order. The app reorders them to the training order and converts them to float64.
  - **Validation:** values must be finite numbers. Strings, booleans, null and NaN return
    422, and a 422 names the missing or unknown columns. Error responses never echo the
    request back, so a bad 6,000-row batch isn't returned in full and a NaN can't crash
    the error encoder.
  - **Failed model load:** the process stays up and `/health` returns 503 with the reason.
* **Evidently monitoring** compares each batch against the **OOF** training predictions, an
  honest baseline that isn't inflated by in-sample fit. It runs `DataDriftPreset` and
  `ClassificationPreset` at the registered threshold. Alerts fire when ≥ 25% of features
  drift, or when F1 or accuracy drops > 0.05.
  - The downturn simulation is caught by drift detection: exactly the 13 shifted columns
    (`LIMIT_BAL`, `PAY_1..PAY_6`, `BILL_AMT1..6`) are flagged, and the untouched `PAY_AMT*`
    are not.
  - Concept drift shows **no feature drift at all**, but F1 drops 0.15. Drift detection
    alone would miss it, which is why label-based performance monitoring is needed as well.

### 4.6 Engineering choices worth knowing

| Choice | Why |
|---|---|
| `uv` + `uv.lock`, `uv sync --frozen` in Docker | byte-identical dependency sets locally and in both images |
| `pandas<3` pin | Evidently 0.7.x imports a pandas internal that pandas 3 removed |
| float64 model signature | robust to clients sending `20000.0`; avoids MLflow's integer-schema pitfalls |
| offline-first data | the CSV and dictionary are committed; `ucimlrepo` runs only on a cache miss or `--refresh` |
| `pathlib` only, LF only | enforced by ruff `PTH` rules and `.gitattributes` / `.editorconfig` |
| pipeline 34–45 s (was 55–60 s) | MLflow start-up overlaps the searches in a background thread; `refit=False` searches plus one parallel batch for the winner's refit and OOF folds; Evidently workers pre-warmed while registration runs. Results are bit-identical to the naive version. Container start-up and exit add about 5 s |
| `PAY_0` → `PAY_1` | consistent 1–6 indexing across all three monthly families; the UCI name is recorded in `schema.UCI_ORIGINAL_NAMES` and the dictionary |
| deck from artifacts only | every slide number is read from `artifacts/*.json` / `*.csv`; the build checks text fit and raises `LayoutError` instead of overflowing |
| non-root containers | `USER 1000:1000`; `HOME`, `USER` and `LOGNAME` are set so MLflow works for any `-u` UID |

## 5. Prerequisites

* Linux `amd64` (or WSL2) with **Docker Engine ≥ 24** and the **compose plugin**. No Docker
  Desktop is needed.
* Internet access only to pull the base images (`python:3.12-slim-bookworm`, `ghcr.io/astral-sh/uv`).
  The dataset is in the repo.
* About 2.5 GB of disk: the two images share their base and dependency layers.
* About 3 GB of RAM at peak during training (one CV worker per core; less on fewer cores).
  The API uses about 300 MB.
* For local development only: [`uv`](https://docs.astral.sh/uv/). On WSL/Ubuntu,
  `sudo apt-get install libgomp1` is also needed for LightGBM; the Docker images include it.

## 6. Run with Docker

`mlruns/` and `artifacts/` are committed as empty folders (`.gitkeep`), so the bind mounts
belong to you. The containers run as your UID/GID, so the generated files do too.

### Option A — docker compose (recommended)

```bash
docker compose up --build
```

`train` runs all stages and exits 0. `serve` starts only after that
(`service_completed_successfully`) and listens on `http://localhost:8000`. It shows
`(healthy)` in `docker compose ps` once the model is loaded, a few seconds after start.
If your UID is not 1000, use `UID=$(id -u) GID=$(id -g) docker compose up --build`.

Batch-score the holdout against the running service (in a second terminal):

```bash
docker compose --profile score run --rm --no-deps score
```

### Option B — sequential `docker build` / `docker run`

```bash
mkdir -p mlruns artifacts

# 1) train, evaluate, register @champion  (writes ./mlruns and ./artifacts)
docker build --platform linux/amd64 -f Dockerfile.train -t credit-default-train .
docker run --rm --platform linux/amd64 -u "$(id -u):$(id -g)" \
  -v "$PWD/mlruns:/app/mlruns" -v "$PWD/artifacts:/app/artifacts" \
  credit-default-train

# 2) serve the registered champion on :8000
docker build --platform linux/amd64 -f Dockerfile.serve -t credit-default-serve .
docker run -d --name credit-default-serve --platform linux/amd64 -u "$(id -u):$(id -g)" \
  -p 8000:8000 -v "$PWD/mlruns:/app/mlruns" \
  credit-default-serve

# the model takes a few seconds to load: wait for the container healthcheck
until [ "$(docker inspect -f '{{.State.Health.Status}}' credit-default-serve)" = healthy ]; do sleep 2; done

# optional: batch-score the holdout through the API
docker run --rm --network host -u "$(id -u):$(id -g)" -v "$PWD/artifacts:/app/artifacts" \
  credit-default-train python -m src.inference.batch_client --url http://localhost:8000
```

Re-running training registers a new version and moves `@champion` to it. Restart `serve`
to load it.

## 7. Verify the API

```bash
curl -s localhost:8000/health
# {"status":"ok","model_loaded":true,"model_name":"credit-default-pipeline","model_alias":"champion",
#  "model_version":"1","model_family":"bagged trees","threshold":0.478358,"n_features":23,"error":null}

curl -s -X POST localhost:8000/invocations \
     -H 'Content-Type: application/json' -d @sample_request.json
# {"predictions":[{"default_probability":0.94,"prediction":1},
#                 {"default_probability":0.32,"prediction":0},
#                 {"default_probability":0.09,"prediction":0}], "threshold":0.478358, ...}
```

[`sample_request.json`](sample_request.json) holds three real holdout clients: a delinquent
defaulter (flagged), a revolving payer, and an inactive card. Interactive docs are at
`http://localhost:8000/docs`.

Python (`pip install requests pandas`):

```python
import json, requests

payload = json.load(open("sample_request.json"))
r = requests.post("http://localhost:8000/invocations", json=payload, timeout=10)
r.raise_for_status()
for p in r.json()["predictions"]:
    print(f"P(default)={p['default_probability']:.3f}  flagged={bool(p['prediction'])}")

# Build a request from a DataFrame (columns may be in any order):
import pandas as pd

df = pd.DataFrame(payload["dataframe_split"]["data"], columns=payload["dataframe_split"]["columns"])
body = {"dataframe_split": df.to_dict(orient="split")}  # index is optional and echoed back
print(requests.post("http://localhost:8000/invocations", json=body, timeout=10).json()["n_rows"])
```

Invalid input is rejected with a precise 422:

```bash
curl -s -X POST localhost:8000/invocations -H 'Content-Type: application/json' \
     -d '{"dataframe_split":{"columns":["LIMIT_BAL"],"data":[[20000]]}}'
# {"detail":{"error":"columns do not match the model signature","missing_columns":["SEX", ...], ...}}

# A pre-rename client still sending PAY_0 is told exactly what changed:
#   "missing_columns": ["PAY_1"], "unknown_columns": ["PAY_0"]   (HTTP 422)
```

## 8. Outputs

| Path | Content |
|---|---|
| `data/data_dictionary.{json,md}` | variables, canonical mapping, timeline, documented and undocumented codes (committed) |
| `artifacts/eda_summary.json`, `eda_report.md` | shapes, missingness, target balance, distributions, code-level default rates |
| `artifacts/cv_comparison.csv` | best candidate per model family (CV F1, ROC-AUC, PR-AUC, precision, recall) |
| `artifacts/iv_table.csv` | WoE/IV ranking and selection |
| `artifacts/train_decile_analysis.csv` | OOF decile lift, bad rate, capture and F1-at-cutoff table |
| `artifacts/test_metrics.json`, `test_confusion_matrix.{json,png}` | holdout evaluation of `@champion` |
| `artifacts/monitoring_report*.html`, `monitoring_summary.json` | Evidently reports and alert summary |
| `artifacts/credit_default_summary.pptx` | **interview deck** (16:9, cover + 4 slides: EDA observations, candidate models and CV grids, top 3 models, OOF vs test and deciles), rebuilt from the artifacts on every run; `uv run python -m src.reporting.ppt` rebuilds it alone |
| `artifacts/model_grids.json`, `train_oof_metrics.json` | deck inputs: model grids, imbalance handling and CV setup (generated from the model specs), and OOF metrics at the threshold |
| `artifacts/batch_predictions.csv` | batch client output: actual vs predicted |
| `mlruns/` | MLflow tracking database, model registry and artifacts |

To browse the runs (optional, read-only UI): `uvx mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.

## 9. Local development

```bash
uv sync                                   # Python 3.12 venv from uv.lock
uv run pytest -q                          # 106 offline tests
uv run ruff check . && uv run ruff format --check .
uv run python -m src.training.run         # all stages (or --stages / --models)
uv run uvicorn src.inference.app:app --port 8000
uv run python -m src.inference.batch_client
uv run python -m src.data.ingest --refresh   # re-download from UCI (optional)
```

The test suite covers:
- data, schema and dictionary
- domain transforms and edge cases
- WoE/IV correctness and CV leakage
- tuning and threshold logic
- the skops allowlist
- a registry round trip in a fresh process with no training code
- the FastAPI contract (`TestClient`)
- the batch client
- import boundaries
- Evidently parsing and alerts
- billing-cycle features (the brief's worked examples, trajectories, finiteness)
- EDA domain-rule checks
- the deck (slide count and titles, populated tables, in-bounds and non-overlapping shapes,
  font minimums, fills, data-driven numbers, overflow errors)

## 10. Cleanup

```bash
docker compose down --rmi all --remove-orphans          # compose path
docker rm -f credit-default-serve                       # sequential path
docker rmi credit-default-train credit-default-serve
git clean -fdX mlruns artifacts                         # delete generated outputs, keep .gitkeep
```

Clear `mlruns` before switching between a host (`uv run`) and a Docker training run, because
their absolute artifact roots differ. The pipeline refuses to mix them and says so.

## 11. Limitations & next steps

* **Cost-based threshold.** Replace F1 with an expected-loss objective once the business
  provides costs for missed defaults and false alarms.
* **Calibration.** Add isotonic or Platt calibration if downstream users need true
  probabilities (e.g. expected-loss pricing).
* **Fairness.** Drop `SEX` or audit outcomes by group; add a fairness section to monitoring.
* **Champion/challenger gate.** Promote a new version only if it beats the current
  `@champion` on a fixed backtest. Today, every training run promotes its winner.
* **Monitoring in production.** Run monitoring on a schedule over real scored batches.
  Labels arrive about a month later, so add a delayed-label join for performance metrics.
* **Leaner serve image.** Split the dependencies into `train`/`serve` groups so the serve
  image doesn't carry Evidently or ucimlrepo.
