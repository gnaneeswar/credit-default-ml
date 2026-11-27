# Credit Default Prediction — End-to-End ML + MLOps Pipeline

Predicts whether a credit-card client will **default on next month's payment**, using the
UCI *Default of Credit Card Clients* dataset (id 350, Taiwan 2005, 30,000 clients).

`Dockerfile.train` ingests the data and builds the EDA. It then tunes four model families
with leak-free 3-fold CV, picks a decision threshold from out-of-fold deciles, and registers
the winner as `models:/credit-default-pipeline@champion` in a SQLite MLflow registry. After
that it evaluates the champion on 20% test data, runs Evidently drift monitoring, and
generates an 8-slide review deck (`artifacts/credit_default_summary.pptx`).
`Dockerfile.serve` loads that champion into **FastAPI**. On a 12-core laptop (WSL2) the
pipeline takes about **40–70 s** on the host and the training container about 50–75 s end to end,
depending on CPU/RAM load.

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

**Model selection** (3-fold stratified CV on the 24,000-row training split). Each family's
hyperparameters come from `GridSearchCV(scoring="f1")`. The **champion is the family with
the highest out-of-fold (OOF) F1 at its own tuned cutoff**, because that is the metric
actually deployed:

| Model | OOF F1 @ own cutoff | Cutoff | CV F1@0.5 (± std) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **LightGBM** ✅ champion | **0.5475** | 0.520 | 0.544 ± 0.008 | 0.785 | 0.560 |
| XGBoost | 0.5473 | 0.581 | 0.543 ± 0.006 | 0.789 | 0.563 |
| Random Forest | 0.5472 | 0.530 | 0.546 ± 0.008 | 0.787 | 0.555 |
| Logistic Regression (WoE/IV) | 0.5424 | 0.592 | 0.534 ± 0.010 | 0.780 | 0.539 |

The top three are **tied** within 0.0002 OOF F1, far less than the fold standard deviation
(about 0.007). Ranked by F1@0.5 instead, the random forest would have won; §4.4 explains
why that metric isn't comparable across families. The champion's settings are LightGBM with
`num_leaves=31` and `learning_rate=0.05`.

**Decision threshold:** 0.520, which means flagging the riskiest 27% of clients. The default
0.5 was an explicit candidate. The tuned cutoff beats it on the OOF data (0.5475 vs 0.5442)
and on the test data below.

**Test data** (6,000 clients the model never saw, scored by the *registered* `@champion`):

| Cutoff | F1 | Precision | Recall | Accuracy | ROC-AUC | PR-AUC | Flagged |
|---|---|---|---|---|---|---|---|
| registered 0.520 | 0.543 | 0.484 | 0.617 | 0.770 | 0.778 | 0.561 | 28.2% |
| default 0.5 (reference) | 0.539 | 0.470 | 0.632 | 0.761 | 0.778 | 0.561 | 29.8% |

Confusion at 0.520: the model catches **819 of 1,327 defaults** and flags 873 good clients
(TN 3,800 · FP 873 · FN 508 · TP 819). The OOF F1 was 0.547 and the test data F1 is 0.543, so
the cutoff generalises.

**Lift:** on the test data, the riskiest 10% of clients default at 3.11× the base rate (3.16× on the
training OOF scores). The top 30% contain 64.5% of all test data defaults (63.8% OOF).

**Test data deciles on the training score bands** (`artifacts/test_decile_analysis.csv`). The
training OOF decile cutoffs are applied unchanged to the 6,000 test data scores, so each row
compares like for like with the training table:

| Decile | Train bad rate | Test data bad rate | Train cum. capture | Test data cum. capture | Test data share |
|---|---|---|---|---|---|
| 1 | 69.8% | 68.8% | 31.6% | 33.2% | 10.7% |
| 3 (operating) | 27.6% | 26.0% | 63.8% | 64.5% | 10.4% |
| 10 | 5.1% | 5.8% | 100% | 100% | 7.8% |

The rank-ordering holds on unseen data. The test data band shares stay within 7.8–11.3% (each
training band holds 10% by construction), giving a **score PSI of 0.009**, far below the
usual 0.1 "stable" threshold. Monitoring reuses this score-band PSI for prediction drift.

**Monitoring** (Evidently, compared against the out-of-fold training baseline):

| Batch | Drifted features | Score PSI | F1 (Δ) | Alerts |
|---|---|---|---|---|
| test data | 0 / 23 | 0.009 | 0.543 (−0.005) | none |
| simulated downturn (limits −40%, revolving / late statuses +1 month, bills +40%) | 13 / 23 | 0.751 | 0.451 (−0.096) | data drift + score drift + performance |
| concept drift (30% of labels flipped) | 0 / 23 | 0.009 | 0.412 (−0.136) | performance only |

## 3. Architecture

```
                         ┌──────────────────────── Dockerfile.train ───────────────────────┐
 data/raw/*.csv ───────▶ │ ingest → EDA → GridSearchCV ×4 (3-fold) → OOF deciles/threshold │
 (committed, offline)    │        → log + register @champion → test data eval → Evidently  │
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
| `src/training` | model specs/grids, tuning, OOF threshold, registry, test data eval, orchestrator | train |
| `src/inference` | FastAPI app, request schemas, batch client — **no imports from training/data** | train + serve |
| `src/monitoring` | Evidently scenarios, reports, alerts | train |
| `src/reporting` | review deck built from `artifacts/` (python-pptx) | train |

The pipeline's stages are `ingest → eda → train → evaluate → monitor → report`. Running
`python -m src.training.run` runs them all by default; `--stages`/`--models` select a subset.
The training run prints a stage-timing table at the end.

## 4. Design Decisions & Trade-offs

### 4.1 Undocumented category codes and domain rules

The EDA ([artifacts/eda_report.md](artifacts/eda_report.md), generated) checks every code
that appears in the data but not in the UCI documentation. **Default rates come from the
24,000-row training split only**, so the test data's labels never informed a design
decision. Row counts are over the full data.

| Column | Codes | Rows | Evidence | Decision |
|---|---|---|---|---|
| `EDUCATION` | 0, 5, 6 | 345 (1.15%) | train default rates 0–17%, close to code 4 "others" (6.2%), far from codes 1–3 (19–25%) | **remap to 4**: supported by the data |
| `MARRIAGE` | 0 | 54 (0.2%) | train default rate 10.6% vs 28.2% for code 3 | **remap to 3** on meaning (unknown ≈ "others"). The rates do *not* match; 54 rows are too few for a reliable level of their own |
| `PAY_n` | −2, 0 | most common codes | `0` revolving (12.8%) and `−2` inactive (12.7%) default **less** than `−1` paid in full (16.6%) | **keep as distinct codes**: collapsing them into −1 would lose real signal |

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
| Credit balance (low risk) | `HAS_CREDIT_BAL_1`, `N_CREDIT_BAL`, `MIN_CREDIT_BAL_RATIO` | `BILL_AMT_n < 0`; negative bills are never clipped | 15.7% |
| Over limit (distress) | `OVER_LIMIT_1`, `N_OVER_LIMIT`, `MAX_OVER_LIMIT_EXCESS` | `BILL_AMT_n > LIMIT_BAL`; excess = `max(BILL − LIMIT, 0) / LIMIT` | 30.4% |
| Intra-cycle paydown (liquidity) | `PAYDOWN_1`, `N_PAYDOWN`, `MAX_PAYDOWN_EXCESS` | `PAY_AMT_n > BILL_AMT_n`; excess = `max(PAY − BILL, 0) / LIMIT` | 15.7% |
| Implied spend | `SPEND_1`, `SPEND_MEAN_3`, `SPEND_MEAN_5` | `BILL_AMT_n − (BILL_AMT_(n+1) − PAY_AMT_n)`, divided by `LIMIT`; 3- and 5-month rolling means smooth calendar-month vs. statement-cutoff timing | — |

**Roll-rate trajectory (11).** These use the status sequence `PAY_6 → … → PAY_1`.
Single-month snapshots treat a never-late account (`0,0,0`) like a recently cured one
(`2,2,0`); these features separate them.
- **`*_RECENT` flags** cover the **latest** transition (`PAY_2 → PAY_1`).
- **The `N_*` counts and `MAX_ROLL_FWD_STEP`** cover only the **4 history transitions**
  (`PAY_6 → … → PAY_2`).

The split is deliberate. Code `1` exists almost only in `PAY_1`, so the latest transition is
coded differently. In the full data it has 1,922 roll-backs against 74–161 in each earlier
transition. Keeping it out of the counts means each feature sees one coding regime, and no
event is counted twice.

| Signal | Features | Logic | Train default rate (any) |
|---|---|---|---|
| Cure | `CURED_RECENT`, `N_CURES` | prev ≥ 2 and cur ≤ 0 | 27.2% (cured, now current) vs **11.5%** never late |
| Roll forward (worsening) | `ROLL_FWD_RECENT`, `N_ROLL_FWD`, `MAX_ROLL_FWD_STEP` | cur > prev **and cur ≥ 1**; step = cur − max(prev, 0) | 42.1% |
| Dormant → revolver | `N_DORMANT_TO_REVOLVER` | prev = −2 and cur = 0 | 18.7% |
| Roll backward (improving, still late) | `ROLL_BACK_RECENT`, `N_ROLL_BACK` | cur < prev, both > 0 | 47.7% |
| Stagnation (chronic delay) | `STAGNANT_RECENT`, `N_STAGNANT`, `MAX_DELINQ_STREAK` | cur = prev ≥ 2; longest run of `PAY ≥ 2` | 57.3% |

**Deliberate choices relative to the literal rules:**
1. **Roll-forward requires `cur ≥ 1`.** Moves among the non-delinquent codes are not
   "worsening into delinquency". `−2 → 0` (dormant → revolver) is its own signal instead:
   its 18.7% default rate sits between never-late (11.5%) and roll-forward (42.1%), so
   merging it into either would dilute both.
2. **The step is measured from `max(prev, 0)`.** So `−2 → 2` counts as 2 missed due dates,
   not 4.
3. **A cure requires `prev ≥ 2`.** Recovering from a single missed due date (`1 → 0`) counts
   as neither a cure nor a roll-back, since roll-back needs `cur > 0`.

There is no recent flag for dormant → revolver, because `PAY_2 = −2 → PAY_1 = 0` occurs in
**0** of 30,000 rows.

**What IV says.**
- **Raw bills vs engineered ratios.** Raw `BILL_AMT1–5` have IV 0.013–0.016 and are dropped.
  `UTIL_*` reach 0.08–0.13, and `PAY_COVER_1` (0.24) beats raw `PAY_AMT1` (0.18).
- **The trajectory features rank high.** `MAX_DELINQ_STREAK` 0.85, `N_STAGNANT` 0.48,
  `STAGNANT_RECENT` 0.38, `N_ROLL_FWD` and `MAX_ROLL_FWD_STEP` 0.30.
- **Sparse flags fall below 0.02 for LR.** `OVER_LIMIT_1`, `N_CREDIT_BAL`, `CURED_RECENT`,
  `N_DORMANT_TO_REVOLVER`, `HAS_CREDIT_BAL_1` and `MIN_CREDIT_BAL_RATIO` are dropped there.
  The tree models still use them.

**What the champion relies on** (`artifacts/feature_group_importance.csv`). This is
permutation importance on the 6,000-row test data, as the drop in ROC-AUC. It answers two
different questions:

| Total importance: raw source shuffled **before** the pipeline | ΔAUC | | Conditional: engineered group shuffled **after** the pipeline | ΔAUC |
|---|---|---|---|---|
| Repayment status (`PAY_1..6`) | **0.160** | | Status counts | 0.058 |
| Payments (`PAY_AMT*`) | 0.044 | | Utilisation | 0.013 |
| Bill amounts (`BILL_AMT*`) | 0.029 | | Implied spend | 0.003 |
| Credit limit | 0.016 | | Payment coverage | 0.003 |
| Demographics | 0.006 | | Stagnation | 0.001 |

- **Total** shuffles the raw columns, so every derived feature changes with them. Shuffling
  `PAY_1..6` also scrambles `N_DELINQ`, the roll-rate features and `MAX_DELINQ_STREAK`. This
  measures how much the model relies on that source of information.
- **Conditional** shuffles one engineered group while the raw columns it came from stay
  intact. It measures what the group adds *on top of everything else*, so it is a lower
  bound. Several sparse groups (cure, over-limit, credit balance) add nothing at the margin.

An earlier version measured only the conditional kind and reported repayment status at
0.016. The total view shows the model's dependence on repayment history is roughly ten
times larger.

Per feature, the champion's own importance (**total gain**) ranks `N_DELINQ`, `MAX_DELINQ`
and `PAY_1` first, then payments and utilisation. Gain is computed on training data and
favours continuous features, so it is reported *alongside* the test data group view. LightGBM's
default `feature_importances_` are split **counts**, not gain, so the code reads total gain
explicitly for both boosters.

**History of the feature work** (same splits, grids and seeds):

| | `main` @ `bfc33f6` (45 features) | + billing features (68) | + review fixes (now) |
|---|---|---|---|
| Champion (selection rule) | RF (CV F1@0.5) | RF (CV F1@0.5) | **LightGBM (OOF F1 @ own cutoff)** |
| Champion CV F1@0.5 | 0.5460 | 0.5493 | 0.5442 |
| Test data F1 @ registered cutoff | 0.5379 | 0.5468 | 0.5426 |
| Test data precision / recall | 0.485 / 0.604 | 0.505 / 0.596 | 0.484 / 0.617 |
| Test data ROC-AUC / PR-AUC | 0.777 / 0.553 | 0.780 / 0.558 | 0.778 / **0.561** |

The billing features gave a consistent, modest gain. The review fixes changed the
methodology:
- family selection by the deployed metric;
- 0.5 as an explicit threshold candidate;
- history-only roll counts.

Those changes trade 0.004 test data F1 for higher recall and PR-AUC. Every change between
columns is within the test data's sampling error (about ±0.015). The methodology change was
made because it is correct, not because it scores higher.

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
  `tree_method="hist"`. Parallelism is one process per fit (`n_jobs=-1` at the search and
  refit level). Each estimator runs single-threaded (`n_jobs=1`, `OMP_NUM_THREADS=1`), which
  avoids nested OpenMP contention. The **served** model is single-threaded too: the registry
  enforces it, and a test checks it on a real tree model.
* **Imbalance and `scoring="f1"`.** Inside each family, `GridSearchCV` ranks candidates by
  F1 at 0.5. Each model is weighted towards the minority class (`class_weight`,
  `is_unbalance`, `scale_pos_weight`), which shifts its probabilities so that 0.5 is a
  sensible cutoff *for that model*. The LR grid shows the effect directly: weighted and
  unweighted LR rank clients equally well (**ROC-AUC 0.780 for both**), but without weights
  F1 at 0.5 collapses to **0.46** (recall 0.35) instead of **0.534**. The trade-off is that
  weighted models are not calibrated probabilities. That is acceptable because decisions use
  a tuned cutoff.
* **Choosing between families: OOF F1 at each family's own cutoff.** The four families
  handle imbalance in different ways, so 0.5 sits at a different point on each one's
  precision/recall curve, and F1@0.5 is not comparable *across* families. Instead:
  1. One parallel batch fits each family's 3 OOF folds plus its full refit (`fit_families`).
  2. Each family gets its own threshold from its own OOF scores.
  3. The family with the highest OOF F1 at that cutoff wins.

  LightGBM, XGBoost and the random forest are tied within 0.0002. **Caveat:** the same 3
  folds pick the hyperparameters, the cutoff and the family, so OOF F1 is mildly optimistic
  (a winner's curse). The untouched test data is the unbiased estimate.
* **Threshold: coarse-to-fine, with 0.5 as a candidate.**
  1. The OOF decile table ([artifacts/train_decile_analysis.csv](artifacts/train_decile_analysis.csv))
     shows bad rate, lift, cumulative capture, and the F1 you would get from flagging the
     top *k* deciles. The best is *k* = 3.
  2. Percentile cutoffs (1% steps) are searched across "flag the top 20–40%", the window
     around that decile. The default 0.5 is added as an explicit candidate, and ties go to
     0.5, so tuning is used only when it actually helps.

  Result: **flag the riskiest 27%, cutoff 0.520**. That gives OOF F1 0.5475 against 0.5442
  at 0.5, and test data F1 0.543 against 0.539.
* **Honest caveat:** the F1 curve is fairly flat around its peak (0.47–0.53). Differences of
  a few thousandths sit inside the CV standard deviation (about 0.007) and the test data's
  sampling error (about ±0.015 with 1,327 defaults). I did **not** retune on the test set,
  which would leak it into the model. In production the cutoff should come from a cost
  matrix (loss per missed default vs. cost per false alarm). The pipeline makes it a single
  registered, auditable number, stored at full precision as a model-version tag.
* **`scale_pos_weight`** for XGBoost is computed once from the full training labels and
  reused inside the CV folds. Strictly, that crosses fold boundaries. Under stratification
  the fold ratios match the full ratio to the third decimal, so the effect is negligible.

### 4.5 Decoupled architecture: FastAPI + MLflow Registry + Evidently

* **MLflow is used as a library only.** It provides tracking, artifacts and the model
  registry in `sqlite:///mlruns/mlflow.db` with artifacts in `./mlruns`. A parent run logs
  the pipeline, with one nested run per model family. The winner is registered as
  `credit-default-pipeline`, with alias `@champion` and version tags (`optimal_threshold` at
  full precision, `cv_f1`, `best_params` …). No `mlflow models serve` or `build-docker` is
  used.
* **Portable paths.** Both containers use `WORKDIR /app` and the same relative URIs, so the
  `file:///app/mlruns/...` locations written by the training container resolve unchanged in
  the serving container. If a database from another working directory is mounted (e.g.
  trained on the host), the pipeline **fails fast with a clear message** instead of loading
  broken paths. The serving container mounts `./mlruns` **read-only**.
* **Version-pinned loading.** The loader resolves `@champion` **once**, then reads the
  threshold, signature and model from `models:/credit-default-pipeline/<version>`. A retrain
  that moves the alias mid-load therefore can't pair one version's model with another
  version's threshold.
* **Safe serialisation.** MLflow 3's sklearn flavour stores the pipeline with **skops**,
  not pickle. Only explicitly trusted types can load, so loading a model cannot run
  arbitrary code.
  - **One shared allowlist.** It lives in `src.common.mlflow_utils.is_trusted`: the prefixes
    `src.features.`, `sklearn.`, `xgboost.`, `lightgbm.`, plus the exact type
    `collections.OrderedDict`, a plain container that LightGBM persists.
  - **Checked twice.** The registry checks at logging time, and the serving loader checks the
    list stored with the model *again* at load time. A tampered `MLmodel` can't widen what
    the API agrees to deserialise.
  - **Every family is tested.** A test serialises each candidate family, so a
    family-specific type can't surface only on the day that family wins.
* **One loader for evaluation, serving and monitoring.** All three call
  `src.common.mlflow_utils.load_champion`. The test data evaluation therefore tests the exact
  code path the API uses: the batch client's F1 over HTTP matches the test data evaluation
  exactly.
* **FastAPI service** (`src/inference`) has no imports from training, data, monitoring or
  reporting code. AST tests (which resolve relative imports) and a runtime test enforce
  this.
  - **Request contract:** `dataframe_split` with exactly the 23 signature columns, in any
    order. The app reorders them to the training order and converts them to float64.
  - **Business ranges:** every column has a rule (`FEATURE_RULES`), and anything outside it
    returns **422** with up to 20 `{row, column, value, rule}` violations. No value is
    silently clipped, imputed or scored. The rules:
    - `LIMIT_BAL` between 1 and 1e9;
    - `SEX` ∈ {1, 2}, `EDUCATION` 0–6, `MARRIAGE` 0–3, `AGE` 18–100, `PAY_n` −2..9 (all
      integer codes);
    - `|BILL_AMT| ≤ 1e9`, `0 ≤ PAY_AMT ≤ 1e9`.
  - **Hard failures are 422, never 500.** These are rejected before scoring: strings,
    booleans, null, NaN/Infinity, magnitudes above 1e15 (checked *before* float conversion,
    so `10**400` can't overflow) and values beyond float32 range. The transformer also
    raises on non-finite input, rather than filling it silently, for callers that bypass
    the API.
  - **Error responses never echo the request,** so a bad 6,000-row batch isn't returned in
    full and a NaN can't crash the error encoder.
  - **Failed model load:** the process stays up and `/health` returns 503 with the
    exception class only. Paths and URIs stay in the server log.
* **Evidently monitoring** compares each batch against the **OOF** training predictions, an
  honest baseline that isn't inflated by in-sample fit (it is mildly optimistic for the same
  winner's-curse reason as in §4.4). It runs `DataDriftPreset` and `ClassificationPreset` at
  the registered threshold, plus a **score PSI** on the training decile bands, which catches
  output drift caused by engineered features that raw-column tests can't see. Alerts fire
  when ≥ 25% of features drift, when score PSI > 0.25, or when F1 or accuracy drops > 0.05.
  - **The downturn simulation** worsens only revolving and late statuses (`PAY ≥ 0`); an
    inactive or paid-in-full status isn't "worse" by one step. It is caught by feature
    drift (exactly the 13 shifted columns), by score drift (PSI 0.75) and by performance.
  - **Concept drift** shows **no feature or score drift at all**, but F1 drops 0.14. Drift
    detection alone would miss it, which is why label-based performance monitoring is
    needed as well.

### 4.6 Engineering choices worth knowing

| Choice | Why |
|---|---|
| `uv` + `uv.lock`, `uv sync --frozen` in Docker | byte-identical dependency sets locally and in both images |
| `pandas<3` pin | Evidently 0.7.x imports a pandas internal that pandas 3 removed |
| float64 model signature | robust to clients sending `20000.0`; avoids MLflow's integer-schema pitfalls |
| offline-first data | the CSV and dictionary are committed; `ucimlrepo` runs only on a cache miss or `--refresh` |
| `pathlib` only, LF only | enforced by ruff `PTH` rules and `.gitattributes` / `.editorconfig` |
| pipeline about 40–70 s on the host (12-core WSL2 reference) | MLflow start-up overlaps the searches in a background thread; `refit=False` searches plus one parallel batch with every family's OOF folds and full refit; Evidently workers pre-warmed while registration runs. Container start-up and exit add about 5 s, and the training container takes about 50–75 s depending on CPU/RAM load |
| single-threaded serving | every estimator keeps `n_jobs=1`, and the registry enforces it before logging. A multi-threaded forest takes about 4× longer per row and spawns a thread pool per request inside uvicorn |
| `PAY_0` → `PAY_1` | consistent 1–6 indexing across all three monthly families; the UCI name is recorded in `schema.UCI_ORIGINAL_NAMES` and the dictionary |
| deck from artifacts only | every slide number is read from `artifacts/*.json` / `*.csv`; the build checks text fit and raises `LayoutError` instead of overflowing |
| non-root containers | `USER 1000:1000`; `HOME`, `USER` and `LOGNAME` are set so MLflow works for any `-u` UID |

## 5. Prerequisites

* **Docker Engine ≥ 24** with the **Docker Compose plugin**. The images run natively on
  `linux/amd64` (Linux, Windows/WSL2) and `linux/arm64` (Apple Silicon). Docker Desktop works on
  macOS and Windows; on Linux or WSL2, Docker Engine alone is enough.
* Internet access only to pull the base images (`python:3.12-slim-bookworm`, `ghcr.io/astral-sh/uv`),
  or the pre-built images from `ghcr.io` for Option B. The dataset is in the repo.
* About 2.5 GB of disk: the two images share their base and dependency layers.
* About 3 GB of RAM at peak during training (one CV worker per core; less on fewer cores).
  The API uses about 300 MB.
* For local development only: [`uv`](https://docs.astral.sh/uv/). On WSL/Ubuntu,
  `sudo apt-get install libgomp1` is also needed for LightGBM; the Docker images include it.

## 6. Run with Docker

`mlruns/` and `artifacts/` are committed as empty folders (`.gitkeep`), so the bind mounts
belong to you. The containers run as your UID/GID, so the generated files do too. If the
folders are missing (e.g. a run without cloning), the one-shot `init-dirs` service creates them
and hands them to the runtime UID before `train` starts, so Docker never leaves them root-owned.

### Option A — Build & Run End-to-End from Source

```bash
git clone https://github.com/gnaneeswar/credit-default-ml.git && cd credit-default-ml
docker compose up --build
```

`train` runs all stages and exits 0. `serve` starts only after that
(`service_completed_successfully`) and listens on `http://localhost:8000`. It shows
`(healthy)` in `docker compose ps` once the model is loaded, a few seconds after start.
If your UID is not 1000, use `HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up --build`.
(`UID=...` does not work, because bash's `UID` is read-only and never exported.) Clone the
repo inside the Linux filesystem (e.g. `~/projects`), **not** under `/mnt/c` on WSL2, where
SQLite locking and UID mapping on bind mounts break.

Batch-score the test data and check the API (in a **second** terminal):

```bash
docker compose --profile score run --rm --no-deps score
curl -s localhost:8000/health
curl -s -X POST localhost:8000/invocations \
     -H 'Content-Type: application/json' -d @sample_request.json
```

### Option B — Run Using Pre-Built Multi-Architecture Images (No Git Clone Needed)

Both images are public on GitHub Container Registry for `linux/amd64` and `linux/arm64`, so
Docker pulls the variant that matches your machine:

* `ghcr.io/gnaneeswar/credit-default-train:latest` (used by `init-dirs`, `train` and `score`)
* `ghcr.io/gnaneeswar/credit-default-serve:latest`

Only `docker-compose.yml` and `sample_request.json` are downloaded. The images are re-tagged to
the names that `docker-compose.yml` uses, and `--no-build` makes compose use them as they are.

#### Linux / macOS (Bash / Zsh)

```bash
mkdir -p credit-default-ml && cd credit-default-ml
curl -fsSLO https://raw.githubusercontent.com/gnaneeswar/credit-default-ml/main/docker-compose.yml
curl -fsSLO https://raw.githubusercontent.com/gnaneeswar/credit-default-ml/main/sample_request.json

docker pull ghcr.io/gnaneeswar/credit-default-train:latest
docker pull ghcr.io/gnaneeswar/credit-default-serve:latest
docker tag ghcr.io/gnaneeswar/credit-default-train:latest credit-default-train:latest
docker tag ghcr.io/gnaneeswar/credit-default-serve:latest credit-default-serve:latest

docker compose up --no-build
```

In a **second** terminal, from the same folder:

```bash
docker compose --profile score run --rm --no-deps score
curl -s localhost:8000/health
curl -s -X POST localhost:8000/invocations \
     -H 'Content-Type: application/json' -d @sample_request.json
```

#### Windows (PowerShell)

```powershell
New-Item -ItemType Directory -Force -Path credit-default-ml | Out-Null
Set-Location credit-default-ml
$base = "https://raw.githubusercontent.com/gnaneeswar/credit-default-ml/main"
Invoke-WebRequest -UseBasicParsing "$base/docker-compose.yml" -OutFile docker-compose.yml
Invoke-WebRequest -UseBasicParsing "$base/sample_request.json" -OutFile sample_request.json

docker pull ghcr.io/gnaneeswar/credit-default-train:latest
docker pull ghcr.io/gnaneeswar/credit-default-serve:latest
docker tag ghcr.io/gnaneeswar/credit-default-train:latest credit-default-train:latest
docker tag ghcr.io/gnaneeswar/credit-default-serve:latest credit-default-serve:latest

docker compose up --no-build
```

In a **second** PowerShell window, from the same folder:

```powershell
docker compose --profile score run --rm --no-deps score
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod -Method Post -Uri http://localhost:8000/invocations `
  -ContentType 'application/json' -InFile sample_request.json | ConvertTo-Json -Depth 5
```

### Option C — sequential `docker build` / `docker run`

```bash
mkdir -p mlruns artifacts

# 1) train, evaluate, register @champion  (writes ./mlruns and ./artifacts)
docker build -f Dockerfile.train -t credit-default-train .
docker run --rm -u "$(id -u):$(id -g)" \
  -v "$PWD/mlruns:/app/mlruns" -v "$PWD/artifacts:/app/artifacts" \
  credit-default-train

# 2) serve the registered champion on :8000
docker build -f Dockerfile.serve -t credit-default-serve .
docker run -d --name credit-default-serve -u "$(id -u):$(id -g)" \
  -p 8000:8000 -v "$PWD/mlruns:/app/mlruns" \
  credit-default-serve

# the model takes a few seconds to load: wait for the container healthcheck
until [ "$(docker inspect -f '{{.State.Health.Status}}' credit-default-serve)" = healthy ]; do sleep 2; done

# optional: batch-score the test data through the API
docker run --rm --network host -u "$(id -u):$(id -g)" -v "$PWD/artifacts:/app/artifacts" \
  credit-default-train python -m src.inference.batch_client --url http://localhost:8000
```

Re-running training registers a new version and moves `@champion` to it. Restart `serve`
to load it.

### Building & Publishing Multi-Platform Images

The `ghcr.io` images for Option B are built with `docker buildx` for both platforms. The
Dockerfiles must not pin `FROM --platform=...`; otherwise every variant contains amd64 binaries.

```bash
# one-time: a builder that can produce multi-platform images
docker buildx create --name multiarch --driver docker-container --use

# standalone Docker Engine on Linux/WSL2 only (Docker Desktop ships QEMU already):
docker run --privileged --rm tonistiigi/binfmt --install arm64

# log in with a GitHub personal access token that has write:packages
echo "$GHCR_TOKEN" | docker login ghcr.io -u <github-user> --password-stdin

docker buildx build --platform linux/amd64,linux/arm64 -f Dockerfile.train \
  -t ghcr.io/gnaneeswar/credit-default-train:latest --push .
docker buildx build --platform linux/amd64,linux/arm64 -f Dockerfile.serve \
  -t ghcr.io/gnaneeswar/credit-default-serve:latest --push .
```

## 7. Verify the API

```bash
curl -s localhost:8000/health
# {"status":"ok","model_loaded":true,"model_name":"credit-default-pipeline","model_alias":"champion",
#  "model_version":"1","model_family":"boosted trees","threshold":0.5197054344940619,
#  "n_features":23,"error":null}

curl -s -X POST localhost:8000/invocations \
     -H 'Content-Type: application/json' -d @sample_request.json
# {"predictions":[{"default_probability":0.95,"prediction":1},
#                 {"default_probability":0.36,"prediction":0},
#                 {"default_probability":0.07,"prediction":0}], "threshold":0.5197..., ...}

# Out-of-contract values are rejected (422), never silently scored:
#   LIMIT_BAL=-5000 -> {"detail":{"error":"values outside the allowed ranges","n_violations":1,
#     "violations":[{"row":0,"column":"LIMIT_BAL","value":-5000.0,"rule":"credit limit: 1 .. 1e9"}]}}
```

[`sample_request.json`](sample_request.json) holds three real test data clients: a delinquent
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
| `artifacts/test_decile_analysis.csv` | test data lift table on the **training** score bands, with band shares; `test_metrics.json` adds `score_psi_vs_train_bands` |
| `artifacts/test_metrics.json`, `test_confusion_matrix.{json,png}` | test data evaluation of `@champion` |
| `artifacts/monitoring_report*.html`, `monitoring_summary.json` | Evidently reports and alert summary |
| `artifacts/credit_default_summary.pptx` | **review deck** (16:9, cover + 7 slides: EDA observations, data preprocessing and missing-value treatment, feature catalogue, feature selection and champion importance, candidate models and CV grids, top 3 models, Train OOF vs Test data metrics with both decile tables), rebuilt from the artifacts on every run; `uv run python -m src.reporting.ppt` rebuilds it alone |
| `artifacts/feature_importance.csv`, `feature_group_importance.csv` | champion importance: per feature from the model (total gain for LightGBM / XGBoost, impurity MDI for random forest, \|coef\| for LR, normalised), and per feature group as test data permutation ΔROC-AUC, either **total** (raw source through the pipeline) or **conditional** (engineered group) |
| `artifacts/model_grids.json`, `train_oof_metrics.json` | deck inputs: model grids, imbalance handling and CV setup (generated from the model specs), and OOF metrics at the threshold |
| `artifacts/batch_predictions.csv` | batch client output: actual vs predicted |
| `mlruns/` | MLflow tracking database, model registry and artifacts |

To browse the runs (optional, read-only UI): `uvx mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.

## 9. Local development

```bash
uv sync                                   # Python 3.12 venv from uv.lock
uv run pytest -q                          # 146 offline tests
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
- the FastAPI contract (`TestClient`), with both a fake model and a **real registered
  random forest**:
  - the served model has `n_jobs == 1`;
  - the load is version-pinned;
  - a `PAY_0` payload gets 422;
  - out-of-range or extreme inputs get 422, never 500;
  - batch and single-row scores match
- per-family serialisation within the skops allowlist, and tamper refusal
- importance units per family (gain vs MDI) and total vs conditional group importance
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
docker rmi ghcr.io/gnaneeswar/credit-default-train:latest \
           ghcr.io/gnaneeswar/credit-default-serve:latest   # pre-built image path
git clean -fdX mlruns artifacts      # cloned repo (Options A / C): delete generated outputs, keep .gitkeep

# pre-built images (Option B): run `docker compose down` inside the folder first, then from its parent
rm -rf credit-default-ml                                # Linux / macOS
# Remove-Item -Recurse -Force credit-default-ml         # Windows PowerShell
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
* **Leaner, pinned serve image.** Split the dependencies into `train`/`serve` groups so the
  serve image doesn't carry Evidently, matplotlib or python-pptx, and pin the base image by
  digest.
* **Model artifact integrity.** Sign or hash the registered model and verify it at load
  time, on top of the skops allowlist.
* **Per-feature permutation importance.** Measured at 7–9 s for 68 inputs, so it is left out
  of the 50–75 s pipeline. Run it offline if per-feature attribution on held-out data is
  needed.
* **Duplicates across the split.** 35 duplicate feature rows (distinct customer IDs) may fall
  on both sides of the 80/20 split. That is too few to matter, but a grouped split would rule
  it out.
* **`PAY_1` timing.** See §4.1. The model keeps `PAY_1`, consistent with the original paper.
  In production it should be used only if September's status is known before the October
  decision.
