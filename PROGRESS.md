# PROGRESS

Plan: leak-free credit-default pipeline (UCI 350) → MLflow registry `@champion` → FastAPI → Evidently → Docker.

## Phases
- [x] P0 Bootstrap — uv/py3.12, LF + gitignore hygiene, CLAUDE.md, /checkpoint skill, `src/common` (config, logging, tables)
- [x] P1 Data — ucimlrepo ingest + cache, data dictionary (json/md), canonical names, stratified split, EDA artifacts
- [x] P2 Features — DomainCleaner, DomainFeatureEngineer, WoEIVSelector + leakage tests
- [x] P3 Training — 4-model GridSearchCV (3-fold), OOF decile threshold, registry `@champion`, holdout eval (<45s)
- [ ] P4 Serving — FastAPI `/health` + `/invocations` (dataframe_split), batch client (drops target col)
- [ ] P5 Monitoring — Evidently drift + classification scenarios, report html + summary json
- [ ] P6 Docker + README + push — Dockerfile.train/serve, compose, sample_request.json, Design Decisions

## Next steps
- P4: `src/inference/schemas.py` (pydantic dataframe_split), `app.py` (lifespan load via `src.common.mlflow_utils.load_champion`; reorder+cast to signature `input_columns`, float64; 422 on missing/unknown cols; /health 503 w/o model), `batch_client.py` (drop target col; one POST; artifacts/batch_predictions.csv with actuals), `sample_request.json`, tests (TestClient + fake model, AST import-boundary test).

## Open issues / notes
- P3 results (full run, 33s wall / 30s pipeline): CV F1 RF 0.5460±0.0065 > LGBM 0.5438 > XGB 0.5420 > LR-WoE 0.5346 (ROC-AUC 0.778-0.785). Champion RF; cutoff 0.4752 (top 27% via coarse decile k=3 -> percentile refine). Holdout F1 0.538 @cutoff vs 0.541 @0.5 (prior run reversed: 0.542 vs 0.539). OOF F1 identical (0.5459) -> flat F1 plateau ~0.47-0.52; diffs << CV std / holdout SE (~0.015). README: say honestly; do NOT retune on test; production cutoff should come from cost matrix.
- Serialization: MLflow 3.18 sklearn flavor = skops (not pickle). Trusted types derived per model + allowlist (src.features., sklearn., xgboost., lightgbm., numpy.), stored in MLmodel. WoEIVSelector stores IV table as records (skops can't persist DataFrames).
- Models signed/trained on float64 inputs (MLflow integer-schema caveat; clients may send 20000.0/null).
- Thread pins + MLflow quieting env in src/__init__.py (runs before any submodule; isort-proof).
- Local WSL needed `sudo apt-get install libgomp1` for LightGBM (Dockerfile must apt-install it too).
- `uv run python -m src.training.run --models a,b` for quick subsets (LR always included for IV table).
- DECISION (user): keep SEX as an input to all tree models; LR drops it via IV<0.02. README must document the fairness trade-off (protected attribute used by trees; IV 0.011 shows little signal; mitigation options: drop it, fairness audit by group).
- EDA evidence (for README Design Decisions): EDUCATION 0/5/6 default 0-16% ~ code 4 (5.7%) -> remap justified by data; MARRIAGE 0 (n=54, 9.3%) vs 3 (26%) -> merged on semantics only, say so honestly; PAY 0 (12.8%) and -2 (13.2%) default LESS than -1 (16.8%) -> preserve codes. 35 duplicate feature rows w/ distinct IDs kept. 1930 rows have negative bills; 866 zero-bill all 6 months.
- PAY_n definitions (user-supplied): -2 no consumption, -1 paid in full, 0 revolving. Data check: 0/-1 consistent, but -2 has non-zero same-month bill in ~75% of PAY_0 rows -> P2 must keep `N_INACTIVE` (PAY==-2 count) and bill<=0 flags as SEPARATE features; never derive one from the other.
- P2 IV evidence (train split, final README): PAY_0 0.88 / N_DELINQ 0.88 / MAX_DELINQ 0.79 top; engineered UTIL_* 0.08-0.13 vs raw BILL_AMT* 0.013-0.023 (raw bills mostly dropped); PAY_COVER_1 0.24 > PAY_AMT1 0.18. Dropped (<0.02): BILL_AMT1-5, SEX, MARRIAGE → LR path excludes protected SEX; trees still see it (mention fairness trade-off in README). WoE PAY_0: -2 +0.67, 0 +0.66, -1 +0.35 (higher=safer). 38/45 features kept. WoE fit 0.09s. Bins: <=12 uniques → per-value; else 10 quantiles; merge bins <1% rows; Laplace alpha 0.5.
- `python -m src.data.ingest --rebuild-dictionary` regenerates the dictionary offline from stored UCI metadata.
- UCI text fields contain CRLF and NaN: ingest normalises to LF + strict JSON (tested).
- Locked: mlflow 3.18, evidently 0.7.23 (new Report/Dataset API), sklearn 1.9.1, pandas 3.1, lightgbm 4.7, xgboost 3.4. uv 0.13 lives in `~/.local/bin`.
- Branch is `main`; pushes authenticate via Windows Git Credential Manager (`ucrt64/bin/git-credential-manager.exe`).
- Docker Engine install (P6) requires the user to run the sudo block.
