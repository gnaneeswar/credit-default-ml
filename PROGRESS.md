# PROGRESS

Plan: leak-free credit-default pipeline (UCI 350) → MLflow registry `@champion` → FastAPI → Evidently → Docker.

## Phases
- [x] P0 Bootstrap — uv/py3.12, LF + gitignore hygiene, CLAUDE.md, /checkpoint skill, `src/common` (config, logging, tables)
- [x] P1 Data — ucimlrepo ingest + cache, data dictionary (json/md), canonical names, stratified split, EDA artifacts
- [x] P2 Features — DomainCleaner, DomainFeatureEngineer, WoEIVSelector + leakage tests
- [x] P3 Training — 4-model GridSearchCV (3-fold), OOF decile threshold, registry `@champion`, holdout eval (<45s)
- [x] P4 Serving — FastAPI `/health` + `/invocations` (dataframe_split), batch client (drops target col)
- [x] P5 Monitoring — Evidently drift + classification scenarios, report html + summary json
- [x] P6 Docker + README + push — Dockerfile.train/serve, compose, sample_request.json, Design Decisions

## Next steps
- All phases complete. Optional follow-ups (README §11): cost-based threshold, calibration, SEX fairness audit/drop, champion/challenger promotion gate, train/serve dependency groups for a leaner serve image.

## Open issues / notes
- P6 verified on Docker 29.10.1 / Compose v5.8: clean `docker compose up --build` (train exit 0 → serve healthy; curl + `--profile score` batch F1 0.5379 identical to host) and README Option B verbatim. Pipeline 41-45s in container, ~50-52s container wall (≈5s fixed start/exit). Images 2.47 GB (shared layers), train peak 2.93 GiB RAM (12 workers), serve ~290 MB. Serve image has no data/ or src/training. Outputs owned by host UID 1000.
- Docker access in this session came from `setfacl -m u:$USER:rw /var/run/docker.sock` (resets on docker restart; a fresh login picks up the docker group permanently).
- `mlruns/.gitkeep` committed (like artifacts/) so bind-mount dirs exist user-owned on clone; `git clean -fdX mlruns artifacts` resets outputs.
- P5 results: holdout drift 0.00, F1 0.538 (Δ-0.008) no alerts; covariate_drift share 0.57 (13 cols: LIMIT_BAL, PAY_*, BILL_AMT*; PAY_AMT* untouched) F1 0.399 → drift+perf alerts; concept_drift (30% labels flipped) drift 0.00, F1 0.404 → perf alert only (argument for label monitoring). Alert policy: drift share >= 0.25, F1/accuracy drop > 0.05 vs OOF reference.
- pandas pinned <3: evidently 0.7.23 imports pandas internals removed in pandas 3 (its own dev stubs pin pandas<3).
- Runtime work (full 5-stage run now 38-41s pipeline / 43-45s wall at loadavg 3-5): MLflow init in background thread overlapping searches; GridSearchCV refit=False + one parallel batch (winner refit w/ spare cores then n_jobs reset to 1, 3 OOF folds, LR refit for IV); monitoring = 3 Evidently reports in a spawn ProcessPool pre-warmed (import evidently ~4.7s) right after searches. Threads don't help Evidently (GIL-bound, measured). Results bit-identical.
- P4 live check: /health ok; sample_request.json (delinquent default p=0.91 flagged, revolving + inactive approved); batch client 6000 rows in one POST 0.13s, F1 0.5379 == holdout eval (serving path == evaluated path). Server scores 6000 rows in ~64 ms.
- API contract: exact 23 signature columns any order (reordered + cast float64); strict finite numbers (no str/bool/null/NaN); 422 lists missing/unknown cols; custom 422 handler never echoes input (default echoed full payload; NaN in it -> 500). /health 503 + error text if model load fails (process stays up).
- `Settings.tracking_uri` honours MLFLOW_TRACKING_URI (default relative). Tests set a per-test absolute URI because MLflow caches stores by URI string.
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
