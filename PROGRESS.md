# PROGRESS

Plan: leak-free credit-default pipeline (UCI 350) → MLflow registry `@champion` → FastAPI → Evidently → Docker.

## Phases
- [x] P0 Bootstrap — uv/py3.12, LF + gitignore hygiene, CLAUDE.md, /checkpoint skill, `src/common` (config, logging, tables)
- [x] P1 Data — ucimlrepo ingest + cache, data dictionary (json/md), canonical names, stratified split, EDA artifacts
- [ ] P2 Features — DomainCleaner, DomainFeatureEngineer, WoEIVSelector + leakage tests
- [ ] P3 Training — 4-model GridSearchCV (3-fold), OOF decile threshold, registry `@champion`, holdout eval (<45s)
- [ ] P4 Serving — FastAPI `/health` + `/invocations` (dataframe_split), batch client (drops target col)
- [ ] P5 Monitoring — Evidently drift + classification scenarios, report html + summary json
- [ ] P6 Docker + README + push — Dockerfile.train/serve, compose, sample_request.json, Design Decisions

## Next steps
- P2: `src/features/domain.py` (DomainCleaner remap EDUCATION{0,5,6}->4, MARRIAGE{0}->3; DomainFeatureEngineer: util, pay coverage, trends, delinquency counts/max) and `src/features/woe.py` (WoEIVSelector, IV>=0.02) + leakage/clone/pickle tests.

## Open issues / notes
- EDA evidence (for README Design Decisions): EDUCATION 0/5/6 default 0-16% ~ code 4 (5.7%) -> remap justified by data; MARRIAGE 0 (n=54, 9.3%) vs 3 (26%) -> merged on semantics only, say so honestly; PAY 0 (12.8%) and -2 (13.2%) default LESS than -1 (16.8%) -> preserve codes. 35 duplicate feature rows w/ distinct IDs kept. 1930 rows have negative bills; 866 zero-bill all 6 months.
- PAY_n definitions (user-supplied): -2 no consumption, -1 paid in full, 0 revolving. Data check: 0/-1 consistent, but -2 has non-zero same-month bill in ~75% of PAY_0 rows -> P2 must keep `N_INACTIVE` (PAY==-2 count) and bill<=0 flags as SEPARATE features; never derive one from the other.
- `python -m src.data.ingest --rebuild-dictionary` regenerates the dictionary offline from stored UCI metadata.
- UCI text fields contain CRLF and NaN: ingest normalises to LF + strict JSON (tested).
- Locked: mlflow 3.18, evidently 0.7.23 (new Report/Dataset API), sklearn 1.9.1, pandas 3.1, lightgbm 4.7, xgboost 3.4. uv 0.13 lives in `~/.local/bin`.
- Branch is `main`; pushes authenticate via Windows Git Credential Manager (`ucrt64/bin/git-credential-manager.exe`).
- Docker Engine install (P6) requires the user to run the sudo block.
