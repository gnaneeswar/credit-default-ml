# PROGRESS

Plan: leak-free credit-default pipeline (UCI 350) → MLflow registry `@champion` → FastAPI → Evidently → Docker.

## Phases
- [x] P0 Bootstrap — uv/py3.12, LF + gitignore hygiene, CLAUDE.md, /checkpoint skill, `src/common` (config, logging, tables)
- [ ] P1 Data — ucimlrepo ingest + cache, data dictionary (json/md), canonical names, stratified split, EDA artifacts
- [ ] P2 Features — DomainCleaner, DomainFeatureEngineer, WoEIVSelector + leakage tests
- [ ] P3 Training — 4-model GridSearchCV (3-fold), OOF decile threshold, registry `@champion`, holdout eval (<45s)
- [ ] P4 Serving — FastAPI `/health` + `/invocations` (dataframe_split), batch client (drops target col)
- [ ] P5 Monitoring — Evidently drift + classification scenarios, report html + summary json
- [ ] P6 Docker + README + push — Dockerfile.train/serve, compose, sample_request.json, Design Decisions

## Next steps
- P1: `src/data/schema.py` (X1..X23→canonical), `ingest.py` (ucimlrepo id=350 → cache CSV + data_dictionary.json/md), `split.py` (stratified 80/20 rs=42), `eda.py` (eda_summary.json, eda_report.md incl. undocumented-code tables), tests.

## Open issues / notes
- Locked: mlflow 3.18, evidently 0.7.23 (new Report/Dataset API), sklearn 1.9.1, pandas 3.1, lightgbm 4.7, xgboost 3.4. uv 0.13 lives in `~/.local/bin`.
- Branch is `main`; pushes authenticate via Windows Git Credential Manager (`ucrt64/bin/git-credential-manager.exe`).
- Docker Engine install (P6) requires the user to run the sudo block.
