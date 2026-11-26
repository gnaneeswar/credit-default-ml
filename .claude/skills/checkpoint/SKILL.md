---
name: checkpoint
description: Verify lint/tests/LF line endings, update PROGRESS.md, make one atomic commit, and push to origin. Use at the end of each phase or when the user says /checkpoint.
---

# /checkpoint

Run every step in order; **stop and report** on the first failure (do not commit broken work).

1. **Quality gates**
   ```bash
   uv run ruff check . && uv run ruff format --check . && uv run pytest -q
   ```
2. **LF line endings** — both must print nothing:
   ```bash
   git add -A --dry-run >/dev/null   # surface unreadable files early
   git ls-files --eol --cached --others --exclude-standard | grep -E 'i/crlf|w/crlf'
   git ls-files -z --cached --others --exclude-standard | xargs -0 grep -lIU $'\r' 2>/dev/null
   ```
3. **Hygiene** — `git status --porcelain` must not list `.venv/`, `mlruns/`, `*.db*`, `artifacts/*` (except `.gitkeep`), `.env*`, or caches. If it does, fix `.gitignore` first.
4. **Update `PROGRESS.md`** — tick the finished phase, rewrite "Next steps", note open issues/decisions. Keep it short.
5. **Atomic commit** — stage only this phase's files (`git add <paths>`, never blind `-A` without reviewing `git status`). Conventional message, e.g. `feat(p1): ingest UCI dataset, data dictionary, EDA`, body = 2-4 bullet summary, ending with the attribution trailer required by the session.
6. **Push** — `git push -u origin HEAD` (pushes the current branch; on a feature branch this does NOT touch `main`). If auth fails, report it to the user; do not change remotes or credentials without asking.
7. Report: phase, commit hash, test count, push result.
