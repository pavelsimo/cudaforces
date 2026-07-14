# CudaForces — Agent Instructions

## Overview

Codeforces-style CUDA kernel problemset with a local nvcc judge

Python 3.13 · Reflex + FastAPI · Passwordless auth · SQLite (WAL)

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.13 |
| Frontend | Reflex — pure-Python components, no hand-written JavaScript |
| API | FastAPI app in `cudaforces/api.py`, mounted via `rx.App(api_transformer=api)` |
| Database | SQLite with WAL mode — SQLModel models, Alembic migrations |
| Auth | Passwordless magic links (no auth library) — ~100 lines in `cudaforces/auth.py` |
| Tooling | uv (packages) · ruff (format + lint) · mypy strict (types) · pytest (tests) |
| Deploy | Kamal — single container, Caddy static frontend + Reflex backend |

## Commands

| Command | Purpose |
|---------|---------|
| `make setup` | Install dependencies, migrate + seed database, install git hooks |
| `make dev` | Start development server (localhost:3000) |
| `make test` | Run all tests with coverage |
| `make ci` | Full CI suite: format check + lint + tests — must pass before commit |
| `make lint` | ruff check + mypy |
| `make fmt` | Auto-format and fix lint offenses |
| `make db-makemigrations` | Generate Alembic migration from model changes |
| `make db-migrate` | Apply pending migrations |
| `make db-reset` | Recreate database and seed |
| `make deploy` | Deploy to production via Kamal |

## The Seven Principles (37signals, adapted)

1. **Rich domain functions** over service classes
2. **Plain functions** for logic; Reflex state only orchestrates
3. **Records as state** over boolean columns
4. **Database-backed everything** (SQLite — no Redis, no external services)
5. **Build it yourself** before adopting packages
6. **Native Reflex components** — no custom React unless unavoidable
7. **Ship to learn**; validate; iterate

## Architecture Rules

### Domain logic

- Business logic lives in plain functions under `cudaforces/` (see `auth.py`) — NOT in Reflex state, NOT in service classes
- Functions take a `sqlmodel.Session` as their first argument so pytest covers them without a Reflex runtime
- State is a separate record, not a boolean column:
  - "closed" → a `Closure` row (not `closed_at` / `is_closed`)
  - "published" → a `Publication` row (not `published: bool`)
- **Never soft-delete** — permanently delete records; add an event log if you need an audit trail

### Models & migrations

- Models subclass `sqlmodel.SQLModel` with `table=True` in `cudaforces/models.py`, with an explicit integer `id` primary key
- Always index foreign keys and filter/sort columns: `sqlmodel.Field(foreign_key="user.id", index=True)`
- Datetimes are naive UTC — use `models.utcnow()`, never `datetime.now()`
- Migrations only via `make db-makemigrations m="describe change"` + `make db-migrate` (Alembic) — never edit the schema by hand
- SQLite PRAGMAs (WAL, busy_timeout, foreign_keys) are set in `cudaforces/db.py` — importing the module registers them

### Reflex state & pages

- One state class per concern; `AuthState` is the base example
- Event handlers stay thin: open `db.session()`, call domain functions, set vars, redirect
- Protected pages guard with `on_load=AuthState.check_auth`
- Pages are functions returning `rx.Component` in `cudaforces/pages/` — one page per file
- Use native Reflex components (`rx.vstack`, `rx.form`, `rx.table`, ...) — no custom JavaScript or React wrapping unless there is no alternative

### API endpoints

- JSON endpoints for external clients go in `cudaforces/api.py` (FastAPI) — they share the backend port and are proxied under the same origin in production
- Authenticate API requests by reading the `session_token` cookie and calling `auth.find_session_by_token`

### Authentication

- Built-in magic link flow — do NOT add an auth library
- `Identity` = global email address; `User` = app-specific profile
- 6-digit codes, 15-minute expiry, 10 codes per 15 minutes rate limit, single-use
- `cudaforces/auth.py` is the single source of truth
- The session token cookie is a random DB-backed credential — revocable server-side (`terminate_session`); note it is readable by frontend JS (`rx.Cookie` is not httponly)

### Testing

- **pytest + plain fixtures** — no factory libraries, no unittest classes
- Domain tests use the in-memory `db` fixture from `tests/conftest.py`
- API tests use FastAPI's `TestClient`
- Tests ship in the same commit as the feature they cover
- Coverage gate is 70% over domain modules (UI modules are excluded — see `pyproject.toml`)

## Dependencies to AVOID

| Avoid | Use instead |
|-------|------------|
| Auth libraries (authlib for login, fastapi-users) | `cudaforces/auth.py` (already in this repo) |
| Service classes / repositories | Plain functions taking a `sqlmodel.Session` |
| pydantic-settings | `os.getenv` in `cudaforces/settings.py` |
| Celery / RQ / Redis | Synchronous work, or a simple background task — this is a single-server app |
| PostgreSQL / MySQL | SQLite with WAL (already tuned in `db.py`) |
| Ad-hoc engines / sessions | The single tuned engine in `db.py`; `db.session()` in state, injected `sqlmodel.Session` in domain functions |
| Custom React components | Native Reflex components |
| poetry / pip / conda | uv (`uv add`, `uv sync`) |
| black / isort / flake8 | ruff (`make fmt`) |

## File Structure

```
cudaforces/
├── cudaforces.py    # App entry — rx.App(api_transformer=api)
├── api.py               # FastAPI JSON endpoints (health check + external API)
├── auth.py              # Magic link auth — the single auth source of truth
├── db.py                # Engine + sessions + SQLite PRAGMA tuning (WAL, busy_timeout)
├── mailer.py            # Outbound email (logs codes in development)
├── models.py            # SQLModel models — one file until it hurts
├── seed.py              # Development seed data
├── settings.py          # Environment-driven configuration
├── state.py             # Reflex state classes (thin orchestration)
├── theme.py             # Design tokens (mockup palette, fonts, verdict colors)
├── judge.py             # nvcc judge: compile, run tests, compare, submit (plain functions)
├── generate.py          # Writes data/tests/<slug>/NN.{in,out} from the NumPy references
├── components/          # header, badges, editor (the Monaco seam)
├── problems/            # content.json + registry + per-problem judge assets
│   └── <slug>/          # harness.cu (stdin -> solve() -> stdout) + ref.py (NumPy reference)
└── pages/               # One page per file (index, problem, submissions, sign_in, verify)
tests/                   # pytest — domain functions + API (GPU judge tests auto-skip)
alembic/                 # Database migrations (make db-makemigrations)
config/deploy.yml        # Kamal deployment config
Caddyfile                # Production: static frontend + backend proxy
rxconfig.py              # Reflex config (app name, db_url, api_url)
```

## Judge Notes

- The judge (`cudaforces/judge.py`) shells out to `nvcc` — settings: `NVCC_PATH`, `NVCC_ARCH` (default `native`), `DATA_DIR`. Never add `--use_fast_math`; it diverges from the NumPy references.
- Test data on disk is a build artifact: regenerate with `make gen-tests`, never hand-edit.
- Judge workspaces live under `data/submissions/`; `make dev` excludes `data/` from hot reload — writing there from a running server must not trigger a restart.
- Float tolerances are per-problem module constants (`RTOL`/`ATOL`/`TIME_LIMIT_MS`) in each `ref.py`. Loosen per-problem, never globally.
- Reflex 0.9 gotchas already encoded here: explicit `set_x` event setters (auto-setters are off), dataclasses for structured vars (`rx.Base` is gone), dynamic route args must not be shadowed by state vars, ORM instances must not escape `db.session()` into state.

## Commit Convention

Gitmoji + lowercase imperative:

| Prefix | When to use |
|--------|-------------|
| `🎉 init:` | First commit only |
| `✨ add:` | New feature or capability |
| `🐛 fix:` | Bug fix |
| `♻️ refactor:` | Restructure without behavior change |
| `🧪 test:` | Add or update tests |
| `📝 docs:` | Documentation only |
| `⚡ perf:` | Performance improvement |
| `🔒 security:` | Security fix |
| `⬆️ upgrade:` | Dependency upgrade |
| `🗑️ remove:` | Delete code or files |
