# CudaForces

Codeforces-style CUDA kernel problemset with a local nvcc judge.

Fifty CUDA kernel exercises organized as a curriculum with difficulty ratings. The first 30 cover essential maps, reductions, scans, atomics, sorting, tiling, sparse workloads, graphs, and simulation; the final 20 apply those skills to the full GPT-2 training loop. Write the kernel in the browser (Monaco editor), hit Submit, and the judge compiles it with your local `nvcc`, runs it on your GPU against generated test data, and marks the problem solved on Accepted.

## Requirements

- **CUDA toolkit** (`nvcc` on PATH, or set `NVCC_PATH`) and an NVIDIA GPU; the judge compiles and runs submissions locally
- Python managed by `uv`

## Setup

```bash
git clone https://github.com/pavelsimo/cudaforces
cd cudaforces
make setup       # deps, migrations, seeds the 50 problems
make gen-tests   # generate judge test data from the NumPy references
make dev
```

Visit [http://localhost:3000](http://localhost:3000). Use "Continue as guest", or sign in with any email: the magic-link code is printed to the server console in development.

## The judge

Each problem ships a `harness.cu` (reads test input, calls your `solve()`, prints outputs) and a NumPy reference (`ref.py`) that generates the test files under `data/tests/<slug>/`. On submit:

1. `nvcc -O2 -arch=native solution.cu harness.cu` (compile error → **CE**)
2. Run each test with the input on stdin (crash → **RE**, timeout → **TLE**)
3. Compare outputs elementwise within per-problem float tolerances (mismatch → **WA**)
4. All tests pass → **Accepted**, and a `Solve` record marks the problem done

Judge a file headless without the UI:

```bash
uv run python -m cudaforces.judge residual-forward my_solution.cu
```

## Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.13 |
| Frontend | Reflex (pure-Python UI, compiles to React) + Monaco editor |
| Judge | local `nvcc` + subprocess, NumPy reference implementations |
| API | FastAPI mounted into the Reflex backend (`api_transformer`) |
| Database | SQLite with WAL mode (SQLModel + Alembic) |
| Auth | Passwordless magic links |
| Tooling | uv · ruff · mypy · pytest · lefthook |
| Deploy | Kamal (single container: Caddy + Reflex backend) |

## Commands

| Command | Description |
|---------|-------------|
| `make dev` | Start development server (excludes `data/` from hot reload) |
| `make test` | Run tests with coverage (GPU judge tests auto-skip without nvcc) |
| `make ci` | Full CI suite (format check + lint + tests) |
| `make lint` | Run ruff + mypy |
| `make fmt` | Auto-format and fix lint offenses |
| `make gen-tests` | Generate judge test data (`data/tests/`) from NumPy references |
| `make db-makemigrations` | Generate a migration from model changes |
| `make db-migrate` | Apply pending migrations |
| `make db-reset` | Recreate the database and seed |
| `make deploy` | Deploy to production |

## Adding a problem

1. Add the statement/starter entry to `cudaforces/problems/content.json`
2. Create `cudaforces/problems/<slug_with_underscores>/` with `__init__.py`, `ref.py` (NumPy `tests()` + `solve()`), and `harness.cu`
3. `make seed && make gen-tests` (the registry upserts by slug)

## Documentation

- [Development guide](docs/development.md): local setup, workflow, architecture overview
- [Deployment guide](docs/deployment.md): Kamal setup and environment variables

## License

MIT © 2026 pavelsimo
