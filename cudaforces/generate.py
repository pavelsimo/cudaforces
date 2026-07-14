"""Generate judge test data (.in/.out files) from the NumPy references.

Run with: uv run python -m cudaforces.generate [--force] [slug ...]
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import problems, settings
from .problems.types import RefCase


def tests_dir(slug: str) -> Path:
    return settings.DATA_DIR / "tests" / slug


def _fmt(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.9g}"


def _write_case(case: RefCase, solve: Any, in_path: Path, out_path: Path) -> None:
    scalars = [v for v in case.inputs.values() if not isinstance(v, np.ndarray)]
    arrays = [v for v in case.inputs.values() if isinstance(v, np.ndarray)]
    with in_path.open("w") as f:
        f.write(" ".join(_fmt(s) for s in scalars) + "\n")
        for arr in arrays:
            f.write("\n".join(_fmt(x) for x in arr.ravel()) + "\n")
    outputs = solve(case.inputs)
    with out_path.open("w") as f:
        for arr in outputs:
            f.write("\n".join(f"{float(x):.9g}" for x in np.asarray(arr).ravel()) + "\n")


def ensure_test_data(slug: str, force: bool = False) -> Path:
    """Materialize data/tests/<slug>/NN.{in,out}; skips work if files exist."""
    ref = problems.ref_module(slug)
    cases: list[RefCase] = ref.tests()
    directory = tests_dir(slug)
    paths = [(directory / f"{i:02d}.in", directory / f"{i:02d}.out") for i in range(1, len(cases) + 1)]
    if not force and all(i.is_file() and o.is_file() for i, o in paths):
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    for case, (in_path, out_path) in zip(cases, paths, strict=True):
        _write_case(case, ref.solve, in_path, out_path)
    return directory


def ensure_all_test_data(force: bool = False) -> None:
    for pd in problems.all_problems():
        ensure_test_data(pd.slug, force=force)
        print(f"tests ready: {pd.slug}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    force = "--force" in argv
    slugs = [a for a in argv if not a.startswith("--")]
    if slugs:
        for s in slugs:
            ensure_test_data(s, force=force)
            print(f"tests ready: {s}")
    else:
        ensure_all_test_data(force=force)
