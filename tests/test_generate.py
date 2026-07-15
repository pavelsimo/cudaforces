from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cudaforces import generate, problems, settings
from cudaforces.problems.types import RefCase


def test_write_case_preserves_exact_int32_output(tmp_path: Path) -> None:
    case = RefCase({"N": 1, "inp": np.array([1], dtype=np.int32)})
    in_path = tmp_path / "case.in"
    out_path = tmp_path / "case.out"

    def solve(inputs: dict[str, Any]) -> list[np.ndarray[Any, Any]]:
        return [np.array([2_147_483_647], dtype=np.int32)]

    generate._write_case(case, solve, in_path, out_path)

    assert out_path.read_text() == "2147483647\n"


def test_ensure_test_data_removes_stale_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cases = [
        RefCase({"N": 1, "inp": np.array([1.0], dtype=np.float32)}),
        RefCase({"N": 1, "inp": np.array([2.0], dtype=np.float32)}),
    ]

    def solve(inputs: dict[str, Any]) -> list[np.ndarray[Any, Any]]:
        return [np.asarray(inputs["inp"])]

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(problems, "ref_module", lambda slug: SimpleNamespace(tests=lambda: cases, solve=solve))
    directory = generate.tests_dir("demo")
    directory.mkdir(parents=True)
    (directory / "03.in").write_text("stale")
    (directory / "03.out").write_text("stale")

    generate.ensure_test_data("demo")

    assert sorted(path.name for path in directory.iterdir()) == ["01.in", "01.out", "02.in", "02.out"]
