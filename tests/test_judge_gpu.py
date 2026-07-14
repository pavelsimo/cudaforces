"""End-to-end judge tests against the real nvcc + GPU. Auto-skips when unavailable."""

import shutil
import subprocess
from pathlib import Path

import pytest

from cudaforces import judge, settings

CORRECT = """
#include <cuda_runtime.h>

__global__ void kernel(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

extern "C" void solve(float* out, const float* inp1, const float* inp2, int N) {
    kernel<<<(N + 255) / 256, 256>>>(out, inp1, inp2, N);
    cudaDeviceSynchronize();
}
"""


def _gpu_available() -> bool:
    if not shutil.which(settings.NVCC_PATH) and not Path(settings.NVCC_PATH).is_file():
        return False
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, timeout=10).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


pytestmark = pytest.mark.skipif(not _gpu_available(), reason="nvcc or GPU unavailable")


def test_correct_solution_is_accepted(tmp_path: Path) -> None:
    result = judge.judge("residual-forward", CORRECT, workdir=tmp_path)
    assert result.verdict == "AC", result.compile_output
    assert all(t.status == "pass" for t in result.tests)


def test_wrong_kernel_is_wa(tmp_path: Path) -> None:
    result = judge.judge("residual-forward", CORRECT.replace("a[i] + b[i]", "a[i] - b[i]"), workdir=tmp_path)
    assert result.verdict == "WA"
    assert result.tests[-1].detail


def test_syntax_error_is_ce(tmp_path: Path) -> None:
    result = judge.judge("residual-forward", CORRECT.replace("cudaDeviceSynchronize();", "oops"), workdir=tmp_path)
    assert result.verdict == "CE"
    assert "error" in result.compile_output.lower()
