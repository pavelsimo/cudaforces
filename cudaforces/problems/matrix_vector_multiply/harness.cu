// Judge harness for matrix-vector-multiply.
// stdin: R C, then matrix[R*C] row-major, then x[C]. stdout: out[R].
#include "judge_io.h"

extern "C" void solve(float* out, const float* matrix, const float* x, int R, int C);

int main() {
    int R, C;
    if (scanf("%d %d", &R, &C) != 2) input_error();
    float* matrix = read_floats_device((size_t)R * C);
    float* x = read_floats_device((size_t)C);
    float* out = device_zeros((size_t)R);
    solve(out, matrix, x, R, C);
    finish_solve();
    print_device_floats(out, (size_t)R);
    return 0;
}
