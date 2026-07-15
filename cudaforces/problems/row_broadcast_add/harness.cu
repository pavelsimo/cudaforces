// Judge harness for row-broadcast-add.
// stdin: R C, then matrix[R*C], row[C]. stdout: out[R*C].
#include "judge_io.h"

extern "C" void solve(float* out, const float* matrix, const float* row, int R, int C);

int main() {
    int R, C;
    if (scanf("%d %d", &R, &C) != 2) input_error();
    size_t n = (size_t)R * C;
    float* matrix = read_floats_device(n);
    float* row = read_floats_device(C);
    float* out = device_zeros(n);
    solve(out, matrix, row, R, C);
    finish_solve();
    print_device_floats(out, n);
    return 0;
}
