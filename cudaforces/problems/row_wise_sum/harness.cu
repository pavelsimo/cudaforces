// Judge harness for row-wise-sum.
// stdin: R C, then inp[R*C]. stdout: out[R].
#include "judge_io.h"

extern "C" void solve(float* out, const float* inp, int R, int C);

int main() {
    int R, C;
    if (scanf("%d %d", &R, &C) != 2) input_error();
    float* inp = read_floats_device((size_t)R * C);
    float* out = device_zeros((size_t)R);
    solve(out, inp, R, C);
    finish_solve();
    print_device_floats(out, (size_t)R);
    return 0;
}
