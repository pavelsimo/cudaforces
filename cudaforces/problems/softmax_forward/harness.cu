// Judge harness for softmax-forward.
// stdin: N C, then inp[N*C]. stdout: out[N*C].
#include "judge_io.h"

extern "C" void solve(float* out, const float* inp, int N, int C);

int main() {
    int N, C;
    if (scanf("%d %d", &N, &C) != 2) input_error();
    float* inp = read_floats_device((size_t)N * C);
    float* out = device_zeros((size_t)N * C);
    solve(out, inp, N, C);
    finish_solve();
    print_device_floats(out, (size_t)N * C);
    return 0;
}
