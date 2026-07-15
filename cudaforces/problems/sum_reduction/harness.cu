// Judge harness for sum-reduction.
// stdin: N, then inp[N]. stdout: out[1].
#include "judge_io.h"

extern "C" void solve(float* out, const float* inp, int N);

int main() {
    int N;
    if (scanf("%d", &N) != 1) input_error();
    float* inp = read_floats_device((size_t)N);
    float* out = device_zeros(1);
    solve(out, inp, N);
    finish_solve();
    print_device_floats(out, 1);
    return 0;
}
