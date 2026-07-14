// Judge harness for gelu-forward.
// stdin: N, then inp[N]. stdout: out[N].
#include "judge_io.h"

extern "C" void solve(float* out, const float* inp, int N);

int main() {
    int N;
    if (scanf("%d", &N) != 1) input_error();
    float* inp = read_floats_device(N);
    float* out = device_zeros(N);
    solve(out, inp, N);
    finish_solve();
    print_device_floats(out, N);
    return 0;
}
