// Judge harness for gelu-backward.
// stdin: N, then inp[N], then dout[N]. stdout: dinp[N].
#include "judge_io.h"

extern "C" void solve(float* dinp, const float* inp, const float* dout, int N);

int main() {
    int N;
    if (scanf("%d", &N) != 1) input_error();
    float* inp = read_floats_device(N);
    float* dout = read_floats_device(N);
    float* dinp = device_zeros(N);
    solve(dinp, inp, dout, N);
    finish_solve();
    print_device_floats(dinp, N);
    return 0;
}
