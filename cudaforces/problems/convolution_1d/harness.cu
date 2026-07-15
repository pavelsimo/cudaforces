// Judge harness for convolution-1d.
// stdin: N K, then inp[N], kernel[K]. stdout: out[N].
#include "judge_io.h"

extern "C" void solve(float* out, const float* inp, const float* kernel, int N, int K);

int main() {
    int N, K;
    if (scanf("%d %d", &N, &K) != 2) input_error();
    float* inp = read_floats_device((size_t)N);
    float* kernel = read_floats_device((size_t)K);
    float* out = device_zeros((size_t)N);
    solve(out, inp, kernel, N, K);
    finish_solve();
    print_device_floats(out, (size_t)N);
    return 0;
}
