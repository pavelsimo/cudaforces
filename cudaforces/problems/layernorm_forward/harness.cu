// Judge harness for layernorm-forward.
// stdin: N C, then inp[N*C], weight[C], bias[C].
// stdout: out[N*C], then mean[N], then rstd[N].
#include "judge_io.h"

extern "C" void solve(float* out, float* mean, float* rstd, const float* inp,
                      const float* weight, const float* bias, int N, int C);

int main() {
    int N, C;
    if (scanf("%d %d", &N, &C) != 2) input_error();
    float* inp = read_floats_device((size_t)N * C);
    float* weight = read_floats_device(C);
    float* bias = read_floats_device(C);
    float* out = device_zeros((size_t)N * C);
    float* mean = device_zeros(N);
    float* rstd = device_zeros(N);
    solve(out, mean, rstd, inp, weight, bias, N, C);
    finish_solve();
    print_device_floats(out, (size_t)N * C);
    print_device_floats(mean, N);
    print_device_floats(rstd, N);
    return 0;
}
