// Judge harness for fused-residual-layernorm.
// stdin: N C, then inp1[N*C], inp2[N*C], weight[C], bias[C].
// stdout: residual[N*C], normed[N*C], mean[N], rstd[N].
#include "judge_io.h"

extern "C" void solve(float* residual, float* normed, float* mean, float* rstd,
                      const float* inp1, const float* inp2, const float* weight,
                      const float* bias, int N, int C);

int main() {
    int N, C;
    if (scanf("%d %d", &N, &C) != 2) input_error();
    size_t NC = (size_t)N * C;
    float* inp1 = read_floats_device(NC);
    float* inp2 = read_floats_device(NC);
    float* weight = read_floats_device(C);
    float* bias = read_floats_device(C);
    float* residual = device_zeros(NC);
    float* normed = device_zeros(NC);
    float* mean = device_zeros(N);
    float* rstd = device_zeros(N);
    solve(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C);
    finish_solve();
    print_device_floats(residual, NC);
    print_device_floats(normed, NC);
    print_device_floats(mean, N);
    print_device_floats(rstd, N);
    return 0;
}
