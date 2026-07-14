// Judge harness for layernorm-backward.
// stdin: B T C, then dout[B*T*C], inp[B*T*C], weight[C], mean[B*T], rstd[B*T].
// stdout: dinp[B*T*C], then dweight[C], then dbias[C].
#include "judge_io.h"

extern "C" void solve(float* dinp, float* dweight, float* dbias,
                      const float* dout, const float* inp, const float* weight,
                      const float* mean, const float* rstd, int B, int T, int C);

int main() {
    int B, T, C;
    if (scanf("%d %d %d", &B, &T, &C) != 3) input_error();
    size_t rows = (size_t)B * T;
    float* dout = read_floats_device(rows * C);
    float* inp = read_floats_device(rows * C);
    float* weight = read_floats_device(C);
    float* mean = read_floats_device(rows);
    float* rstd = read_floats_device(rows);
    float* dinp = device_zeros(rows * C);
    float* dweight = device_zeros(C);
    float* dbias = device_zeros(C);
    solve(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);
    finish_solve();
    print_device_floats(dinp, rows * C);
    print_device_floats(dweight, C);
    print_device_floats(dbias, C);
    return 0;
}
