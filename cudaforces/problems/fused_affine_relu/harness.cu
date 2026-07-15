// Judge harness for fused-affine-relu.
// stdin: N scale bias, then x[N]. stdout: out[N].
#include "judge_io.h"

extern "C" void solve(float* out, const float* x, float scale, float bias, int N);

int main() {
    int N;
    float scale, bias;
    if (scanf("%d %f %f", &N, &scale, &bias) != 3) input_error();
    float* x = read_floats_device(N);
    float* out = device_zeros(N);
    solve(out, x, scale, bias, N);
    finish_solve();
    print_device_floats(out, N);
    return 0;
}
