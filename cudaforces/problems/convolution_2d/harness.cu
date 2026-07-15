// Judge harness for convolution-2d.
// stdin: H W K, then inp[H*W], kernel[K*K]. stdout: out[H*W].
#include "judge_io.h"

extern "C" void solve(float* out, const float* inp, const float* kernel,
                      int H, int W, int K);

int main() {
    int H, W, K;
    if (scanf("%d %d %d", &H, &W, &K) != 3) input_error();
    float* inp = read_floats_device((size_t)H * W);
    float* kernel = read_floats_device((size_t)K * K);
    float* out = device_zeros((size_t)H * W);
    solve(out, inp, kernel, H, W, K);
    finish_solve();
    print_device_floats(out, (size_t)H * W);
    return 0;
}
