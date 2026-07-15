// Judge harness for five-point-stencil.
// stdin: H W, then inp[H*W] row-major. stdout: out[H*W].
#include "judge_io.h"

extern "C" void solve(float* out, const float* inp, int H, int W);

int main() {
    int H, W;
    if (scanf("%d %d", &H, &W) != 2) input_error();
    float* inp = read_floats_device((size_t)H * W);
    float* out = device_zeros((size_t)H * W);
    solve(out, inp, H, W);
    finish_solve();
    print_device_floats(out, (size_t)H * W);
    return 0;
}
