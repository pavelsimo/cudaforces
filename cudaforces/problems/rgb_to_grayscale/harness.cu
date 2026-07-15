// Judge harness for rgb-to-grayscale.
// stdin: H W, then rgb[H*W*3] in HWC order. stdout: out[H*W].
#include "judge_io.h"

extern "C" void solve(float* out, const float* rgb, int H, int W);

int main() {
    int H, W;
    if (scanf("%d %d", &H, &W) != 2) input_error();
    float* rgb = read_floats_device((size_t)H * W * 3);
    float* out = device_zeros((size_t)H * W);
    solve(out, rgb, H, W);
    finish_solve();
    print_device_floats(out, (size_t)H * W);
    return 0;
}
