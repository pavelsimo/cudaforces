// Judge harness for matrix-transpose.
// stdin: H W, then inp[H*W] row-major. stdout: out[W*H] row-major.
#include "judge_io.h"

extern "C" void solve(float* out, const float* inp, int H, int W);

int main() {
    int H, W;
    if (scanf("%d %d", &H, &W) != 2) input_error();
    size_t n = (size_t)H * W;
    float* inp = read_floats_device(n);
    float* out = device_zeros(n);
    solve(out, inp, H, W);
    finish_solve();
    print_device_floats(out, n);
    return 0;
}
