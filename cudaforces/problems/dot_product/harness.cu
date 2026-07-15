// Judge harness for dot-product.
// stdin: N, then a[N], b[N]. stdout: out[1].
#include "judge_io.h"

extern "C" void solve(float* out, const float* a, const float* b, int N);

int main() {
    int N;
    if (scanf("%d", &N) != 1) input_error();
    float* a = read_floats_device((size_t)N);
    float* b = read_floats_device((size_t)N);
    float* out = device_zeros(1);
    solve(out, a, b, N);
    finish_solve();
    print_device_floats(out, 1);
    return 0;
}
