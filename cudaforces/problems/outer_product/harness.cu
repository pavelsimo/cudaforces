// Judge harness for outer-product.
// stdin: M N, then a[M], then b[N]. stdout: out[M*N] row-major.
#include "judge_io.h"

extern "C" void solve(float* out, const float* a, const float* b, int M, int N);

int main() {
    int M, N;
    if (scanf("%d %d", &M, &N) != 2) input_error();
    float* a = read_floats_device((size_t)M);
    float* b = read_floats_device((size_t)N);
    float* out = device_zeros((size_t)M * N);
    solve(out, a, b, M, N);
    finish_solve();
    print_device_floats(out, (size_t)M * N);
    return 0;
}
