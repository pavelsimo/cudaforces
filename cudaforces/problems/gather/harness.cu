// Judge harness for gather.
// stdin: N M, then src[N], indices[M]. stdout: out[M].
#include "judge_io.h"

extern "C" void solve(float* out, const float* src, const int* indices, int N, int M);

int main() {
    int N, M;
    if (scanf("%d %d", &N, &M) != 2) input_error();
    float* src = read_floats_device(N);
    int* indices = read_ints_device(M);
    float* out = device_zeros(M);
    solve(out, src, indices, N, M);
    finish_solve();
    print_device_floats(out, M);
    return 0;
}
