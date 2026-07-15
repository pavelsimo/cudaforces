// Judge harness for collision-free-scatter.
// stdin: N M, then values[N], unique indices[N]. stdout: out[M], initially zero.
#include "judge_io.h"

extern "C" void solve(float* out, const float* values, const int* indices, int N, int M);

int main() {
    int N, M;
    if (scanf("%d %d", &N, &M) != 2) input_error();
    float* values = read_floats_device(N);
    int* indices = read_ints_device(N);
    float* out = device_zeros(M);
    solve(out, values, indices, N, M);
    finish_solve();
    print_device_floats(out, M);
    return 0;
}
