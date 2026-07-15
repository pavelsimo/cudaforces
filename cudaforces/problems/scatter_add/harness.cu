// Judge harness for scatter-add.
// stdin: N M, then values[N], indices[N]. stdout: out[M].
#include "judge_io.h"

extern "C" void solve(float* out, const float* values, const int* indices,
                      int N, int M);

int main() {
    int N, M;
    if (scanf("%d %d", &N, &M) != 2) input_error();
    float* values = read_floats_device((size_t)N);
    int* indices = read_ints_device((size_t)N);
    float* out = device_zeros((size_t)M);
    solve(out, values, indices, N, M);
    finish_solve();
    print_device_floats(out, (size_t)M);
    return 0;
}
