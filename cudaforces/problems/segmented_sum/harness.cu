// Judge harness for segmented-sum.
// stdin: N S, then values[N], offsets[S+1]. stdout: out[S].
#include "judge_io.h"

extern "C" void solve(float* out, const float* values, const int* offsets,
                      int N, int S);

int main() {
    int N, S;
    if (scanf("%d %d", &N, &S) != 2) input_error();
    float* values = read_floats_device((size_t)N);
    int* offsets = read_ints_device((size_t)S + 1);
    float* out = device_zeros((size_t)S);
    solve(out, values, offsets, N, S);
    finish_solve();
    print_device_floats(out, (size_t)S);
    return 0;
}
