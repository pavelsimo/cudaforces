// Judge harness for top-k-selection.
// stdin: N K, then inp[N]. stdout: values_out[K], then indices_out[K].
#include "judge_io.h"

extern "C" void solve(float* values_out, int* indices_out, const float* inp,
                      int N, int K);

int main() {
    int N, K;
    if (scanf("%d %d", &N, &K) != 2) input_error();
    float* inp = read_floats_device((size_t)N);
    float* values_out = device_zeros((size_t)K);
    int* indices_out = device_int_zeros((size_t)K);
    solve(values_out, indices_out, inp, N, K);
    finish_solve();
    print_device_floats(values_out, (size_t)K);
    print_device_ints(indices_out, (size_t)K);
    return 0;
}
