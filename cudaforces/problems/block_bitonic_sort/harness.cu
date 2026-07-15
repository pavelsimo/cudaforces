// Judge harness for block-bitonic-sort.
// stdin: N, then inp[N]. stdout: out[N] in ascending order.
#include "judge_io.h"

extern "C" void solve(int* out, const int* inp, int N);

int main() {
    int N;
    if (scanf("%d", &N) != 1) input_error();
    int* inp = read_ints_device((size_t)N);
    int* out = device_int_zeros((size_t)N);
    solve(out, inp, N);
    finish_solve();
    print_device_ints(out, (size_t)N);
    return 0;
}
