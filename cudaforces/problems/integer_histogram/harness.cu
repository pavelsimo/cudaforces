// Judge harness for integer-histogram.
// stdin: N B, then inp[N]. stdout: bins[B].
#include "judge_io.h"

extern "C" void solve(int* bins, const int* inp, int N, int B);

int main() {
    int N, B;
    if (scanf("%d %d", &N, &B) != 2) input_error();
    int* inp = read_ints_device((size_t)N);
    int* bins = device_int_zeros((size_t)B);
    solve(bins, inp, N, B);
    finish_solve();
    print_device_ints(bins, (size_t)B);
    return 0;
}
