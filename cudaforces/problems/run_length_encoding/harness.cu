// Judge harness for run-length-encoding.
// stdin: N, then inp[N]. stdout: runs, unique[runs], counts[runs].
#include "judge_io.h"

extern "C" void solve(int* unique, int* counts, int* runs, const int* inp, int N);

int main() {
    int N;
    if (scanf("%d", &N) != 1 || N < 1) input_error();
    int* inp = read_ints_device(N);
    int* unique = device_int_zeros(N);
    int* counts = device_int_zeros(N);
    int* runs = device_int_zeros(1);
    solve(unique, counts, runs, inp, N);
    finish_solve();
    int host_runs = read_device_int(runs);
    if (host_runs < 1 || host_runs > N) input_error();
    printf("%d\n", host_runs);
    print_device_ints(unique, host_runs);
    print_device_ints(counts, host_runs);
    return 0;
}
