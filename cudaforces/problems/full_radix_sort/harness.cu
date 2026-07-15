// Judge harness for full-radix-sort.
// stdin: N, then keys[N], values[N]. stdout: keys_out[N], values_out[N].
#include "judge_io.h"

extern "C" void solve(int* keys_out, int* values_out, const int* keys,
                      const int* values, int N);

int main() {
    int N;
    if (scanf("%d", &N) != 1) input_error();
    int* keys = read_ints_device(N);
    int* values = read_ints_device(N);
    int* keys_out = device_int_zeros(N);
    int* values_out = device_int_zeros(N);
    solve(keys_out, values_out, keys, values, N);
    finish_solve();
    print_device_ints(keys_out, N);
    print_device_ints(values_out, N);
    return 0;
}
