// Judge harness for radix-digit-pass.
// stdin: N shift, then keys[N], values[N]. stdout: keys_out[N], then values_out[N].
#include "judge_io.h"

extern "C" void solve(int* keys_out, int* values_out, const int* keys,
                      const int* values, int N, int shift);

int main() {
    int N, shift;
    if (scanf("%d %d", &N, &shift) != 2) input_error();
    int* keys = read_ints_device((size_t)N);
    int* values = read_ints_device((size_t)N);
    int* keys_out = device_int_zeros((size_t)N);
    int* values_out = device_int_zeros((size_t)N);
    solve(keys_out, values_out, keys, values, N, shift);
    finish_solve();
    print_device_ints(keys_out, (size_t)N);
    print_device_ints(values_out, (size_t)N);
    return 0;
}
