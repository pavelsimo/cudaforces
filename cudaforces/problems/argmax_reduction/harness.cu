// Judge harness for argmax-reduction.
// stdin: N, then inp[N]. stdout: max_value[1], then max_index[1].
#include "judge_io.h"

extern "C" void solve(float* max_value, int* max_index, const float* inp, int N);

int main() {
    int N;
    if (scanf("%d", &N) != 1) input_error();
    float* inp = read_floats_device((size_t)N);
    float* max_value = device_zeros(1);
    int* max_index = device_int_zeros(1);
    solve(max_value, max_index, inp, N);
    finish_solve();
    print_device_floats(max_value, 1);
    print_device_ints(max_index, 1);
    return 0;
}
