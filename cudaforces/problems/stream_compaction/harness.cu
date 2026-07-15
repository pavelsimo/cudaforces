// Judge harness for stream-compaction.
// stdin: N, then inp[N]. stdout: count[1], then out[:count].
#include "judge_io.h"

extern "C" void solve(float* out, int* count, const float* inp, int N);

int main() {
    int N;
    if (scanf("%d", &N) != 1) input_error();
    float* inp = read_floats_device((size_t)N);
    float* out = device_zeros((size_t)N);
    int* count = device_int_zeros(1);
    solve(out, count, inp, N);
    finish_solve();
    int host_count = read_device_int(count);
    if (host_count < 0 || host_count > N) input_error();
    print_device_ints(count, 1);
    if (host_count > 0) print_device_floats(out, (size_t)host_count);
    return 0;
}
