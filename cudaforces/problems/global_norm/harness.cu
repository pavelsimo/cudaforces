// Judge harness for global-norm.
// stdin: count, then data[count]. stdout: out[1] = sum of squares.
#include "judge_io.h"

extern "C" void solve(float* out, const float* data, size_t count);

int main() {
    size_t count;
    if (scanf("%zu", &count) != 1) input_error();
    float* data = read_floats_device(count);
    float* out = device_zeros(1);
    solve(out, data, count);
    finish_solve();
    print_device_floats(out, 1);
    return 0;
}
