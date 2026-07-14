// Judge harness for residual-forward.
// stdin: N, then inp1[N], then inp2[N]. stdout: out[N].
#include "judge_io.h"

extern "C" void solve(float* out, const float* inp1, const float* inp2, int N);

int main() {
    int N;
    if (scanf("%d", &N) != 1) input_error();
    float* inp1 = read_floats_device(N);
    float* inp2 = read_floats_device(N);
    float* out = device_zeros(N);
    solve(out, inp1, inp2, N);
    finish_solve();
    print_device_floats(out, N);
    return 0;
}
