// Judge harness for grid-stride-saxpy.
// stdin: N a, then x[N], y[N]. stdout: updated y[N].
#include "judge_io.h"

extern "C" void solve(float* y, const float* x, float a, int N);

int main() {
    int N;
    float a;
    if (scanf("%d %f", &N, &a) != 2) input_error();
    float* x = read_floats_device(N);
    float* y = read_floats_device(N);
    solve(y, x, a, N);
    finish_solve();
    print_device_floats(y, N);
    return 0;
}
