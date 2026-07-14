// Judge harness for matmul-backward-bias.
// stdin: BT OC, then dout[BT*OC]. stdout: dbias[OC] (zero-initialized before solve).
#include "judge_io.h"

extern "C" void solve(float* dbias, const float* dout, int BT, int OC);

int main() {
    int BT, OC;
    if (scanf("%d %d", &BT, &OC) != 2) input_error();
    float* dout = read_floats_device((size_t)BT * OC);
    float* dbias = device_zeros((size_t)OC);
    solve(dbias, dout, BT, OC);
    finish_solve();
    print_device_floats(dbias, (size_t)OC);
    return 0;
}
