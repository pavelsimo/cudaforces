// Judge harness for matmul-forward.
// stdin: BT C OC, then inp[BT*C], weight[OC*C], bias[OC]. stdout: out[BT*OC].
#include "judge_io.h"

extern "C" void solve(float* out, const float* inp, const float* weight,
                      const float* bias, int BT, int C, int OC);

int main() {
    int BT, C, OC;
    if (scanf("%d %d %d", &BT, &C, &OC) != 3) input_error();
    float* inp = read_floats_device((size_t)BT * C);
    float* weight = read_floats_device((size_t)OC * C);
    float* bias = read_floats_device((size_t)OC);
    float* out = device_zeros((size_t)BT * OC);
    solve(out, inp, weight, bias, BT, C, OC);
    finish_solve();
    print_device_floats(out, (size_t)BT * OC);
    return 0;
}
