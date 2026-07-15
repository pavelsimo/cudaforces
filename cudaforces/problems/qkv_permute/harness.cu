// Judge harness for qkv-permute.
// stdin: B T NH HS, then inp[B*T*3*NH*HS] laid out (B,T,3,NH,HS).
// stdout: q, k, v, each B*NH*T*HS values laid out (B,NH,T,HS).
#include "judge_io.h"

extern "C" void solve(float* q, float* k, float* v, const float* inp,
                      int B, int T, int NH, int HS);

int main() {
    int B, T, NH, HS;
    if (scanf("%d %d %d %d", &B, &T, &NH, &HS) != 4) input_error();
    size_t N = (size_t)B * NH * T * HS;
    float* inp = read_floats_device(3 * N);
    float* q = device_zeros(N);
    float* k = device_zeros(N);
    float* v = device_zeros(N);
    solve(q, k, v, inp, B, T, NH, HS);
    finish_solve();
    print_device_floats(q, N);
    print_device_floats(k, N);
    print_device_floats(v, N);
    return 0;
}
