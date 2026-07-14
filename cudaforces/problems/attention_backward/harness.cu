// Judge harness for attention-backward.
// stdin: B NH T HS, then dout, q, k, v — each B*NH*T*HS laid out (B,NH,T,HS) —
// then att (B*NH*T*T, lower-triangular). dq, dk, dv are zero-initialized;
// datt and dpreatt are zero-initialized (B,NH,T,T) scratch, not printed.
// stdout: dq, dk, dv.
#include "judge_io.h"

extern "C" void solve(float* dq, float* dk, float* dv, float* datt, float* dpreatt,
                      const float* dout, const float* q, const float* k,
                      const float* v, const float* att, int B, int NH, int T, int HS);

int main() {
    int B, NH, T, HS;
    if (scanf("%d %d %d %d", &B, &NH, &T, &HS) != 4) input_error();
    size_t N = (size_t)B * NH * T * HS;
    size_t A = (size_t)B * NH * T * T;
    float* dout = read_floats_device(N);
    float* q = read_floats_device(N);
    float* k = read_floats_device(N);
    float* v = read_floats_device(N);
    float* att = read_floats_device(A);
    float* dq = device_zeros(N);
    float* dk = device_zeros(N);
    float* dv = device_zeros(N);
    float* datt = device_zeros(A);
    float* dpreatt = device_zeros(A);
    solve(dq, dk, dv, datt, dpreatt, dout, q, k, v, att, B, NH, T, HS);
    finish_solve();
    print_device_floats(dq, N);
    print_device_floats(dk, N);
    print_device_floats(dv, N);
    return 0;
}
