// Judge harness for attention-forward.
// stdin: B NH T HS, then q, k, v, each B*NH*T*HS values laid out (B,NH,T,HS).
// preatt and att are zero-initialized (B,NH,T,T) scratch; only out is printed.
#include "judge_io.h"

extern "C" void solve(float* out, float* preatt, float* att, const float* q,
                      const float* k, const float* v, int B, int NH, int T, int HS);

int main() {
    int B, NH, T, HS;
    if (scanf("%d %d %d %d", &B, &NH, &T, &HS) != 4) input_error();
    size_t N = (size_t)B * NH * T * HS;
    float* q = read_floats_device(N);
    float* k = read_floats_device(N);
    float* v = read_floats_device(N);
    float* preatt = device_zeros((size_t)B * NH * T * T);
    float* att = device_zeros((size_t)B * NH * T * T);
    float* out = device_zeros(N);
    solve(out, preatt, att, q, k, v, B, NH, T, HS);
    finish_solve();
    print_device_floats(out, N);
    return 0;
}
