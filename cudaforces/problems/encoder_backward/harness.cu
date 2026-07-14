// Judge harness for encoder-backward.
// stdin: B T C V, then dout[B*T*C], inp[B*T] (token ids).
// stdout: dwte[V*C], then dwpe[T*C]. V sizes dwte but is not passed to solve().
#include "judge_io.h"

extern "C" void solve(float* dwte, float* dwpe, const float* dout,
                      const int* inp, int B, int T, int C);

int main() {
    int B, T, C, V;
    if (scanf("%d %d %d %d", &B, &T, &C, &V) != 4) input_error();
    float* dout = read_floats_device((size_t)B * T * C);
    int* inp = read_ints_device((size_t)B * T);
    float* dwte = device_zeros((size_t)V * C);
    float* dwpe = device_zeros((size_t)T * C);
    solve(dwte, dwpe, dout, inp, B, T, C);
    finish_solve();
    print_device_floats(dwte, (size_t)V * C);
    print_device_floats(dwpe, (size_t)T * C);
    return 0;
}
