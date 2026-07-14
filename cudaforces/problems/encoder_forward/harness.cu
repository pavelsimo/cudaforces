// Judge harness for encoder-forward.
// stdin: B T C V, then inp[B*T] (token ids), wte[V*C], wpe[T*C].
// stdout: out[B*T*C]. V sizes the wte allocation but is not passed to solve().
#include "judge_io.h"

extern "C" void solve(float* out, const int* inp, const float* wte,
                      const float* wpe, int B, int T, int C);

int main() {
    int B, T, C, V;
    if (scanf("%d %d %d %d", &B, &T, &C, &V) != 4) input_error();
    int* inp = read_ints_device((size_t)B * T);
    float* wte = read_floats_device((size_t)V * C);
    float* wpe = read_floats_device((size_t)T * C);
    float* out = device_zeros((size_t)B * T * C);
    solve(out, inp, wte, wpe, B, T, C);
    finish_solve();
    print_device_floats(out, (size_t)B * T * C);
    return 0;
}
