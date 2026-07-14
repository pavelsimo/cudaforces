// Judge harness for crossentropy-softmax-backward.
// stdin: B T V, then dlosses[B*T], probs[B*T*V], targets[B*T].
// stdout: dlogits[B*T*V].
#include "judge_io.h"

extern "C" void solve(float* dlogits, const float* dlosses, const float* probs,
                      const int* targets, int B, int T, int V);

int main() {
    int B, T, V;
    if (scanf("%d %d %d", &B, &T, &V) != 3) input_error();
    size_t BT = (size_t)B * T;
    float* dlosses = read_floats_device(BT);
    float* probs = read_floats_device(BT * V);
    int* targets = read_ints_device(BT);
    float* dlogits = device_zeros(BT * V);
    solve(dlogits, dlosses, probs, targets, B, T, V);
    finish_solve();
    print_device_floats(dlogits, BT * V);
    return 0;
}
