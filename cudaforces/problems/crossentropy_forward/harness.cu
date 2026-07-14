// Judge harness for crossentropy-forward.
// stdin: B T V, then probs[B*T*V], then targets[B*T]. stdout: losses[B*T].
#include "judge_io.h"

extern "C" void solve(float* losses, const float* probs, const int* targets,
                      int B, int T, int V);

int main() {
    int B, T, V;
    if (scanf("%d %d %d", &B, &T, &V) != 3) input_error();
    size_t BT = (size_t)B * T;
    float* probs = read_floats_device(BT * V);
    int* targets = read_ints_device(BT);
    float* losses = device_zeros(BT);
    solve(losses, probs, targets, B, T, V);
    finish_solve();
    print_device_floats(losses, BT);
    return 0;
}
