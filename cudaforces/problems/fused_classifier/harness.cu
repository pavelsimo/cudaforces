// Judge harness for fused-classifier.
// stdin: B T V dloss, then logits[B*T*V], targets[B*T].
// stdout: losses[B*T], then logits[B*T*V] (overwritten in place with dlogits).
#include "judge_io.h"

extern "C" void solve(float* logits, float* losses, const int* targets,
                      int B, int T, int V, float dloss);

int main() {
    int B, T, V;
    float dloss;
    if (scanf("%d %d %d %f", &B, &T, &V, &dloss) != 4) input_error();
    size_t BT = (size_t)B * T;
    float* logits = read_floats_device(BT * V);
    int* targets = read_ints_device(BT);
    float* losses = device_zeros(BT);
    solve(logits, losses, targets, B, T, V, dloss);
    finish_solve();
    print_device_floats(losses, BT);
    print_device_floats(logits, BT * V);
    return 0;
}
