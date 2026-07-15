// Judge harness for n-body-acceleration.
// stdin: N softening, then position[N*3], mass[N]. stdout: acceleration[N*3].
#include "judge_io.h"

extern "C" void solve(float* acceleration, const float* position, const float* mass,
                      float softening, int N);

int main() {
    int N;
    float softening;
    if (scanf("%d %f", &N, &softening) != 2) input_error();
    size_t position_elems = (size_t)N * 3;
    float* position = read_floats_device(position_elems);
    float* mass = read_floats_device(N);
    float* acceleration = device_zeros(position_elems);
    solve(acceleration, position, mass, softening, N);
    finish_solve();
    print_device_floats(acceleration, position_elems);
    return 0;
}
