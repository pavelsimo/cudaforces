// Judge harness for csr-spmv.
// stdin: R C NNZ, then row_ptr[R+1], col_idx[NNZ], values[NNZ], x[C].
// stdout: y[R]. Empty rows are valid and produce zero.
#include "judge_io.h"

extern "C" void solve(float* y, const int* row_ptr, const int* col_idx,
                      const float* values, const float* x,
                      int R, int C, int NNZ);

int main() {
    int R, C, NNZ;
    if (scanf("%d %d %d", &R, &C, &NNZ) != 3) input_error();
    int* row_ptr = read_ints_device((size_t)R + 1);
    int* col_idx = read_ints_device((size_t)NNZ);
    float* values = read_floats_device((size_t)NNZ);
    float* x = read_floats_device((size_t)C);
    float* y = device_zeros((size_t)R);
    solve(y, row_ptr, col_idx, values, x, R, C, NNZ);
    finish_solve();
    print_device_floats(y, (size_t)R);
    return 0;
}
