// Judge harness for coo-to-csr.
// stdin: R C NNZ, then rows[NNZ], cols[NNZ], values[NNZ].
// stdout: row_ptr[R+1], col_out[NNZ], values_out[NNZ].
#include "judge_io.h"

extern "C" void solve(int* row_ptr, int* col_out, float* values_out,
                      const int* rows, const int* cols, const float* values,
                      int R, int C, int NNZ);

int main() {
    int R, C, NNZ;
    if (scanf("%d %d %d", &R, &C, &NNZ) != 3) input_error();
    int* rows = read_ints_device(NNZ);
    int* cols = read_ints_device(NNZ);
    float* values = read_floats_device(NNZ);
    int* row_ptr = device_int_zeros((size_t)R + 1);
    int* col_out = device_int_zeros(NNZ);
    float* values_out = device_zeros(NNZ);
    solve(row_ptr, col_out, values_out, rows, cols, values, R, C, NNZ);
    finish_solve();
    print_device_ints(row_ptr, (size_t)R + 1);
    print_device_ints(col_out, NNZ);
    print_device_floats(values_out, NNZ);
    return 0;
}
