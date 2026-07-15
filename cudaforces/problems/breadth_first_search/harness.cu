// Judge harness for breadth-first-search.
// stdin: V E source, then row_ptr[V+1], col_idx[E]. stdout: distance[V].
#include "judge_io.h"

extern "C" void solve(int* distance, const int* row_ptr, const int* col_idx,
                      int V, int E, int source);

int main() {
    int V, E, source;
    if (scanf("%d %d %d", &V, &E, &source) != 3) input_error();
    int* row_ptr = read_ints_device((size_t)V + 1);
    int* col_idx = read_ints_device((size_t)E);
    int* distance = device_int_zeros((size_t)V);
    CUDA_CHECK(cudaMemset(distance, 0xff, (size_t)V * sizeof(int)));
    solve(distance, row_ptr, col_idx, V, E, source);
    finish_solve();
    print_device_ints(distance, (size_t)V);
    return 0;
}
