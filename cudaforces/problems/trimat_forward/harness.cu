// Judge harness for trimat-forward.
// stdin: B NH T HS, then q[B*NH*T*HS], k[B*NH*T*HS], each laid out (B,NH,T,HS).
// preatt (B,NH,T,T) is zero-initialized; only the lower triangle is printed,
// so entries above the diagonal may be left untouched or set to -INFINITY.
#include "judge_io.h"

extern "C" void solve(float* preatt, const float* q, const float* k,
                      int B, int NH, int T, int HS);

int main() {
    int B, NH, T, HS;
    if (scanf("%d %d %d %d", &B, &NH, &T, &HS) != 4) input_error();
    size_t N = (size_t)B * NH * T * HS;
    float* q = read_floats_device(N);
    float* k = read_floats_device(N);
    size_t att_elems = (size_t)B * NH * T * T;
    float* preatt = device_zeros(att_elems);
    solve(preatt, q, k, B, NH, T, HS);
    finish_solve();
    float* h = (float*)malloc(att_elems * sizeof(float));
    if (h == NULL) input_error();
    CUDA_CHECK(cudaMemcpy(h, preatt, att_elems * sizeof(float), cudaMemcpyDeviceToHost));
    for (int b = 0; b < B; b++)
        for (int nh = 0; nh < NH; nh++)
            for (int t1 = 0; t1 < T; t1++)
                for (int t2 = 0; t2 <= t1; t2++)
                    printf("%.9g\n", h[(((size_t)b * NH + nh) * T + t1) * T + t2]);
    free(h);
    return 0;
}
