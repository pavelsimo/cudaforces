// Shared I/O helpers for CudaForces judge harnesses.
//
// Protocol: stdin line 1 holds all scalar parameters space-separated, then
// each input array's values follow whitespace-separated, in signature order.
// Outputs are printed one value per line with %.9g (float32 round-trip safe).
#ifndef JUDGE_IO_H
#define JUDGE_IO_H

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err_),    \
                    __FILE__, __LINE__);                                              \
            exit(2);                                                                  \
        }                                                                             \
    } while (0)

static inline void input_error() {
    fprintf(stderr, "harness: malformed input\n");
    exit(2);
}

static inline float* read_floats_host(size_t n) {
    float* p = (float*)malloc(n * sizeof(float));
    if (p == NULL) input_error();
    for (size_t i = 0; i < n; i++)
        if (scanf("%f", &p[i]) != 1) input_error();
    return p;
}

static inline int* read_ints_host(size_t n) {
    int* p = (int*)malloc(n * sizeof(int));
    if (p == NULL) input_error();
    for (size_t i = 0; i < n; i++)
        if (scanf("%d", &p[i]) != 1) input_error();
    return p;
}

// Read a host array from stdin and copy it to a fresh device buffer.
static inline float* read_floats_device(size_t n) {
    float* h = read_floats_host(n);
    float* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice));
    free(h);
    return d;
}

static inline int* read_ints_device(size_t n) {
    int* h = read_ints_host(n);
    int* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(int), cudaMemcpyHostToDevice));
    free(h);
    return d;
}

static inline float* device_zeros(size_t n) {
    float* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d, 0, n * sizeof(float)));
    return d;
}

static inline void print_device_floats(const float* d, size_t n) {
    float* h = (float*)malloc(n * sizeof(float));
    if (h == NULL) input_error();
    CUDA_CHECK(cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; i++)
        printf("%.9g\n", h[i]);
    free(h);
}

// Call after solve(): surfaces async kernel errors as a runtime failure.
static inline void finish_solve() {
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

#endif  // JUDGE_IO_H
