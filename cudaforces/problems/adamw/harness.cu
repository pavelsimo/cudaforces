// Judge harness for adamw.
// stdin: num_parameters lr beta1 beta2 t eps weight_decay, then
// params[n], grads[n], m[n], v[n]. stdout: params[n], m[n], v[n] (updated).
#include "judge_io.h"

extern "C" void solve(float* params, const float* grads, float* m, float* v,
                      long num_parameters, float lr, float beta1, float beta2,
                      int t, float eps, float weight_decay);

int main() {
    long num_parameters;
    float lr, beta1, beta2, eps, weight_decay;
    int t;
    if (scanf("%ld %f %f %f %d %f %f", &num_parameters, &lr, &beta1, &beta2,
              &t, &eps, &weight_decay) != 7)
        input_error();
    size_t n = (size_t)num_parameters;
    float* params = read_floats_device(n);
    float* grads = read_floats_device(n);
    float* m = read_floats_device(n);
    float* v = read_floats_device(n);
    solve(params, grads, m, v, num_parameters, lr, beta1, beta2, t, eps, weight_decay);
    finish_solve();
    print_device_floats(params, n);
    print_device_floats(m, n);
    print_device_floats(v, n);
    return 0;
}
