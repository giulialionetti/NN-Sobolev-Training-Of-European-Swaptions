#ifndef MC_ENGINE_CUH
#define MC_ENGINE_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include "hw_model.cuh"
#include "hw_constants.cuh"

struct HWParams {
    float a;
    float sigma;
    float dt;
    float mean_reversion_factor;   // exp(-a*dt)
    float std_gaussian_shock;      // sigma * sqrt((1-exp(-2a*dt))/(2a))
};

inline HWParams params(float a, float sigma) {
    float dt                   = T_FINAL / N_STEPS;
    float mean_reversion_factor = expf(-a * dt);
    float std_gaussian_shock    = sigma * sqrtf((1.0f - expf(-2.0f * a * dt)) / (2.0f * a));
    return { a, sigma, dt, mean_reversion_factor, std_gaussian_shock };
}


__global__ void init_rng(curandState* states, unsigned long seed){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N_PATHS) curand_init(seed, idx, 0, &states[idx]);
}

__device__ inline void evolve_short_rate(float& r,
                                          float& discount_integral,
                                          float  drift,
                                          float  G,
                                          const HWParams p) {
    float r_next = r * p.mean_reversion_factor + drift + p.std_gaussian_shock * G;
    discount_integral += 0.5f * (r + r_next) * p.dt;
    r = r_next;
}

__device__ inline void evolve_short_rate_derivative(float& dr_dsigma,
                                                      float& dr_dsigma_integral,
                                                      float  sensitivity_drift,
                                                      float  G,
                                                      const HWParams p) {
    float dr_dsigma_next = dr_dsigma * p.mean_reversion_factor
                         + sensitivity_drift
                         + (p.std_gaussian_shock / p.sigma) * G;
    dr_dsigma_integral += 0.5f * (dr_dsigma + dr_dsigma_next) * p.dt;
    dr_dsigma           = dr_dsigma_next;
}


#endif // MC_ENGINE_CUH