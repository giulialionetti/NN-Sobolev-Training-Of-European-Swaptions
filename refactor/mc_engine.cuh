#ifndef MC_ENGINE_CUH
#define MC_ENGINE_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include "hw_model.cuh"
#include "hw_constants.cuh"

__constant__ float device_a;
__constant__ float device_sigma;
__constant__ float device_dt;
__constant__ float device_mean_reversion_factor;
__constant__ float device_std_gaussian_shock;

inline void init(float a, float sigma){
    float mean_reversion_factor = expf(-a * host_dt);
    float std_gaussian_shock = sigma * sqrtf((1.0f - expf(-2.0f * a * host_dt)) / (2.0f * a));
    cudaMemcpyToSymbol(device_a, &a, sizeof(float));
    cudaMemcpyToSymbol(device_sigma, &sigma,sizeof(float));
    cudaMemcpyToSymbol(device_dt, &host_dt, sizeof(float));
    cudaMemcpyToSymbol(device_mean_reversion_factor, &mean_reversion_factor, sizeof(float));
    cudaMemcpyToSymbol(device_std_gaussian_shock, &std_gaussian_shock, sizeof(float));
}

__global__ void init_rng(curandState* states, unsigned long seed){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N_PATHS) curand_init(seed, idx, 0, &states[idx]);
}

__device__ inline void evolve_short_rate(float& r,
                                          float& discount_integral,
                                          float  drift,
                                          float  G){
    float r_next = r * device_mean_reversion_factor + drift
                       + device_std_gaussian_shock * G;

    discount_integral += 0.5f * (r + r_next) * device_dt;
    r = r_next;
}

__device__ inline void evolve_short_rate_derivative(float& dr_dsigma,
                                                      float& dr_dsigma_integral,
                                                      float  sensitivity_drift,
                                                      float  G){
    float dr_dsigma_next  = dr_dsigma * device_mean_reversion_factor
                         + sensitivity_drift
                         + (device_std_gaussian_shock / device_sigma) * G;

    dr_dsigma_integral  += 0.5f * (dr_dsigma + dr_dsigma_next) * device_dt;
    dr_dsigma            = dr_dsigma_next;
}


#endif // MC_ENGINE_CUH