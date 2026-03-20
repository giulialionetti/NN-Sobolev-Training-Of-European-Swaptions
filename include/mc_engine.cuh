#ifndef MC_ENGINE_CUH
#define MC_ENGINE_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>
#include "hw_primitives.cuh"
#include "logger.h"
#include "calibration.cuh"

#define N_PATHS     (1024 * 1024)
#define N_STEPS     1000
#define T_FINAL     10.0f
#define N_MAT       100
#define MAT_SPACING 0.1f
#define SAVE_STRIDE 10
#define MAX_TENORS  10

#include "cuda_config.cuh"

const float host_a     = 1.0f;
const float host_sigma = 0.1f;
const float host_r0    = 0.012f;
const float host_dt    = T_FINAL / N_STEPS;

__constant__ float device_a;
__constant__ float device_sigma;
__constant__ float device_r0;
__constant__ float device_dt;
__constant__ float device_mean_reversion_factor;
__constant__ float device_std_gaussian_shock;
__constant__ float device_drift_table[N_STEPS];
__constant__ float device_sensitivity_drift_table[N_STEPS];

enum class CurveType { FLAT, PIECEWISE_LINEAR };

float compute_drift_flat(int i){
    float mean_reversion_factor = expf(-host_a * host_dt);
    return host_r0 * (1.0f - mean_reversion_factor) / host_a;
}

float compute_drift_piecewise_linear(int i){
    float s         = i * host_dt;
    float s_plus_dt = s + host_dt;
    float alpha     = (s < 5.0f) ? 0.012f  : 0.019f;
    float beta      = (s < 5.0f) ? 0.0014f : 0.001f;
    return (alpha + beta * s_plus_dt)
           * ((1.0f - expf(-host_a * host_dt)) / host_a)
           - beta * (1.0f - expf(-host_a * host_dt) * (1.0f + host_a * host_dt))
           / (host_a * host_a);
}

void init_device_constants(float fd_sigma = host_sigma,
                            CurveType curve = CurveType::FLAT){
    float mean_reversion_factor = expf(-host_a * host_dt);
    float std_gaussian_shock    = fd_sigma
                                  * sqrtf((1.0f - expf(-2.0f * host_a * host_dt))
                                          / (2.0f * host_a));
    float host_drift_table[N_STEPS];
    float host_sensitivity_drift_table[N_STEPS];

    for(int i = 0; i < N_STEPS; i++){
        host_drift_table[i] = (curve == CurveType::FLAT)
                              ? compute_drift_flat(i)
                              : compute_drift_piecewise_linear(i);

        float s_plus_dt = (i + 1) * host_dt;
        float s         = i * host_dt;
        host_sensitivity_drift_table[i] = (2.0f * fd_sigma
                                           * expf(-host_a * s_plus_dt)
                                           * (coshf(host_a * s_plus_dt)
                                              - coshf(host_a * s)))
                                          / (host_a * host_a);
    }

    cudaMemcpyToSymbol(device_a,                      &host_a,                sizeof(float));
    cudaMemcpyToSymbol(device_sigma,                  &fd_sigma,              sizeof(float));
    cudaMemcpyToSymbol(device_r0,                     &host_r0,               sizeof(float));
    cudaMemcpyToSymbol(device_dt,                     &host_dt,               sizeof(float));
    cudaMemcpyToSymbol(device_mean_reversion_factor,  &mean_reversion_factor, sizeof(float));
    cudaMemcpyToSymbol(device_std_gaussian_shock,     &std_gaussian_shock,    sizeof(float));
    cudaMemcpyToSymbol(device_drift_table,             host_drift_table,       N_STEPS * sizeof(float));
    cudaMemcpyToSymbol(device_sensitivity_drift_table, host_sensitivity_drift_table,
                       N_STEPS * sizeof(float));
}

void init_device_constants_calibrated(const float* h_f){
    float mean_reversion_factor = expf(-host_a * host_dt);
    float std_gaussian_shock    = host_sigma
                                  * sqrtf((1.0f - expf(-2.0f * host_a * host_dt))
                                          / (2.0f * host_a));
    float host_drift_table[N_STEPS];
    float host_sensitivity_drift_table[N_STEPS];

    compute_calibrated_drift_table(host_drift_table, h_f, host_a, host_sigma,
                                   host_dt, MAT_SPACING, N_STEPS, N_MAT);

    for(int i = 0; i < N_STEPS; i++){
        float s_plus_dt = (i + 1) * host_dt;
        float s         = i * host_dt;
        host_sensitivity_drift_table[i] = (2.0f * host_sigma
                                           * expf(-host_a * s_plus_dt)
                                           * (coshf(host_a * s_plus_dt)
                                              - coshf(host_a * s)))
                                          / (host_a * host_a);
    }

    cudaMemcpyToSymbol(device_a,                      &host_a,                sizeof(float));
    cudaMemcpyToSymbol(device_sigma,                  &host_sigma,            sizeof(float));
    cudaMemcpyToSymbol(device_r0,                     &host_r0,               sizeof(float));
    cudaMemcpyToSymbol(device_dt,                     &host_dt,               sizeof(float));
    cudaMemcpyToSymbol(device_mean_reversion_factor,  &mean_reversion_factor, sizeof(float));
    cudaMemcpyToSymbol(device_std_gaussian_shock,     &std_gaussian_shock,    sizeof(float));
    cudaMemcpyToSymbol(device_drift_table,             host_drift_table,       N_STEPS * sizeof(float));
    cudaMemcpyToSymbol(device_sensitivity_drift_table, host_sensitivity_drift_table,
                       N_STEPS * sizeof(float));
}


__global__ void init_rng(curandState* states, unsigned long seed){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N_PATHS) curand_init(seed, idx, 0, &states[idx]);
}

__device__ inline void evolve_short_rate(float& r,
                                          float& discount_integral,
                                          float  drift,
                                          float  G){
    float r_new        = r * device_mean_reversion_factor
                       + drift
                       + device_std_gaussian_shock * G;
    discount_integral += 0.5f * (r + r_new) * device_dt;
    r                  = r_new;
}

__device__ inline void evolve_short_rate_derivative(float& dr_dsigma,
                                                     float& dr_dsigma_integral,
                                                     float  sensitivity_drift,
                                                     float  G){
    float dr_dsigma_new  = dr_dsigma * device_mean_reversion_factor
                         + sensitivity_drift
                         + (device_std_gaussian_shock / device_sigma) * G;
    dr_dsigma_integral  += 0.5f * (dr_dsigma + dr_dsigma_new) * device_dt;
    dr_dsigma            = dr_dsigma_new;
}

#endif // MC_ENGINE_CUH