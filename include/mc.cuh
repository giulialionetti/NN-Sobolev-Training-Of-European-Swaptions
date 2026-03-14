

#ifndef MC_CUH
#define MC_CUH


#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>
#include "hw_primitives.cuh"
#include "hw_pricing.cuh"
#include "hw_greeks_first.cuh"
#include "calibration.cuh"

#define N_PATHS (1024 * 1024)
#define NTPB 1024
#define NB ((N_PATHS + NTPB - 1) / NTPB)
#define N_STEPS 1000
#define T_FINAL 10.0f
#define N_MAT 100
#define MAT_SPACING 0.1f
#define SAVE_STRIDE 10
#define MAX_TENORS 10

const float host_a = 1.0f;
const float host_sigma = 0.1f;
const float host_r0 = 0.012f;
const float host_dt = T_FINAL / N_STEPS; // 0.01

__constant__ float device_a;
__constant__ float device_sigma;
__constant__ float device_r0;
__constant__ float device_dt;
__constant__ float device_mean_reversion_factor;
__constant__ float device_std_gaussian_shock;
__constant__ float device_drift_table[N_STEPS];
__constant__ float device_sensitivity_drift_table[N_STEPS];

__constant__ float device_tenor_dates[MAX_TENORS];
__constant__ float device_c[MAX_TENORS];
__constant__ int device_n_tenors;

enum class CurveType { FLAT, PIECEWISE_LINEAR};

float compute_drift_flat(int i){
   
    float host_mean_reversion_factor = expf(-host_a * host_dt);
    float host_drift_term = host_r0 * (1.0f - host_mean_reversion_factor) / host_a;
    
    return host_drift_term;
}

float compute_drift_piecewise_linear(int i){

    float s = i * host_dt;
    float s_plus_dt  = s + host_dt;

    float alpha = (s < 5.0f) ? 0.012f  : 0.019f;
    float beta  = (s < 5.0f) ? 0.0014f : 0.001f;
    return (alpha + beta * s_plus_dt) * ((1.0f - expf(-host_a * host_dt))/ host_a) -
     beta * (1.0f - expf(-host_a * host_dt) * (1.0f + host_a * host_dt)) / (host_a * host_a);
}

void init_device_constants(float fd_sigma = host_sigma, CurveType curve = CurveType::FLAT){
    float host_mean_reversion_factor = expf(-host_a * host_dt);
    float host_std_gaussian_shock = fd_sigma * sqrtf((1.0f- expf(-2.0f*host_a*host_dt))/(2.0f*host_a));
    
   
    float host_drift_table[N_STEPS];
    float host_sensitivity_drift_table[N_STEPS];

    for(int i=0; i< N_STEPS; i++){
        if(curve == CurveType::FLAT)
            host_drift_table[i] = compute_drift_flat(i);
        else
            host_drift_table[i] = compute_drift_piecewise_linear(i);

        float s = i * host_dt;
        float s_plus_dt = i * host_dt + host_dt;

        host_sensitivity_drift_table[i] = (2.0f * fd_sigma * expf(-host_a * s_plus_dt) * 
                                      (coshf(host_a * s_plus_dt) - coshf(host_a * s))) 
                                      / (host_a * host_a);
    }

    cudaMemcpyToSymbol(device_a, &host_a, sizeof(float));
    cudaMemcpyToSymbol(device_sigma, &fd_sigma, sizeof(float));
    cudaMemcpyToSymbol(device_r0, &host_r0, sizeof(float));
    cudaMemcpyToSymbol(device_dt, &host_dt, sizeof(float));
    cudaMemcpyToSymbol(device_mean_reversion_factor, &host_mean_reversion_factor, sizeof(float));
    cudaMemcpyToSymbol(device_std_gaussian_shock, &host_std_gaussian_shock, sizeof(float));
    cudaMemcpyToSymbol(device_drift_table, host_drift_table, N_STEPS * sizeof(float));
    cudaMemcpyToSymbol(device_sensitivity_drift_table, host_sensitivity_drift_table, 
                   N_STEPS * sizeof(float));

}

void init_device_constants_calibrated(const float* h_f){
    float host_mean_reversion_factor = expf(-host_a * host_dt);
    float host_std_gaussian_shock    = host_sigma * sqrtf((1.0f - expf(-2.0f * host_a * host_dt))
                                                          / (2.0f * host_a));
    float host_drift_table[N_STEPS];
    float host_sensitivity_drift_table[N_STEPS];

    compute_calibrated_drift_table(host_drift_table, h_f, host_a, host_sigma,
                                   host_dt, MAT_SPACING, N_STEPS, N_MAT);

    for(int i = 0; i < N_STEPS; i++){
        float s_plus_dt = (i + 1) * host_dt;
        host_sensitivity_drift_table[i] = (2.0f * host_sigma * expf(-host_a * s_plus_dt) *
                                          (coshf(host_a * s_plus_dt) - coshf(host_a * (i * host_dt))))
                                          / (host_a * host_a);
    }

    cudaMemcpyToSymbol(device_a,                      &host_a,                      sizeof(float));
    cudaMemcpyToSymbol(device_sigma,                  &host_sigma,                  sizeof(float));
    cudaMemcpyToSymbol(device_r0,                     &host_r0,                     sizeof(float));
    cudaMemcpyToSymbol(device_dt,                     &host_dt,                     sizeof(float));
    cudaMemcpyToSymbol(device_mean_reversion_factor,  &host_mean_reversion_factor,  sizeof(float));
    cudaMemcpyToSymbol(device_std_gaussian_shock,     &host_std_gaussian_shock,     sizeof(float));
    cudaMemcpyToSymbol(device_drift_table,             host_drift_table,             N_STEPS * sizeof(float));
    cudaMemcpyToSymbol(device_sensitivity_drift_table, host_sensitivity_drift_table, N_STEPS * sizeof(float));
}


void init_swaption_constants(const float* tenor_dates, const float* c, int n_tenors){
    cudaMemcpyToSymbol(device_tenor_dates, tenor_dates, n_tenors * sizeof(float));
    cudaMemcpyToSymbol(device_c,           c,           n_tenors * sizeof(float));
    cudaMemcpyToSymbol(device_n_tenors,    &n_tenors,   sizeof(int));
}


__global__ void init_rng(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_PATHS) curand_init(seed, idx, 0, &states[idx]);
}

__device__ inline void evolve_short_rate(float& r_step_i, 
 float& discount_factor_integral, float drift_step_i, float G){

    float r_step_i_plus_one = r_step_i * device_mean_reversion_factor +
                     drift_step_i + device_std_gaussian_shock * G;

    discount_factor_integral += 0.5f * (r_step_i + r_step_i_plus_one) * device_dt;
    r_step_i = r_step_i_plus_one;
    
}

__device__ inline void evolve_short_rate_derivative(float& drdsigma_step_i,
     float& drdsigma_integral, float drift_sensitivity_step_i, float G){
        
        float drdsigma_step_i_plus_one = drdsigma_step_i * device_mean_reversion_factor +
        drift_sensitivity_step_i + (device_std_gaussian_shock/ device_sigma) *G;

        drdsigma_integral += 0.5f * (drdsigma_step_i + drdsigma_step_i_plus_one) * device_dt;
        drdsigma_step_i = drdsigma_step_i_plus_one;
}

__global__ void mc_P0T(float* P_estimator, curandState* states){

    int path_id = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float s_P[N_MAT];

    // each thread clears N_MAT/NTPB slots
    for(int m = threadIdx.x; m < N_MAT; m += blockDim.x)
        s_P[m] = 0.0f;
    __syncthreads();

    if(path_id < N_PATHS){
        curandState local_state = states[path_id];

        float r_step_i = device_r0;
        float discount_factor_integral = 0.0f;
        int maturity_index = 0;

        for(int i = 0; i < N_STEPS; i++){
            float G = curand_normal(&local_state);
            evolve_short_rate(r_step_i, discount_factor_integral, device_drift_table[i], G);

            if((i + 1) % SAVE_STRIDE == 0){
                atomicAdd(&s_P[maturity_index], expf(-discount_factor_integral));
                maturity_index++;
            }
        }
        states[path_id] = local_state;
    }

    __syncthreads();

    // one atomicAdd per maturity per block to global memory
    for(int m = threadIdx.x; m < N_MAT; m += blockDim.x)
        atomicAdd(&P_estimator[m], s_P[m]);
}

__global__ void mc_zbc_vega(float* ZBC_estimator, float* vega_estimator,
 curandState* states, float T_maturity, float S, float K,
 const float* P_market, const float* f_market){

        int path_id = blockDim.x * blockIdx.x + threadIdx.x;

        __shared__ float shared_zbc[NTPB];
        __shared__ float shared_vega[NTPB];

        float thread_zbc = 0.0f;
        float thread_vega = 0.0f;

        if(path_id < N_PATHS){

            curandState local_state = states[path_id];

            float r_step_i = device_r0;
            float discount_factor_integral = 0.0f;
            float drdsigma_step_i = 0.0f;
            float drdsigma_integral = 0.0f;

            int n_steps_T = (int)(T_maturity / device_dt);
            for(int i=0; i < n_steps_T; i++){
                float G = curand_normal(&local_state);
                evolve_short_rate(r_step_i, discount_factor_integral, device_drift_table[i], G);
                evolve_short_rate_derivative(drdsigma_step_i, drdsigma_integral,
                                            device_sensitivity_drift_table[i], G);
            }

            float bond_price_at_maturity = (P_market == nullptr) ?
            FlatCurve{device_a, device_sigma, device_r0}.P(T_maturity, S, r_step_i) :
            MarketCurve{device_a, device_sigma, P_market, f_market, MAT_SPACING, N_MAT}.P(T_maturity, S, r_step_i);
            float discount_factor = expf(-discount_factor_integral);
            float B_val = BtT(T_maturity, S, device_a);
            float convexity_dsigma = (device_sigma / (2.0f * device_a))
                       * (1.0f - expf(-2.0f * device_a * T_maturity))
                       * B_val * B_val;

            float dPricedsigma = -bond_price_at_maturity
                   * (B_val * drdsigma_step_i + convexity_dsigma);

            thread_zbc = discount_factor * fmaxf(bond_price_at_maturity - K, 0.0f);
            thread_vega = discount_factor * dPricedsigma * (bond_price_at_maturity > K ? 1.0f : 0.0f)
                        - drdsigma_integral * discount_factor * fmaxf(bond_price_at_maturity - K, 0.0f);
            states[path_id] = local_state;
        }

        shared_zbc[threadIdx.x] = thread_zbc;
        shared_vega[threadIdx.x] = thread_vega;
        __syncthreads();

        for(int i = NTPB/2; i > 0 ; i >>= 1){
            if(threadIdx.x < i){
                shared_zbc[threadIdx.x] += shared_zbc[threadIdx.x + i];
                shared_vega[threadIdx.x] += shared_vega[threadIdx.x + i];
            }
            __syncthreads();
        }

        if(threadIdx.x == 0){
            atomicAdd(ZBC_estimator, shared_zbc[0]);
            atomicAdd(vega_estimator, shared_vega[0]);
        }
}
__global__ void mc_swaption(float* swaption_estimator, float* vega_estimator,
                             curandState* states,
                             float T, const float* d_P_market, const float* d_f_market){

    int path_id = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float shared_swaption[NTPB];
    __shared__ float shared_vega[NTPB];
    float thread_swaption = 0.0f;
    float thread_vega     = 0.0f;

    if(path_id < N_PATHS){
        curandState local_state = states[path_id];

        float r_step_i                 = device_r0;
        float discount_factor_integral = 0.0f;
        float drdsigma_step_i          = 0.0f;
        float drdsigma_integral        = 0.0f;

        int n_steps_T = (int)(T / device_dt);
        for(int i = 0; i < n_steps_T; i++){
            float G = curand_normal(&local_state);
            evolve_short_rate(r_step_i, discount_factor_integral,
                              device_drift_table[i], G);
            evolve_short_rate_derivative(drdsigma_step_i, drdsigma_integral,
                                        device_sensitivity_drift_table[i], G);
        }

        MarketCurve curve{device_a, device_sigma, d_P_market, d_f_market,
                          MAT_SPACING, N_MAT};

        float swap_value      = 0.0f;
        float dswap_dsigma    = 0.0f;
        for(int i = 0; i < device_n_tenors; i++){
            float P_T_Ti       = curve.P(T, device_tenor_dates[i], r_step_i);
            float B_T_Ti       = BtT(T, device_tenor_dates[i], device_a);
            swap_value        += device_c[i] * P_T_Ti;
            dswap_dsigma      += device_c[i] * (-B_T_Ti) * P_T_Ti * drdsigma_step_i;
        }

        float discount_factor  = expf(-discount_factor_integral);
        float in_the_money     = (swap_value > 1.0f) ? 1.0f : 0.0f;

        thread_swaption = discount_factor * fmaxf(swap_value - 1.0f, 0.0f);
        thread_vega     = discount_factor * dswap_dsigma * in_the_money
                        - drdsigma_integral * discount_factor
                          * fmaxf(swap_value - 1.0f, 0.0f);

        states[path_id] = local_state;
    }

    shared_swaption[threadIdx.x] = thread_swaption;
    shared_vega[threadIdx.x]     = thread_vega;
    __syncthreads();

    for(int i = NTPB/2; i > 0; i >>= 1){
        if(threadIdx.x < i){
            shared_swaption[threadIdx.x] += shared_swaption[threadIdx.x + i];
            shared_vega[threadIdx.x]     += shared_vega[threadIdx.x + i];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        atomicAdd(swaption_estimator, shared_swaption[0]);
        atomicAdd(vega_estimator,     shared_vega[0]);
    }
}


#endif // MC_CUH