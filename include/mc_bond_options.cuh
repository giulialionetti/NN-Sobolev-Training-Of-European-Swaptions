#ifndef MC_BOND_OPTIONS_CUH
#define MC_BOND_OPTIONS_CUH

#include "mc_engine.cuh"
#include "hw_pricing.cuh"
#include "hw_greeks_first.cuh"


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


void monteCarlo_vega(float T, float S, float K, curandState* d_states, CurveType curve,
                     float* d_P_market = nullptr, float* d_f_market = nullptr,
                     float* h_P = nullptr, float* h_f = nullptr){

    float* d_ZBC  = nullptr;
    float* d_vega = nullptr;
    cudaMalloc(&d_ZBC,  sizeof(float));
    cudaMalloc(&d_vega, sizeof(float));
    cudaMemset(d_ZBC,  0, sizeof(float));
    cudaMemset(d_vega, 0, sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    const char* label = (curve == CurveType::FLAT)   ? "FLAT"      :
                        (d_P_market == nullptr)        ? "PIECEWISE" : "PIECEWISE+MARKET";
    LOG_INFO("=== MC Pricing [%s curve] ===", label);

    mc_zbc_vega<<<NB, NTPB>>>(d_ZBC, d_vega, d_states, T, S, K, d_P_market, d_f_market);

    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    float h_ZBC, h_vega;
    cudaMemcpy(&h_ZBC,  d_ZBC,  sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_vega, d_vega, sizeof(float), cudaMemcpyDeviceToHost);
    h_ZBC  /= N_PATHS;
    h_vega /= N_PATHS;

    if(d_P_market == nullptr){
        // Flat curve — build PricingState with FlatCurve
        auto ps           = make_pricing_state(0.0f, T, S, K, host_r0,
                                               host_a, host_sigma,
                                               FlatCurve{host_a, host_sigma, host_r0});
        float analytical_zbc  = ZBC_from_state(ps);
        float analytical_vega = vega_ZBC_from_state(ps);
        LOG_INFO("MC ZBC:          %.6f  |  Analytical: %.6f  |  Error: %.2e",
                 h_ZBC, analytical_zbc, fabsf(h_ZBC - analytical_zbc));
        LOG_INFO("MC Vega:         %.6f  |  Analytical: %.6f  |  Error: %.2e",
                 h_vega, analytical_vega, fabsf(h_vega - analytical_vega));
    } else {
        // Market curve — build PricingState with MarketCurve
        auto ps           = make_pricing_state(0.0f, T, S, K, host_r0,
                                               host_a, host_sigma,
                                               MarketCurve{host_a, host_sigma,
                                                           h_P, h_f,
                                                           MAT_SPACING, N_MAT});
        float analytical_zbc  = ZBC_from_state(ps);
        float analytical_vega = vega_ZBC_from_state(ps);
        LOG_INFO("MC ZBC:          %.6f  |  Analytical (market): %.6f  |  Error: %.2e",
                 h_ZBC, analytical_zbc, fabsf(h_ZBC - analytical_zbc));
        LOG_INFO("MC Vega:         %.6f  |  Analytical (market): %.6f  |  Error: %.2e",
                 h_vega, analytical_vega, fabsf(h_vega - analytical_vega));
    }

    cudaFree(d_ZBC);
    cudaFree(d_vega);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void finitedifferences_mc_vega(float T, float S, float K, curandState* d_states, CurveType curve,
                                float* d_P_market = nullptr, float* d_f_market = nullptr,
                                float* h_P = nullptr, float* h_f = nullptr){

    float eps        = 0.001f;
    unsigned long seed = time(NULL);

    float* d_ZBC_plus   = nullptr;
    float* d_ZBC_minus  = nullptr;
    float* d_vega_dummy = nullptr;
    cudaMalloc(&d_ZBC_plus,   sizeof(float));
    cudaMalloc(&d_ZBC_minus,  sizeof(float));
    cudaMalloc(&d_vega_dummy, sizeof(float));

    cudaMemset(d_ZBC_plus,   0, sizeof(float));
    cudaMemset(d_vega_dummy, 0, sizeof(float));
    init_device_constants(host_sigma + eps, curve);
    init_rng<<<NB, NTPB>>>(d_states, seed);
    cudaDeviceSynchronize();
    mc_zbc_vega<<<NB, NTPB>>>(d_ZBC_plus, d_vega_dummy, d_states, T, S, K,
                               d_P_market, d_f_market);
    cudaDeviceSynchronize();

    cudaMemset(d_ZBC_minus,  0, sizeof(float));
    cudaMemset(d_vega_dummy, 0, sizeof(float));
    init_device_constants(host_sigma - eps, curve);
    init_rng<<<NB, NTPB>>>(d_states, seed);
    cudaDeviceSynchronize();
    mc_zbc_vega<<<NB, NTPB>>>(d_ZBC_minus, d_vega_dummy, d_states, T, S, K,
                               d_P_market, d_f_market);
    cudaDeviceSynchronize();

    float h_plus, h_minus;
    cudaMemcpy(&h_plus,  d_ZBC_plus,  sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_minus, d_ZBC_minus, sizeof(float), cudaMemcpyDeviceToHost);
    h_plus  /= N_PATHS;
    h_minus /= N_PATHS;

    float vega_fd = (h_plus - h_minus) / (2.0f * eps);

    if(d_P_market == nullptr){
        auto ps               = make_pricing_state(0.0f, T, S, K, host_r0,
                                                   host_a, host_sigma,
                                                   FlatCurve{host_a, host_sigma, host_r0});
        float analytical_vega = vega_ZBC_from_state(ps);
        LOG_INFO("FD Vega:         %.6f  |  Analytical: %.6f  |  Error: %.2e",
                 vega_fd, analytical_vega, fabsf(vega_fd - analytical_vega));
    } else {
        auto ps               = make_pricing_state(0.0f, T, S, K, host_r0,
                                                   host_a, host_sigma,
                                                   MarketCurve{host_a, host_sigma,
                                                               h_P, h_f,
                                                               MAT_SPACING, N_MAT});
        float analytical_vega = vega_ZBC_from_state(ps);
        LOG_INFO("FD Vega:         %.6f  |  Analytical (market): %.6f  |  Error: %.2e",
                 vega_fd, analytical_vega, fabsf(vega_fd - analytical_vega));
    }

    cudaFree(d_ZBC_plus);
    cudaFree(d_ZBC_minus);
    cudaFree(d_vega_dummy);
}


#endif // MC_BOND_OPTIONS_CUH