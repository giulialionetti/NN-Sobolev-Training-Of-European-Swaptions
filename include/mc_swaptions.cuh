#ifndef MC_SWAPTION_CUH
#define MC_SWAPTION_CUH

#include "mc_engine.cuh"
#include "swaptions.cuh"


__constant__ float device_tenor_dates[MAX_TENORS];
__constant__ float device_c[MAX_TENORS];
__constant__ int   device_n_tenors;


void init_swaption_constants(const float* tenor_dates, const float* c, int n_tenors){
    cudaMemcpyToSymbol(device_tenor_dates, tenor_dates, n_tenors * sizeof(float));
    cudaMemcpyToSymbol(device_c,           c,           n_tenors * sizeof(float));
    cudaMemcpyToSymbol(device_n_tenors,    &n_tenors,   sizeof(int));
}

__global__ void mc_payer_swaption_volga(float* swaption_estimator,
                                         float* vega_estimator,
                                         float* volga_estimator,
                                         curandState* states,
                                         float T, const float* d_P_market,
                                         const float* d_f_market){

    int path_id = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float shared_swaption[NTPB];
    __shared__ float shared_vega[NTPB];
    __shared__ float shared_volga[NTPB];

    float thread_swaption = 0.0f;
    float thread_vega     = 0.0f;
    float thread_volga    = 0.0f;

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

        float swap_value     = 0.0f;
        float dswap_dsigma   = 0.0f;
        float d2swap_dsigma2 = 0.0f;

      for(int i = 0; i < device_n_tenors; i++){
    float P_T_Ti = curve.P(T, device_tenor_dates[i], r_step_i);
    float B_T_Ti = BtT(T, device_tenor_dates[i], device_a);

    swap_value     += device_c[i] * P_T_Ti;
    dswap_dsigma   += device_c[i] * (-B_T_Ti) * P_T_Ti * drdsigma_step_i;
    d2swap_dsigma2 += device_c[i] * B_T_Ti * B_T_Ti * P_T_Ti
                    * drdsigma_step_i * drdsigma_step_i;
}

        float discount_factor = expf(-discount_factor_integral);
        float payoff          = fmaxf(1.0f - swap_value, 0.0f);
        float in_the_money    = (swap_value < 1.0f) ? 1.0f : 0.0f;

        thread_swaption = discount_factor * payoff;
        thread_vega =   discount_factor * (-dswap_dsigma) * in_the_money
                      - drdsigma_integral * discount_factor * payoff;

        // volga:
        // (1)  ∂²D/∂σ² · (1-V)       =  drdsigma_integral² · D · payoff
        // (2)  2·(∂D/∂σ)·(-∂V/∂σ)    = +2·drdsigma_integral · D · dswap_dsigma · itm
        //      sign flips vs receiver because ∂(1-V)/∂σ = -∂V/∂σ
        // (3)  D · (-∂²V/∂σ²)         = -D · d2swap_dsigma2 · itm
        //      sign flips vs receiver
        thread_volga =
              drdsigma_integral * drdsigma_integral * discount_factor * payoff
            + 2.0f * drdsigma_integral * discount_factor * dswap_dsigma * in_the_money
            - discount_factor * d2swap_dsigma2 * in_the_money;

        states[path_id] = local_state;
    }

    shared_swaption[threadIdx.x] = thread_swaption;
    shared_vega[threadIdx.x]     = thread_vega;
    shared_volga[threadIdx.x]    = thread_volga;
    __syncthreads();

    for(int i = NTPB/2; i > 0; i >>= 1){
        if(threadIdx.x < i){
            shared_swaption[threadIdx.x] += shared_swaption[threadIdx.x + i];
            shared_vega[threadIdx.x]     += shared_vega[threadIdx.x + i];
            shared_volga[threadIdx.x]    += shared_volga[threadIdx.x + i];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        atomicAdd(swaption_estimator, shared_swaption[0]);
        atomicAdd(vega_estimator,     shared_vega[0]);
        atomicAdd(volga_estimator,    shared_volga[0]);
    }
}


void monteCarlo_swaption(float T_expiry, curandState* d_states,
                         float* d_P_market, float* d_f_market,
                         float analytical_price, float analytical_vega, float analytical_volga){

    float* d_swaption = nullptr;
    float* d_vega     = nullptr;
    float* d_volga    = nullptr;
    cudaMalloc(&d_swaption, sizeof(float));
    cudaMalloc(&d_vega,     sizeof(float));
    cudaMalloc(&d_volga,    sizeof(float));
    cudaMemset(d_swaption, 0, sizeof(float));
    cudaMemset(d_vega,     0, sizeof(float));
    cudaMemset(d_volga,    0, sizeof(float));

    init_device_constants(host_sigma, CurveType::PIECEWISE_LINEAR);
    mc_payer_swaption_volga<<<NB, NTPB>>>(d_swaption, d_vega, d_volga, d_states,
                                           T_expiry, d_P_market, d_f_market);
    cudaDeviceSynchronize();

    float h_swaption, h_vega, h_volga;
    cudaMemcpy(&h_swaption, d_swaption, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_vega,     d_vega,     sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_volga,    d_volga,    sizeof(float), cudaMemcpyDeviceToHost);
    h_swaption /= N_PATHS;
    h_vega     /= N_PATHS;
    h_volga    /= N_PATHS;

    LOG_INFO("=== Swaption MC vs Analytical ===");
    LOG_INFO("MC swaption      : %.6f  |  Analytical: %.6f  |  Error: %.2e",
             h_swaption, analytical_price, fabsf(h_swaption - analytical_price));
    LOG_INFO("MC pathwise vega : %.6f  |  Analytical: %.6f  |  Error: %.2e",
             h_vega, analytical_vega, fabsf(h_vega - analytical_vega));
    LOG_INFO("MC pathwise volga: %.6f  |  Analytical: %.6f  |  Error: %.2e",
             h_volga,    analytical_volga,  fabsf(h_volga    - analytical_volga));

    cudaFree(d_swaption);
    cudaFree(d_vega);
    cudaFree(d_volga);
}

void finitedifferences_mc_swaption_vega(float T_expiry, curandState* d_states,
                                         float* d_P_market, float* d_f_market){

    float eps          = 0.001f;
    unsigned long seed = time(NULL);

    float* d_swaption_plus  = nullptr;
    float* d_swaption_minus = nullptr;
    float* d_vega_dummy     = nullptr;
    float* d_volga_dummy    = nullptr;
    cudaMalloc(&d_swaption_plus,  sizeof(float));
    cudaMalloc(&d_swaption_minus, sizeof(float));
    cudaMalloc(&d_vega_dummy,     sizeof(float));
    cudaMalloc(&d_volga_dummy,    sizeof(float));

    auto run_pass = [&](float sig, float* d_out){
        float shock = sig * sqrtf((1.0f - expf(-2.0f*host_a*host_dt)) / (2.0f*host_a));
        cudaMemset(d_out,         0, sizeof(float));
        cudaMemset(d_vega_dummy,  0, sizeof(float));
        cudaMemset(d_volga_dummy, 0, sizeof(float));
        cudaMemcpyToSymbol(device_sigma,              &sig,   sizeof(float));
        cudaMemcpyToSymbol(device_std_gaussian_shock, &shock, sizeof(float));
        init_rng<<<NB, NTPB>>>(d_states, seed);
        cudaDeviceSynchronize();
        mc_payer_swaption_volga<<<NB, NTPB>>>(d_out, d_vega_dummy, d_volga_dummy,
                                               d_states, T_expiry,
                                               d_P_market, d_f_market);
        cudaDeviceSynchronize();
    };

    run_pass(host_sigma + eps, d_swaption_plus);
    run_pass(host_sigma - eps, d_swaption_minus);

    float shock_orig = host_sigma * sqrtf((1.0f - expf(-2.0f*host_a*host_dt)) / (2.0f*host_a));
    cudaMemcpyToSymbol(device_sigma,              &host_sigma,  sizeof(float));
    cudaMemcpyToSymbol(device_std_gaussian_shock, &shock_orig,  sizeof(float));

    float h_plus, h_minus;
    cudaMemcpy(&h_plus,  d_swaption_plus,  sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_minus, d_swaption_minus, sizeof(float), cudaMemcpyDeviceToHost);
    h_plus  /= N_PATHS;
    h_minus /= N_PATHS;

    LOG_INFO("FD vega          : %.6f", (h_plus - h_minus) / (2.0f * eps));

    cudaFree(d_swaption_plus);
    cudaFree(d_swaption_minus);
    cudaFree(d_vega_dummy);
    cudaFree(d_volga_dummy);
}

void finitedifferences_mc_swaption_volga(float T_expiry, curandState* d_states,
                                          float* d_P_market, float* d_f_market){

    float eps          = 0.001f;
    unsigned long seed = time(NULL);

    float* d_swaption_plus  = nullptr;
    float* d_swaption_mid   = nullptr;
    float* d_swaption_minus = nullptr;
    float* d_vega_dummy     = nullptr;
    float* d_volga_dummy    = nullptr;
    cudaMalloc(&d_swaption_plus,  sizeof(float));
    cudaMalloc(&d_swaption_mid,   sizeof(float));
    cudaMalloc(&d_swaption_minus, sizeof(float));
    cudaMalloc(&d_vega_dummy,     sizeof(float));
    cudaMalloc(&d_volga_dummy,    sizeof(float));

    auto run_pass = [&](float sig, float* d_out){
        float shock = sig * sqrtf((1.0f - expf(-2.0f*host_a*host_dt)) / (2.0f*host_a));
        cudaMemset(d_out,         0, sizeof(float));
        cudaMemset(d_vega_dummy,  0, sizeof(float));
        cudaMemset(d_volga_dummy, 0, sizeof(float));
        cudaMemcpyToSymbol(device_sigma,              &sig,   sizeof(float));
        cudaMemcpyToSymbol(device_std_gaussian_shock, &shock, sizeof(float));
        init_rng<<<NB, NTPB>>>(d_states, seed);
        cudaDeviceSynchronize();
        mc_payer_swaption_volga<<<NB, NTPB>>>(d_out, d_vega_dummy, d_volga_dummy,
                                               d_states, T_expiry,
                                               d_P_market, d_f_market);
        cudaDeviceSynchronize();
    };

    run_pass(host_sigma + eps, d_swaption_plus);
    run_pass(host_sigma,       d_swaption_mid);
    run_pass(host_sigma - eps, d_swaption_minus);

    float shock_orig = host_sigma * sqrtf((1.0f - expf(-2.0f*host_a*host_dt)) / (2.0f*host_a));
    cudaMemcpyToSymbol(device_sigma,              &host_sigma,  sizeof(float));
    cudaMemcpyToSymbol(device_std_gaussian_shock, &shock_orig,  sizeof(float));

    float h_plus, h_mid, h_minus;
    cudaMemcpy(&h_plus,  d_swaption_plus,  sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_mid,   d_swaption_mid,   sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_minus, d_swaption_minus, sizeof(float), cudaMemcpyDeviceToHost);
    h_plus  /= N_PATHS;
    h_mid   /= N_PATHS;
    h_minus /= N_PATHS;

    LOG_INFO("FD volga         : %.6f",
             (h_plus - 2.0f * h_mid + h_minus) / (eps * eps));

    cudaFree(d_swaption_plus);
    cudaFree(d_swaption_mid);
    cudaFree(d_swaption_minus);
    cudaFree(d_vega_dummy);
    cudaFree(d_volga_dummy);
}


#endif // MC_SWAPTIONS_CUH  