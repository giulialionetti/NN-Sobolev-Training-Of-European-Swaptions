#ifndef MC_SWAPTIONS_CUH
#define MC_SWAPTIONS_CUH


__constant__ int   d_n_tenors;
__constant__ float d_tenor_dates[MAX_TENORS];
__constant__ float d_c[MAX_TENORS];

inline void init_swaption(const float* tenor_dates, const float* c, int n_tenors){
    cudaMemcpyToSymbol(d_n_tenors,    &n_tenors,   sizeof(int));
    cudaMemcpyToSymbol(d_tenor_dates, tenor_dates, n_tenors * sizeof(float));
    cudaMemcpyToSymbol(d_c,           c,           n_tenors * sizeof(float));
}

__global__ void simulate_swaption(float* out,
                                   curandState* states,
                                   const float* P0, const float* f0,
                                   float T, float a, float sigma, float r0){

    int id = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float s_price[NTPB];
    __shared__ float s_vega [NTPB];

    s_price[threadIdx.x] = 0.0f;
    s_vega [threadIdx.x] = 0.0f;

    if(id < N_PATHS){
        curandState local_state  = states[id];

        float r                  = r0;
        float discount_integral  = 0.0f;
        float dr_dsigma          = 0.0f;
        float dr_dsigma_integral = 0.0f;

        int n_steps = (int)(T / device_dt);
        for(int i = 0; i < n_steps; i++){
            float G = curand_normal(&local_state);
            evolve_short_rate(r, discount_integral, device_drift_table[i], G);
            evolve_short_rate_derivative(dr_dsigma, dr_dsigma_integral,
                                         device_sensitivity_drift_table[i], G);
        }

        float disc = expf(-discount_integral);

        
        float swap_val   = 0.0f;
        float dswap_ds   = 0.0f;
       

        for(int i = 0; i < d_n_tenors; i++){
            float Ti         = d_tenor_dates[i];
            float B_T_Ti     = B(T, Ti, a);
            float P_T_Ti     = P(P0, f0, T, Ti, r, a, sigma);

            float sens_A     = (sigma / (2.0f * a)) * (1.0f - expf(-2.0f * a * T)) * B_T_Ti * B_T_Ti;
            float dsens_A_ds = (1.0f  / (2.0f * a)) * (1.0f - expf(-2.0f * a * T)) * B_T_Ti * B_T_Ti;
            float sens_rT    = B_T_Ti * dr_dsigma;
            float dP_T_Ti    = -P_T_Ti * (sens_A + sens_rT);
           

            swap_val   += d_c[i] * P_T_Ti;
            dswap_ds   += d_c[i] * dP_T_Ti;
         
        }

        float pay = fmaxf(1.0f - swap_val, 0.0f);
        float itm = (swap_val < 1.0f) ? 1.0f : 0.0f;

        s_price[threadIdx.x] = disc * pay;
        s_vega [threadIdx.x] = - disc * dswap_ds * itm
                             - dr_dsigma_integral * disc * pay;

        states[id] = local_state;
    }

    __syncthreads();
    for(int i = NTPB/2; i > 0; i >>= 1){
        if(threadIdx.x < i){
            s_price[threadIdx.x] += s_price[threadIdx.x + i];
            s_vega [threadIdx.x] += s_vega [threadIdx.x + i];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        atomicAdd(&out[0], s_price[0]);
        atomicAdd(&out[1], s_vega [0]);
    }
}

__global__ void simulate_swaption_delta(float* out,
                                         curandState* states,
                                         const float* P0, const float* f0,
                                         float T, float a, float sigma, float r0){

    int id = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float s_delta[NTPB];
    s_delta[threadIdx.x] = 0.0f;

    if(id < N_PATHS){
        curandState local_state  = states[id];

        float r                  = r0;
        float discount_integral  = 0.0f;

        int n_steps = (int)(T / device_dt);
        for(int i = 0; i < n_steps; i++){
            float G = curand_normal(&local_state);
            evolve_short_rate(r, discount_integral, device_drift_table[i], G);
        }

        float disc   = expf(-discount_integral);
        float e_aT   = expf(-a * T);
        float B_0T   = B(0.0f, T, a);

        float swap_val = 0.0f;
        float dswap_dr = 0.0f;

        for(int i = 0; i < d_n_tenors; i++){
            float Ti     = d_tenor_dates[i];
            float B_T_Ti = B(T, Ti, a);
            float P_i    = P(P0, f0, T, Ti, r, a, sigma);
            swap_val    += d_c[i] * P_i;
            dswap_dr    += d_c[i] * (-B_T_Ti * P_i * e_aT);
        }

        float pay = fmaxf(1.0f - swap_val, 0.0f);
        float itm = (swap_val < 1.0f) ? 1.0f : 0.0f;

        s_delta[threadIdx.x] = - disc * dswap_dr * itm - B_0T * disc * pay;

        states[id] = local_state;
    }

    __syncthreads();
    for(int i = NTPB/2; i > 0; i >>= 1){
        if(threadIdx.x < i)
            s_delta[threadIdx.x] += s_delta[threadIdx.x + i];
        __syncthreads();
    }

    if(threadIdx.x == 0)
        atomicAdd(&out[0], s_delta[0]);
}

__global__ void simulate_swaption_gamma(float* out,
                                         curandState* states,
                                         const float* P0, const float* f0,
                                         float T, float a, float sigma, float r0,
                                         float eps){

    int id = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float s_gamma[NTPB];
    s_gamma[threadIdx.x] = 0.0f;

    if(id < N_PATHS){
        curandState local_state  = states[id];

        float r                  = r0;
        float discount_integral  = 0.0f;

        int n_steps = (int)(T / device_dt);
        for(int i = 0; i < n_steps; i++){
            float G = curand_normal(&local_state);
            evolve_short_rate(r, discount_integral, device_drift_table[i], G);
        }

        float disc      = expf(-discount_integral);
        float e_aT      = expf(-a * T);
        float B_0T      = B(0.0f, T, a);
        float disc_up   = disc * expf(-eps * B_0T);
        float disc_down = disc * expf(+eps * B_0T);

        float swap_base = 0.0f;
        float swap_up   = 0.0f;
        float swap_down = 0.0f;

        for(int i = 0; i < d_n_tenors; i++){
            float Ti     = d_tenor_dates[i];
            swap_base   += d_c[i] * P(P0, f0, T, Ti, r,              a, sigma);
            swap_up     += d_c[i] * P(P0, f0, T, Ti, r + eps * e_aT, a, sigma);
            swap_down   += d_c[i] * P(P0, f0, T, Ti, r - eps * e_aT, a, sigma);
        }


        float pay_base = fmaxf(1.0f - swap_base, 0.0f);
        float pay_up   = fmaxf(1.0f - swap_up, 0.0f);
        float pay_down = fmaxf(1.0f - swap_down, 0.0f);

        s_gamma[threadIdx.x] = disc_up   * pay_up
                             - 2.0f * disc * pay_base
                             + disc_down * pay_down;

        states[id] = local_state;
    }

    __syncthreads();
    for(int i = NTPB/2; i > 0; i >>= 1){
        if(threadIdx.x < i)
            s_gamma[threadIdx.x] += s_gamma[threadIdx.x + i];
        __syncthreads();
    }

    if(threadIdx.x == 0)
        atomicAdd(&out[0], s_gamma[0]);
}


#endif // MC_SWAPTIONS_CUH