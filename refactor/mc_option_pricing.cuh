
#ifndef MC_OPTION_PRICING_CUH
#define MC_OPTION_PRICING_CUH


__global__ void simulate_option(float* out,
                                 curandState* states,
                                 const float* P0, const float* f0,
                                 float T, float S, float X,
                                 float a, float sigma, float r0){

    int id = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float s_zbc      [NTPB];
    __shared__ float s_zbp      [NTPB];
    __shared__ float s_vega_zbc [NTPB];
    __shared__ float s_vega_zbp [NTPB];
    __shared__ float s_volga_zbc[NTPB];
    __shared__ float s_volga_zbp[NTPB];

    s_zbc      [threadIdx.x] = 0.0f;
    s_zbp      [threadIdx.x] = 0.0f;
    s_vega_zbc [threadIdx.x] = 0.0f;
    s_vega_zbp [threadIdx.x] = 0.0f;
    s_volga_zbc[threadIdx.x] = 0.0f;
    s_volga_zbp[threadIdx.x] = 0.0f;

    if(id < N_PATHS){
        curandState local_state      = states[id];

        float r                      = r0;
        float discount_integral      = 0.0f;
        float dr_dsigma              = 0.0f;
        float dr_dsigma_integral     = 0.0f;


        int n_steps = (int)(T / device_dt);
        for(int i = 0; i < n_steps; i++){
            float G = curand_normal(&local_state);
            evolve_short_rate(r, discount_integral, device_drift_table[i], G);
            evolve_short_rate_derivative(dr_dsigma, dr_dsigma_integral,
                                         device_sensitivity_drift_table[i], G);
        }

        float disc                = expf(-discount_integral);
        float B_TS                = B(T, S, a);
        float P_S                 = P(P0, f0, T, S, r, a, sigma);

        float sens_through_A_T_S  = (sigma / (2.0f * a)) * (1.0f - expf(-2.0f * a * T)) * B_TS * B_TS;
        float dsens_through_A_ds  = (1.0f  / (2.0f * a)) * (1.0f - expf(-2.0f * a * T)) * B_TS * B_TS;
        float sens_through_rT     = B_TS * dr_dsigma;

        float dP_S_ds = -P_S * (sens_through_A_T_S + sens_through_rT);
        float d2P_S_ds2 = P_S * (sens_through_A_T_S + sens_through_rT) * (sens_through_A_T_S + sens_through_rT) - dsens_through_A_ds;

        float pay_zbc = fmaxf(P_S - X, 0.0f);
        float pay_zbp = fmaxf(X - P_S, 0.0f);
        float itm_zbc = (P_S > X) ? 1.0f : 0.0f;
        float itm_zbp = (P_S < X) ? 1.0f : 0.0f;

        s_zbc      [threadIdx.x] = disc * pay_zbc;
        s_zbp      [threadIdx.x] = disc * pay_zbp;
        s_vega_zbc [threadIdx.x] = disc * dP_S_ds * itm_zbc
                                  - dr_dsigma_integral * disc * pay_zbc;
        s_vega_zbp [threadIdx.x] = -disc * dP_S_ds * itm_zbp
                                  - dr_dsigma_integral * disc * pay_zbp;

        s_volga_zbc[threadIdx.x] = disc * d2P_S_ds2 * itm_zbc
                          -  dr_dsigma_integral * disc * dP_S_ds * itm_zbc
                          + dr_dsigma_integral * dr_dsigma_integral * disc * pay_zbc;

        s_volga_zbp[threadIdx.x] = - disc * d2P_S_ds2 * itm_zbp
                          - dr_dsigma_integral * disc * dP_S_ds * itm_zbp
                          + dr_dsigma_integral * dr_dsigma_integral * disc * pay_zbp;
        states[id] = local_state;

    }

    __syncthreads();
    for(int i = NTPB/2; i > 0; i >>= 1){
        if(threadIdx.x < i){
            s_zbc      [threadIdx.x] += s_zbc      [threadIdx.x + i];
            s_zbp      [threadIdx.x] += s_zbp      [threadIdx.x + i];
            s_vega_zbc [threadIdx.x] += s_vega_zbc [threadIdx.x + i];
            s_vega_zbp [threadIdx.x] += s_vega_zbp [threadIdx.x + i];
            s_volga_zbc[threadIdx.x] += s_volga_zbc[threadIdx.x + i];
            s_volga_zbp[threadIdx.x] += s_volga_zbp[threadIdx.x + i];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        atomicAdd(&out[0], s_zbc      [0]);
        atomicAdd(&out[1], s_zbp      [0]);
        atomicAdd(&out[2], s_vega_zbc [0]);
        atomicAdd(&out[3], s_vega_zbp [0]);
        atomicAdd(&out[4], s_volga_zbc[0]);
        atomicAdd(&out[5], s_volga_zbp[0]);
        
    }

   

}

#endif // MC_OPTION_PRICING_CUH