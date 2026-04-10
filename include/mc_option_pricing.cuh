#ifndef MC_OPTION_PRICING_CUH
#define MC_OPTION_PRICING_CUH

#include "mc_engine.cuh"


__global__ void simulate_option(float* out,
                                 curandState* states,
                                 const float* P0, const float* f0,
                                 const float* d_drift,
                                 const float* d_sens_drift,
                                 HWParams p,
                                 float T, float S, float X, float r0) {
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

    if (id < N_PATHS) {
        curandState local_state      = states[id];
        float r                      = r0;
        float discount_integral      = 0.0f;
        float dr_dsigma              = 0.0f;
        float dr_dsigma_integral     = 0.0f;

        int n_steps = (int)(T / p.dt);
        for (int i = 0; i < n_steps; i++) {
            float G = curand_normal(&local_state);
            evolve_short_rate(r, discount_integral, d_drift[i], G, p);
            evolve_short_rate_derivative(dr_dsigma, dr_dsigma_integral,
                                         d_sens_drift[i], G, p);
        }

        float disc               = expf(-discount_integral);
        float B_TS               = B(T, S, p.a);
        float P_S                = P(P0, f0, T, S, r, p.a, p.sigma);

        float sens_A             = (p.sigma / (2.0f * p.a))
                                   * (1.0f - expf(-2.0f * p.a * T)) * B_TS * B_TS;
        float dsens_A_ds         = (1.0f   / (2.0f * p.a))
                                   * (1.0f - expf(-2.0f * p.a * T)) * B_TS * B_TS;
        float sens_rT            = B_TS * dr_dsigma;

        float dP_S_ds   = -P_S * (sens_A + sens_rT);
        float d2P_S_ds2 =  P_S * (sens_A + sens_rT) * (sens_A + sens_rT)
                         - P_S * dsens_A_ds;

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
        s_volga_zbc[threadIdx.x] = -disc * d2P_S_ds2 * itm_zbc
                                   - 2.0f * dr_dsigma_integral * disc * dP_S_ds * itm_zbc
                                   + dr_dsigma_integral * dr_dsigma_integral * disc * pay_zbc;
        s_volga_zbp[threadIdx.x] = -disc * d2P_S_ds2 * itm_zbp
                                   - 2.0f * dr_dsigma_integral * disc * dP_S_ds * itm_zbp
                                   + dr_dsigma_integral * dr_dsigma_integral * disc * pay_zbp;

        states[id] = local_state;
    }

    __syncthreads();
    for (int i = NTPB/2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            s_zbc      [threadIdx.x] += s_zbc      [threadIdx.x + i];
            s_zbp      [threadIdx.x] += s_zbp      [threadIdx.x + i];
            s_vega_zbc [threadIdx.x] += s_vega_zbc [threadIdx.x + i];
            s_vega_zbp [threadIdx.x] += s_vega_zbp [threadIdx.x + i];
            s_volga_zbc[threadIdx.x] += s_volga_zbc[threadIdx.x + i];
            s_volga_zbp[threadIdx.x] += s_volga_zbp[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(&out[0], s_zbc      [0]);
        atomicAdd(&out[1], s_zbp      [0]);
        atomicAdd(&out[2], s_vega_zbc [0]);
        atomicAdd(&out[3], s_vega_zbp [0]);
        atomicAdd(&out[4], s_volga_zbc[0]);
        atomicAdd(&out[5], s_volga_zbp[0]);
    }
}

__global__ void simulate_option_delta(float* out,
                                       curandState* states,
                                       const float* P0, const float* f0,
                                       const float* d_drift,
                                       HWParams p,
                                       float T, float S, float X, float r0) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float s_delta_zbc[NTPB];
    __shared__ float s_delta_zbp[NTPB];

    s_delta_zbc[threadIdx.x] = 0.0f;
    s_delta_zbp[threadIdx.x] = 0.0f;

    if (id < N_PATHS) {
        curandState local_state = states[id];
        float r                 = r0;
        float discount_integral = 0.0f;

        int n_steps = (int)(T / p.dt);
        for (int i = 0; i < n_steps; i++) {
            float G = curand_normal(&local_state);
            evolve_short_rate(r, discount_integral, d_drift[i], G, p);
        }

        float disc    = expf(-discount_integral);
        float B_TS    = B(T, S, p.a);
        float B_0T    = B(0.0f, T, p.a);
        float e_aT    = expf(-p.a * T);
        float P_S     = P(P0, f0, T, S, r, p.a, p.sigma);

        float dP_S_dr = -B_TS * P_S * e_aT;

        float pay_zbc = fmaxf(P_S - X, 0.0f);
        float pay_zbp = fmaxf(X - P_S, 0.0f);
        float itm_zbc = (P_S > X) ? 1.0f : 0.0f;
        float itm_zbp = (P_S < X) ? 1.0f : 0.0f;

        s_delta_zbc[threadIdx.x] =  disc * dP_S_dr * itm_zbc - B_0T * disc * pay_zbc;
        s_delta_zbp[threadIdx.x] = -disc * dP_S_dr * itm_zbp - B_0T * disc * pay_zbp;

        states[id] = local_state;
    }

    __syncthreads();
    for (int i = NTPB/2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            s_delta_zbc[threadIdx.x] += s_delta_zbc[threadIdx.x + i];
            s_delta_zbp[threadIdx.x] += s_delta_zbp[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(&out[0], s_delta_zbc[0]);
        atomicAdd(&out[1], s_delta_zbp[0]);
    }
}

__global__ void simulate_option_gamma(float* out,
                                       curandState* states,
                                       const float* P0, const float* f0,
                                       const float* d_drift,
                                       HWParams p,
                                       float T, float S, float X, float r0,
                                       float eps) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float s_gamma_zbc[NTPB];
    __shared__ float s_gamma_zbp[NTPB];

    s_gamma_zbc[threadIdx.x] = 0.0f;
    s_gamma_zbp[threadIdx.x] = 0.0f;

    if (id < N_PATHS) {
        curandState local_state = states[id];
        float r                 = r0;
        float discount_integral = 0.0f;

        int n_steps = (int)(T / p.dt);
        for (int i = 0; i < n_steps; i++) {
            float G = curand_normal(&local_state);
            evolve_short_rate(r, discount_integral, d_drift[i], G, p);
        }

        float disc      = expf(-discount_integral);
        float e_aT      = expf(-p.a * T);
        float B_0T      = B(0.0f, T, p.a);
        float disc_up   = disc * expf(-eps * B_0T);
        float disc_down = disc * expf(+eps * B_0T);

        float P_S_base  = P(P0, f0, T, S, r,              p.a, p.sigma);
        float P_S_up    = P(P0, f0, T, S, r + eps * e_aT, p.a, p.sigma);
        float P_S_down  = P(P0, f0, T, S, r - eps * e_aT, p.a, p.sigma);

        float zbc_base  = disc      * fmaxf(P_S_base - X, 0.0f);
        float zbc_up    = disc_up   * fmaxf(P_S_up   - X, 0.0f);
        float zbc_down  = disc_down * fmaxf(P_S_down - X, 0.0f);

        float zbp_base  = disc      * fmaxf(X - P_S_base, 0.0f);
        float zbp_up    = disc_up   * fmaxf(X - P_S_up,   0.0f);
        float zbp_down  = disc_down * fmaxf(X - P_S_down, 0.0f);

        s_gamma_zbc[threadIdx.x] = zbc_up - 2.0f * zbc_base + zbc_down;
        s_gamma_zbp[threadIdx.x] = zbp_up - 2.0f * zbp_base + zbp_down;

        states[id] = local_state;
    }

    __syncthreads();
    for (int i = NTPB/2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            s_gamma_zbc[threadIdx.x] += s_gamma_zbc[threadIdx.x + i];
            s_gamma_zbp[threadIdx.x] += s_gamma_zbp[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(&out[0], s_gamma_zbc[0]);
        atomicAdd(&out[1], s_gamma_zbp[0]);
    }
}

#endif // MC_OPTION_PRICING_CUH
