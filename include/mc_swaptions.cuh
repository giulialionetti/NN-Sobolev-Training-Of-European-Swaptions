#ifndef MC_SWAPTIONS_CUH
#define MC_SWAPTIONS_CUH

#include "mc_engine.cuh"


__global__ void simulate_swaption(float* out,
                                   curandState* states,
                                   const float* P0, const float* f0,
                                   const float* d_drift,
                                   const float* d_sens_drift,
                                   HWParams p,
                                   float T, float r0,
                                   const float* tenor_dates,
                                   const float* c,
                                   int n_tenors) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float s_price[NTPB];
    __shared__ float s_vega [NTPB];

    s_price[threadIdx.x] = 0.0f;
    s_vega [threadIdx.x] = 0.0f;

    if (id < N_PATHS) {
        curandState local_state  = states[id];
        float r                  = r0;
        float discount_integral  = 0.0f;
        float dr_dsigma          = 0.0f;
        float dr_dsigma_integral = 0.0f;

        int n_steps = (int)(T / p.dt);
        for (int i = 0; i < n_steps; i++) {
            float G = curand_normal(&local_state);
            evolve_short_rate(r, discount_integral, d_drift[i], G, p);
            evolve_short_rate_derivative(dr_dsigma, dr_dsigma_integral,
                                         d_sens_drift[i], G, p);
        }

        float disc     = expf(-discount_integral);
        float swap_val = 0.0f;
        float dswap_ds = 0.0f;

        for (int i = 0; i < n_tenors; i++) {
            float Ti         = tenor_dates[i];
            float B_T_Ti     = B(T, Ti, p.a);
            float P_T_Ti     = P(P0, f0, T, Ti, r, p.a, p.sigma);

            float sens_A     = (p.sigma / (2.0f * p.a))
                               * (1.0f - expf(-2.0f * p.a * T)) * B_T_Ti * B_T_Ti;
            float sens_rT    = B_T_Ti * dr_dsigma;
            float dP_T_Ti_ds = -P_T_Ti * (sens_A + sens_rT);

            swap_val += c[i] * P_T_Ti;
            dswap_ds += c[i] * dP_T_Ti_ds;
        }

        float pay = fmaxf(1.0f - swap_val, 0.0f);
        float itm = (swap_val < 1.0f) ? 1.0f : 0.0f;

        s_price[threadIdx.x] = disc * pay;
        s_vega [threadIdx.x] = -disc * dswap_ds * itm
                               - dr_dsigma_integral * disc * pay;

        states[id] = local_state;
    }

    __syncthreads();
    for (int i = NTPB/2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            s_price[threadIdx.x] += s_price[threadIdx.x + i];
            s_vega [threadIdx.x] += s_vega [threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(&out[0], s_price[0]);
        atomicAdd(&out[1], s_vega [0]);
    }
}


__global__ __launch_bounds__(1024, 1) void simulate_swaption_volga(float* out,
                                         curandState* states,
                                         const float* P0, const float* f0_base,
                                         const float* f0_up, const float* f0_down,
                                         const float* d_drift_base,
                                         const float* d_drift_up,
                                         const float* d_drift_down,
                                         HWParams p_base,
                                         HWParams p_up,
                                         HWParams p_down,
                                         float T, float r0,
                                         const float* tenor_dates,
                                         const float* c,
                                         int n_tenors,
                                         float eps_v) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float s_volga[NTPB];
    s_volga[threadIdx.x] = 0.0f;

    if (id < N_PATHS) {
        curandState local_state = states[id];

        // three paths with same Gaussian draws, different sigma
        float r_base = r0, r_up = r0, r_down = r0;
        float di_base = 0.0f, di_up = 0.0f, di_down = 0.0f;

        int n_steps = (int)(T / p_base.dt);
        for (int i = 0; i < n_steps; i++) {
            float G = curand_normal(&local_state);
            evolve_short_rate(r_base, di_base, d_drift_base[i], G, p_base);
            evolve_short_rate(r_up,   di_up,   d_drift_up  [i], G, p_up);
            evolve_short_rate(r_down, di_down, d_drift_down[i], G, p_down);
        }

        float disc_base  = expf(-di_base);
        float disc_up    = expf(-di_up);
        float disc_down  = expf(-di_down);

        float swap_base = 0.0f, swap_up = 0.0f, swap_down = 0.0f;
        for (int i = 0; i < n_tenors; i++) {
            float Ti = tenor_dates[i];
            swap_base += c[i] * P(P0, f0_base, T, Ti, r_base, p_base.a, p_base.sigma);
            swap_up   += c[i] * P(P0, f0_up,   T, Ti, r_up,   p_up.a,   p_up.sigma);
            swap_down += c[i] * P(P0, f0_down, T, Ti, r_down, p_down.a, p_down.sigma);
        }

        float pay_base = fmaxf(1.0f - swap_base, 0.0f);
        float pay_up   = fmaxf(1.0f - swap_up,   0.0f);
        float pay_down = fmaxf(1.0f - swap_down,  0.0f);

        s_volga[threadIdx.x] = (disc_up   * pay_up
                              - 2.0f * disc_base * pay_base
                              +  disc_down * pay_down) / (eps_v * eps_v);

        states[id] = local_state;
    }

    __syncthreads();
    for (int i = NTPB/2; i > 0; i >>= 1) {
        if (threadIdx.x < i)
            s_volga[threadIdx.x] += s_volga[threadIdx.x + i];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(&out[0], s_volga[0]);
}


__global__ void simulate_swaption_delta(float* out,
                                         curandState* states,
                                         const float* P0, const float* f0,
                                         const float* d_drift,
                                         HWParams p,
                                         float T, float r0,
                                         const float* tenor_dates,
                                         const float* c,
                                         int n_tenors) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float s_delta[NTPB];
    s_delta[threadIdx.x] = 0.0f;

    if (id < N_PATHS) {
        curandState local_state = states[id];
        float r                 = r0;
        float discount_integral = 0.0f;

        int n_steps = (int)(T / p.dt);
        for (int i = 0; i < n_steps; i++) {
            float G = curand_normal(&local_state);
            evolve_short_rate(r, discount_integral, d_drift[i], G, p);
        }

        float disc     = expf(-discount_integral);
        float e_aT     = expf(-p.a * T);
        float B_0T     = B(0.0f, T, p.a);
        float swap_val = 0.0f;
        float dswap_dr = 0.0f;

        for (int i = 0; i < n_tenors; i++) {
            float Ti     = tenor_dates[i];
            float B_T_Ti = B(T, Ti, p.a);
            float P_i    = P(P0, f0, T, Ti, r, p.a, p.sigma);
            swap_val    += c[i] * P_i;
            dswap_dr    += c[i] * (-B_T_Ti * P_i * e_aT);
        }

        float pay = fmaxf(1.0f - swap_val, 0.0f);
        float itm = (swap_val < 1.0f) ? 1.0f : 0.0f;

        s_delta[threadIdx.x] = -disc * dswap_dr * itm - B_0T * disc * pay;

        states[id] = local_state;
    }

    __syncthreads();
    for (int i = NTPB/2; i > 0; i >>= 1) {
        if (threadIdx.x < i)
            s_delta[threadIdx.x] += s_delta[threadIdx.x + i];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(&out[0], s_delta[0]);
}

__global__ void simulate_swaption_gamma(float* out,
                                         curandState* states,
                                         const float* P0, const float* f0,
                                         const float* d_drift,
                                         HWParams p,
                                         float T, float r0,
                                         const float* tenor_dates,
                                         const float* c,
                                         int n_tenors,
                                         float eps) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float s_gamma[NTPB];
    s_gamma[threadIdx.x] = 0.0f;

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

        float swap_base = 0.0f, swap_up = 0.0f, swap_down = 0.0f;
        for (int i = 0; i < n_tenors; i++) {
            float Ti = tenor_dates[i];
            swap_base += c[i] * P(P0, f0, T, Ti, r,              p.a, p.sigma);
            swap_up   += c[i] * P(P0, f0, T, Ti, r + eps * e_aT, p.a, p.sigma);
            swap_down += c[i] * P(P0, f0, T, Ti, r - eps * e_aT, p.a, p.sigma);
        }

        float pay_base = fmaxf(1.0f - swap_base, 0.0f);
        float pay_up   = fmaxf(1.0f - swap_up,   0.0f);
        float pay_down = fmaxf(1.0f - swap_down,  0.0f);

        s_gamma[threadIdx.x] = disc_up   * pay_up
                             - 2.0f * disc * pay_base
                             +  disc_down * pay_down;

        states[id] = local_state;
    }

    __syncthreads();
    for (int i = NTPB/2; i > 0; i >>= 1) {
        if (threadIdx.x < i)
            s_gamma[threadIdx.x] += s_gamma[threadIdx.x + i];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(&out[0], s_gamma[0]);
}

#endif // MC_SWAPTIONS_CUH
