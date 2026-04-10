#include "mc_calibration.cuh"
#include "mc_market_price.cuh"
#include "mc_swaptions.cuh"
#include "hw_swaptions.cuh"
#include <cstdio>
#include <ctime>

const float a     = 1.0f;
const float sigma = 0.1f;
const float r0    = 0.012f;

inline void price_swaption(float* h_out,
                            curandState* d_states,
                            const float* d_P0, const float* d_f0,
                            const float* d_drift, const float* d_sens_drift,
                            HWParams p,
                            float T, float r0,
                            const float* d_tenor_dates, const float* d_c, int n_tenors) {
    float* d_out;
    cudaMalloc(&d_out, 2 * sizeof(float));
    cudaMemset(d_out,  0, 2 * sizeof(float));
    simulate_swaption<<<NB, NTPB>>>(d_out, d_states, d_P0, d_f0,
                                     d_drift, d_sens_drift, p,
                                     T, r0,
                                     d_tenor_dates, d_c, n_tenors);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    h_out[0] /= N_PATHS;
    h_out[1] /= N_PATHS;
    cudaFree(d_out);
}

inline void price_swaption_delta(float* h_out,
                                  curandState* d_states,
                                  const float* d_P0, const float* d_f0,
                                  const float* d_drift,
                                  HWParams p,
                                  float T, float r0,
                                  const float* d_tenor_dates, const float* d_c, int n_tenors) {
    float* d_out;
    cudaMalloc(&d_out, sizeof(float));
    cudaMemset(d_out,  0, sizeof(float));
    simulate_swaption_delta<<<NB, NTPB>>>(d_out, d_states, d_P0, d_f0,
                                           d_drift, p,
                                           T, r0,
                                           d_tenor_dates, d_c, n_tenors);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    h_out[0] /= N_PATHS;
    cudaFree(d_out);
}

inline void price_swaption_gamma(float* h_out,
                                  curandState* d_states,
                                  const float* d_P0, const float* d_f0,
                                  const float* d_drift,
                                  HWParams p,
                                  float T, float r0,
                                  const float* d_tenor_dates, const float* d_c, int n_tenors,
                                  float eps) {
    float* d_out;
    cudaMalloc(&d_out, sizeof(float));
    cudaMemset(d_out,  0, sizeof(float));
    simulate_swaption_gamma<<<NB, NTPB>>>(d_out, d_states, d_P0, d_f0,
                                           d_drift, p,
                                           T, r0,
                                           d_tenor_dates, d_c, n_tenors, eps);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    h_out[0] /= (N_PATHS * eps * eps);
    cudaFree(d_out);
}

int main() {
    curandState* d_states;
    cudaMalloc(&d_states, N_PATHS * sizeof(curandState));

    float* d_drift;
    float* d_sens_drift;
    alloc_drift_tables(&d_drift, &d_sens_drift);

    HWParams p = params(a, sigma);
    init_drift(a, sigma, r0, d_drift, d_sens_drift);

    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    cudaDeviceSynchronize();

    float h_P[N_MAT];
    simulate_market_price(h_P, d_states, d_drift, p, r0);

    float f0[N_MAT];
    calibrate(h_P, f0, a, sigma, d_drift, d_sens_drift);

    float* d_P0;  
    cudaMalloc(&d_P0, N_MAT * sizeof(float));
    float* d_f0; 
    cudaMalloc(&d_f0, N_MAT * sizeof(float));
    cudaMemcpy(d_P0, h_P, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f0, f0,  N_MAT * sizeof(float), cudaMemcpyHostToDevice);

    const float T       = 10.0f;
    const float eps_r   = 0.001f;
    const float eps_v   = 0.01f;
    const int   n_tenors = 8;

    float tenor_dates[n_tenors] = { 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f };

    float K = par_swap_rate(T, tenor_dates, n_tenors, h_P);

    float c[n_tenors];
    for (int i = 0; i < n_tenors; i++) {
        float delta_i = (i == 0) ? tenor_dates[0] - T : tenor_dates[i] - tenor_dates[i-1];
        c[i] = K * delta_i;
    }
    c[n_tenors - 1] += 1.0f;

    /* Upload swaption data as regular device buffers */
    float* d_tenor_dates;  cudaMalloc(&d_tenor_dates, n_tenors * sizeof(float));
    float* d_c;            cudaMalloc(&d_c,            n_tenors * sizeof(float));
    cudaMemcpy(d_tenor_dates, tenor_dates, n_tenors * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c,           c,           n_tenors * sizeof(float), cudaMemcpyHostToDevice);

    // ── Analytical ────────────────────────────────────────────────────────
    float an_price = analytical_swaption      (T, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);
    float an_vega  = analytical_swaption_vega (T, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);
    float an_volga = analytical_swaption_volga(T, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);
    float an_delta = analytical_swaption_delta(T, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);
    float an_gamma = analytical_swaption_gamma(T, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);

    // ── MC: price + pathwise vega ──────────────────────────────────────────
    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    cudaDeviceSynchronize();
    float h_pv[2];
    price_swaption(h_pv, d_states, d_P0, d_f0, d_drift, d_sens_drift, p,
                   T, r0, d_tenor_dates, d_c, n_tenors);

    // ── MC FD volga (2nd order central on sigma) ───────────────────────────
    unsigned long volga_seed = time(NULL);

    HWParams p_vup = params(a, sigma + eps_v);
    float f0_vup[N_MAT];
    calibrate(h_P, f0_vup, a, sigma + eps_v, d_drift, d_sens_drift);
    cudaMemcpy(d_f0, f0_vup, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
    init_rng<<<NB, NTPB>>>(d_states, volga_seed); 
    cudaDeviceSynchronize();
    float h_vup[2];
    price_swaption(h_vup, d_states, d_P0, d_f0, d_drift, d_sens_drift, p_vup,
                   T, r0, d_tenor_dates, d_c, n_tenors);

    HWParams p_vdown = params(a, sigma - eps_v);
    float f0_vdown[N_MAT];
    calibrate(h_P, f0_vdown, a, sigma - eps_v, d_drift, d_sens_drift);
    cudaMemcpy(d_f0, f0_vdown, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
    init_rng<<<NB, NTPB>>>(d_states, volga_seed); 
    
    cudaDeviceSynchronize();
    float h_vdown[2];
    price_swaption(h_vdown, d_states, d_P0, d_f0, d_drift, d_sens_drift, p_vdown,
                   T, r0, d_tenor_dates, d_c, n_tenors);

    // restore base for mid
    calibrate(h_P, f0, a, sigma, d_drift, d_sens_drift);
    cudaMemcpy(d_f0, f0, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
    init_rng<<<NB, NTPB>>>(d_states, volga_seed); cudaDeviceSynchronize();
    float h_vmid[2];
    price_swaption(h_vmid, d_states, d_P0, d_f0, d_drift, d_sens_drift, p,
                   T, r0, d_tenor_dates, d_c, n_tenors);

    float mc_fd_volga = (h_vup[0] - 2.0f * h_vmid[0] + h_vdown[0]) / (eps_v * eps_v);

    
    init_rng<<<NB, NTPB>>>(d_states, time(NULL)); cudaDeviceSynchronize();
    float h_delta[1];
    price_swaption_delta(h_delta, d_states, d_P0, d_f0, d_drift, p,
                          T, r0, d_tenor_dates, d_c, n_tenors);

    
    init_rng<<<NB, NTPB>>>(d_states, time(NULL)); cudaDeviceSynchronize();
    float h_gamma[1];
    price_swaption_gamma(h_gamma, d_states, d_P0, d_f0, d_drift, p,
                          T, r0, d_tenor_dates, d_c, n_tenors, eps_r);

    
    printf("\n=== Payer Swaption %.0fYx%.0fY  |  a=%.1f  sigma=%.2f  r0=%.3f  K=%.4f ===\n",
           T, tenor_dates[n_tenors - 1] - T, a, sigma, r0, K);
    printf("%-12s  %-14s  %-14s  %-12s\n", "", "MC", "Analytical", "Error");
    printf("%-12s  %-14.6f  %-14.6f  %-12.2e\n", "Price",     h_pv[0],     an_price, fabsf(h_pv[0]     - an_price));
    printf("%-12s  %-14.6f  %-14.6f  %-12.2e\n", "Vega",      h_pv[1],     an_vega,  fabsf(h_pv[1]     - an_vega));
    printf("%-12s  %-14.6f  %-14.6f  %-12.2e\n", "Delta",     h_delta[0],  an_delta, fabsf(h_delta[0]  - an_delta));
    printf("%-12s  %-14.6f  %-14.6f  %-12.2e\n", "Gamma",     h_gamma[0],  an_gamma, fabsf(h_gamma[0]  - an_gamma));
    printf("%-12s  %-14.6f  %-14.6f  %-12.2e\n", "Volga (FD)", mc_fd_volga, an_volga, fabsf(mc_fd_volga - an_volga));

    free_drift_tables(d_drift, d_sens_drift);
    cudaFree(d_P0); cudaFree(d_f0); cudaFree(d_states);
    cudaFree(d_tenor_dates); cudaFree(d_c);
    return 0;
}
