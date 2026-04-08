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
                            float T, float a, float sigma, float r0){
    float* d_out;
    cudaMalloc(&d_out, 3 * sizeof(float));
    cudaMemset(d_out,  0, 3 * sizeof(float));
    simulate_swaption<<<NB, NTPB>>>(d_out, d_states, d_P0, d_f0, T, a, sigma, r0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    h_out[0] /= N_PATHS;
    h_out[1] /= N_PATHS;
    cudaFree(d_out);
}

inline void price_swaption_delta(float* h_out,
                                  curandState* d_states,
                                  const float* d_P0, const float* d_f0,
                                  float T, float a, float sigma, float r0){
    float* d_out;
    cudaMalloc(&d_out, sizeof(float));
    cudaMemset(d_out,  0, sizeof(float));
    simulate_swaption_delta<<<NB, NTPB>>>(d_out, d_states, d_P0, d_f0, T, a, sigma, r0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    h_out[0] /= N_PATHS;
    cudaFree(d_out);
}

inline void price_swaption_gamma(float* h_out,
                                  curandState* d_states,
                                  const float* d_P0, const float* d_f0,
                                  float T, float a, float sigma, float r0,
                                  float eps){
    float* d_out;
    cudaMalloc(&d_out, sizeof(float));
    cudaMemset(d_out,  0, sizeof(float));
    simulate_swaption_gamma<<<NB, NTPB>>>(d_out, d_states, d_P0, d_f0, T, a, sigma, r0, eps);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    h_out[0] /= (N_PATHS * eps * eps);
    cudaFree(d_out);
}

int main(){
    init(a, sigma);

    curandState* d_states;
    cudaMalloc(&d_states, N_PATHS * sizeof(curandState));

    init_drift(a, sigma, r0);

    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    cudaDeviceSynchronize();
    float h_P[N_MAT];
    simulate_market_price(h_P, d_states, r0);

    float f0[N_MAT];
    calibrate(h_P, f0, a, sigma);

    float* d_P0;
    float* d_f0;
    cudaMalloc(&d_P0, N_MAT * sizeof(float));
    cudaMalloc(&d_f0, N_MAT * sizeof(float));
    cudaMemcpy(d_P0, h_P, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f0, f0,  N_MAT * sizeof(float), cudaMemcpyHostToDevice);


    const float T     = 10.0f;
    const float eps_r = 0.001f;
    const int   n_tenors = 8;

    float tenor_dates[n_tenors] = { 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 
        18.0f};

    // forward par swap rate = ATM strike
    float K = par_swap_rate(T, tenor_dates, n_tenors, h_P);

    // c[i] = K * delta_i  (year fractions matching par_swap_rate convention)
    // c[n-1] += 1  (notional repayment at last tenor)
    float c[n_tenors];
    for(int i = 0; i < n_tenors; i++){
        float delta_i = (i == 0) ? tenor_dates[0] - T : tenor_dates[i] - tenor_dates[i - 1];
        c[i] = K * delta_i;
    }
    c[n_tenors - 1] += 1.0f;

    init_swaption(tenor_dates, c, n_tenors);

    // ── analytical ────────────────────────────────────────────────────────────
    float an_price = analytical_swaption      (T, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);
    float an_vega  = analytical_swaption_vega (T, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);
    float an_volga = analytical_swaption_volga(T, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);
    float an_delta = analytical_swaption_delta(T, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);
    float an_gamma = analytical_swaption_gamma(T, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);

    // ── MC: price + pathwise vega ─────────────────────────────────────────────
    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    cudaDeviceSynchronize();
    float h_pv[3];
    price_swaption(h_pv, d_states, d_P0, d_f0, T, a, sigma, r0);

    // ── MC FD volga from price (2nd-order central FD on sigma) ────────────────
const float eps_v    = 0.01f;                // larger bump: 1/h^2 amplifies noise
unsigned long volga_seed = time(NULL);

// ── up bump ───────────────────────────────────────────────────────────────
init(a, sigma + eps_v);
calibrate(h_P, f0, a, sigma + eps_v);
init_swaption(tenor_dates, c, n_tenors);     // recompute device swaption constants
cudaMemcpy(d_f0, f0, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
init_rng<<<NB, NTPB>>>(d_states, volga_seed);
cudaDeviceSynchronize();
float h_vup[3];
price_swaption(h_vup, d_states, d_P0, d_f0, T, a, sigma + eps_v, r0);

// ── down bump ─────────────────────────────────────────────────────────────
init(a, sigma - eps_v);
calibrate(h_P, f0, a, sigma - eps_v);
init_swaption(tenor_dates, c, n_tenors);
cudaMemcpy(d_f0, f0, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
init_rng<<<NB, NTPB>>>(d_states, volga_seed);
cudaDeviceSynchronize();
float h_vdown[3];
price_swaption(h_vdown, d_states, d_P0, d_f0, T, a, sigma - eps_v, r0);

// ── mid (base sigma, same seed) ───────────────────────────────────────────
init(a, sigma);
calibrate(h_P, f0, a, sigma);
init_swaption(tenor_dates, c, n_tenors);
cudaMemcpy(d_f0, f0, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
init_rng<<<NB, NTPB>>>(d_states, volga_seed);
cudaDeviceSynchronize();
float h_vmid[3];
price_swaption(h_vmid, d_states, d_P0, d_f0, T, a, sigma, r0);

float mc_fd_volga = (h_vup[0] - 2.0f * h_vmid[0] + h_vdown[0]) / (eps_v * eps_v);

    // ── MC: delta ─────────────────────────────────────────────────────────────
    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    cudaDeviceSynchronize();
    float h_delta[1];
    price_swaption_delta(h_delta, d_states, d_P0, d_f0, T, a, sigma, r0);

    // ── MC: gamma ─────────────────────────────────────────────────────────────
    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    cudaDeviceSynchronize();
    float h_gamma[1];
    price_swaption_gamma(h_gamma, d_states, d_P0, d_f0, T, a, sigma, r0, eps_r);

    // ── print ─────────────────────────────────────────────────────────────────
    printf("\n=== Payer Swaption %.0fYx%.0fY  |  a=%.1f  sigma=%.2f  r0=%.3f  K=%.4f ===\n",
           T, tenor_dates[n_tenors - 1] - T, a, sigma, r0, K);
    printf("%-12s  %-14s  %-14s  %-12s\n", "", "MC", "Analytical", "Error");
    printf("%-12s  %-14.6f  %-14.6f  %-12.2e\n", "Price",     h_pv[0],    an_price, fabsf(h_pv[0]    - an_price));
    printf("%-12s  %-14.6f  %-14.6f  %-12.2e\n", "Vega",  h_pv[1], an_vega,  fabsf(h_pv[1] - an_vega));
    printf("%-12s  %-14.6f  %-14.6f  %-12.2e\n", "Delta",      h_delta[0], an_delta, fabsf(h_delta[0]   - an_delta));
    printf("%-12s  %-14.6f  %-14.6f  %-12.2e\n", "Gamma",     h_gamma[0], an_gamma, fabsf(h_gamma[0] - an_gamma));
    printf("%-12s  %-14.6f  %-14.6f  %-12.2e\n", "Volga (FD)", mc_fd_volga, an_volga, fabsf(mc_fd_volga - an_volga));

    cudaFree(d_P0);
    cudaFree(d_f0);
    cudaFree(d_states);
    return 0;
}
