#include "mc_calibration.cuh"
#include "mc_market_price.cuh"
#include "mc_option_pricing.cuh"
#include "hw_option_pricing.cuh"
#include "hw_option_sensitivities.cuh"
#include <cstdio>
#include <ctime>

const float a     = 1.0f;
const float sigma = 0.1f;
const float r0    = 0.012f;

inline void price_option(float* h_out,
                          curandState* d_states,
                          const float* d_P0, const float* d_f0,
                          float T, float S, float X,
                          float a, float sigma, float r0){
    float* d_out;
    cudaMalloc(&d_out, 6 * sizeof(float));
    cudaMemset(d_out,  0, 6 * sizeof(float));
    simulate_option<<<NB, NTPB>>>(d_out, d_states, d_P0, d_f0, T, S, X, a, sigma, r0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 6 * sizeof(float), cudaMemcpyDeviceToHost);
    h_out[0] /= N_PATHS;
    h_out[1] /= N_PATHS;
    h_out[2] /= N_PATHS;
    h_out[3] /= N_PATHS;
    h_out[4] /= N_PATHS;
    h_out[5] /= N_PATHS;
    cudaFree(d_out);
}

inline void price_option_delta(float* h_out,
                                curandState* d_states,
                                const float* d_P0, const float* d_f0,
                                float T, float S, float X,
                                float a, float sigma, float r0){
    float* d_out;
    cudaMalloc(&d_out, 2 * sizeof(float));
    cudaMemset(d_out,  0, 2 * sizeof(float));
    simulate_option_delta<<<NB, NTPB>>>(d_out, d_states, d_P0, d_f0, T, S, X, a, sigma, r0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    h_out[0] /= N_PATHS;
    h_out[1] /= N_PATHS;
    cudaFree(d_out);
}

inline void price_option_gamma(float* h_out,
                                curandState* d_states,
                                const float* d_P0, const float* d_f0,
                                float T, float S, float X,
                                float a, float sigma, float r0,
                                float eps){
    float* d_out;
    cudaMalloc(&d_out, 2 * sizeof(float));
    cudaMemset(d_out,  0, 2 * sizeof(float));
    simulate_option_gamma<<<NB, NTPB>>>(d_out, d_states, d_P0, d_f0, T, S, X, a, sigma, r0, eps);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    h_out[0] /= (N_PATHS * eps * eps);
    h_out[1] /= (N_PATHS * eps * eps);
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

    const float T      = 5.0f;
    const float S      = 10.0f;
    const float X      = expf(-0.1f);
    const float eps_fd = 0.001f;   // bump for FD vega (1st order)
    const float eps_v  = 0.01f;    // bump for FD volga (2nd order)
    const float eps_r  = 0.001f;   // bump for gamma

    // ── analytical ────────────────────────────────────────────────────────────
    EuroOption o   = euro_option(h_P, f0, 0.0f, T, S, X, r0, a, sigma);
    float an_price = ZBC(o);
    float an_vega  = vega_zbc (o, 0.0f, T, S, a, sigma);
    float an_volga = volga_zbc(o, 0.0f, T, S, a, sigma);
    float an_delta = delta_zbc(o, 0.0f, T, S, a);
    float an_gamma = gamma_zbc(o, 0.0f, T, S, a);

    // ── MC: price + pathwise vega + pathwise volga ────────────────────────────
    // out[0] = ZBC price
    // out[2] = ZBC vega  (pathwise)
    // out[4] = ZBC volga (pathwise)
    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    cudaDeviceSynchronize();
    float h_pv[6];
    price_option(h_pv, d_states, d_P0, d_f0, T, S, X, a, sigma, r0);
    float mc_vega_pw  = h_pv[2];
    float mc_volga_pw = h_pv[4];

    // ── MC: delta ─────────────────────────────────────────────────────────────
    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    cudaDeviceSynchronize();
    float h_delta[2];
    price_option_delta(h_delta, d_states, d_P0, d_f0, T, S, X, a, sigma, r0);

    // ── MC: gamma ─────────────────────────────────────────────────────────────
    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    cudaDeviceSynchronize();
    float h_gamma[2];
    price_option_gamma(h_gamma, d_states, d_P0, d_f0, T, S, X, a, sigma, r0, eps_r);

    // ── MC FD vega: 1st order central FD on MC price ──────────────────────────
    unsigned long fd_seed = time(NULL);

    init(a, sigma + eps_fd);
    calibrate(h_P, f0, a, sigma + eps_fd);
    cudaMemcpy(d_f0, f0, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
    init_rng<<<NB, NTPB>>>(d_states, fd_seed);
    cudaDeviceSynchronize();
    float h_up[6];
    price_option(h_up, d_states, d_P0, d_f0, T, S, X, a, sigma + eps_fd, r0);

    init(a, sigma - eps_fd);
    calibrate(h_P, f0, a, sigma - eps_fd);
    cudaMemcpy(d_f0, f0, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
    init_rng<<<NB, NTPB>>>(d_states, fd_seed);
    cudaDeviceSynchronize();
    float h_down[6];
    price_option(h_down, d_states, d_P0, d_f0, T, S, X, a, sigma - eps_fd, r0);

    float mc_fd_vega = (h_up[0] - h_down[0]) / (2.0f * eps_fd);

    float mc_fd_volga_from_vega = (h_up[2] - h_down[2]) / (2.0f * eps_fd);

    // ── MC FD volga: 2nd order central FD on MC price ─────────────────────────
    unsigned long volga_seed = time(NULL);

    init(a, sigma + eps_v);
    calibrate(h_P, f0, a, sigma + eps_v);
    cudaMemcpy(d_f0, f0, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
    init_rng<<<NB, NTPB>>>(d_states, volga_seed);
    cudaDeviceSynchronize();
    float h_vup[6];
    price_option(h_vup, d_states, d_P0, d_f0, T, S, X, a, sigma + eps_v, r0);

    init(a, sigma - eps_v);
    calibrate(h_P, f0, a, sigma - eps_v);
    cudaMemcpy(d_f0, f0, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
    init_rng<<<NB, NTPB>>>(d_states, volga_seed);
    cudaDeviceSynchronize();
    float h_vdown[6];
    price_option(h_vdown, d_states, d_P0, d_f0, T, S, X, a, sigma - eps_v, r0);

    init(a, sigma);
    calibrate(h_P, f0, a, sigma);
    cudaMemcpy(d_f0, f0, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
    init_rng<<<NB, NTPB>>>(d_states, volga_seed);
    cudaDeviceSynchronize();
    float h_vmid[6];
    price_option(h_vmid, d_states, d_P0, d_f0, T, S, X, a, sigma, r0);

    float mc_fd_volga = (h_vup[0] - 2.0f * h_vmid[0] + h_vdown[0]) / (eps_v * eps_v);


    // ── MC FD delta: 1st order central FD on MC price wrt r0 ─────────────────
unsigned long delta_seed = time(NULL);

init_rng<<<NB, NTPB>>>(d_states, delta_seed);
cudaDeviceSynchronize();
float h_rup[6];
price_option(h_rup, d_states, d_P0, d_f0, T, S, X, a, sigma, r0 + eps_fd);

init_rng<<<NB, NTPB>>>(d_states, delta_seed);
cudaDeviceSynchronize();
float h_rdown[6];
price_option(h_rdown, d_states, d_P0, d_f0, T, S, X, a, sigma, r0 - eps_fd);

float mc_fd_delta = (h_rup[0] - h_rdown[0]) / (2.0f * eps_fd);

// ── MC FD gamma: 2nd order central FD on MC price wrt r0 ─────────────────
init_rng<<<NB, NTPB>>>(d_states, delta_seed);
cudaDeviceSynchronize();
float h_rmid[6];
price_option(h_rmid, d_states, d_P0, d_f0, T, S, X, a, sigma, r0);

float mc_fd_gamma = (h_rup[0] - 2.0f * h_rmid[0] + h_rdown[0]) / (0.001f * 0.001f);
    // ── print ─────────────────────────────────────────────────────────────────
    printf("\n=== ZBC(%.0f, %.0f, e^-0.1)  |  a=%.1f  sigma=%.2f  r0=%.3f ===\n",
           T, S, a, sigma, r0);
    printf("%-14s  %-14s  %-14s  %-12s\n", "", "MC", "Analytical", "Error");
    printf("%-14s  %-14.6f  %-14.6f  %-12.2e\n", "Price",      h_pv[0],      an_price, fabsf(h_pv[0]      - an_price));
    printf("%-14s  %-14.6f  %-14.6f  %-12.2e\n", "Vega (pw)",  mc_vega_pw,   an_vega,  fabsf(mc_vega_pw   - an_vega));
    printf("%-14s  %-14.6f  %-14.6f  %-12.2e\n", "Vega (FD)",  mc_fd_vega,   an_vega,  fabsf(mc_fd_vega   - an_vega));
    printf("%-14s  %-14.6f  %-14.6f  %-12.2e\n", "Volga (pw)", mc_volga_pw,  an_volga, fabsf(mc_volga_pw  - an_volga));
    printf("%-14s  %-14.6f  %-14.6f  %-12.2e\n", "Volga (FD)", mc_fd_volga,  an_volga, fabsf(mc_fd_volga  - an_volga));
    printf("%-14s  %-14.6f  %-14.6f  %-12.2e\n", "Volga (FD v)", mc_fd_volga_from_vega, an_volga, fabsf(mc_fd_volga_from_vega - an_volga));
    printf("%-14s  %-14.6f  %-14.6f  %-12.2e\n", "Delta",      h_delta[0],   an_delta, fabsf(h_delta[0]   - an_delta));
    printf("%-14s  %-14.6f  %-14.6f  %-12.2e\n", "Gamma",      h_gamma[0],   an_gamma, fabsf(h_gamma[0]   - an_gamma));
    printf("%-14s  %-14.6f  %-14.6f  %-12.2e\n", "Delta (FD)", mc_fd_delta,  an_delta, fabsf(mc_fd_delta  - an_delta));
    printf("%-14s  %-14.6f  %-14.6f  %-12.2e\n", "Gamma (FD)", mc_fd_gamma,  an_gamma, fabsf(mc_fd_gamma  - an_gamma));

    
    cudaFree(d_P0);
    cudaFree(d_f0);
    cudaFree(d_states);
    return 0;
}