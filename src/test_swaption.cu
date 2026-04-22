#include "mc_calibration.cuh"
#include "mc_market_price.cuh"
#include "mc_swaptions.cuh"
#include "hw_swaptions.cuh"
#include <cstdio>
#include <ctime>

const float a     = 1.0f;
const float sigma = 0.1f;
const float r0    = 0.012f;

struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin() { cudaEventRecord(start); }
    float end(const char* label) {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        fprintf(stderr, "%-30s : %.3f ms\n", label, ms);
        return ms;
    }
};

inline void price_swaption(float* h_out,
                            curandState* d_states,
                            const float* d_P0, const float* d_f0,
                            const float* d_drift, const float* d_sens_drift,
                            HWParams p,
                            float T, float r0,
                            const float* d_tenor_dates, const float* d_c, int n_tenors,
                            const char* label) {
    float* d_out;
    cudaMalloc(&d_out, 2 * sizeof(float));
    cudaMemset(d_out,  0, 2 * sizeof(float));
    GpuTimer t;
    t.begin();
    simulate_swaption<<<NB, NTPB>>>(d_out, d_states, d_P0, d_f0,
                                     d_drift, d_sens_drift, p,
                                     T, r0, d_tenor_dates, d_c, n_tenors);
    t.end(label);
    cudaMemcpy(h_out, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    h_out[0] /= N_PATHS; h_out[1] /= N_PATHS;
    cudaFree(d_out);
}

inline void price_swaption_volga(float* h_out,
                                  curandState* d_states,
                                  const float* d_P0,
                                  const float* d_f0_base,
                                  const float* d_f0_up,
                                  const float* d_f0_down,
                                  const float* d_drift_base,
                                  const float* d_drift_up,
                                  const float* d_drift_down,
                                  HWParams p_base,
                                  HWParams p_up,
                                  HWParams p_down,
                                  float T, float r0,
                                  const float* d_tenor_dates, const float* d_c, int n_tenors,
                                  float eps_v,
                                  const char* label) {
    float* d_out;
    cudaMalloc(&d_out, sizeof(float));
    cudaMemset(d_out,  0, sizeof(float));
    GpuTimer t;
    t.begin();
    simulate_swaption_volga<<<NB, NTPB>>>(d_out, d_states,
                                           d_P0,
                                           d_f0_base, d_f0_up, d_f0_down,
                                           d_drift_base, d_drift_up, d_drift_down,
                                           p_base, p_up, p_down,
                                           T, r0,
                                           d_tenor_dates, d_c, n_tenors,
                                           eps_v);
    t.end(label);
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    h_out[0] /= N_PATHS;
    cudaFree(d_out);
}



inline void price_swaption_delta(float* h_out,
                                  curandState* d_states,
                                  const float* d_P0, const float* d_f0,
                                  const float* d_drift,
                                  HWParams p,
                                  float T, float r0,
                                  const float* d_tenor_dates, const float* d_c, int n_tenors,
                                  const char* label) {
    float* d_out;
    cudaMalloc(&d_out, sizeof(float));
    cudaMemset(d_out,  0, sizeof(float));
    GpuTimer t;
    t.begin();
    simulate_swaption_delta<<<NB, NTPB>>>(d_out, d_states, d_P0, d_f0,
                                           d_drift, p,
                                           T, r0, d_tenor_dates, d_c, n_tenors);
    t.end(label);
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
                                  float eps,
                                  const char* label) {
    float* d_out;
    cudaMalloc(&d_out, sizeof(float));
    cudaMemset(d_out,  0, sizeof(float));
    GpuTimer t;
    t.begin();
    simulate_swaption_gamma<<<NB, NTPB>>>(d_out, d_states, d_P0, d_f0,
                                           d_drift, p,
                                           T, r0, d_tenor_dates, d_c, n_tenors, eps);
    t.end(label);
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

    init_rng<<<NB, NTPB>>>(d_states, time(NULL), N_PATHS);
    cudaDeviceSynchronize();

    float h_P[N_MAT];
    // time the market price kernel
    {
        GpuTimer t;
        // launch_market_price is inside simulate_market_price so we time the whole call
        t.begin();
        simulate_market_price(h_P, d_states, d_drift, p, r0, N_PATHS, NB);
        t.end("market_price kernel");
    }

    float f0[N_MAT];
    calibrate(h_P, f0, a, sigma, d_drift, d_sens_drift);

    float* d_P0;  cudaMalloc(&d_P0, N_MAT * sizeof(float));
    float* d_f0;  cudaMalloc(&d_f0, N_MAT * sizeof(float));
    cudaMemcpy(d_P0, h_P, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f0, f0,  N_MAT * sizeof(float), cudaMemcpyHostToDevice);



    

    const float T        = 5.0f;
    const float eps_r    = 0.001f;
    const float eps_v    = 0.01f;
    const int   n_tenors = 5;

    float tenor_dates[n_tenors] = { 6.f, 7.f, 8.f, 9.f, 10.f };

    float K = par_swap_rate(T, tenor_dates, n_tenors, h_P);

    float c[n_tenors];
    for (int i = 0; i < n_tenors; i++) {
        float delta_i = (i == 0) ? tenor_dates[0] - T : tenor_dates[i] - tenor_dates[i-1];
        c[i] = K * delta_i;
    }
    c[n_tenors - 1] += 1.0f;

    float* d_tenor_dates; cudaMalloc(&d_tenor_dates, n_tenors * sizeof(float));
    float* d_c;           cudaMalloc(&d_c,            n_tenors * sizeof(float));
    cudaMemcpy(d_tenor_dates, tenor_dates, n_tenors * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c,           c,           n_tenors * sizeof(float), cudaMemcpyHostToDevice);

    float an_price = analytical_swaption      (T, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);
    float an_vega  = analytical_swaption_vega (T, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);
    float an_volga = analytical_swaption_volga(T, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);
    float an_delta = analytical_swaption_delta(T, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);
    float an_gamma = analytical_swaption_gamma(T, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);

    fprintf(stderr, "\n--- Kernel timings ---\n");

    // ── MC: price + Vega ──────────────────────────────────────────
    init_rng<<<NB, NTPB>>>(d_states, time(NULL), N_PATHS);
    cudaDeviceSynchronize();
    float h_pv[2];
    price_swaption(h_pv, d_states, d_P0, d_f0, d_drift, d_sens_drift, p,
                   T, r0, d_tenor_dates, d_c, n_tenors,
                   "simulate_swaption (price+vega)");

    // ── MC volga ───────────────────────────────────────────────────────────
    HWParams p_vup   = params(a, sigma + eps_v);
    HWParams p_vdown = params(a, sigma - eps_v);
    float f0_vup[N_MAT], f0_vdown[N_MAT];

    float* d_drift_up;   float* d_sens_drift_up;
    float* d_drift_down; float* d_sens_drift_down;
    alloc_drift_tables(&d_drift_up,   &d_sens_drift_up);
    alloc_drift_tables(&d_drift_down, &d_sens_drift_down);

    calibrate(h_P, f0_vup,   a, sigma + eps_v, d_drift_up,   d_sens_drift_up);
    calibrate(h_P, f0_vdown, a, sigma - eps_v, d_drift_down, d_sens_drift_down);

    float* d_f0_up;   cudaMalloc(&d_f0_up,   N_MAT * sizeof(float));
    float* d_f0_down; cudaMalloc(&d_f0_down, N_MAT * sizeof(float));
    cudaMemcpy(d_f0_up,   f0_vup,   N_MAT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f0_down, f0_vdown, N_MAT * sizeof(float), cudaMemcpyHostToDevice);

    init_rng<<<NB, NTPB>>>(d_states, time(NULL), N_PATHS);
    cudaDeviceSynchronize();
    float h_volga[1];
    price_swaption_volga(h_volga, d_states,
                          d_P0,
                          d_f0, d_f0_up, d_f0_down,
                          d_drift, d_drift_up, d_drift_down,
                          p, p_vup, p_vdown,
                          T, r0, d_tenor_dates, d_c, n_tenors,
                          eps_v,
                          "simulate_swaption_volga");
                                     

    // ── MC: delta ──────────────────────────────────────────────────────────
    init_rng<<<NB, NTPB>>>(d_states, time(NULL), N_PATHS);
    cudaDeviceSynchronize();
    float h_delta[1];
    price_swaption_delta(h_delta, d_states, d_P0, d_f0, d_drift, p,
                          T, r0, d_tenor_dates, d_c, n_tenors,
                          "simulate_swaption_delta");

    // ── MC: gamma ──────────────────────────────────────────────────────────
    init_rng<<<NB, NTPB>>>(d_states, time(NULL), N_PATHS);
    cudaDeviceSynchronize();
    float h_gamma[1];
    price_swaption_gamma(h_gamma, d_states, d_P0, d_f0, d_drift, p,
                          T, r0, d_tenor_dates, d_c, n_tenors, eps_r,
                          "simulate_swaption_gamma");

    float eps_a = 0.001f;
float an_da = analytical_swaption_da(T, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);

unsigned long da_seed = time(NULL);

// ── up ────────────────────────────────────────────────────────────────────
HWParams p_aup = params(a + eps_a, sigma);
float f0_aup[N_MAT];
float* d_drift_aup; float* d_sens_drift_aup;
alloc_drift_tables(&d_drift_aup, &d_sens_drift_aup);
calibrate(h_P, f0_aup, a + eps_a, sigma, d_drift_aup, d_sens_drift_aup);
float* d_f0_aup; cudaMalloc(&d_f0_aup, N_MAT * sizeof(float));
cudaMemcpy(d_f0_aup, f0_aup, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
init_rng<<<NB, NTPB>>>(d_states, da_seed, N_PATHS);
cudaDeviceSynchronize();
float* d_out_da; cudaMalloc(&d_out_da, 2 * sizeof(float));
cudaMemset(d_out_da, 0, 2 * sizeof(float));
simulate_swaption<<<NB, NTPB>>>(d_out_da, d_states, d_P0, d_f0_aup,
                                  d_drift_aup, d_sens_drift_aup, p_aup,
                                  T, r0, d_tenor_dates, d_c, n_tenors);
cudaDeviceSynchronize();
float h_aup[2]; cudaMemcpy(h_aup, d_out_da, 2*sizeof(float), cudaMemcpyDeviceToHost);
float v_up = h_aup[0] / N_PATHS;

// ── down ──────────────────────────────────────────────────────────────────
HWParams p_adown = params(a - eps_a, sigma);
float f0_adown[N_MAT];
float* d_drift_adown; float* d_sens_drift_adown;
alloc_drift_tables(&d_drift_adown, &d_sens_drift_adown);
calibrate(h_P, f0_adown, a - eps_a, sigma, d_drift_adown, d_sens_drift_adown);
float* d_f0_adown; cudaMalloc(&d_f0_adown, N_MAT * sizeof(float));
cudaMemcpy(d_f0_adown, f0_adown, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
init_rng<<<NB, NTPB>>>(d_states, da_seed, N_PATHS);
cudaDeviceSynchronize();
cudaMemset(d_out_da, 0, 2 * sizeof(float));
simulate_swaption<<<NB, NTPB>>>(d_out_da, d_states, d_P0, d_f0_adown,
                                  d_drift_adown, d_sens_drift_adown, p_adown,
                                  T, r0, d_tenor_dates, d_c, n_tenors);
cudaDeviceSynchronize();
float h_adown[2]; cudaMemcpy(h_adown, d_out_da, 2*sizeof(float), cudaMemcpyDeviceToHost);
float v_down = h_adown[0] / N_PATHS;

float mc_fd_da = (v_up - v_down) / (2.0f * eps_a);

printf("dV/da  analytical=%.6f  MC FD=%.6f  err=%.2e\n",
       an_da, mc_fd_da, fabsf(an_da - mc_fd_da));

free_drift_tables(d_drift_aup,   d_sens_drift_aup);
free_drift_tables(d_drift_adown, d_sens_drift_adown);
cudaFree(d_f0_aup);
cudaFree(d_f0_adown);
cudaFree(d_out_da);

    

    printf("\n=== Payer Swaption %.0fYx%.0fY  |  a=%.1f  sigma=%.2f  r0=%.3f  K=%.4f ===\n",
           T, tenor_dates[n_tenors - 1] - T, a, sigma, r0, K);
    printf("%-12s  %-14s  %-14s  %-12s\n", "", "MC", "Analytical", "Error");
    printf("%-12s  %-14.6f  %-14.6f  %-12.2e\n", "Price",      h_pv[0],     an_price, fabsf(h_pv[0]     - an_price));
    printf("%-12s  %-14.6f  %-14.6f  %-12.2e\n", "Vega",       h_pv[1],     an_vega,  fabsf(h_pv[1]     - an_vega));
     printf("%-12s  %-14.6f  %-14.6f  %-12.2e\n", "Volga", h_volga[0],      an_volga, fabsf(h_volga[0]   - an_volga));
    printf("%-12s  %-14.6f  %-14.6f  %-12.2e\n", "Delta",      h_delta[0],  an_delta, fabsf(h_delta[0]  - an_delta));
    printf("%-12s  %-14.6f  %-14.6f  %-12.2e\n", "Gamma",      h_gamma[0],  an_gamma, fabsf(h_gamma[0]  - an_gamma));
 
printf("dV/da  analytical=%.6f  MC FD=%.6f  err=%.2e\n", an_da, mc_fd_da, fabsf(an_da - mc_fd_da));
float an_d2a = analytical_swaption_d2a(T, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);
float fd_d2a = (analytical_swaption_da(T, tenor_dates, n_tenors, c, h_P, f0, a + eps_a, sigma, r0)
              - analytical_swaption_da(T, tenor_dates, n_tenors, c, h_P, f0, a - eps_a, sigma, r0))
             / (2.0f * eps_a);
printf("d²V/da²  analytical=%.6f  FD=%.6f  err=%.2e\n", an_d2a, fd_d2a, fabsf(an_d2a - fd_d2a));

    free_drift_tables(d_drift, d_sens_drift);
    cudaFree(d_P0); cudaFree(d_f0); cudaFree(d_states);
    cudaFree(d_tenor_dates); cudaFree(d_c);
    free_drift_tables(d_drift_up,   d_sens_drift_up);
    free_drift_tables(d_drift_down, d_sens_drift_down);
    cudaFree(d_f0_up);
    cudaFree(d_f0_down);
    return 0;
}