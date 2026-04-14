#include "mc_calibration.cuh"
#include "mc_market_price.cuh"
#include "mc_swaptions.cuh"
#include "hw_swaptions.cuh"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

static float rand_uniform(float lo, float hi) {
    return lo + (hi - lo) * (float)rand() / (float)RAND_MAX;
}

static int rand_int(int lo, int hi) {
    return lo + rand() % (hi - lo + 1);
}

int main(int argc, char** argv) {
    const int   N_SAMPLES = (argc > 1) ? atoi(argv[1]) : 500;
    const char* out_path  = (argc > 2) ? argv[2]       : "data/swaption_validation.csv";

    srand((unsigned)time(NULL) + 99999);

    curandState* d_states;
    cudaMalloc(&d_states, N_PATHS * sizeof(curandState));

    float* d_P0;   cudaMalloc(&d_P0,   N_MAT * sizeof(float));
    float* d_f0;   cudaMalloc(&d_f0,   N_MAT * sizeof(float));
    float* d_out2; cudaMalloc(&d_out2, 2     * sizeof(float));
    float* d_out1; cudaMalloc(&d_out1, 1     * sizeof(float));

    float* d_drift;
    float* d_sens_drift;
    alloc_drift_tables(&d_drift, &d_sens_drift);

    
    const float eps_v = 0.01f;
    float* d_drift_up;   float* d_sens_drift_up;
    float* d_drift_down; float* d_sens_drift_down;
    alloc_drift_tables(&d_drift_up,   &d_sens_drift_up);
    alloc_drift_tables(&d_drift_down, &d_sens_drift_down);
    float* d_f0_up;   cudaMalloc(&d_f0_up,   N_MAT * sizeof(float));
    float* d_f0_down; cudaMalloc(&d_f0_down, N_MAT * sizeof(float));

    float* d_tenor_dates; cudaMalloc(&d_tenor_dates, MAX_TENORS * sizeof(float));
    float* d_c;           cudaMalloc(&d_c,            MAX_TENORS * sizeof(float));

    FILE* fp = fopen(out_path, "w");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", out_path); return 1; }
    fprintf(fp, "a,sigma,r0,T,swap_length,K,"
                "mc_price,mc_vega,mc_delta,mc_gamma,mc_volga,"
                "an_price,an_vega,an_volga,an_delta,an_gamma\n");

    unsigned long base_seed = (unsigned long)time(NULL) + 99999UL;
    const float   eps_r     = 0.001f;

    for (int s = 0; s < N_SAMPLES; s++) {

        float a     = rand_uniform(0.10f,  2.00f);
        float sigma = rand_uniform(0.01f,  0.30f);
        float r0    = rand_uniform(0.001f, 0.05f);

        int   swap_length = rand_int(1, 4);
        float T_max       = 9.0f - (float)swap_length;
        float T           = rand_uniform(1.0f, T_max);

        int   n_tenors = swap_length;
        float tenor_dates[MAX_TENORS];
        for (int i = 0; i < n_tenors; i++)
            tenor_dates[i] = T + (float)(i + 1);

      
        HWParams p = params(a, sigma);

        init_drift(a, sigma, r0, d_drift, d_sens_drift);

        init_rng<<<NB, NTPB>>>(d_states,
                                base_seed + (unsigned long)s,
                                N_PATHS);
        cudaDeviceSynchronize();

        float h_P[N_MAT];
        simulate_market_price(h_P, d_states, d_drift, p, r0, N_PATHS, NB);

        float f0[N_MAT];
        calibrate(h_P, f0, a, sigma, d_drift, d_sens_drift);

        
        float K_atm     = par_swap_rate(T, tenor_dates, n_tenors, h_P);
        float moneyness = rand_uniform(0.80f, 1.20f);
        float K         = K_atm * moneyness;

        float c[MAX_TENORS];
        for (int i = 0; i < n_tenors; i++)
            c[i] = K;
        c[n_tenors - 1] += 1.0f;

        cudaMemcpy(d_P0,          h_P,         N_MAT    * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_f0,          f0,          N_MAT    * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tenor_dates, tenor_dates, n_tenors * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_c,           c,           n_tenors * sizeof(float), cudaMemcpyHostToDevice);

     
        init_rng<<<NB, NTPB>>>(d_states,
                                base_seed + (unsigned long)s
                                          + 1UL * (unsigned long)N_SAMPLES,
                                N_PATHS);
        cudaDeviceSynchronize();
        cudaMemset(d_out2, 0, 2 * sizeof(float));
        simulate_swaption<<<NB, NTPB>>>(d_out2, d_states, d_P0, d_f0,
                                         d_drift, d_sens_drift, p,
                                         T, r0,
                                         d_tenor_dates, d_c, n_tenors);
        cudaDeviceSynchronize();

        float h_pv[2];
        cudaMemcpy(h_pv, d_out2, 2 * sizeof(float), cudaMemcpyDeviceToHost);
        float mc_price = h_pv[0] / N_PATHS;
        float mc_vega  = h_pv[1] / N_PATHS;

     
        init_rng<<<NB, NTPB>>>(d_states,
                                base_seed + (unsigned long)s
                                          + 2UL * (unsigned long)N_SAMPLES,
                                N_PATHS);
        cudaDeviceSynchronize();
        cudaMemset(d_out1, 0, sizeof(float));
        simulate_swaption_delta<<<NB, NTPB>>>(d_out1, d_states, d_P0, d_f0,
                                               d_drift, p,
                                               T, r0,
                                               d_tenor_dates, d_c, n_tenors);
        cudaDeviceSynchronize();

        float h_delta_mc;
        cudaMemcpy(&h_delta_mc, d_out1, sizeof(float), cudaMemcpyDeviceToHost);
        float mc_delta = h_delta_mc / N_PATHS;

       
        init_rng<<<NB, NTPB>>>(d_states,
                                base_seed + (unsigned long)s
                                          + 3UL * (unsigned long)N_SAMPLES,
                                N_PATHS);
        cudaDeviceSynchronize();
        cudaMemset(d_out1, 0, sizeof(float));
        simulate_swaption_gamma<<<NB, NTPB>>>(d_out1, d_states, d_P0, d_f0,
                                               d_drift, p,
                                               T, r0,
                                               d_tenor_dates, d_c, n_tenors,
                                               eps_r);
        cudaDeviceSynchronize();

        float h_gamma_mc;
        cudaMemcpy(&h_gamma_mc, d_out1, sizeof(float), cudaMemcpyDeviceToHost);
        float mc_gamma = h_gamma_mc / (N_PATHS * eps_r * eps_r);

        
        HWParams p_vup   = params(a, sigma + eps_v);
        HWParams p_vdown = params(a, sigma - eps_v);
        float f0_vup[N_MAT], f0_vdown[N_MAT];

        calibrate(h_P, f0_vup,   a, sigma + eps_v, d_drift_up,   d_sens_drift_up);
        calibrate(h_P, f0_vdown, a, sigma - eps_v, d_drift_down, d_sens_drift_down);

        cudaMemcpy(d_f0_up,   f0_vup,   N_MAT * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_f0_down, f0_vdown, N_MAT * sizeof(float), cudaMemcpyHostToDevice);

        init_rng<<<NB, NTPB>>>(d_states,
                                base_seed + (unsigned long)s
                                          + 4UL * (unsigned long)N_SAMPLES,
                                N_PATHS);
        cudaDeviceSynchronize();
        cudaMemset(d_out1, 0, sizeof(float));
        simulate_swaption_volga<<<NB, NTPB>>>(d_out1, d_states,
                                               d_P0,
                                               d_f0, d_f0_up, d_f0_down,
                                               d_drift, d_drift_up, d_drift_down,
                                               p, p_vup, p_vdown,
                                               T, r0,
                                               d_tenor_dates, d_c, n_tenors,
                                               eps_v);
        cudaDeviceSynchronize();
        float h_volga_mc;
        cudaMemcpy(&h_volga_mc, d_out1, sizeof(float), cudaMemcpyDeviceToHost);
        float mc_volga = h_volga_mc / N_PATHS;

        
        float an_price = analytical_swaption      (T, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);
        float an_vega  = analytical_swaption_vega (T, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);
        float an_volga = analytical_swaption_volga(T, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);
        float an_delta = analytical_swaption_delta(T, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);
        float an_gamma = analytical_swaption_gamma(T, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);

        fprintf(fp, "%.6f,%.6f,%.6f,%.6f,%d,%.6f,"
                    "%.8f,%.8f,%.8f,%.8f,%.8f,"
                    "%.8f,%.8f,%.8f,%.8f,%.8f\n",
                a, sigma, r0, T, swap_length, K,
                mc_price, mc_vega, mc_delta, mc_gamma, mc_volga,
                an_price, an_vega, an_volga, an_delta, an_gamma);

        if ((s + 1) % 100 == 0)
            fprintf(stderr, "  %d / %d\n", s + 1, N_SAMPLES);
    }

    fclose(fp);
    fprintf(stderr, "Done: %d samples written to %s\n", N_SAMPLES, out_path);

    free_drift_tables(d_drift,      d_sens_drift);
    free_drift_tables(d_drift_up,   d_sens_drift_up);
    free_drift_tables(d_drift_down, d_sens_drift_down);
    cudaFree(d_states);
    cudaFree(d_P0);   cudaFree(d_f0);
    cudaFree(d_out2); cudaFree(d_out1);
    cudaFree(d_f0_up); cudaFree(d_f0_down);
    cudaFree(d_tenor_dates); cudaFree(d_c);
    return 0;
}