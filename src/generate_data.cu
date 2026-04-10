/*
 * generate_data.cu
 *
 * Generates a CSV training dataset for swaption pricing with sensitivities.
 * For each sample, parameters (a, sigma, r0, T, swap_length, K) are drawn
 * randomly, the Hull-White model is calibrated against a simulated yield
 * curve, and the analytical formulas produce:
 *   - price  : analytical swaption price
 *   - vega   : d(price)/d(sigma)
 *   - volga  : d²(price)/d(sigma)²
 *   - delta  : d(price)/d(r0)
 *   - gamma  : d²(price)/d(r0)²
 *
 * Swaption structure: payer swaption with annual payment dates
 *   T+1, T+2, ..., T+swap_length  (swap_length in {1,2,3,4} years)
 * Constraint: T + swap_length <= 9 (fits within the bond-price grid).
 *
 * All state is local to the loop — no __constant__ memory, so this loop
 * can be trivially parallelised across CPU threads or CUDA streams.
 *
 * Usage:
 *   ./bin/generate_data [N_SAMPLES] [output_csv]
 *   ./bin/generate_data 2000 data/swaption_data.csv
 */

#include "mc_calibration.cuh"
#include "mc_market_price.cuh"
#include "hw_swaptions.cuh"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

static float rand_uniform(float lo, float hi) {
    return lo + (hi - lo) * (float)rand() / (float)RAND_MAX;
}

static int rand_int(int lo, int hi) {   /* inclusive on both ends */
    return lo + rand() % (hi - lo + 1);
}

int main(int argc, char** argv) {
    const int   N_SAMPLES = (argc > 1) ? atoi(argv[1]) : 2000;
    const char* out_path  = (argc > 2) ? argv[2]       : "data/swaption_data.csv";

    srand((unsigned)time(NULL));

    /* ── GPU allocations (reused across all samples) ────────────────────── */
    curandState* d_states;
    cudaMalloc(&d_states, N_PATHS * sizeof(curandState));

    /* Per-sample drift tables: allocated once, reused every iteration.      */
    float* d_drift;
    float* d_sens_drift;
    alloc_drift_tables(&d_drift, &d_sens_drift);

    /* ── Output CSV ─────────────────────────────────────────────────────── */
    FILE* fp = fopen(out_path, "w");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", out_path); return 1; }
    fprintf(fp, "a,sigma,r0,T,swap_length,K,price,vega,volga,delta,gamma\n");

    unsigned long base_seed = (unsigned long)time(NULL);

    for (int s = 0; s < N_SAMPLES; s++) {

        /* ── Random HW parameters ───────────────────────────────────────── */
        float a     = rand_uniform(0.10f,  2.00f);
        float sigma = rand_uniform(0.01f,  0.30f);
        float r0    = rand_uniform(0.001f, 0.05f);

        /* ── Random swaption structure (annual payments) ────────────────── */
        int   swap_length = rand_int(1, 4);
        float T_max       = 9.0f - (float)swap_length;
        float T           = rand_uniform(1.0f, T_max);

        int   n_tenors = swap_length;
        float tenor_dates[MAX_TENORS];
        for (int i = 0; i < n_tenors; i++)
            tenor_dates[i] = T + (float)(i + 1);

        /* ── Calibrate to a simulated initial yield curve ───────────────── */
        HWParams p = params(a, sigma);

        init_drift(a, sigma, r0, d_drift, d_sens_drift);

        init_rng<<<NB, NTPB>>>(d_states, base_seed + (unsigned long)s);
        cudaDeviceSynchronize();

        float h_P[N_MAT];
        simulate_market_price(h_P, d_states, d_drift, p, r0);

        float f0[N_MAT];
        calibrate(h_P, f0, a, sigma, d_drift, d_sens_drift);

        /* ── Strike: ATM par rate scaled by a random moneyness ─────────── */
        float K_atm     = par_swap_rate(T, tenor_dates, n_tenors, h_P);
        float moneyness = rand_uniform(0.80f, 1.20f);
        float K         = K_atm * moneyness;

        /* ── Cash-flow vector (annual unit year-fractions) ──────────────── */
        float c[MAX_TENORS];
        for (int i = 0; i < n_tenors; i++)
            c[i] = K;
        c[n_tenors - 1] += 1.0f;

        /* ── Analytical price and sensitivities ─────────────────────────── */
        float price = analytical_swaption      (T, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);
        float vega  = analytical_swaption_vega (T, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);
        float volga = analytical_swaption_volga(T, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);
        float delta = analytical_swaption_delta(T, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);
        float gamma = analytical_swaption_gamma(T, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);

        fprintf(fp, "%.6f,%.6f,%.6f,%.6f,%d,%.6f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                a, sigma, r0, T, swap_length, K,
                price, vega, volga, delta, gamma);

        if ((s + 1) % 100 == 0)
            fprintf(stderr, "  %d / %d\n", s + 1, N_SAMPLES);
    }

    fclose(fp);
    fprintf(stderr, "Done: %d samples written to %s\n", N_SAMPLES, out_path);

    free_drift_tables(d_drift, d_sens_drift);
    cudaFree(d_states);
    return 0;
}
