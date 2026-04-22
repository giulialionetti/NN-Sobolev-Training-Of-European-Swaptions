#include "mc_calibration.cuh"
#include "mc_market_price.cuh"
#include <cstdio>
#include <ctime>

const float a     = 1.0f;
const float sigma = 0.1f;
const float r0    = 0.012f;

int main() {
    curandState* d_states;
    CUDA_CHECK(cudaMalloc(&d_states, N_PATHS * sizeof(curandState)));

    float* d_drift;
    float* d_sens_drift;
    alloc_drift_tables(&d_drift, &d_sens_drift);

    HWParams p = params(a, sigma);
    init_drift(a, sigma, r0, d_drift, d_sens_drift);

    init_rng<<<NB, NTPB>>>(d_states, time(NULL), N_PATHS);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_P;
    float* f0;
    CUDA_CHECK(cudaMallocHost(&h_P, N_MAT * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&f0,  N_MAT * sizeof(float)));

    simulate_market_price(h_P, d_states, d_drift, p, r0, N_PATHS, NB);
    CUDA_CHECK(cudaDeviceSynchronize());

    calibrate(h_P, f0, a, sigma, d_drift, d_sens_drift);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\n=== Bond Prices and Forward Rates ===\n");
    printf("%-10s  %-14s  %-14s\n", "Maturity", "P(0,T)", "f(0,T)");
    for (int i = 0; i < N_MAT; i++) {
        float t = (i + 1) * MAT_SPACING;
        printf("%-10.2f  %-14.6f  %-14.6f\n", t, h_P[i], f0[i]);
    }

    CUDA_CHECK(cudaFreeHost(h_P));
    CUDA_CHECK(cudaFreeHost(f0));
    free_drift_tables(d_drift, d_sens_drift);
    CUDA_CHECK(cudaFree(d_states));
    return 0;
}