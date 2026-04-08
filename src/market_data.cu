#include "mc_calibration.cuh"
#include "mc_market_price.cuh"
#include <cstdio>
#include <ctime>

const float a     = 1.0f;
const float sigma = 0.1f;
const float r0    = 0.012f;


// nvcc -o print_curve src/print_curve.cu -I include -lcurand
// ./print_curve


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

    printf("\n=== Bond Prices and Forward Rates ===\n");
    printf("%-10s  %-14s  %-14s\n", "Maturity", "P(0,T)", "f(0,T)");
    for(int i = 0; i < N_MAT; i++){
        float t = (i + 1) * MAT_SPACING;
        printf("%-10.2f  %-14.6f  %-14.6f\n", t, h_P[i], f0[i]);
    }

    cudaFree(d_states);
    return 0;
}