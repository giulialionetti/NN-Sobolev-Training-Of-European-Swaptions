#ifndef MC_MARKET_PRICE_CUH
#define MC_MARKET_PRICE_CUH

#include "mc_engine.cuh"


__global__ void market_price(float* P0_sum,
                              curandState* states,
                              const float* d_drift,
                              HWParams p,
                              float r0) {
    int path_id = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float s_P[N_MAT];
    for (int m = threadIdx.x; m < N_MAT; m += blockDim.x)
        s_P[m] = 0.0f;
    __syncthreads();

    if (path_id < N_PATHS) {
        curandState local_state    = states[path_id];
        float r                    = r0;
        float discount_integral    = 0.0f;
        int   maturity_index       = 0;

        for (int i = 0; i < N_STEPS; i++) {
            float G = curand_normal(&local_state);
            evolve_short_rate(r, discount_integral, d_drift[i], G, p);

            if ((i + 1) % SAVE_STRIDE == 0) {
                atomicAdd(&s_P[maturity_index], expf(-discount_integral));
                maturity_index++;
            }
        }
        states[path_id] = local_state;
    }

    __syncthreads();
    for (int m = threadIdx.x; m < N_MAT; m += blockDim.x)
        atomicAdd(&P0_sum[m], s_P[m]);
}

inline void simulate_market_price(float* h_P,
                                   curandState* d_states,
                                   const float* d_drift,
                                   HWParams p,
                                   float r0) {
    float* d_P0_sum;
    cudaMalloc(&d_P0_sum, N_MAT * sizeof(float));
    cudaMemset(d_P0_sum, 0, N_MAT * sizeof(float));

    market_price<<<NB, NTPB>>>(d_P0_sum, d_states, d_drift, p, r0);
    cudaDeviceSynchronize();

    cudaMemcpy(h_P, d_P0_sum, N_MAT * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N_MAT; i++)
        h_P[i] /= N_PATHS;

    cudaFree(d_P0_sum);
}

#endif // MC_MARKET_PRICE_CUH
