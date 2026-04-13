#ifndef MC_MARKET_PRICE_CUH
#define MC_MARKET_PRICE_CUH

#include "mc_engine.cuh"

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d — %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

__global__ void market_price(float* P0_sum,
                              curandState* states,
                              const float* d_drift,
                              HWParams p,
                              float r0,
                              int n_paths) {
    int path_id = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float s_P[N_MAT];
    for (int m = threadIdx.x; m < N_MAT; m += blockDim.x)
        s_P[m] = 0.0f;
    __syncthreads();

    if (path_id < n_paths) {
        curandState local_state = states[path_id];
        float r                 = r0;
        float discount_integral = 0.0f;
        int   maturity_index    = 0;

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

// Launch only — returns immediately, no sync.
inline void launch_market_price(float* d_P0_sum,
                                 curandState* d_states,
                                 const float* d_drift,
                                 HWParams p,
                                 float r0,
                                 int n_paths,
                                 int nb,
                                 cudaStream_t stream) {
    cudaMemsetAsync(d_P0_sum, 0, N_MAT * sizeof(float), stream);
    market_price<<<nb, NTPB, 0, stream>>>(d_P0_sum, d_states, d_drift, p, r0, n_paths);
}

// Collect — async memcpy on stream then syncs only that stream.
// Using cudaMemcpyAsync avoids the implicit full-device barrier
// that cudaMemcpy (non-async) imposes across all streams.
inline void collect_market_price(float* h_P,
                                  const float* d_P0_sum,
                                  int n_paths,
                                  cudaStream_t stream) {
    cudaMemcpyAsync(h_P, d_P0_sum, N_MAT * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    for (int i = 0; i < N_MAT; i++)
        h_P[i] /= n_paths;
}

// Synchronous convenience wrapper for serial callers.
inline void simulate_market_price(float* h_P,
                                   curandState* d_states,
                                   const float* d_drift,
                                   HWParams p,
                                   float r0,
                                   int n_paths,
                                   int nb) {
    float* d_P0_sum;
    cudaMalloc(&d_P0_sum, N_MAT * sizeof(float));
    launch_market_price(d_P0_sum, d_states, d_drift, p, r0, n_paths, nb, 0);
    cudaDeviceSynchronize();
    collect_market_price(h_P, d_P0_sum, n_paths, 0);
    cudaFree(d_P0_sum);
}

#endif // MC_MARKET_PRICE_CUH