#ifndef MC_DISCOUNT_CURVE_CUH
#define MC_DISCOUNT_CURVE_CUH

#include "mc_engine.cuh"


// this kernel estimates the initial discount curve P^M(0, T) for
// T in {delta, 2*delta, 3*delta, ... N*delta} where delta = MAT_SPACING
// from the risk neutral pricing formula the Zero Coupon Bond price is:
// P(0, T) = E^Q[exp(-integral_0_T(r(t)dt))]
// the kernel estimates this expectation by MC.
// The short rate is evolved in Nsteps discrete steps of size dt and
// the discount factor integral is accumulated using the trapezoid rule

// once the estimated price is obtained the instantanous forward rate 
// is recovered by finite differences with one sided differneces at the
// boundaries 

__global__ void mc_P0T(float* P_estimator, curandState* states){

    int path_id = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float s_P[N_MAT];

    // each thread clears N_MAT/NTPB slots
    for(int m = threadIdx.x; m < N_MAT; m += blockDim.x)
        s_P[m] = 0.0f;
    __syncthreads();

    if(path_id < N_PATHS){
        curandState local_state = states[path_id];

        float r_step_i = device_r0;
        float discount_factor_integral = 0.0f;
        int maturity_index = 0;

        for(int i = 0; i < N_STEPS; i++){
            float G = curand_normal(&local_state);
            evolve_short_rate(r_step_i, discount_factor_integral, device_drift_table[i], G);

            if((i + 1) % SAVE_STRIDE == 0){
                atomicAdd(&s_P[maturity_index], expf(-discount_factor_integral));
                maturity_index++;
            }
        }
        states[path_id] = local_state;
    }

    __syncthreads();

    // one atomicAdd per maturity per block to global memory
    for(int m = threadIdx.x; m < N_MAT; m += blockDim.x)
        atomicAdd(&P_estimator[m], s_P[m]);
}


void compute_market_data(float* h_P, float* h_f, curandState* d_states){
    float* d_P_sum;
    cudaMalloc(&d_P_sum, N_MAT * sizeof(float));
    cudaMemset(d_P_sum, 0, N_MAT * sizeof(float));

    init_device_constants(host_sigma, CurveType::PIECEWISE_LINEAR);
    mc_P0T<<<NB, NTPB>>>(d_P_sum, d_states);
    cudaDeviceSynchronize();

    cudaMemcpy(h_P, d_P_sum, N_MAT * sizeof(float), cudaMemcpyDeviceToHost);
    for(int k = 0; k < N_MAT; k++)
        h_P[k] /= N_PATHS;

    h_f[0] = -(logf(h_P[1]) - logf(1.0f)) / MAT_SPACING;
    for(int k = 1; k < N_MAT - 1; k++)
        h_f[k] = -(logf(h_P[k+1]) - logf(h_P[k-1])) / (2.0f * MAT_SPACING);
    h_f[N_MAT-1] = -(logf(h_P[N_MAT-1]) - logf(h_P[N_MAT-2])) / MAT_SPACING;

    LOG_INFO("=== Market Data (Piecewise theta) ===");
    LOG_INFO("P(0,1)=%.6f  P(0,5)=%.6f  P(0,10)=%.6f", h_P[9], h_P[49], h_P[99]);
    LOG_INFO("f(0,1)=%.6f  f(0,5)=%.6f  f(0,10)=%.6f", h_f[9], h_f[49], h_f[99]);

    cudaFree(d_P_sum);
}

#endif // MC_DISCOUNT_CURVE_CUH