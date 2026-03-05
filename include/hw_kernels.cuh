#ifndef HW_KERNELS_CUH
#define HW_KERNELS_CUH

#include <cmath>

__host__ __device__ inline float P0T(float T_maturity, float r0){
    return expf(-r0 * T_maturity);
}

__host__ __device__ inline float BtT(float t, float T_maturity, float a){
    return (1.0f - expf(-a * (T_maturity - t))) / a;
}

__host__ __device__ inline float AtT(float t, float T_maturity, float a, float sigma, float r0){
    float B_t_T      = BtT(t, T_maturity, a);
    float fwd_discount = expf(-r0 * (T_maturity - t));
    float convexity_adj = -(sigma * sigma) * (1.0f - expf(-2.0f * a * t)) / (4.0f * a) * B_t_T * B_t_T;
    return fwd_discount * expf(B_t_T * r0 + convexity_adj);
}

__host__ __device__ inline float PtT(float t, float T_maturity, float rt, float a, float sigma, float r0){
    float A = AtT(t, T_maturity, a, sigma, r0);
    float B = BtT(t, T_maturity, a);
    return A * expf(-B * rt);
}

__host__ __device__ inline float ZBC(float t, float T_maturity, float S, float K, float rt, float a, float sigma, float r0){
    float P_t_S  = PtT(t, S, rt, a, sigma, r0);
    float P_t_T  = PtT(t, T_maturity, rt, a, sigma, r0);
    float B_T_S  = BtT(T_maturity, S, a);
    float sigma_p = sigma * sqrtf((1.0f - expf(-2.0f * a * (T_maturity - t))) / (2.0f * a)) * B_T_S;
    float h      = (1.0f / sigma_p) * logf(P_t_S / (P_t_T * K)) + sigma_p / 2.0f;
    return P_t_S * normcdff(h) - K * P_t_T * normcdff(h - sigma_p);
}

__host__ __device__ inline float ZBP(float t, float T_maturity, float S, float K, float rt, float a, float sigma, float r0){
    float P_t_T  = PtT(t, T_maturity, rt, a, sigma, r0);
    float P_t_S  = PtT(t, S, rt, a, sigma, r0);
    float B_T_S  = BtT(T_maturity, S, a);
    float sigma_p = sigma * sqrtf((1.0f - expf(-2.0f * a * (T_maturity - t))) / (2.0f * a)) * B_T_S;
    float h      = (1.0f / sigma_p) * logf(P_t_S / (P_t_T * K)) + sigma_p / 2.0f;
    return K * P_t_T * normcdff(-h + sigma_p) - P_t_S * normcdff(-h);
}

__host__ __device__ inline float vega_ZBC(float t, float T_maturity, float S, float K, float rt, float a, float sigma, float r0){
    float B_T_S  = BtT(T_maturity, S, a);
    float P_t_S  = PtT(t, S, rt, a, sigma, r0);
    float P_t_T  = PtT(t, T_maturity, rt, a, sigma, r0);
    float sigma_p = sigma * sqrtf((1.0f - expf(-2.0f * a * (T_maturity - t))) / (2.0f * a)) * B_T_S;
    float h      = (1.0f / sigma_p) * logf(P_t_S / (P_t_T * K)) + sigma_p / 2.0f;
    float phi_h  = expf(-h * h * 0.5f) / sqrtf(2.0f * 3.14159265f);
    return P_t_S * phi_h * (sigma_p / sigma);
}

__host__ __device__ inline float vega_ZBP(float t, float T_maturity, float S, float K, float rt, float a, float sigma, float r0){
    float B_T_S  = BtT(T_maturity, S, a);
    float P_t_S  = PtT(t, S, rt, a, sigma, r0);
    float P_t_T  = PtT(t, T_maturity, rt, a, sigma, r0);
    float sigma_p = sigma * sqrtf((1.0f - expf(-2.0f * a * (T_maturity - t))) / (2.0f * a)) * B_T_S;
    float h      = (1.0f / sigma_p) * logf(P_t_S / (P_t_T * K)) + sigma_p / 2.0f;
    float phi_h  = expf(-h * h * 0.5f) / sqrtf(2.0f * 3.14159265f);
    return P_t_S * phi_h * (sigma_p / sigma);
}

__host__ __device__ inline float delta_ZBC(float t, float T_maturity, float S, float K, float rt, float a, float sigma, float r0){
    float B_t_S  = BtT(t, S, a);
    float B_t_T  = BtT(t, T_maturity, a);
    float B_T_S  = BtT(T_maturity, S, a);
    float P_t_S  = PtT(t, S, rt, a, sigma, r0);
    float P_t_T  = PtT(t, T_maturity, rt, a, sigma, r0);
    float sigma_p = sigma * sqrtf((1.0f - expf(-2.0f * a * (T_maturity - t))) / (2.0f * a)) * B_T_S;
    float h      = (1.0f / sigma_p) * logf(P_t_S / (P_t_T * K)) + sigma_p / 2.0f;
    return -(B_t_S * P_t_S * normcdff(h)) + (K * B_t_T * P_t_T * normcdff(h - sigma_p));
}

__host__ __device__ inline float delta_ZBP(float t, float T_maturity, float S, float K, float rt, float a, float sigma, float r0){
    float B_t_S  = BtT(t, S, a);
    float B_t_T  = BtT(t, T_maturity, a);
    float B_T_S  = BtT(T_maturity, S, a);
    float P_t_S  = PtT(t, S, rt, a, sigma, r0);
    float P_t_T  = PtT(t, T_maturity, rt, a, sigma, r0);
    float sigma_p = sigma * sqrtf((1.0f - expf(-2.0f * a * (T_maturity - t))) / (2.0f * a)) * B_T_S;
    float h      = (1.0f / sigma_p) * logf(P_t_S / (P_t_T * K)) + sigma_p / 2.0f;
    return -(K * B_t_T * P_t_T * normcdff(-h + sigma_p)) + (B_t_S * P_t_S * normcdff(-h));
}

__host__ __device__ inline float interpolate(const float* data, float T, float spacing, int n_mat){
    int idx = (int)(T / spacing);
    if(idx >= n_mat - 1) return data[n_mat - 1];
    float t0    = idx * spacing;
    float alpha = (T - t0) / spacing;
    return data[idx] * (1.0f - alpha) + data[idx + 1] * alpha;
}

__host__ __device__ inline float AtT_market(float t, float T_maturity, float a, float sigma,
                                             const float* P_market, const float* f_market,
                                             float mat_spacing, int n_mat){
    float B     = BtT(t, T_maturity, a);
    float P0T   = interpolate(P_market, T_maturity, mat_spacing, n_mat);
    float P0t   = (t == 0.0f) ? 1.0f : interpolate(P_market, t, mat_spacing, n_mat);
    float f0t   = interpolate(f_market, t, mat_spacing, n_mat);
    float ratio = P0T / P0t;
    float term2 = B * f0t;
    float term3 = (sigma * sigma / (4.0f * a)) * (1.0f - expf(-2.0f * a * t)) * B * B;
    return ratio * expf(term2 - term3);
}

__host__ __device__ inline float PtT_market(float t, float T_maturity, float rt, float a, float sigma,
                                             const float* P_market, const float* f_market,
                                             float mat_spacing, int n_mat){
    float A = AtT_market(t, T_maturity, a, sigma, P_market, f_market, mat_spacing, n_mat);
    float B = BtT(t, T_maturity, a);
    return A * expf(-B * rt);
}

__host__ __device__ inline float ZBC_market(float t, float T_maturity, float S, float K, float rt,
                                             float a, float sigma,
                                             const float* P_market, const float* f_market,
                                             float mat_spacing, int n_mat){
    float P_t_S  = PtT_market(t, S, rt, a, sigma, P_market, f_market, mat_spacing, n_mat);
    float P_t_T  = PtT_market(t, T_maturity, rt, a, sigma, P_market, f_market, mat_spacing, n_mat);
    float B_T_S  = BtT(T_maturity, S, a);
    float sigma_p = sigma * sqrtf((1.0f - expf(-2.0f * a * (T_maturity - t))) / (2.0f * a)) * B_T_S;
    float h      = (1.0f / sigma_p) * logf(P_t_S / (P_t_T * K)) + sigma_p / 2.0f;
    return P_t_S * normcdff(h) - K * P_t_T * normcdff(h - sigma_p);
}

__host__ __device__ inline float vega_ZBC_market(float t, float T_maturity, float S, float K, float rt,
                                                  float a, float sigma,
                                                  const float* P_market, const float* f_market,
                                                  float mat_spacing, int n_mat){
    float B_T_S  = BtT(T_maturity, S, a);
    float P_t_S  = PtT_market(t, S, rt, a, sigma, P_market, f_market, mat_spacing, n_mat);
    float P_t_T  = PtT_market(t, T_maturity, rt, a, sigma, P_market, f_market, mat_spacing, n_mat);
    float sigma_p = sigma * sqrtf((1.0f - expf(-2.0f * a * (T_maturity - t))) / (2.0f * a)) * B_T_S;
    float h      = (1.0f / sigma_p) * logf(P_t_S / (P_t_T * K)) + sigma_p / 2.0f;
    float phi_h  = expf(-h * h * 0.5f) / sqrtf(2.0f * 3.14159265f);
    return P_t_S * phi_h * (sigma_p / sigma);
}


#endif // HW_KERNELS_CUH