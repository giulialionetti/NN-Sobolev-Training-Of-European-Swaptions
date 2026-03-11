#ifndef HW_KERNELS_CUH
#define HW_KERNELS_CUH

#include <cmath>

__host__ __device__ inline float P0T(float T_maturity, float r0){
    return expf(-r0 * T_maturity);
}

__host__ __device__ inline float BtT(float t, float T_maturity, float a){
    return (1.0f - expf(-a * (T_maturity - t))) / a;
}

__host__ __device__ inline float interpolate(const float* data, float T_maturity,
                                              float mat_spacing, int n_mat){
    int idx = (int)(T_maturity / mat_spacing);
    if(idx >= n_mat - 1) return data[n_mat - 1];
    float t0    = idx * mat_spacing;
    float alpha = (T_maturity - t0) / mat_spacing;
    return data[idx] * (1.0f - alpha) + data[idx + 1] * alpha;
}

// ============================================================================
// Curve policies
// Each policy exposes a single method: P(t, T_maturity, rt) -> float
// ============================================================================

// Flat curve: uses analytical A(t,T) with constant r0
// P(t,T) = A(t,T) * exp(-B(t,T) * r(t))
// A(t,T) derived under flat curve assumption f(0,T) = r0
struct FlatCurve {
    float a, sigma, r0;

    __host__ __device__ float P(float t, float T_maturity, float rt) const {
        float B_t_T             = BtT(t, T_maturity, a);
        float forward_discount  = expf(-r0 * (T_maturity - t));
        float convexity_adj     = -(sigma * sigma) * (1.0f - expf(-2.0f * a * t))
                                  / (4.0f * a) * B_t_T * B_t_T;
        float A_t_T             = forward_discount * expf(B_t_T * r0 + convexity_adj);
        return A_t_T * expf(-B_t_T * rt);
    }
};

// Market curve: uses real P(0,T) and f(0,T) arrays with linear interpolation
// Brigo-Mercurio eq. 3.39
struct MarketCurve {
    float a, sigma;
    const float* P_market;
    const float* f_market;
    float mat_spacing;
    int   n_mat;

    __host__ __device__ float P(float t, float T_maturity, float rt) const {
        float B_t_T              = BtT(t, T_maturity, a);
        float P_zero_T_maturity  = interpolate(P_market, T_maturity, mat_spacing, n_mat);
        float P_zero_t           = (t == 0.0f) ? 1.0f
                                               : interpolate(P_market, t, mat_spacing, n_mat);
        float f_zero_t           = interpolate(f_market, t, mat_spacing, n_mat);
        float forward_discount   = P_zero_T_maturity / P_zero_t;
        float drift_adjustment   = B_t_T * f_zero_t;
        float convexity_adj      = (sigma * sigma / (4.0f * a))
                                   * (1.0f - expf(-2.0f * a * t)) * B_t_T * B_t_T;
        float A_t_T              = forward_discount * expf(drift_adjustment - convexity_adj);
        return A_t_T * expf(-B_t_T * rt);
    }
};

template<typename Curve>
__host__ __device__ inline float ZBC_impl(float t, float T_maturity, float S, float K,
                                           float rt, float a, float sigma,
                                           const Curve& curve){
    float bond_price_t_S  = curve.P(t, S, rt);
    float bond_price_t_T  = curve.P(t, T_maturity, rt);
    float B_T_maturity_S  = BtT(T_maturity, S, a);
    float sigma_p         = sigma * sqrtf((1.0f - expf(-2.0f * a * (T_maturity - t)))
                                          / (2.0f * a)) * B_T_maturity_S;
    float h               = (1.0f / sigma_p) * logf(bond_price_t_S
                             / (bond_price_t_T * K)) + sigma_p / 2.0f;
    return bond_price_t_S * normcdff(h) - K * bond_price_t_T * normcdff(h - sigma_p);
}

template<typename Curve>
__host__ __device__ inline float ZBP_impl(float t, float T_maturity, float S, float K,
                                           float rt, float a, float sigma,
                                           const Curve& curve){
    float bond_price_t_S  = curve.P(t, S, rt);
    float bond_price_t_T  = curve.P(t, T_maturity, rt);
    float B_T_maturity_S  = BtT(T_maturity, S, a);
    float sigma_p         = sigma * sqrtf((1.0f - expf(-2.0f * a * (T_maturity - t)))
                                          / (2.0f * a)) * B_T_maturity_S;
    float h               = (1.0f / sigma_p) * logf(bond_price_t_S
                             / (bond_price_t_T * K)) + sigma_p / 2.0f;
    return K * bond_price_t_T * normcdff(-h + sigma_p)
           - bond_price_t_S * normcdff(-h);
}

template<typename Curve>
__host__ __device__ inline float vega_ZBC_impl(float t, float T_maturity, float S, float K,
                                                float rt, float a, float sigma,
                                                const Curve& curve){
    float bond_price_t_S  = curve.P(t, S, rt);
    float bond_price_t_T  = curve.P(t, T_maturity, rt);
    float B_T_maturity_S  = BtT(T_maturity, S, a);
    float B_t_S           = BtT(t, S, a);
    float B_t_T           = BtT(t, T_maturity, a);
    float sigma_p         = sigma * sqrtf((1.0f - expf(-2.0f * a * (T_maturity - t)))
                                          / (2.0f * a)) * B_T_maturity_S;
    float h               = (1.0f / sigma_p) * logf(bond_price_t_S
                             / (bond_price_t_T * K)) + sigma_p / 2.0f;
    float phi_h           = expf(-h * h * 0.5f) / sqrtf(2.0f * 3.14159265f);

    // d(sigma_p)/dsigma
    float dsigmap_dsigma  = sigma_p / sigma;

    // d(P(t,S))/dsigma and d(P(t,T))/dsigma from convexity term in A
    float conv            = (1.0f - expf(-2.0f * a * t)) / (2.0f * a);
    float dPS_dsigma      = -bond_price_t_S * sigma * conv * B_t_S * B_t_S;
    float dPT_dsigma      = -bond_price_t_T * sigma * conv * B_t_T * B_t_T;

    return bond_price_t_S * phi_h * dsigmap_dsigma
         + normcdff(h)         * dPS_dsigma
         - K * normcdff(h - sigma_p) * dPT_dsigma;
}

template<typename Curve>
__host__ __device__ inline float vega_ZBP_impl(float t, float T_maturity, float S, float K,
                                                float rt, float a, float sigma,
                                                const Curve& curve){
    // vega ZBP = vega ZBC by put-call parity
    return vega_ZBC_impl(t, T_maturity, S, K, rt, a, sigma, curve);
}

template<typename Curve>
__host__ __device__ inline float delta_ZBC_impl(float t, float T_maturity, float S, float K,
                                                 float rt, float a, float sigma,
                                                 const Curve& curve){
    float B_t_S           = BtT(t, S, a);
    float B_t_T           = BtT(t, T_maturity, a);
    float bond_price_t_S  = curve.P(t, S, rt);
    float bond_price_t_T  = curve.P(t, T_maturity, rt);
    float B_T_maturity_S  = BtT(T_maturity, S, a);
    float sigma_p         = sigma * sqrtf((1.0f - expf(-2.0f * a * (T_maturity - t)))
                                          / (2.0f * a)) * B_T_maturity_S;
    float h               = (1.0f / sigma_p) * logf(bond_price_t_S
                             / (bond_price_t_T * K)) + sigma_p / 2.0f;
    return -(B_t_S * bond_price_t_S * normcdff(h))
           + (K * B_t_T * bond_price_t_T * normcdff(h - sigma_p));
}

template<typename Curve>
__host__ __device__ inline float delta_ZBP_impl(float t, float T_maturity, float S, float K,
                                                 float rt, float a, float sigma,
                                                 const Curve& curve){
    float B_t_S           = BtT(t, S, a);
    float B_t_T           = BtT(t, T_maturity, a);
    float bond_price_t_S  = curve.P(t, S, rt);
    float bond_price_t_T  = curve.P(t, T_maturity, rt);
    float B_T_maturity_S  = BtT(T_maturity, S, a);
    float sigma_p         = sigma * sqrtf((1.0f - expf(-2.0f * a * (T_maturity - t)))
                                          / (2.0f * a)) * B_T_maturity_S;
    float h               = (1.0f / sigma_p) * logf(bond_price_t_S
                             / (bond_price_t_T * K)) + sigma_p / 2.0f;
    return -(K * B_t_T * bond_price_t_T * normcdff(-h + sigma_p))
           + (B_t_S * bond_price_t_S * normcdff(-h));
}

// wrappers for backwards compatibility in 

__host__ __device__ inline float ZBC(float t, float T_maturity, float S, float K,
                                      float rt, float a, float sigma, float r0){
    return ZBC_impl(t, T_maturity, S, K, rt, a, sigma, FlatCurve{a, sigma, r0});
}

__host__ __device__ inline float ZBP(float t, float T_maturity, float S, float K,
                                      float rt, float a, float sigma, float r0){
    return ZBP_impl(t, T_maturity, S, K, rt, a, sigma, FlatCurve{a, sigma, r0});
}

__host__ __device__ inline float vega_ZBC(float t, float T_maturity, float S, float K,
                                           float rt, float a, float sigma, float r0){
    return vega_ZBC_impl(t, T_maturity, S, K, rt, a, sigma, FlatCurve{a, sigma, r0});
}

__host__ __device__ inline float vega_ZBP(float t, float T_maturity, float S, float K,
                                           float rt, float a, float sigma, float r0){
    return vega_ZBP_impl(t, T_maturity, S, K, rt, a, sigma, FlatCurve{a, sigma, r0});
}

__host__ __device__ inline float delta_ZBC(float t, float T_maturity, float S, float K,
                                            float rt, float a, float sigma, float r0){
    return delta_ZBC_impl(t, T_maturity, S, K, rt, a, sigma, FlatCurve{a, sigma, r0});
}

__host__ __device__ inline float delta_ZBP(float t, float T_maturity, float S, float K,
                                            float rt, float a, float sigma, float r0){
    return delta_ZBP_impl(t, T_maturity, S, K, rt, a, sigma, FlatCurve{a, sigma, r0});
}

__host__ __device__ inline float ZBC_market(float t, float T_maturity, float S, float K,
                                             float rt, float a, float sigma,
                                             const float* P_market, const float* f_market,
                                             float mat_spacing, int n_mat){
    return ZBC_impl(t, T_maturity, S, K, rt, a, sigma,
                    MarketCurve{a, sigma, P_market, f_market, mat_spacing, n_mat});
}

__host__ __device__ inline float vega_ZBC_market(float t, float T_maturity, float S, float K,
                                                  float rt, float a, float sigma,
                                                  const float* P_market, const float* f_market,
                                                  float mat_spacing, int n_mat){
    return vega_ZBC_impl(t, T_maturity, S, K, rt, a, sigma,
                         MarketCurve{a, sigma, P_market, f_market, mat_spacing, n_mat});
}
__host__ __device__ inline float PtT(float t, float T_maturity, float rt,
                                      float a, float sigma, float r0){
    return FlatCurve{a, sigma, r0}.P(t, T_maturity, rt);
}

__host__ __device__ inline float PtT_market(float t, float T_maturity, float rt,
                                             float a, float sigma,
                                             const float* P_market, const float* f_market,
                                             float mat_spacing, int n_mat){
    return MarketCurve{a, sigma, P_market, f_market, mat_spacing, n_mat}.P(t, T_maturity, rt);
}

#endif // HW_KERNELS_CUH
