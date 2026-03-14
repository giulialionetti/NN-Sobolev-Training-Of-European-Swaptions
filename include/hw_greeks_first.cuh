#ifndef HW_GREEKS_FIRST_CUH
#define HW_GREEKS_FIRST_CUH

#include "hw_primitives.cuh"
#include "hw_pricing.cuh"

// Section 1: Delta — ∂/∂r(t)
//
// Differentiating ZBC w.r.t. r(t). Since dP/dr = -B*P for any bond,
// the chain rule produces two direct contributions plus terms proportional
// to ∂h/∂r multiplied by [bS.P*phi_h - K*bT.P*phi(h-sigma_p)].
// That bracket is zero by the identity stored in PS_phi_h, so the
// ∂h/∂r terms vanish and only the direct dP/dr contributions survive.
//
// ∂ZBC/∂r = bS.dP_dr * Φ(h) - K * bT.dP_dr * Φ(h - sigma_p)

__host__ __device__ inline float delta_ZBC_from_state(const PricingState& ps){
    return  ps.bS.dP_dr * normcdff( ps.h)
          - ps.K * ps.bT.dP_dr * normcdff( ps.h - ps.sigma_p);
}

__host__ __device__ inline float delta_ZBP_from_state(const PricingState& ps){
    // parity: ZBC - ZBP = bS.P - K*bT.P
    // differentiating: delta_ZBC - delta_ZBP = bS.dP_dr - K*bT.dP_dr
    return delta_ZBC_from_state(ps)
         - ps.bS.dP_dr
         + ps.K * ps.bT.dP_dr;
}

// Section 2: Vega — ∂/∂σ
//
// σ appears in three places simultaneously: sigma_p (through σ_p ∝ σ),
// bS.P and bT.P .
// After the ∂h/∂σ terms cancel via the same identity as delta, three
// contributions survive:
//
//   ∂ZBC/∂σ = PS_phi_h * dsp_ds                    (sigma_p sensitivity)
//           + Φ(h)          * bS.dP_ds              (underlying bond moves)
//           - K*Φ(h-sigma_p) * bT.dP_ds             (discount bond moves)
//


__host__ __device__ inline float vega_ZBC_from_state(const PricingState& ps){
    return  ps.PS_phi_h * ps.dsp_ds
          + normcdff( ps.h)                * ps.bS.dP_ds
          - ps.K * normcdff( ps.h - ps.sigma_p) * ps.bT.dP_ds;
}

__host__ __device__ inline float vega_ZBP_from_state(const PricingState& ps){
    // parity: vega_ZBC - vega_ZBP = bS.dP_ds - K*bT.dP_ds
    return vega_ZBC_from_state(ps)
         - ps.bS.dP_ds
         + ps.K * ps.bT.dP_ds;
}

__host__ __device__ inline float vega_ZBC_market(float t, float T, float S, float K,
                                                  float rt, float a, float sigma,
                                                  const float* P_market,
                                                  const float* f_market,
                                                  float mat_spacing, int n_mat){
    return vega_ZBC_from_state(
               make_pricing_state(t, T, S, K, rt, a, sigma,
                   MarketCurve{a, sigma, P_market, f_market, mat_spacing, n_mat}));
}

// Section 3: Templated entry points
// Mirror the pattern from hw_pricing.cuh — build a PricingState internally.
// If you need multiple Greeks for the same option, call make_pricing_state()
// yourself and use the _from_state functions to avoid redundant computation.

template<typename Curve>
__host__ __device__ inline float delta_ZBC_impl(float t, float T, float S, float K,
                                                 float rt, float a, float sigma,
                                                 const Curve& curve){
    return delta_ZBC_from_state(make_pricing_state(t, T, S, K, rt, a, sigma, curve));
}

template<typename Curve>
__host__ __device__ inline float delta_ZBP_impl(float t, float T, float S, float K,
                                                 float rt, float a, float sigma,
                                                 const Curve& curve){
    return delta_ZBP_from_state(make_pricing_state(t, T, S, K, rt, a, sigma, curve));
}

template<typename Curve>
__host__ __device__ inline float vega_ZBC_impl(float t, float T, float S, float K,
                                                float rt, float a, float sigma,
                                                const Curve& curve){
    return vega_ZBC_from_state(make_pricing_state(t, T, S, K, rt, a, sigma, curve));
}

template<typename Curve>
__host__ __device__ inline float vega_ZBP_impl(float t, float T, float S, float K,
                                                float rt, float a, float sigma,
                                                const Curve& curve){
    return vega_ZBP_from_state(make_pricing_state(t, T, S, K, rt, a, sigma, curve));
}

// Backward-compatible flat-curve wrappers
__host__ __device__ inline float vega_ZBC(float t, float T, float S, float K,
                                           float rt, float a, float sigma, float r0){
    return vega_ZBC_impl(t, T, S, K, rt, a, sigma, FlatCurve{a, sigma, r0});
}

__host__ __device__ inline float vega_ZBP(float t, float T, float S, float K,
                                           float rt, float a, float sigma, float r0){
    return vega_ZBP_impl(t, T, S, K, rt, a, sigma, FlatCurve{a, sigma, r0});
}

__host__ __device__ inline float delta_ZBC(float t, float T, float S, float K,
                                            float rt, float a, float sigma, float r0){
    return delta_ZBC_impl(t, T, S, K, rt, a, sigma, FlatCurve{a, sigma, r0});
}

__host__ __device__ inline float delta_ZBP(float t, float T, float S, float K,
                                            float rt, float a, float sigma, float r0){
    return delta_ZBP_impl(t, T, S, K, rt, a, sigma, FlatCurve{a, sigma, r0});
}

#endif // HW_GREEKS_FIRST_CUH