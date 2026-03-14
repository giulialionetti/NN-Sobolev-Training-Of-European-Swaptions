#ifndef HW_PRICING_CUH
#define HW_PRICING_CUH

#include "hw_primitives.cuh"

// Section 1: Zero-coupon bond option pricing
// Consumes a PricingState built by make_pricing_state().
// No transcendental functions are called here — all intermediate quantities
// were already computed in make_pricing_state().
// Ref: Brigo-Mercurio eq. 3.40-3.41, p.76

__host__ __device__ inline float ZBC_from_state(const PricingState& ps){
    // P(t,S)*Φ(h) - K*P(t,T)*Φ(h - sigma_p)
    return  ps.bS.P * normcdff( ps.h)
          - ps.K * ps.bT.P * normcdff( ps.h - ps.sigma_p);
}

__host__ __device__ inline float ZBP_from_state(const PricingState& ps){
    // K*P(t,T)*Φ(-h + sigma_p) - P(t,S)*Φ(-h)
    return  ps.K * ps.bT.P * normcdff(-ps.h + ps.sigma_p)
          - ps.bS.P * normcdff(-ps.h);
}

#endif // HW_PRICING_CUH