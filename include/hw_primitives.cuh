#ifndef HW_PRIMITIVES_CUH
#define HW_PRIMITIVES_CUH

#include <cmath>

// Section 1:
// this section contains the foundational blocks of the Hull-White model,
// its functions of time and parameters. Every higher-level formula
// is composed from these.


__host__ __device__ inline float BtT(float t, float T, float a){
    // this functions returns the mean reversion factor B(t, T) in HW.
    // when a -> 0 (no mean reversion) B(t, T) -> (T, t) which is the
    // Time To Maturity and the pure duration of a Zero-Coupon Bond.

    // appears in: P(t,T) = A(t,T)*exp(-B(t,T)*r(t))
    // Ref: Brigo, Mercurio eq. 3.39 page 75.
    return (1.0f - expf(-a * (T - t))) / a;
}

__host__ __device__ inline float short_rate_variance_norm(float t, float a){
    // this functions returns the normalized variance of the short rate
    // which quantifies its accumulated randomness from time 0 to time t.
    // Var(r(t)) is then recovered as σ² * short_rate_variance_norm(t, a) whenever needed.

    // (1 - exp(-2at)) / (2a)  =  Var(r(t)) / σ²
    // Ref: Brigo, Mercurio page 76
    return (1.0f - expf(-2.0f * a * t)) / (2.0f * a);
}


__host__ __device__ inline float interpolate(const float* data, float T,
                                              float mat_spacing, int n_mat){
    
    // Market curves (P^M(0,T) and f^M(0,T)) are stored at discrete maturities
    // {0, mat_spacing, 2*mat_spacing, ..., (n_mat-1)*mat_spacing}.
    // For any query maturity T that falls between grid points, we interpolate
    // linearly between the two bracketing values. For T beyond the last grid
    // point we return the last value (flat extrapolation).
    int   idx   = (int)(T / mat_spacing);
    if   (idx  >= n_mat - 1) return data[n_mat - 1];
    float t0    = idx * mat_spacing;
    float alpha = (T - t0) / mat_spacing;
    return data[idx] * (1.0f - alpha) + data[idx + 1] * alpha;
}


// Section 2: Curve Policies
// A Curve policy implements a single method P(t, T, rt) that returns the
// zero-coupon bond price at time t for maturity T given short rate rt.
// Both policies share the same affine structure P = A(t,T)*exp(-B(t,T)*r(t))
// from Brigo-Mercurio — they differ only in how A(t,T) is constructed.
//
// The with_sigma() and with_a() methods return a new curve with one parameter
// replaced. These are used exclusively by the finite-difference second
// derivative d²/da², and nowhere else. Keeping them here means the Greek
// layer never needs to know the internal layout of a curve struct.

struct FlatCurve {
    float a, sigma, r0;

    // Bond price under the flat initial forward curve assumption f^M(0,T) = r0.
    //
    // This is a degenerate special case of eq 3.39 (see above)
    // obtained by substituting P^M(0,T) = exp(-r0*T) and f^M(0,T) = r0.
    // The full bond price is then A(t,T) * exp(-B(t,T)*r(t)).
    // This formula is a flat-forward specialization of Hull-White, used for backward compatibility
    // when no market curve is available.
    __host__ __device__ float P(float t, float T, float rt) const {
        float B    = BtT(t, T, a);
        float srvn = short_rate_variance_norm(t, a);
        float A    = expf(-r0 * (T - t))   // forward discount          
                   * expf(B * r0)     // drift adjustment                   
                   * expf(-0.5f * sigma * sigma * srvn * B * B); // convexity correction
        return A * expf(-B * rt);
    }

    __host__ __device__ FlatCurve   with_sigma(float s) const { return {a, s,     r0}; }
    __host__ __device__ FlatCurve   with_a    (float x) const { return {x, sigma, r0}; }
};

struct MarketCurve {
    float        a, sigma;
    const float* P_market;    // P^M(0, i*mat_spacing), i = 0..n_mat-1
    const float* f_market;    // f^M(0, i*mat_spacing), i = 0..n_mat-1
    float        mat_spacing;
    int          n_mat;

    // Bond price fitted exactly to the observed initial term structure.

    // The key difference from FlatCurve is the first two factors: instead of
    // reconstructing the forward discount from a constant r0, we read it
    // directly from the observed market prices. This is what "fitting the
    // initial term structure" means — the model is calibrated to exactly
    // reproduce every market bond price at t=0.
    //
    // Note on the special case t=0: P^M(0,0) = 1 by definition (the price
    // today of receiving 1 today is 1), so we handle it explicitly to avoid
    // a division by a potentially small interpolated value near zero.
    __host__ __device__ float P(float t, float T, float rt) const {
        float B    = BtT(t, T, a);
        float srvn = short_rate_variance_norm(t, a);
        float P0T  = interpolate(P_market, T, mat_spacing, n_mat);
        float P0t  = (t == 0.0f) ? 1.0f
                                 : interpolate(P_market, t, mat_spacing, n_mat);
        float f0t  = interpolate(f_market, t, mat_spacing, n_mat);
        float A    = (P0T / P0t)
                   * expf(B * f0t)
                   * expf(-0.5f * sigma * sigma * srvn * B * B);
        return A * expf(-B * rt);
    }

    __host__ __device__ MarketCurve with_sigma(float s) const {
        return {a, s,     P_market, f_market, mat_spacing, n_mat};
    }
    __host__ __device__ MarketCurve with_a(float x) const {
        return {x, sigma, P_market, f_market, mat_spacing, n_mat};
    }
};

// Section 3: Bond Derivatives

// BondDerivatives bundles a bond price with all its first and second
// derivatives with respect to r(t) and σ. Computing them together in a
// single pass avoids repeating the B and short_rate_variance_norm calculations
// across every Greek function.

// All four derivative formulas follow from P(t,T) = A(t,T)*exp(-B*r) by
// noting that A depends on σ but not on r, while the exp(-B*r) 
// factor depends on r but not on σ.

struct BondDerivatives {
    float P;        // P(t, T)
    float B;        // B(t, T)      
    float dP_dr;    // ∂P/∂r(t)   
    float dP_ds;    // ∂P/∂σ    
    float d2P_dr2;  // ∂²P/∂r²  
    float d2P_ds2;  // ∂²P/∂σ²  
};


// note: srvn = short_rate_variance_norm(t, a) is passed in rather than recomputed
// here because make_pricing_state (see next Section) calls bond_derivs twice (once for bS, once
// for bT) and srvn depends only on t and a — the same for both bonds.
// Computing it once and passing it down avoids a redundant expf() call.
template<typename Curve>
__host__ __device__ inline BondDerivatives bond_derivs(float t, float T_mat,
                                                        float rt, float sigma,
                                                        float srvn,
                                                        const Curve& curve){
    BondDerivatives d;
    d.B       = BtT(t, T_mat, curve.a);
    d.P       = curve.P(t, T_mat, rt);
    d.dP_dr   = -d.B * d.P;
    d.dP_ds   = -d.P * sigma * srvn * d.B * d.B;
    d.d2P_dr2 =  d.B * d.B * d.P;
    d.d2P_ds2 =  d.P * srvn * d.B * d.B
                 * (sigma * sigma * srvn * d.B * d.B - 1.0f);
    return d;
}

// Section 4: Pricing State

// This struct encapulates everything that is relevant to an option
// at a particular moment in time computed once in make_pricing_state
// and then consumed read-only every pricing and Greek function.

struct PricingState {

    // the two bonds involved in the option
    BondDerivatives bS; // the underlying bond P(t,S) that the option is written on  
    // maturing at S > T     
    BondDerivatives bT; // the discount bond P(t,T) used to
    //  present-value the option payoff at expiry T.

    // Option volatility structure 

    float sigma_p;  // effective volatility of the Bond Option  
    // σ * sqrt((1 - e^{-2a(T-t)}) / (2a)) * B(T,S)
                             
    float dsp_ds;   // ∂sigma_p/∂σ = sigma_p/σ
                    // exact because sigma_p is linear in σ — the sqrt(...)
                    // and B(T,S) factors do not depend on σ, so the whole
                    // expression differentiates trivially. Used in vega.         

    float h;      //  // Log-moneyness variable h (Brigo-Mercurio eq. 3.40, p. 76)     
    float phi_h;             // φ(h) — standard normal PDF at h
    float PS_phi_h;          // P(t,S)*φ(h) = K*P(t,T)*φ(h-sigma_p)

    float srvn;             
    float K;                 
};

// Build a PricingState from raw inputs.
//
// This is the only place where intermediate quantities are computed.
// Every downstream function (pricing, Greeks) reads from this struct —
// none of them recompute σ_p, h, or φ(h) independently.
template<typename Curve>
__host__ __device__ inline PricingState make_pricing_state(float t, float T,
                                                            float S, float K,
                                                            float rt, float a,
                                                            float sigma,
                                                            const Curve& curve){
    PricingState ps;
    ps.K    = K;

    // Compute srvn once and share it between bS and bT.
    // Both bonds are evaluated at the same current time t under the same
    // model parameters, so their short_rate_variance_norm is identical.
    ps.srvn = short_rate_variance_norm(t, a);

    ps.bS   = bond_derivs(t, S, rt, sigma, ps.srvn, curve);
    ps.bT   = bond_derivs(t, T, rt, sigma, ps.srvn, curve);

    // B(T,S) is the forward duration of the underlying bond at option expiry.
    // It appears in sigma_p because the uncertainty in P(T,S) at time T is
    // driven by how sensitive P(T,S) is to r(T), which is exactly B(T,S).
    float B_TS  = BtT(T, S, a);
    ps.sigma_p  = sigma
                  * sqrtf((1.0f - expf(-2.0f * a * (T - t))) / (2.0f * a))
                  * B_TS;
    ps.dsp_ds   = ps.sigma_p / sigma;

    ps.h        = (1.0f / ps.sigma_p)
                  * logf(ps.bS.P / (ps.bT.P * K))
                  + ps.sigma_p / 2.0f;
    ps.phi_h    = expf(-ps.h * ps.h * 0.5f) / sqrtf(2.0f * 3.14159265f);

   
    ps.PS_phi_h = ps.bS.P * ps.phi_h;

    return ps;
}


#endif // HW_PRIMITVES_CUH