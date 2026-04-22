#ifndef SWAPTIONS_CUH
#define SWAPTIONS_CUH

#include "hw_model.cuh"
#include "hw_option_pricing.cuh"
#include "hw_option_sensitivities.cuh"

__host__ __device__ inline float par_swap_rate(float T, const float* tenor_dates,
                                                int n_tenors, const float* P0){
    float P_T  = interpolate(P0, T);
    float P_Tn = interpolate(P0, tenor_dates[n_tenors - 1]);
    float annuity = 0.0f;
    for(int i = 0; i < n_tenors; i++){
        float delta_i = (i == 0) ? tenor_dates[0] - T
                                 : tenor_dates[i] - tenor_dates[i-1];
        annuity += delta_i * interpolate(P0, tenor_dates[i]);
    }
    return (P_T - P_Tn) / annuity;
}

__host__ __device__ inline float swap_value_at_r(float r, float T,
                                                   const float* tenor_dates, int n_tenors,
                                                   const float* c,
                                                   const float* P0, const float* f0,
                                                   float a, float sigma){
    float val = 0.0f;
    for(int i = 0; i < n_tenors; i++)
        val += c[i] * P(P0, f0, T, tenor_dates[i], r, a, sigma);
    return val;
}

__host__ __device__ inline float critical_rate_r_star(float T, const float* tenor_dates,
                                                       int n_tenors, const float* c,
                                                       const float* P0, const float* f0,
                                                       float a, float sigma){
    float lo = -0.5f, hi = 0.5f;
    for(int iter = 0; iter < 100; iter++){
        float mid = 0.5f * (lo + hi);
        float val = swap_value_at_r(mid, T, tenor_dates, n_tenors, c, P0, f0, a, sigma);
        if(val > 1.0f) lo = mid; else hi = mid;
    }
    return 0.5f * (lo + hi);
}

__host__ __device__ inline float analytical_swaption(float T, const float* tenor_dates,
                                                      int n_tenors, const float* c,
                                                      const float* P0, const float* f0,
                                                      float a, float sigma, float r0){
    float r_star = critical_rate_r_star(T, tenor_dates, n_tenors, c, P0, f0, a, sigma);
    float price = 0.0f;
    for(int i = 0; i < n_tenors; i++){
        float X_i = P(P0, f0, T, tenor_dates[i], r_star, a, sigma);
        EuroOption o = euro_option(P0, f0, 0.0f, T, tenor_dates[i], X_i, r0, a, sigma);
        price += c[i] * ZBP(o);
    }
    return price;
}

__host__ __device__ inline float analytical_swaption_vega(float T, const float* tenor_dates,
                                                           int n_tenors, const float* c,
                                                           const float* P0, const float* f0,
                                                           float a, float sigma, float r0){
    float r_star = critical_rate_r_star(T, tenor_dates, n_tenors, c, P0, f0, a, sigma);

    // σ/(2a)·(1−e^{−2aT}) — shared factor in ∂X_i/∂σ|_{r*} and dr*/dσ
    float coeff = (sigma / (2.0f * a)) * (1.0f - expf(-2.0f * a * T));

    // Implicit differentiation of  Σ_j c_j·P(T,T_j,r*,σ) = 0  w.r.t. σ gives:
    //   dr*/dσ = −coeff · [Σ_j c_j·B_j²·X_j] / [Σ_j c_j·B_j·X_j]
    float num_dr = 0.0f, den_dr = 0.0f;
    for(int j = 0; j < n_tenors; j++){
        float X_j = P(P0, f0, T, tenor_dates[j], r_star, a, sigma);
        float B_j = B(T, tenor_dates[j], a);
        num_dr += c[j] * B_j * B_j * X_j;
        den_dr += c[j] * B_j * X_j;
    }
    float dr_star_dsigma = -coeff * num_dr / den_dr;

    float vega = 0.0f;
    for(int i = 0; i < n_tenors; i++){
        float X_i    = P(P0, f0, T, tenor_dates[i], r_star, a, sigma);
        float B_T_Ti = B(T, tenor_dates[i], a);

        EuroOption o = euro_option(P0, f0, 0.0f, T, tenor_dates[i], X_i, r0, a, sigma);

        // Full total derivative: dX_i/dσ = ∂X_i/∂σ|_{r*} + ∂X_i/∂r · dr*/dσ
        //   ∂X_i/∂σ|_{r*} = −X_i · coeff · B_i²
        //   ∂X_i/∂r       = −B_i · X_i
        float dXi_dsigma     = -X_i * (coeff * B_T_Ti * B_T_Ti + B_T_Ti * dr_star_dsigma);
        float sensitivity_via_Xi = dXi_dsigma * o.P_T * normcdff(-o.h + o.sigma_p);
        vega += c[i] * (vega_zbp(o, 0.0f, T, tenor_dates[i], a, sigma) + sensitivity_via_Xi);
    }
    return vega;
}

__host__ __device__ inline float analytical_swaption_volga(float T, const float* tenor_dates,
                                                            int n_tenors, const float* c,
                                                            const float* P0, const float* f0,
                                                            float a, float sigma, float r0){
    float r_star = critical_rate_r_star(T, tenor_dates, n_tenors, c, P0, f0, a, sigma);
    float coeff  = (sigma / (2.0f * a)) * (1.0f - expf(-2.0f * a * T));

    // Pass 1: dr*/dσ = −coeff · [Σ c_j B_j² X_j] / [Σ c_j B_j X_j]
    float num_dr = 0.0f, den_dr = 0.0f;
    for(int j = 0; j < n_tenors; j++){
        float X_j = P(P0, f0, T, tenor_dates[j], r_star, a, sigma);
        float B_j = B(T, tenor_dates[j], a);
        num_dr += c[j] * B_j * B_j * X_j;
        den_dr += c[j] * B_j * X_j;
    }
    float dr_star_dsigma = -coeff * num_dr / den_dr;

    // Pass 2: d²r*/dσ² via second differentiation of Σ c_j X_j B_j Q_j = 0
    //   r*'' = [Σ c_j X_j B_j² (Q_j² − coeff/σ)] / [Σ c_j X_j B_j]
    //   where Q_j = coeff B_j + dr*/dσ
    float num_d2r = 0.0f, den_d2r = 0.0f;
    for(int j = 0; j < n_tenors; j++){
        float X_j = P(P0, f0, T, tenor_dates[j], r_star, a, sigma);
        float B_j = B(T, tenor_dates[j], a);
        float Q_j = coeff * B_j + dr_star_dsigma;
        num_d2r += c[j] * X_j * B_j * B_j * (Q_j * Q_j - coeff / sigma);
        den_d2r += c[j] * X_j * B_j;
    }
    float d2r_star_dsigma2 = num_d2r / den_d2r;

    float volga = 0.0f;
    for(int i = 0; i < n_tenors; i++){
        float X_i    = P(P0, f0, T, tenor_dates[i], r_star, a, sigma);
        float B_T_Ti = B(T, tenor_dates[i], a);
        EuroOption o = euro_option(P0, f0, 0.0f, T, tenor_dates[i], X_i, r0, a, sigma);

        float P_0T     = o.P_T;
        float phi_h_sp = expf(-0.5f * (-o.h + o.sigma_p) * (-o.h + o.sigma_p))
                       / sqrtf(2.0f * 3.14159265f);

        // Full dX_i/dσ = −X_i B_i Q_i,  Q_i = coeff B_i + dr*/dσ
        float Q_i        = coeff * B_T_Ti + dr_star_dsigma;
        float dXi_dsigma = -X_i * B_T_Ti * Q_i;

        // Full d²X_i/dσ² = X_i B_i [B_i(Q_i² − coeff/σ) − d²r*/dσ²]
        float d2Xi_dsigma2 = X_i * B_T_Ti
                           * (B_T_Ti * (Q_i * Q_i - coeff / sigma) - d2r_star_dsigma2);

        // ∂²ZBP/∂X_i²          = P_0T φ(-h+σ_p) / (X_i σ_p)
        // ∂²ZBP/(∂X_i ∂σ)|_{Xi} = P_0T φ(-h+σ_p) · h/σ   [d(-h+σ_p)/dσ|_{Xi} = h/σ]
        float d2ZBP_dXi2       = P_0T * phi_h_sp / (X_i * o.sigma_p);
        float d2ZBP_dXi_dsigma = P_0T * phi_h_sp * o.h / sigma;

        // d²(ZBP_i)/dσ² by the full chain rule:
        //   term1:  ∂ZBP/∂X_i        · d²X_i/dσ²
        //   term2:  2 ∂²ZBP/(∂X_i∂σ) · dX_i/dσ
        //   term3:  ∂²ZBP/∂X_i²      · (dX_i/dσ)²
        float term1 = d2Xi_dsigma2    * P_0T * normcdff(-o.h + o.sigma_p);
        float term2 = 2.0f * dXi_dsigma * d2ZBP_dXi_dsigma;
        float term3 = d2ZBP_dXi2      * dXi_dsigma * dXi_dsigma;

        volga += c[i] * (volga_zbp(o, 0.0f, T, tenor_dates[i], a, sigma) + term1 + term2 + term3);
    }
    return volga;
}

__host__ __device__ inline float analytical_swaption_delta(float T, const float* tenor_dates,
                                                            int n_tenors, const float* c,
                                                            const float* P0, const float* f0,
                                                            float a, float sigma, float r0){
    float r_star = critical_rate_r_star(T, tenor_dates, n_tenors, c, P0, f0, a, sigma);
    float delta = 0.0f;
    for(int i = 0; i < n_tenors; i++){
        float X_i = P(P0, f0, T, tenor_dates[i], r_star, a, sigma);
        EuroOption o = euro_option(P0, f0, 0.0f, T, tenor_dates[i], X_i, r0, a, sigma);
        delta += c[i] * delta_zbp(o, 0.0f, T, tenor_dates[i], a);
    }
    return delta;
}

__host__ __device__ inline float analytical_swaption_gamma(float T, const float* tenor_dates,
                                                            int n_tenors, const float* c,
                                                            const float* P0, const float* f0,
                                                            float a, float sigma, float r0){
    float r_star = critical_rate_r_star(T, tenor_dates, n_tenors, c, P0, f0, a, sigma);
    float gamma = 0.0f;
    for(int i = 0; i < n_tenors; i++){
        float X_i = P(P0, f0, T, tenor_dates[i], r_star, a, sigma);
        EuroOption o = euro_option(P0, f0, 0.0f, T, tenor_dates[i], X_i, r0, a, sigma);
        gamma += c[i] * gamma_zbp(o, 0.0f, T, tenor_dates[i], a);
    }
    return gamma;
}

__host__ __device__ inline float analytical_swaption_da(float T, const float* tenor_dates,
                                                          int n_tenors, const float* c,
                                                          const float* P0, const float* f0,
                                                          float a, float sigma, float r0) {
    float r_star = critical_rate_r_star(T, tenor_dates, n_tenors, c, P0, f0, a, sigma);
    float f0T    = interpolate(f0, T);

    // ── dr*/da via implicit differentiation of Σ cⱼ P(T,Tⱼ,r*,a) = 0 ────
    // d/da [Σ cⱼ P_j] = 0
    // Σ cⱼ (∂P_j/∂a|_{r*} + ∂P_j/∂r * dr*/da) = 0
    // dr*/da = -[Σ cⱼ dP_j/da] / [Σ cⱼ dP_j/dr]
    float num_dr = 0.0f, den_dr = 0.0f;
    for (int j = 0; j < n_tenors; j++) {
        float Tj      = tenor_dates[j];
        float X_j     = P(P0, f0, T, Tj, r_star, a, sigma);
        float B_j     = B(T, Tj, a);
        float dB_j    = dB_da(T, Tj, a);
        float dlnA_j  = dlnA_da(T, Tj, a, sigma, f0T, B_j, dB_j);
        float dP_j_da = dP_da(X_j, r_star, dB_j, dlnA_j);
        float dP_j_dr = -B_j * X_j;
        num_dr += c[j] * dP_j_da;
        den_dr += c[j] * dP_j_dr;
    }
    float dr_star_da = -num_dr / den_dr;


    float result = 0.0f;
    for (int i = 0; i < n_tenors; i++) {
        float Ti      = tenor_dates[i];
        float X_i     = P(P0, f0, T, Ti, r_star, a, sigma);
        float B_i     = B(T, Ti, a);
        float dB_i    = dB_da(T, Ti, a);
        float dlnA_i  = dlnA_da(T, Ti, a, sigma, f0T, B_i, dB_i);

        // dX_i/da = ∂X_i/∂a|_{r*} + ∂X_i/∂r * dr*/da
        float dXi_da_partial = dP_da(X_i, r_star, dB_i, dlnA_i);
        float dXi_dr         = -B_i * X_i;
        float dXi_da         = dXi_da_partial + dXi_dr * dr_star_da;

        EuroOption o = euro_option(P0, f0, 0.0f, T, Ti, X_i, r0, a, sigma);

        // dZBP_i/da total = ∂ZBP_i/∂a|_{X_i} + ∂ZBP_i/∂X_i * dX_i/da
        // ∂ZBP_i/∂X_i = -P_0T * normcdf(-h + sigma_p)  (from ZBP formula)
        float dZBPi_dXi   = -o.P_T * normcdff(-o.h + o.sigma_p);
        float dZBPi_da    = dZBP_da(o, 0.0f, T, Ti, a, sigma, r0, P0, f0);

        result += c[i] * (dZBPi_da + dZBPi_dXi * dXi_da);
    }
    return result;
}

__host__ __device__ inline float analytical_swaption_d2a(float T, const float* tenor_dates,
                                                           int n_tenors, const float* c,
                                                           const float* P0, const float* f0,
                                                           float a, float sigma, float r0) {
    float r_star = critical_rate_r_star(T, tenor_dates, n_tenors, c, P0, f0, a, sigma);
    float f0T    = interpolate(f0, T);

    // ── Pass 1: dr*/da ────────────────────────────────────────────────────
    float num_dr = 0.0f, den_dr = 0.0f;
    for (int j = 0; j < n_tenors; j++) {
        float Tj     = tenor_dates[j];
        float X_j    = P(P0, f0, T, Tj, r_star, a, sigma);
        float B_j    = B(T, Tj, a);
        float dB_j   = dB_da(T, Tj, a);
        float dlnA_j = dlnA_da(T, Tj, a, sigma, f0T, B_j, dB_j);
        num_dr += c[j] * dP_da(X_j, r_star, dB_j, dlnA_j);
        den_dr += c[j] * (-B_j * X_j);
    }
    float dr_star_da = -num_dr / den_dr;

    // ── Pass 2: d²r*/da² ─────────────────────────────────────────────────
    float num_d2r = 0.0f;
    for (int j = 0; j < n_tenors; j++) {
        float Tj       = tenor_dates[j];
        float X_j      = P(P0, f0, T, Tj, r_star, a, sigma);
        float B_j      = B(T, Tj, a);
        float dB_j     = dB_da(T, Tj, a);
        float d2B_j    = d2B_da2(T, Tj, a);
        float dlnA_j   = dlnA_da(T, Tj, a, sigma, f0T, B_j, dB_j);
        float d2lnA_j  = d2lnA_da2(T, Tj, a, sigma, f0T, B_j, dB_j, d2B_j);
        float dP_j_da  = dP_da(X_j, r_star, dB_j, dlnA_j);
        float d2P_j_da2  = d2P_da2(X_j, r_star, dB_j, d2B_j, dlnA_j, d2lnA_j);
        float d2P_j_dr2  = B_j * B_j * X_j;
        float d2P_j_drda = -dB_j * X_j - B_j * dP_j_da;
        num_d2r += c[j] * (d2P_j_da2
                         + 2.0f * d2P_j_drda * dr_star_da
                         + d2P_j_dr2 * dr_star_da * dr_star_da);
    }
    float d2r_star_da2 = -num_d2r / den_dr;

    // ── Pass 3: sum over tenors ───────────────────────────────────────────
    float f0_0 = interpolate(f0, 0.0f);
    float result = 0.0f;

    for (int i = 0; i < n_tenors; i++) {
        float Ti       = tenor_dates[i];
        float X_i      = P(P0, f0, T, Ti, r_star, a, sigma);
        float B_i      = B(T, Ti, a);
        float dB_i     = dB_da(T, Ti, a);
        float d2B_i    = d2B_da2(T, Ti, a);
        float dlnA_i   = dlnA_da(T, Ti, a, sigma, f0T, B_i, dB_i);
        float d2lnA_i  = d2lnA_da2(T, Ti, a, sigma, f0T, B_i, dB_i, d2B_i);

        float dP_i_da    = dP_da(X_i, r_star, dB_i, dlnA_i);
        float d2P_i_da2  = d2P_da2(X_i, r_star, dB_i, d2B_i, dlnA_i, d2lnA_i);
        float dP_i_dr    = -B_i * X_i;
        float d2P_i_dr2  =  B_i * B_i * X_i;
        float d2P_i_drda = -dB_i * X_i - B_i * dP_i_da;

        float dXi_da   = dP_i_da + dP_i_dr * dr_star_da;
        float d2Xi_da2 = d2P_i_da2
                       + 2.0f * d2P_i_drda  * dr_star_da
                       + dP_i_dr            * d2r_star_da2
                       + d2P_i_dr2          * dr_star_da * dr_star_da;

        EuroOption o = euro_option(P0, f0, 0.0f, T, Ti, X_i, r0, a, sigma);

        float phi_h_sp = expf(-0.5f * (-o.h + o.sigma_p) * (-o.h + o.sigma_p))
                       / sqrtf(2.0f * 3.14159265f);

        // ∂ZBP/∂Xi = P_T * Φ(-h + σ_p)
        float dZBPi_dXi  = o.P_T * normcdff(-o.h + o.sigma_p);

        // ∂²ZBP/∂Xi² = P_T * φ(-h+σ_p) / (Xi * σ_p)
        float d2ZBPi_dXi2 = o.P_T * phi_h_sp / (X_i * o.sigma_p);

        // ∂²ZBP/(∂Xi∂a): differentiate ∂ZBP/∂Xi = P_T * Φ(-h+σ_p) w.r.t. a at fixed Xi
        float B_0Ti    = B(0.0f, Ti, a);
        float B_0T_    = B(0.0f, T,  a);
        float dB_0Ti   = dB_da(0.0f, Ti, a);
        float dB_0T_   = dB_da(0.0f, T,  a);
        float dlnA_0Ti = dlnA_da(0.0f, Ti, a, sigma, f0_0, B_0Ti, dB_0Ti);
        float dlnA_0T_ = dlnA_da(0.0f, T,  a, sigma, f0_0, B_0T_,  dB_0T_);
        float dP_T_da  = dP_da(o.P_T, r0, dB_0T_, dlnA_0T_);

        float dlnPS_da     = dlnA_0Ti - r0 * dB_0Ti;
        float dlnPT_da     = dlnA_0T_ - r0 * dB_0T_;
        float d_logPSPT_da = dlnPS_da - dlnPT_da;
        float dsigmap      = dsigmap_da(0.0f, T, Ti, a, o.sigma_p, sigma);
        float dh_sp_da     = -(1.0f / o.sigma_p) * d_logPSPT_da
                            + (o.h / o.sigma_p) * dsigmap;

        float d2ZBPi_dXi_da = dP_T_da * normcdff(-o.h + o.sigma_p)
                             + o.P_T * phi_h_sp * dh_sp_da / X_i;

        // ∂²ZBP/∂a²|_{Xi} via FD of dZBP_da at fixed Xi
        float eps_a2 = 0.0001f;
        EuroOption o_up   = euro_option(P0, f0, 0.0f, T, Ti, X_i, r0, a + eps_a2, sigma);
        EuroOption o_down = euro_option(P0, f0, 0.0f, T, Ti, X_i, r0, a - eps_a2, sigma);
        float d2ZBPi_da2_fixed_Xi =
            (dZBP_da(o_up,   0.0f, T, Ti, a + eps_a2, sigma, r0, P0, f0)
           - dZBP_da(o_down, 0.0f, T, Ti, a - eps_a2, sigma, r0, P0, f0))
          / (2.0f * eps_a2);

        result += c[i] * (d2ZBPi_da2_fixed_Xi
                        + 2.0f * d2ZBPi_dXi_da * dXi_da
                        + dZBPi_dXi            * d2Xi_da2
                        + d2ZBPi_dXi2          * dXi_da * dXi_da);
    }
    return result;
}

#endif // SWAPTIONS_CUH
