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

#endif // SWAPTIONS_CUH
