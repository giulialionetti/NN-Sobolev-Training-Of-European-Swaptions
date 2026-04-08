#ifndef HW_OPTION_SENSITIVITIES_CUH
#define HW_OPTION_SENSITIVITIES_CUH

#include "hw_option_pricing.cuh"
#include "hw_bond_dsigma.cuh"
#include "hw_bond_dr.cuh"
#include "hw_bond_da.cuh"


__host__ __device__ inline float vega_zbc(const EuroOption& o,
                                           float t, float T, float S,
                                           float a, float sigma){

    // probability density function of a Normal Gaussian                                       
    float phi_h    = expf(-o.h * o.h * 0.5f) / sqrtf(2.0f * 3.14159265f);
    
    float PS_phi_h = o.P_S * phi_h; // P(t, S)*phi(h)

    float dsp_ds   = o.sigma_p / sigma;
    float B_S      = B(t, S, a);
    float B_T      = B(t, T, a);
    float dP_S_ds  = dP_dsigma(B_S, o.P_S, sigma, a, t);
    float dP_T_ds  = dP_dsigma(B_T, o.P_T, sigma, a, t);

    
// P(t,S)*phi(h)* dsigma_p/dsigma + N(h)*dP(t,S)dsigma - X*N(h-sigma_p)*dP(t,T)dsigma
    return PS_phi_h * dsp_ds + normcdff(o.h) * dP_S_ds - o.X * normcdff(o.h - o.sigma_p) * dP_T_ds;
}

__host__ __device__ inline float vega_zbp(const EuroOption& o,
                                           float t, float T, float S,
                                           float a, float sigma){

    float B_S      = B(t, S, a);
    float B_T      = B(t, T, a);
    float dP_S_ds  = dP_dsigma(B_S, o.P_S, sigma, a, t);
    float dP_T_ds  = dP_dsigma(B_T, o.P_T, sigma, a, t);

    // from Put-Call Parity ZBC-ZBP = P(t,S) - X* P(t,T)
    // dZBPdsigma = dZBCdsigma - d(P(t,S) + X* P(t,T))dsigma
    return vega_zbc(o, t, T, S, a, sigma) - dP_S_ds + o.X * dP_T_ds;

}

__host__ __device__ inline float volga_zbc(const EuroOption& o,
                                            float t, float T, float S,
                                            float a, float sigma){
    float B_S       = B(t, S, a);
    float B_T       = B(t, T, a);
    float dsp_ds    = o.sigma_p / sigma;
    float phi_h     = expf(-o.h * o.h * 0.5f) / sqrtf(2.0f * 3.14159265f);
   //  float phi_h_sp  = expf(-(o.h - o.sigma_p) * (o.h - o.sigma_p) * 0.5f)
                   // / sqrtf(2.0f * 3.14159265f); // phi(h- sigma_p)

   // float dP_S_ds   = dP_dsigma(B_S, o.P_S, sigma, a, t);
    // float dP_T_ds   = dP_dsigma(B_T, o.P_T, sigma, a, t);
   // float d2P_S_ds2 = d2P_dsigma2(B_S, o.P_S, sigma, a, t); 
   // float d2P_T_ds2 = d2P_dsigma2(B_T, o.P_T, sigma, a, t);

    // float srvn      = (1.0f - expf(-2.0f * a * t)) / (2.0f * a);
    float dh_ds = dsp_ds * (o.sigma_p - o.h) / o.sigma_p;
           // - sigma * srvn * (B_S * B_S - B_T * B_T) / o.sigma_p;    

   // float term1     = (dP_S_ds * phi_h - o.P_S * phi_h * o.h * dh_ds) * dsp_ds;
   // float term2     = phi_h * dh_ds * dP_S_ds + normcdff(o.h) * d2P_S_ds2;
                   // + o.X * normcdff(o.h - o.sigma_p) * d2P_T_ds2;

    return - o.P_S * phi_h * o.h * dh_ds * dsp_ds;
}

__host__ __device__ inline float volga_zbp(const EuroOption& o,
                                            float t, float T, float S,
                                            float a, float sigma){
    float dP_S_ds   = dP_dsigma(B(t, S, a), o.P_S, sigma, a, t);
    float dP_T_ds   = dP_dsigma(B(t, T, a), o.P_T, sigma, a, t);
    float d2P_S_ds2 = d2P_dsigma2(B(t, S, a), o.P_S, sigma, a, t);
    float d2P_T_ds2 = d2P_dsigma2(B(t, T, a), o.P_T, sigma, a, t);

    return volga_zbc(o, t, T, S, a, sigma)
         - d2P_S_ds2
         + o.X * d2P_T_ds2;
}


__host__ __device__ inline float delta_zbc(const EuroOption& o,
                                            float t, float T, float S,
                                            float a){
    float B_S    = B(t, S, a);
    float B_T    = B(t, T, a);
    float dP_S_dr = dP_dr(B_S, o.P_S);
    float dP_T_dr = dP_dr(B_T, o.P_T);

    return dP_S_dr * normcdff(o.h)
         - o.X * dP_T_dr * normcdff(o.h - o.sigma_p);
}

__host__ __device__ inline float delta_zbp(const EuroOption& o,
                                            float t, float T, float S,
                                            float a){
    float B_S     = B(t, S, a);
    float B_T     = B(t, T, a);
    float dP_S_dr = dP_dr(B_S, o.P_S);
    float dP_T_dr = dP_dr(B_T, o.P_T);

    return delta_zbc(o, t, T, S, a)
         - dP_S_dr
         + o.X * dP_T_dr;
}

__host__ __device__ inline float gamma_zbc(const EuroOption& o,
                                            float t, float T, float S,
                                            float a){
    float B_S      = B(t, S, a);
    float B_T      = B(t, T, a);
    float phi_h    = expf(-o.h * o.h * 0.5f) / sqrtf(2.0f * 3.14159265f);

    return d2P_dr2(B_S, o.P_S) * normcdff(o.h)
         - o.X * d2P_dr2(B_T, o.P_T) * normcdff(o.h - o.sigma_p)
         + (o.P_S * phi_h)* (B_T - B_S) * (B_T - B_S) / o.sigma_p;
}

__host__ __device__ inline float gamma_zbp(const EuroOption& o,
                                            float t, float T, float S,
                                            float a){
    return gamma_zbc(o, t, T, S, a)
         - d2P_dr2(B(t, S, a), o.P_S)
         + o.X * d2P_dr2(B(t, T, a), o.P_T);
}

__host__ __device__ inline float dZBC_da(const EuroOption& o,
                                          float t, float T, float S,
                                          float a, float sigma, float rt,
                                          const float* P0, const float* f0){
    float phi_h    = expf(-o.h * o.h * 0.5f) / sqrtf(2.0f * 3.14159265f);
    float B_S      = B(t, S, a);
    float B_T      = B(t, T, a);
    float dB_S     = dB_da(t, S, a);
    float dB_T     = dB_da(t, T, a);
    float f0t      = interpolate(f0, t);
    float dlnA_S   = dlnA_da(t, S, a, sigma, f0t, B_S, dB_S);
    float dlnA_T   = dlnA_da(t, T, a, sigma, f0t, B_T, dB_T);
    float dP_S_da  = dP_da(o.P_S, rt, dB_S, dlnA_S);
    float dP_T_da  = dP_da(o.P_T, rt, dB_T, dlnA_T);
    float dsigmap  = dsigmap_da(t, T, S, a, o.sigma_p, sigma);

    return dP_S_da * normcdff(o.h)
         - o.X * dP_T_da * normcdff(o.h - o.sigma_p)
         + o.P_S * phi_h * dsigmap;
}

__host__ __device__ inline float dZBP_da(const EuroOption& o,
                                          float t, float T, float S,
                                          float a, float sigma, float rt,
                                          const float* P0, const float* f0){
    float B_S     = B(t, S, a);
    float B_T     = B(t, T, a);
    float dB_S    = dB_da(t, S, a);
    float dB_T    = dB_da(t, T, a);
    float f0t     = interpolate(f0, t);
    float dlnA_S  = dlnA_da(t, S, a, sigma, f0t, B_S, dB_S);
    float dlnA_T  = dlnA_da(t, T, a, sigma, f0t, B_T, dB_T);
    float dP_S_da = dP_da(o.P_S, rt, dB_S, dlnA_S);
    float dP_T_da = dP_da(o.P_T, rt, dB_T, dlnA_T);

    return dZBC_da(o, t, T, S, a, sigma, rt, P0, f0)
         - dP_S_da
         + o.X * dP_T_da;
}


#endif // HW_OPTION_SENSITIVITIES_CUH