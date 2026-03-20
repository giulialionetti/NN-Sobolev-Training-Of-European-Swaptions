#include "mc.cuh"
#include "logger.h"
#include "calibration.cuh"

void analytical_greeks(float t, float T, float S, float K, float rt){

    auto ps = make_pricing_state(t, T, S, K, rt,
                                 host_a, host_sigma,
                                 FlatCurve{host_a, host_sigma, host_r0});

    float zbc       = ZBC_from_state(ps);
    float zbp       = ZBP_from_state(ps);
    float vega_zbc  = vega_ZBC_from_state(ps);
    float vega_zbp  = vega_ZBP_from_state(ps);
    float delta_zbc = delta_ZBC_from_state(ps);
    float delta_zbp = delta_ZBP_from_state(ps);

    
    float parity     = expf(-host_r0 * S) - K * expf(-host_r0 * T);
    float parity_err = (zbc - zbp) - parity;

    LOG_INFO("=== Analytical Pricing ===");
    LOG_INFO("Params: t=%.2f  T=%.2f  S=%.2f  K=%.6f  r0=%.6f", t, T, S, K, rt);
    LOG_INFO("ZBC               : %.6f", zbc);
    LOG_INFO("ZBP               : %.6f", zbp);
    LOG_INFO("ZBC - ZBP         : %.6f", zbc - zbp);
    LOG_INFO("P(0,S) - K*P(0,T) : %.6f", parity);
    if (fabsf(parity_err) < 1e-5f)
        LOG_INFO("Put-call parity   : OK   (err=%.2e)", parity_err);
    else
        LOG_WARN("Put-call parity   : FAIL (err=%.2e)", parity_err);
    LOG_INFO("Vega  ZBC         : %.6f", vega_zbc);
    LOG_INFO("Delta ZBC         : %.6f", delta_zbc);
    LOG_INFO("Vega  ZBP         : %.6f", vega_zbp);
    LOG_INFO("Delta ZBP         : %.6f", delta_zbp);
}

int main(){
    Logger& log = Logger::instance();
    log.open_file("hw_output.log");
    log.set_level(LogLevel::DEBUG);

    float t = 0.0f;
    float T = 1.0f;
    float S = 5.0f;
    float K = expf(-host_r0 * S);

    analytical_greeks(t, T, S, K, host_r0);

    curandState* d_states;
    cudaMalloc(&d_states, N_PATHS * sizeof(curandState));
    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    cudaDeviceSynchronize();

    float h_P[N_MAT], h_f[N_MAT];
    compute_market_data(h_P, h_f, d_states);

    float* d_P_market = nullptr;
    float* d_f_market = nullptr;
    cudaMalloc(&d_P_market, N_MAT * sizeof(float));
    cudaMalloc(&d_f_market, N_MAT * sizeof(float));
    cudaMemcpy(d_P_market, h_P, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f_market, h_f, N_MAT * sizeof(float), cudaMemcpyHostToDevice);

    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    cudaDeviceSynchronize();

    init_device_constants(host_sigma, CurveType::FLAT);
    monteCarlo_vega(T, S, K, d_states, CurveType::FLAT);
    finitedifferences_mc_vega(T, S, K, d_states, CurveType::FLAT);

    init_device_constants(host_sigma, CurveType::PIECEWISE_LINEAR);
    monteCarlo_vega(T, S, K, d_states, CurveType::PIECEWISE_LINEAR);
    finitedifferences_mc_vega(T, S, K, d_states, CurveType::PIECEWISE_LINEAR);

    init_device_constants_calibrated(h_f);
    monteCarlo_vega(T, S, K, d_states, CurveType::PIECEWISE_LINEAR,
                    d_P_market, d_f_market, h_P, h_f);
    finitedifferences_mc_vega(T, S, K, d_states, CurveType::PIECEWISE_LINEAR,
                               d_P_market, d_f_market, h_P, h_f);

    float tenor_dates[] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    int   n_tenors      = 5;

    float K_swap = par_swap_rate(1.0f, tenor_dates, n_tenors, h_P, MAT_SPACING, N_MAT);

    float c[5];
    for(int i = 0; i < n_tenors - 1; i++) c[i] = K_swap;
    c[n_tenors - 1] = 1.0f + K_swap;

    float ps_analytical = analytical_swaption(1.0f, tenor_dates, n_tenors, c,
                                               h_P, h_f, host_a, host_sigma, host_r0,
                                               MAT_SPACING, N_MAT);

    LOG_INFO("=== Swaption Analytical ===");
    LOG_INFO("Par swap rate K  : %.6f", K_swap);
    LOG_INFO("Analytical price : %.6f", ps_analytical);

    init_swaption_constants(tenor_dates, c, n_tenors);
    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    cudaDeviceSynchronize();

    float ps_analytical_vega  = analytical_payer_swaption_vega(1.0f, tenor_dates, n_tenors, c,
                                                            h_P, h_f, host_a, host_sigma,
                                                            host_r0, MAT_SPACING, N_MAT);
    float ps_analytical_volga = analytical_payer_swaption_volga(1.0f, tenor_dates, n_tenors, c,
                                                             h_P, h_f, host_a, host_sigma,
                                                             host_r0, MAT_SPACING, N_MAT);
    monteCarlo_swaption(1.0f, d_states, d_P_market, d_f_market,
                        ps_analytical, ps_analytical_vega, ps_analytical_volga);

    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    cudaDeviceSynchronize();
    finitedifferences_mc_swaption_vega(1.0f, d_states, d_P_market, d_f_market);

    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    cudaDeviceSynchronize();
    finitedifferences_mc_swaption_volga(1.0f, d_states, d_P_market, d_f_market);

    cudaFree(d_states);
    cudaFree(d_P_market);
    cudaFree(d_f_market);
    return 0;
    cudaFree(d_states);
    cudaFree(d_P_market);
    cudaFree(d_f_market);
    return 0;
}