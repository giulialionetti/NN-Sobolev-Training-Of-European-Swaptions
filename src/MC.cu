#include "mc.cuh"
#include "swaptions.cuh"
#include "logger.h"
#include "calibration.cuh"


void analytical_greeks(float t, float T, float S, float K, float rt){
    float zbc       = ZBC(t, T, S, K, rt, host_a, host_sigma, host_r0);
    float zbp       = ZBP(t, T, S, K, rt, host_a, host_sigma, host_r0);
    float parity    = P0T(S, host_r0) - K * P0T(T, host_r0);
    float vega_zbc  = vega_ZBC(t, T, S, K, rt, host_a, host_sigma, host_r0);
    float vega_zbp  = vega_ZBP(t, T, S, K, rt, host_a, host_sigma, host_r0);
    float delta_zbc = delta_ZBC(t, T, S, K, rt, host_a, host_sigma, host_r0);
    float delta_zbp = delta_ZBP(t, T, S, K, rt, host_a, host_sigma, host_r0);

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


void monteCarlo_vega(float T, float S, float K, curandState* d_states, CurveType curve,
                     float* d_P_market = nullptr, float* d_f_market = nullptr,
                     float* h_P = nullptr, float* h_f = nullptr){

    float* d_ZBC  = nullptr;
    float* d_vega = nullptr;
    cudaMalloc(&d_ZBC,  sizeof(float));
    cudaMalloc(&d_vega, sizeof(float));
    cudaMemset(d_ZBC,  0, sizeof(float));
    cudaMemset(d_vega, 0, sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    const char* label = (curve == CurveType::FLAT) ? "FLAT" :
                        (d_P_market == nullptr)     ? "PIECEWISE" : "PIECEWISE+MARKET";
    LOG_INFO("=== MC Pricing [%s curve] ===", label);

    mc_zbc_vega<<<NB, NTPB>>>(d_ZBC, d_vega, d_states, T, S, K, d_P_market, d_f_market);

    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    float h_ZBC, h_vega;
    cudaMemcpy(&h_ZBC,  d_ZBC,  sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_vega, d_vega, sizeof(float), cudaMemcpyDeviceToHost);
    h_ZBC  /= N_PATHS;
    h_vega /= N_PATHS;

    if(d_P_market == nullptr){
    float analytical_zbc  = ZBC(0.0f, T, S, K, host_r0, host_a, host_sigma, host_r0);
    float analytical_vega = vega_ZBC(0.0f, T, S, K, host_r0, host_a, host_sigma, host_r0);
    LOG_INFO("MC ZBC:          %.6f  |  Analytical: %.6f  |  Error: %.2e",
             h_ZBC, analytical_zbc, fabsf(h_ZBC - analytical_zbc));
    LOG_INFO("MC Vega:         %.6f  |  Analytical: %.6f  |  Error: %.2e",
             h_vega, analytical_vega, fabsf(h_vega - analytical_vega));
} else {
    float analytical_zbc  = ZBC_market(0.0f, T, S, K, host_r0, host_a, host_sigma,
                                       h_P, h_f, MAT_SPACING, N_MAT);
    float analytical_vega = vega_ZBC_market(0.0f, T, S, K, host_r0, host_a, host_sigma,
                                            h_P, h_f, MAT_SPACING, N_MAT);
    LOG_INFO("MC ZBC:          %.6f  |  Analytical (market): %.6f  |  Error: %.2e",
             h_ZBC, analytical_zbc, fabsf(h_ZBC - analytical_zbc));
    LOG_INFO("MC Vega:         %.6f  |  Analytical (market): %.6f  |  Error: %.2e",
             h_vega, analytical_vega, fabsf(h_vega - analytical_vega));
}
    cudaFree(d_ZBC);
    cudaFree(d_vega);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void finitedifferences_mc_vega(float T, float S, float K, curandState* d_states, CurveType curve,
                                float* d_P_market = nullptr, float* d_f_market = nullptr,
                                float* h_P = nullptr, float* h_f = nullptr){

    float eps  = 0.001f;
    unsigned long seed = time(NULL);

    float* d_ZBC_plus   = nullptr;
    float* d_ZBC_minus  = nullptr;
    float* d_vega_dummy = nullptr;
    cudaMalloc(&d_ZBC_plus,   sizeof(float));
    cudaMalloc(&d_ZBC_minus,  sizeof(float));
    cudaMalloc(&d_vega_dummy, sizeof(float));

    // bump up
    cudaMemset(d_ZBC_plus,   0, sizeof(float));
    cudaMemset(d_vega_dummy, 0, sizeof(float));
    init_device_constants(host_sigma + eps, curve);
    init_rng<<<NB, NTPB>>>(d_states, seed);
    cudaDeviceSynchronize();
    mc_zbc_vega<<<NB, NTPB>>>(d_ZBC_plus, d_vega_dummy, d_states, T, S, K,
                               d_P_market, d_f_market);
    cudaDeviceSynchronize();

    // bump down — same seed
    cudaMemset(d_ZBC_minus,  0, sizeof(float));
    cudaMemset(d_vega_dummy, 0, sizeof(float));
    init_device_constants(host_sigma - eps, curve);
    init_rng<<<NB, NTPB>>>(d_states, seed);
    cudaDeviceSynchronize();
    mc_zbc_vega<<<NB, NTPB>>>(d_ZBC_minus, d_vega_dummy, d_states, T, S, K,
                               d_P_market, d_f_market);
    cudaDeviceSynchronize();

    float h_plus, h_minus;
    cudaMemcpy(&h_plus,  d_ZBC_plus,  sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_minus, d_ZBC_minus, sizeof(float), cudaMemcpyDeviceToHost);
    h_plus  /= N_PATHS;
    h_minus /= N_PATHS;

    float vega_fd = (h_plus - h_minus) / (2.0f * eps);
   if(d_P_market == nullptr){
    float analytical_vega = vega_ZBC(0.0f, T, S, K, host_r0, host_a, host_sigma, host_r0);
    LOG_INFO("FD Vega:         %.6f  |  Analytical: %.6f  |  Error: %.2e",
             vega_fd, analytical_vega, fabsf(vega_fd - analytical_vega));
} else {
    float analytical_vega = vega_ZBC_market(0.0f, T, S, K, host_r0, host_a, host_sigma,
                                            h_P, h_f, MAT_SPACING, N_MAT);
    LOG_INFO("FD Vega:         %.6f  |  Analytical (market): %.6f  |  Error: %.2e",
             vega_fd, analytical_vega, fabsf(vega_fd - analytical_vega));
}

    cudaFree(d_ZBC_plus);
    cudaFree(d_ZBC_minus);
    cudaFree(d_vega_dummy);
}


void compute_market_data(float* h_P, float* h_f, curandState* d_states){

    float* d_P_sum;
    cudaMalloc(&d_P_sum, N_MAT * sizeof(float));
    cudaMemset(d_P_sum, 0, N_MAT * sizeof(float));

    init_device_constants(host_sigma, CurveType::PIECEWISE_LINEAR);
    mc_P0T<<<NB, NTPB>>>(d_P_sum, d_states);
    cudaDeviceSynchronize();

    cudaMemcpy(h_P, d_P_sum, N_MAT * sizeof(float), cudaMemcpyDeviceToHost);


    // normalize
    for(int k = 0; k < N_MAT; k++)
        h_P[k] /= N_PATHS;

    // P(0,0) = 1 by definition
    // f(0,T) via finite difference: f(0,T) = -d/dT ln P(0,T)
    h_f[0] = -(logf(h_P[1]) - logf(1.0f)) / MAT_SPACING;
    for(int k = 1; k < N_MAT - 1; k++)
    h_f[k] = -(logf(h_P[k+1]) - logf(h_P[k-1])) / (2.0f * MAT_SPACING);
    h_f[N_MAT-1] = -(logf(h_P[N_MAT-1]) - logf(h_P[N_MAT-2])) / MAT_SPACING;

    LOG_INFO("=== Market Data (Piecewise theta) ===");
    LOG_INFO("P(0,1)=%.6f  P(0,5)=%.6f  P(0,10)=%.6f",
             h_P[9], h_P[49], h_P[99]);
    LOG_INFO("f(0,1)=%.6f  f(0,5)=%.6f  f(0,10)=%.6f",
             h_f[9], h_f[49], h_f[99]);

    cudaFree(d_P_sum);
}
void monteCarlo_swaption(float T_expiry, curandState* d_states,
                         float* d_P_market, float* d_f_market,
                         float analytical_price){

    float* d_swaption = nullptr;
    float* d_vega     = nullptr;
    cudaMalloc(&d_swaption, sizeof(float));
    cudaMalloc(&d_vega,     sizeof(float));
    cudaMemset(d_swaption, 0, sizeof(float));
    cudaMemset(d_vega,     0, sizeof(float));

    init_device_constants(host_sigma, CurveType::PIECEWISE_LINEAR);
    mc_swaption<<<NB, NTPB>>>(d_swaption, d_vega, d_states,
                               T_expiry, d_P_market, d_f_market);
    cudaDeviceSynchronize();

    float h_swaption, h_vega;
    cudaMemcpy(&h_swaption, d_swaption, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_vega,     d_vega,     sizeof(float), cudaMemcpyDeviceToHost);
    h_swaption /= N_PATHS;
    h_vega     /= N_PATHS;

    LOG_INFO("=== Swaption MC vs Analytical ===");
    LOG_INFO("MC swaption      : %.6f  |  Analytical: %.6f  |  Error: %.2e",
             h_swaption, analytical_price, fabsf(h_swaption - analytical_price));
    LOG_INFO("MC pathwise vega : %.6f", h_vega);

    cudaFree(d_swaption);
    cudaFree(d_vega);
}

void finitedifferences_mc_swaption_vega(float T_expiry, curandState* d_states,
                                         float* d_P_market, float* d_f_market){
    float eps          = 0.001f;
    unsigned long seed = time(NULL);

    float* d_swaption_plus  = nullptr;
    float* d_swaption_minus = nullptr;
    float* d_vega_dummy     = nullptr;
    cudaMalloc(&d_swaption_plus,  sizeof(float));
    cudaMalloc(&d_swaption_minus, sizeof(float));
    cudaMalloc(&d_vega_dummy,     sizeof(float));

    // bump up — only sigma and std_gaussian_shock, drift table unchanged
    float sig_p    = host_sigma + eps;
    float shock_p  = sig_p * sqrtf((1.0f - expf(-2.0f*host_a*host_dt)) / (2.0f*host_a));
    cudaMemset(d_swaption_plus, 0, sizeof(float));
    cudaMemset(d_vega_dummy,    0, sizeof(float));
    cudaMemcpyToSymbol(device_sigma,             &sig_p,   sizeof(float));
    cudaMemcpyToSymbol(device_std_gaussian_shock, &shock_p, sizeof(float));
    init_rng<<<NB, NTPB>>>(d_states, seed);
    cudaDeviceSynchronize();
    mc_swaption<<<NB, NTPB>>>(d_swaption_plus, d_vega_dummy, d_states,
                               T_expiry, d_P_market, d_f_market);
    cudaDeviceSynchronize();

    // bump down — same seed
    float sig_m    = host_sigma - eps;
    float shock_m  = sig_m * sqrtf((1.0f - expf(-2.0f*host_a*host_dt)) / (2.0f*host_a));
    cudaMemset(d_swaption_minus, 0, sizeof(float));
    cudaMemset(d_vega_dummy,     0, sizeof(float));
    cudaMemcpyToSymbol(device_sigma,             &sig_m,   sizeof(float));
    cudaMemcpyToSymbol(device_std_gaussian_shock, &shock_m, sizeof(float));
    init_rng<<<NB, NTPB>>>(d_states, seed);
    cudaDeviceSynchronize();
    mc_swaption<<<NB, NTPB>>>(d_swaption_minus, d_vega_dummy, d_states,
                               T_expiry, d_P_market, d_f_market);
    cudaDeviceSynchronize();

    // restore
    float shock_orig = host_sigma * sqrtf((1.0f - expf(-2.0f*host_a*host_dt)) / (2.0f*host_a));
    cudaMemcpyToSymbol(device_sigma,             &host_sigma,  sizeof(float));
    cudaMemcpyToSymbol(device_std_gaussian_shock, &shock_orig, sizeof(float));

    float h_plus, h_minus;
    cudaMemcpy(&h_plus,  d_swaption_plus,  sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_minus, d_swaption_minus, sizeof(float), cudaMemcpyDeviceToHost);
    h_plus  /= N_PATHS;
    h_minus /= N_PATHS;

    float vega_fd = (h_plus - h_minus) / (2.0f * eps);
    LOG_INFO("FD vega          : %.6f", vega_fd);

    cudaFree(d_swaption_plus);
    cudaFree(d_swaption_minus);
    cudaFree(d_vega_dummy);
}
int main(){
    Logger& log = Logger::instance();
    log.open_file("hw_output.log");
    log.set_level(LogLevel::DEBUG);

    float t = 0.0f;
    float T = 1.0f;
    float S = 5.0f;
    float K = P0T(S, host_r0);

    analytical_greeks(t, T, S, K, host_r0);

    curandState* d_states;
    cudaMalloc(&d_states, N_PATHS * sizeof(curandState));
    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    cudaDeviceSynchronize();

    // compute market data from piecewise theta
    float h_P[N_MAT], h_f[N_MAT];
    compute_market_data(h_P, h_f, d_states);

    // upload market data to device
    float* d_P_market = nullptr;
    float* d_f_market = nullptr;
    cudaMalloc(&d_P_market, N_MAT * sizeof(float));
    cudaMalloc(&d_f_market, N_MAT * sizeof(float));
    cudaMemcpy(d_P_market, h_P, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f_market, h_f, N_MAT * sizeof(float), cudaMemcpyHostToDevice);

    // reinitialize RNG
    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    cudaDeviceSynchronize();

    // flat curve — no market data
init_device_constants(host_sigma, CurveType::FLAT);
monteCarlo_vega(T, S, K, d_states, CurveType::FLAT);
finitedifferences_mc_vega(T, S, K, d_states, CurveType::FLAT);

// piecewise — flat curve AtT, no market data
init_device_constants(host_sigma, CurveType::PIECEWISE_LINEAR);
monteCarlo_vega(T, S, K, d_states, CurveType::PIECEWISE_LINEAR);
finitedifferences_mc_vega(T, S, K, d_states, CurveType::PIECEWISE_LINEAR);

// piecewise — market data AtT
init_device_constants_calibrated(h_f);
monteCarlo_vega(T, S, K, d_states, CurveType::PIECEWISE_LINEAR, d_P_market, d_f_market, h_P, h_f);

finitedifferences_mc_vega(T, S, K, d_states, CurveType::PIECEWISE_LINEAR, 
                           d_P_market, d_f_market, h_P, h_f);

float tenor_dates[] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    int   n_tenors      = 5;

    float K_swap = par_swap_rate(1.0f, tenor_dates, n_tenors, h_P, MAT_SPACING, N_MAT);

    float c[5];
    for(int i = 0; i < n_tenors - 1; i++) c[i] = K_swap;
    c[n_tenors - 1] = 1.0f + K_swap;
    float ps_analytical = analytical_swaption(1.0f, tenor_dates, n_tenors, c,
                                           h_P, h_f,
                                           host_a, host_sigma, host_r0,
                                           MAT_SPACING, N_MAT);

    LOG_INFO("=== Swaption Analytical ===");
    LOG_INFO("Par swap rate K  : %.6f", K_swap);
    LOG_INFO("Analytical price : %.6f", ps_analytical);

    init_swaption_constants(tenor_dates, c, n_tenors);

    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    cudaDeviceSynchronize();
monteCarlo_swaption(1.0f, d_states, d_P_market, d_f_market, ps_analytical);

init_rng<<<NB, NTPB>>>(d_states, time(NULL));
cudaDeviceSynchronize();
float ps_vega_analytical = analytical_swaption_vega(1.0f, tenor_dates, n_tenors, c,
                                                     h_P, h_f, host_a, host_sigma,
                                                     host_r0, MAT_SPACING, N_MAT);
LOG_INFO("Analytical Swaption Vega  : %.6f", ps_vega_analytical);
finitedifferences_mc_swaption_vega(1.0f, d_states, d_P_market, d_f_market);

cudaFree(d_states);
    cudaFree(d_P_market);
    cudaFree(d_f_market);
    return 0;
}