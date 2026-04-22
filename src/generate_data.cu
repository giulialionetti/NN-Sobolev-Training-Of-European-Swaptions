
#include "mc_calibration.cuh"
#include "mc_market_price.cuh"
#include "hw_swaptions.cuh"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <thread>
#include <vector>
#include <string>
#include <atomic>
#include <chrono>

static const int DEFAULT_STREAMS_PER_GPU  = 4;
static const int DEFAULT_PATHS_PER_STREAM = 1024 * 1024;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d — %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

struct StreamContext {
    cudaStream_t stream;
    curandState* d_states;
    float*       d_drift;
    float*       d_sens_drift;
    float*       h_drift;       
    float*       h_sens_drift;  
    float*       d_P0_sum;
    float*       d_P0;
    float*       d_f0;
    float*       h_P;
    float*       h_f0;
    int          id;
    int          paths;
    int          blocks;
};
static void alloc_context(StreamContext& context, int id, int paths, unsigned long seed) {
    context.id     = id;
    context.paths  = paths;
    context.blocks = (paths + NTPB - 1) / NTPB;
    CUDA_CHECK(cudaStreamCreate(&context.stream));
    CUDA_CHECK(cudaMalloc(&context.d_states, paths * sizeof(curandState)));
    alloc_drift_tables(&context.d_drift, &context.d_sens_drift);
    CUDA_CHECK(cudaMalloc(&context.d_P0_sum, N_MAT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&context.d_P0, N_MAT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&context.d_f0, N_MAT * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&context.h_P, N_MAT   * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&context.h_f0, N_MAT   * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&context.h_drift, N_STEPS * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&context.h_sens_drift, N_STEPS * sizeof(float)));
    init_rng<<<context.blocks, NTPB, 0, context.stream>>>(context.d_states, seed, context.paths);
    CUDA_CHECK(cudaStreamSynchronize(context.stream));
}

static void free_context(StreamContext& context) {
    cudaStreamSynchronize(context.stream);
    cudaStreamDestroy(context.stream);
    cudaFree(context.d_states);
    free_drift_tables(context.d_drift, context.d_sens_drift);
    cudaFreeHost(context.h_drift);
    cudaFreeHost(context.h_sens_drift);
    cudaFree(context.d_P0_sum);
    cudaFree(context.d_P0);
    cudaFree(context.d_f0);
    cudaFreeHost(context.h_P);
    cudaFreeHost(context.h_f0);
}

struct SampleParams {
    float a, sigma, r0, T, K;
    int   swap_length, n_tenors;
    float tenor_dates[MAX_TENORS];
    float c[MAX_TENORS];
};

struct SampleResult {
    SampleParams sp;
    float price, vega, volga, delta, gamma;
    bool  valid;
};

struct PhaseTimes {
    double gpu_wait_s  = 0.0;  
    double calibrate_s = 0.0;
    double greeks_s    = 0.0;
    double launch_s    = 0.0;
    double write_s     = 0.0;
    long   n_samples   = 0;
};

using Clock = std::chrono::steady_clock;
using Sec   = std::chrono::duration<double>;

static float rand_uniform(float lo, float hi) {
    return lo + (hi - lo) * (float)rand() / (float)RAND_MAX;
}
static int rand_int(int lo, int hi) {
    return lo + rand() % (hi - lo + 1);
}

static SampleParams draw_params() {
    SampleParams sp;
    sp.a          = rand_uniform(0.10f, 2.00f);
    sp.sigma      = rand_uniform(0.01f, 0.30f);
    sp.r0         = rand_uniform(0.001f, 0.05f);
    sp.swap_length = rand_int(1, 4);
    float T_max   = 9.0f - (float)sp.swap_length;
    sp.T          = rand_uniform(1.0f, T_max);
    sp.n_tenors   = sp.swap_length;
    for (int i = 0; i < sp.n_tenors; i++)
        sp.tenor_dates[i] = sp.T + (float)(i + 1);
    return sp;
}

static void launch_sample(StreamContext& ctx, const SampleParams& sp,
                           PhaseTimes& pt) {
    auto t0    = Clock::now();
    HWParams p = params(sp.a, sp.sigma);
    init_drift(sp.a, sp.sigma, sp.r0,
               ctx.d_drift, ctx.d_sens_drift,
               ctx.h_drift, ctx.h_sens_drift,
               ctx.stream);
    launch_market_price(ctx.d_P0_sum, ctx.d_states, ctx.d_drift,
                        p, sp.r0, ctx.paths, ctx.blocks, ctx.stream);
    pt.launch_s += Sec(Clock::now() - t0).count();
}

static bool collect_sample(StreamContext& ctx,
                            SampleParams& sp,
                            float& price, float& vega, float& volga,
                            float& delta, float& gamma,
                            PhaseTimes& pt) {
    // async memcpy + stream-only sync — no full-device barrier
    auto t0 = Clock::now();
    collect_market_price(ctx.h_P, ctx.d_P0_sum, ctx.paths, ctx.stream);
    auto t1 = Clock::now();
    pt.gpu_wait_s += Sec(t1 - t0).count();

    // CPU calibration + strike
   calibrate(ctx.h_P, ctx.h_f0, sp.a, sp.sigma,
              ctx.d_drift, ctx.d_sens_drift,
              ctx.h_drift, ctx.h_sens_drift,
              ctx.stream);
    float K_atm     = par_swap_rate(sp.T, sp.tenor_dates, sp.n_tenors, ctx.h_P);
    float moneyness = rand_uniform(0.80f, 1.20f);
    sp.K            = K_atm * moneyness;
    for (int i = 0; i < sp.n_tenors; i++)
        sp.c[i] = sp.K;
    sp.c[sp.n_tenors - 1] += 1.0f;
    auto t2 = Clock::now();
    pt.calibrate_s += Sec(t2 - t1).count();

    // Analytical greeks
    price = analytical_swaption      (sp.T, sp.tenor_dates, sp.n_tenors, sp.c,
                                      ctx.h_P, ctx.h_f0, sp.a, sp.sigma, sp.r0);
    vega  = analytical_swaption_vega (sp.T, sp.tenor_dates, sp.n_tenors, sp.c,
                                      ctx.h_P, ctx.h_f0, sp.a, sp.sigma, sp.r0);
    volga = analytical_swaption_volga(sp.T, sp.tenor_dates, sp.n_tenors, sp.c,
                                      ctx.h_P, ctx.h_f0, sp.a, sp.sigma, sp.r0);
    delta = analytical_swaption_delta(sp.T, sp.tenor_dates, sp.n_tenors, sp.c,
                                      ctx.h_P, ctx.h_f0, sp.a, sp.sigma, sp.r0);
    gamma = analytical_swaption_gamma(sp.T, sp.tenor_dates, sp.n_tenors, sp.c,
                                      ctx.h_P, ctx.h_f0, sp.a, sp.sigma, sp.r0);
    auto t3 = Clock::now();
    pt.greeks_s += Sec(t3 - t2).count();

    pt.n_samples++;
    return true;
}

static void gpu_worker(int gpu_id, int n_streams, int n_samples, int sample_start, unsigned long base_seed, const std::string& out_path) {

    cudaSetDevice(gpu_id);

    int actual_device = -1;
    cudaGetDevice(&actual_device);
    if (actual_device != gpu_id) {
        fprintf(stderr, "[GPU %d] ERROR: bound to device %d instead\n", gpu_id, actual_device);
        return;
    }
    fprintf(stderr, "[GPU %d] Thread bound successfully\n", gpu_id);

    auto t_start = Clock::now();

    srand((unsigned)(base_seed ^ ((unsigned long)gpu_id * 2654435761UL)));

    std::string fname = out_path + "_gpu" + std::to_string(gpu_id) + ".csv";
    FILE* fp = fopen(fname.c_str(), "w");
    if (!fp) { fprintf(stderr, "[GPU %d] Cannot open %s\n", gpu_id, fname.c_str());
        return;
    }
    fprintf(fp, "a,sigma,r0,T,swap_length,K,price,vega,volga,delta,gamma\n");

    std::vector<StreamContext> ctxs(n_streams);
    for (int i = 0; i < n_streams; i++) {
        unsigned long stream_seed = base_seed + (unsigned long)gpu_id * 500009UL + (unsigned long)i* 100003UL;
        alloc_context(ctxs[i], i, DEFAULT_PATHS_PER_STREAM, stream_seed);
    }

    std::vector<SampleResult> results(n_streams);
    std::vector<bool> in_flight(n_streams, false);
    PhaseTimes pt;
    int n_written = 0;

    for (int s = 0; s < n_samples + n_streams; s++) {
        int slot = s % n_streams;

       
        if (s >= n_streams && in_flight[slot]) {
            collect_sample(ctxs[slot], results[slot].sp, results[slot].price, results[slot].vega, results[slot].volga, results[slot].delta, results[slot].gamma, pt);
            in_flight[slot] = false;

            if (n_written < n_samples) {
                const SampleResult& r  = results[slot];
                const SampleParams& sp = r.sp;
                auto tw = Clock::now();
                fprintf(fp, "%.6f,%.6f,%.6f,%.6f,%d,%.6f," "%.8f,%.8f,%.8f,%.8f,%.8f\n", sp.a, sp.sigma, sp.r0, sp.T, sp.swap_length, sp.K, 
                     r.price, r.vega, r.volga, r.delta, r.gamma);
                pt.write_s += Sec(Clock::now() - tw).count();
                n_written++;

                if (n_written % 10000 == 0)
                    fprintf(stderr, "[GPU %d] %d / %d\n",
                            gpu_id, n_written, n_samples);
            }
        }

        
        if (s < n_samples) {
            results[slot].sp = draw_params();
            launch_sample(ctxs[slot], results[slot].sp, pt);
            in_flight[slot]  = true;
        }
    }

    fclose(fp);

    auto t_end = Clock::now();
    double elapsed = Sec(t_end - t_start).count();
    fprintf(stderr, "[GPU %d] Done — %d samples in %.1fs (%.0f samples/sec) → %s\n",
            gpu_id, n_written, elapsed, n_written / elapsed, fname.c_str());

    double total_measured = pt.gpu_wait_s + pt.calibrate_s
                      + pt.greeks_s   + pt.launch_s  + pt.write_s;
    double ms = 1000.0 / pt.n_samples;
    fprintf(stderr,
        "[GPU %d] Phase breakdown (avg ms/sample, %ld samples):\n"
        "         gpu_wait  %6.3f ms  (%4.1f%%)  — kernel + async DtoH\n"
        "         calibrate %6.3f ms  (%4.1f%%)  — forward_rate + theta + upload_drift\n"
        "         greeks    %6.3f ms  (%4.1f%%)  — 5 analytical swaption functions\n"
        "         launch    %6.3f ms  (%4.1f%%)  — init_drift + kernel enqueue\n"
        "         write     %6.3f ms  (%4.1f%%)  — fprintf CSV\n",
        gpu_id, pt.n_samples,
        pt.gpu_wait_s  * ms, 100.0 * pt.gpu_wait_s  / total_measured,
        pt.calibrate_s * ms, 100.0 * pt.calibrate_s / total_measured,
        pt.greeks_s    * ms, 100.0 * pt.greeks_s    / total_measured,
        pt.launch_s    * ms, 100.0 * pt.launch_s    / total_measured,
        pt.write_s     * ms, 100.0 * pt.write_s     / total_measured);

    for (int i = 0; i < n_streams; i++)
        free_context(ctxs[i]);
}

int main(int argc, char** argv) {
    const int   N_SAMPLES = (argc > 1) ? atoi(argv[1]) : 1000000;
    const int   N_STREAMS = (argc > 2) ? atoi(argv[2]) : DEFAULT_STREAMS_PER_GPU;
    const int   PATHS     = (argc > 3) ? atoi(argv[3]) : DEFAULT_PATHS_PER_STREAM;
    const char* out_base  = (argc > 4) ? argv[4]       : "data/swaption_data";

    int n_gpus = 0;
    cudaGetDeviceCount(&n_gpus);
    if (n_gpus == 0) { fprintf(stderr, "No CUDA devices found.\n"); return 1; }
    if (n_gpus > 4)  n_gpus = 4;

    for (int g = 0; g < n_gpus; g++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, g);
        fprintf(stderr, "[GPU %d] %s | %zu MB | %d SMs\n",
                g, prop.name,
                prop.totalGlobalMem / (1024*1024),
                prop.multiProcessorCount);
    }

    fprintf(stderr,
            "Config: %d samples | %d GPUs | %d streams/GPU | %d paths/stream\n"
            "        total paths in flight per GPU: %d\n",
            N_SAMPLES, n_gpus, N_STREAMS, PATHS, N_STREAMS * PATHS);

    const unsigned long base_seed = (unsigned long)time(NULL);

    std::vector<int> gpu_samples(n_gpus);
    std::vector<int> gpu_offsets(n_gpus);
    int base_count = N_SAMPLES / n_gpus;
    int remainder  = N_SAMPLES % n_gpus;
    int offset     = 0;
    for (int g = 0; g < n_gpus; g++) {
        gpu_samples[g] = base_count + (g < remainder ? 1 : 0);
        gpu_offsets[g] = offset;
        offset        += gpu_samples[g];
    }

    std::vector<std::thread> threads;
    threads.reserve(n_gpus);
    for (int g = 0; g < n_gpus; g++) {
        threads.emplace_back(gpu_worker,
                             g,
                             N_STREAMS,
                             gpu_samples[g],
                             gpu_offsets[g],
                             base_seed,
                             std::string(out_base));
    }

    for (auto& t : threads)
        t.join();

    fprintf(stderr,
            "\nAll GPUs done. Merge output files with:\n"
            "  head -1 %s_gpu0.csv > %s.csv\n"
            "  tail -n +2 -q %s_gpu*.csv >> %s.csv\n",
            out_base, out_base, out_base, out_base);

    return 0;
}