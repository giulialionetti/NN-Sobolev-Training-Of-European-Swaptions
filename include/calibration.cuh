#ifndef CALIBRATION_CUH
#define CALIBRATION_CUH

#include "mc.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Theta calibration from forward rates  (Brigo-Mercurio eq. 3.10)
//
//   theta(t) = df(0,t)/dt  +  a * f(0,t)  +  (sigma^2 / 2a)(1 - e^{-2at})
// ─────────────────────────────────────────────────────────────────────────────


// df(0,t)/dt via finite differences on h_f
// All boundary stencils are O(h^2) consistent with the interior centered diff.
// inv_2h = 1 / (2 * mat_spacing) — caller precomputes once.
inline float dfdt(const float* __restrict__ h_f,
                  int i, float inv_2h, int n_mat)
{
    if (i == 0)
        return (-3.0f*h_f[0] + 4.0f*h_f[1] - h_f[2]) * inv_2h;
    if (i >= n_mat - 1)
        return (h_f[n_mat-3] - 4.0f*h_f[n_mat-2] + 3.0f*h_f[n_mat-1]) * inv_2h;
    return (h_f[i+1] - h_f[i-1]) * inv_2h;
}


// theta(t) at t = i * mat_spacing
// sigma2_2a = sigma^2 / (2*a) — caller precomputes once.
inline float theta(int i,
                   const float* __restrict__ h_f,
                   float a,
                   float sigma2_2a,       // sigma^2 / (2*a)
                   float mat_spacing,
                   int   n_mat,
                   float inv_2h)
{
    float t      = i * mat_spacing;
    float f0t    = h_f[i];
    float df0t   = dfdt(h_f, i, inv_2h, n_mat);
    float convex = sigma2_2a * (1.0f - expf(-2.0f * a * t));
    return df0t + a * f0t + convex;
}


// Build drift table for Hull-White MC.
//
//   drift[i]  =  integral_{s}^{s+dt}  e^{-a(s+dt-u)} theta(u) du
//             ≈  theta(s + dt/2) * (1 - e^{-a*dt}) / a
//
// where s = i * dt.
//
// Parameters
//   host_drift_table  output, length n_steps
//   h_f               instantaneous forward rates f(0, i*mat_spacing), length n_mat
//   a                 mean-reversion speed
//   sigma             short-rate volatility
//   dt                MC time step
//   mat_spacing       spacing between forward-rate nodes
//   n_steps           number of MC time steps
//   n_mat             number of forward-rate maturities (100)
inline void compute_calibrated_drift_table(float* __restrict__ host_drift_table,
                                            const float* __restrict__ h_f,
                                            float a,
                                            float sigma,
                                            float dt,
                                            float mat_spacing,
                                            int   n_steps,
                                            int   n_mat)
{

    const float inv_2h    = 1.0f / (2.0f * mat_spacing);
    const float sigma2_2a = (sigma * sigma) / (2.0f * a);
    const float factor    = (1.0f - expf(-a * dt)) / a;   // same for every step

    // ── precompute theta at every mat node (avoids recomputing dfdt twice per  
    //    interpolation interval and keeps h_f in L1 for the 100-float window) ──
    float theta_table[n_mat];   // 100 floats = 400 B, lives on stack / L1
    for (int i = 0; i < n_mat; i++)
        theta_table[i] = theta(i, h_f, a, sigma2_2a, mat_spacing, n_mat, inv_2h);

    // build drift table 
    for (int i = 0; i < n_steps; i++) {
        float s_mid = (i + 0.5f) * dt;
        float t_idx = s_mid / mat_spacing;
        int   idx   = (int)t_idx;

        float theta_mid;
        if (idx >= n_mat - 1) {
            theta_mid = theta_table[n_mat - 1];
        } else {
            float alpha = t_idx - idx;                      // in [0,1)
            theta_mid   = (1.0f - alpha) * theta_table[idx]
                        +          alpha * theta_table[idx + 1];
        }

        host_drift_table[i] = theta_mid * factor;
    }
}

#endif // CALIBRATION_CUH