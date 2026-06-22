/**
 * pyHT C extension — hot loops (momentum + two-phase energy).
 * Optimised v2: pre-allocated arrays, interp2d index caching, sqrt for pow(., 0.5).
 * Perf profile: gcc -O3 -march=native -shared -DPROFILE -o _solver_c.dll _solver_c.c -lm
 * Normal build: gcc -O3 -march=native -shared -o _solver_c.dll _solver_c.c -lm
 */

#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <stdio.h>

#ifdef PROFILE
#include <windows.h>
/* ---- high-resolution performance accumulators ---- */
static LARGE_INTEGER _pfreq = {0};
static double _psec = 0.0;
enum { P_HTC, P_MOM, P_MOM_SP, P_MOM_TP, P_EN, 
       P_EN_P1, P_EN_P1B, P_EN_P2, P_EN_P2B, P_EN_P3,
       P_WALL, P_FFCNT, P_FRCNT, P_NPROBE };
static const char *pname[P_NPROBE] = {
    "htc_bulk_c", "momentum_c", "momentum_sp", "momentum_tp",
    "energy_c", "en_pass1", "en_pass1b", "en_pass2", "en_pass2b", "en_pass3",
    "update_wall", "ff_scalar_cnt", "friedel_cnt"};
static LARGE_INTEGER _pacc[P_NPROBE], _pcnt[P_NPROBE];
static LARGE_INTEGER _pstack[8];
static int _pstack_ptr = 0;
static int _pinit = 0;

static inline void prof_init(void) {
    if (_pinit) return; _pinit = 1;
    QueryPerformanceFrequency(&_pfreq);
    _psec = 1.0 / _pfreq.QuadPart;
    for (int i = 0; i < P_NPROBE; i++) {
        _pacc[i].QuadPart = 0; _pcnt[i].QuadPart = 0;
    }
}
static inline void prof_start(void) {
    QueryPerformanceCounter(&_pstack[_pstack_ptr++]);
}
static inline void prof_stop(int id) {
    _pstack_ptr--;
    LARGE_INTEGER now; QueryPerformanceCounter(&now);
    _pacc[id].QuadPart += now.QuadPart - _pstack[_pstack_ptr].QuadPart;
    _pcnt[id].QuadPart++;
}
static inline void prof_inc(int id) { _pcnt[id].QuadPart++; }
static inline void prof_print(void) {
    fprintf(stderr, "\n===== C PROBE =====\n");
    for (int i = 0; i < P_NPROBE; i++) {
        double t = _pacc[i].QuadPart * _psec;
        long long n = _pcnt[i].QuadPart;
        if (t > 1e-9 && n > 0)
            fprintf(stderr, "  %-16s %9.3f ms  %8lld calls  %8.3f us/call\n",
                    pname[i], t * 1000.0, n, t * 1e6 / n);
        else if (n > 0)
            fprintf(stderr, "  %-16s %9s       %8lld calls  (counter)\n",
                    pname[i], "", n);
    }
    fprintf(stderr, "===================\n");
}
#define PROBE_START() do { prof_init(); prof_start(); } while(0)
#define PROBE_STOP(id) prof_stop(id)
#define PROBE_INC(id) prof_inc(id)
__declspec(dllexport) void prof_print_c(void) {
    prof_print(); fflush(stderr);
}
#else
enum { P_HTC, P_MOM, P_MOM_SP, P_MOM_TP, P_EN, 
       P_EN_P1, P_EN_P1B, P_EN_P2, P_EN_P2B, P_EN_P3,
       P_WALL, P_FFCNT, P_FRCNT, P_NPROBE };
#define PROBE_START()
#define PROBE_STOP(id)
#define PROBE_INC(id)
__declspec(dllexport) void prof_print_c(void) {}
#endif

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define CLIP(x,lo,hi) MIN(MAX(x,lo),hi)

/* ---- 1D linear interpolation ---- */
static inline double interp1d(const double *x_grid, const double *y_grid,
                               int n, double x) {
    if (x <= x_grid[0]) return y_grid[0];
    if (x >= x_grid[n-1]) return y_grid[n-1];
    int i = 0;
    double w;
    while (i < n-2 && x_grid[i+1] < x) i++;
    w = (x - x_grid[i]) / (x_grid[i+1] - x_grid[i]);
    return y_grid[i] + w * (y_grid[i+1] - y_grid[i]);
}

/* ---- 1D index + weight (binary search for monotonic grids) ---- */
static inline int interp1d_idx(const double *x_grid, int n, double x, double *w) {
    if (x <= x_grid[0]) { *w = 0.0; return 0; }
    if (x >= x_grid[n-1]) { *w = 0.0; return n - 2; }
    int lo = 0, hi = n - 2;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (x_grid[mid] <= x) lo = mid; else hi = mid - 1;
    }
    *w = (x - x_grid[lo]) / (x_grid[lo + 1] - x_grid[lo]);
    return lo;
}

static inline double interp1d_at_w(const double *y_grid, int i, double w) {
    return y_grid[i] + w * (y_grid[i+1] - y_grid[i]);
}

/* ---- 2D bilinear interpolation with pre-computed grid constants ---- */
/* ir_log = 1 / (log(p_grid[-1]) - log(p_grid[0])), ir_T = 1 / (T_grid[-1] - T_grid[0]) */
static inline void interp2d_indices(
    const double *p_grid, int n_p,
    const double *T_grid, int n_T,
    double ir_log, double ir_T,
    double p, double T,
    int *i, int *j, double *wp, double *wT) {
    p = CLIP(p, p_grid[0], p_grid[n_p-1]);
    T = CLIP(T, T_grid[0], T_grid[n_T-1]);
    double ratio = (log(p) - log(p_grid[0])) * ir_log;
    *i = CLIP((int)(ratio * (n_p - 1)), 0, n_p - 2);
    double ratio_T = (T - T_grid[0]) * ir_T;
    *j = CLIP((int)(ratio_T * (n_T - 1)), 0, n_T - 2);
    *wp = (p - p_grid[*i]) / (p_grid[*i+1] - p_grid[*i]);
    *wT = (T - T_grid[*j]) / (T_grid[*j+1] - T_grid[*j]);
}

static inline double interp2d_at(const double *tab, int n_T,
                                  int i, int j, double wp, double wT) {
    return (1-wp)*(1-wT)*tab[i*n_T + j] +
           wp*(1-wT)*tab[(i+1)*n_T + j] +
           (1-wp)*wT*tab[i*n_T + j+1] +
           wp*wT*tab[(i+1)*n_T + j+1];
}

/* wrapper for one-off calls (rare path, computes grid constants on the fly) */
static inline double interp2d(const double *tab,
                               const double *p_grid, int n_p,
                               const double *T_grid, int n_T,
                               double p, double T) {
    int i, j; double wp, wT;
    double l0 = log(p_grid[0]), ir_log = 1.0 / (log(p_grid[n_p-1]) - l0);
    double ir_T = 1.0 / (T_grid[n_T-1] - T_grid[0]);
    interp2d_indices(p_grid, n_p, T_grid, n_T, ir_log, ir_T, p, T, &i, &j, &wp, &wT);
    return interp2d_at(tab, n_T, i, j, wp, wT);
}

/* ---- single-phase Churchill friction (optimised: mul replaces pow for integer exponents) ---- */
static double ff_scalar_c(double Re, double Dh) {
    PROBE_INC(P_FFCNT);
    const double roughness = 1.5e-6;
    if (Re < 1.0) Re = 1.0;
    double f_lam = 64.0 / Re;
    /* A = [2.457 * log( 1 / ( (7/Re)^0.9 + 0.27*roughness/Dh ) )]^16 */
    double t1 = pow(7.0 / Re, 0.9) + 0.27 * roughness / Dh;
    double t1l = log(1.0 / t1);
    double a0 = 2.457 * t1l;
    double a2 = a0 * a0, a4 = a2 * a2, a8 = a4 * a4, a16 = a8 * a8;
    /* B = (37530/Re)^16 */
    double b0 = 37530.0 / Re;
    double b2 = b0 * b0, b4 = b2 * b2, b8 = b4 * b4, b16 = b8 * b8;
    /* f_turb = 8 * [ (8/Re)^12 + 1/(A+B)^1.5 ]^(1/12) */
    double c0 = 8.0 / Re;
    double c2 = c0 * c0, c4 = c2 * c2, c8 = c4 * c4, c12 = c8 * c4;
    double d0 = a16 + b16;
    double d15 = d0 * sqrt(d0);  /* (A+B)^1.5 */
    double ft0 = c12 + 1.0 / d15;
    double ft1 = pow(ft0, 1.0 / 12.0);
    double f_turb = 8.0 * ft1;
    if (Re <= 2300) return f_lam;
    if (Re < 4000) return f_lam + (f_turb - f_lam) * (Re - 2300.0) / 1700.0;
    return f_turb;
}

/* ---- Gnielinski Nusselt (optimised) ---- */
static double nu_scalar_c(double Re, double Pr) {
    if (Re < 2300) return 3.66;
    if (Re < 2301) Re = 2301;
    double t = 0.79 * log(Re) - 1.64;
    double f = 1.0 / (t * t);                /* f = t^{-2} */
    double Pr23 = exp(2.0/3.0 * log(Pr));    /* Pr^{2/3} */
    double Nu = (f/8) * (Re - 1000) * Pr / (1 + 12.7 * sqrt(f/8) * (Pr23 - 1));
    return MAX(Nu, 0.1);
}

/* ---- Friedel two-phase multiplier (pow→exp*log for fractional exponents) ---- */
static double friedel_c(double x, double rho_l, double rho_g,
                         double mu_l, double mu_g, double G, double Dh,
                         double sigma) {
    PROBE_INC(P_FRCNT);
    if (x <= 0.0 || x >= 1.0 || G <= 1e-10) return 1.0;
    double Re_lo = G * Dh / (mu_l > 1e-10 ? mu_l : 1e-10);
    double Fr = G*G / (9.81 * Dh * rho_l * rho_l);
    if (Fr < 1e-10) Fr = 1e-10;
    double We = G*G*Dh / (sigma * rho_l);
    if (We < 1e-10) We = 1e-10;
    double rho_ratio = rho_l / rho_g;
    double mu_ratio = mu_l / mu_g;
    double f_mu = ff_scalar_c(Re_lo * mu_ratio, Dh);
    double f_lo = ff_scalar_c(Re_lo, Dh);
    if (f_lo < 1e-10) f_lo = 1e-10;
    double E = (1-x)*(1-x) + x*x*rho_ratio*f_mu/f_lo;
    double F = exp(0.78 * log(x)) * exp(0.224 * log(1.0 - x));
    double base = 1.0 - mu_ratio/rho_ratio;
    double H = exp(0.91 * log(rho_ratio)) * exp(0.19 * log(mu_ratio));
    if (base > 0.0) H *= exp(0.7 * log(base));
    double s_rho = sqrt(rho_ratio);
    double Fr_h = Fr / s_rho; if (Fr_h < 1e-10) Fr_h = 1e-10;
    double We_h = We / s_rho; if (We_h < 1e-10) We_h = 1e-10;
    double phi2 = E + 3.24*F*H / (exp(0.045 * log(Fr_h)) * exp(0.035 * log(We_h)));
    return MAX(phi2, 1.0);
}

/* ---- HEM quality ---- */
static inline double quality_c(double h, double hl, double hg) {
    if (hl >= hg - 1) return 0.0;
    return CLIP((h - hl) / (hg - hl), 0.0, 1.0);
}

/* ---- Zuber-Findlay drift-flux void fraction (falls back to HEM for G=0) ---- */
static inline double void_drift_flux_c(double x, double rho_l, double rho_g,
                                        double G, double Dh) {
    if (x <= 0) return 0.0;
    if (x >= 1) return 1.0;
    if (fabs(G) < 1e-6) return 1.0 / (1.0 + (1.0-x)/x * rho_g/rho_l);
    double C0 = (Dh > 0.01) ? 1.2 : 1.13;
    double arg = 0.07 * 9.81 * (rho_l - rho_g) / fmax(rho_l * rho_l, 1e-10);
    double V_gj = 1.53 * sqrt(sqrt(arg));          /* pow(..., 0.25) → sqrt(sqrt) */
    double G_eff = fmax(fabs(G), 1.0);
    double alpha = x / (rho_g * (C0 * (x/rho_g + (1-x)/rho_l) + V_gj/G_eff));
    return CLIP(alpha, 0.0, 1.0);
}

/* ---- Chen two-phase heat transfer (optimised pow→sqrt) ---- */
static double chen_two_phase_c(double x, double G, double Dh,
                                double rho_l, double rho_g,
                                double mu_l, double mu_g,
                                double k_l, double cp_l,
                                double sigma, double h_fg,
                                double dT_sat, double Tsat_K) {
    if (x <= 0.0 || x >= 1.0 || dT_sat <= 1.0) return 0.0;
    double Pr_l = mu_l * cp_l / MAX(k_l, 1e-10);
    double xs = x > 0 ? x : 1e-10;
    double X_tt = pow((1.0-xs)/xs, 0.9) * sqrt(rho_g/rho_l) * pow(mu_l/MAX(mu_g,1e-10), 0.1);
    double X_inv = 1.0 / MAX(X_tt, 1e-10);
    double F = (X_inv <= 0.1) ? 1.0 : 2.35 * pow(X_inv + 0.213, 0.736);
    double Re_l = G * (1.0-xs) * Dh / MAX(mu_l, 1e-10);
    double h_macro = 0.023 * pow(Re_l, 0.8) * pow(Pr_l, 0.4) * k_l / Dh * F;
    double Re_tp = Re_l * pow(F, 1.25);
    double S = 1.0 / (1.0 + 2.53e-6 * pow(Re_tp, 1.17));
    double dP_sat = dT_sat * h_fg * rho_g / MAX(Tsat_K, 1.0);
    double h_micro = (0.00122 * pow(dT_sat,0.24) * pow(dP_sat,0.75)) /
        (sqrt(sigma) * pow(mu_l,0.29) * pow(h_fg,0.24) * pow(rho_g,0.24)) * S;
    return h_macro + (h_micro > 0 ? h_micro : 0);
}


/* ===================================================================
 * Exported: momentum_loop_c — pre-allocated, no per-call malloc
 * =================================================================== */
__declspec(dllexport) void momentum_loop_c(
    double *p, double *T, double *rho, double *h,
    double *G, const double *Dh, double dz,
    int n_ch, int n_ax,
    const int *ch_tp,
    const double *p_out,
    const int *flow_sign,
    double urf_p,
    const int *fluids,
    const double *mu_tab_w, const double *k_tab_w,
    const double *cp_tab_w, const double *rho_tab_w, const double *h_tab_w,
    const double *p_grid_w, int n_p_w,
    const double *T_grid_w, int n_T_w,
    const double *mu_tab_he, const double *k_tab_he,
    const double *cp_tab_he, const double *rho_tab_he, const double *h_tab_he,
    const double *p_grid_he, int n_p_he,
    const double *T_grid_he, int n_T_he,
    const double *Tsat_grid,
    const double *hl_sat_grid, const double *hg_sat_grid,
    const double *rho_l_sat_grid, const double *rho_g_sat_grid,
    const double *mu_l_sat_grid, const double *mu_g_sat_grid,
    const double *sigma_tab
) {
    PROBE_START();
    if (n_ax > 40) return; /* fixed-size stack guard */
    double p_new[40];
    for (int c = 0; c < n_ch; c++) {
        double *p_c   = p   + c * n_ax;
        double *T_c   = T   + c * n_ax;
        double *rho_c = rho + c * n_ax;
        double *h_c   = h   + c * n_ax;
        double *G_c   = G   + c * (n_ax + 1);

        double Dh_c_val = Dh[c];
        double p_out_val = p_out[c];
        int is_tp = ch_tp[c];
        int is_he = fluids[c];

        /* select property table set per fluid */
        const double *mu_tab  = is_he ? mu_tab_he  : mu_tab_w;
        const double *k_tab   = is_he ? k_tab_he   : k_tab_w;
        const double *cp_tab  = is_he ? cp_tab_he  : cp_tab_w;
        const double *rho_tab = is_he ? rho_tab_he : rho_tab_w;
        const double *h_tab   = is_he ? h_tab_he   : h_tab_w;
        const double *p_grid  = is_he ? p_grid_he  : p_grid_w;
        const double *T_grid  = is_he ? T_grid_he  : T_grid_w;
        int n_p = is_he ? n_p_he : n_p_w;
        int n_T = is_he ? n_T_he : n_T_w;
        double _ir_log = 1.0 / (log(p_grid[n_p-1]) - log(p_grid[0]));
        double _ir_T = 1.0 / (T_grid[n_T-1] - T_grid[0]);

        for (int j = 0; j < n_ax; j++) p_new[j] = p_c[j];
        int fs = flow_sign[c];

        /* pressure drops are integrated from outlet toward inlet in the flow direction.
           up   : cell closer to the inlet   (higher pressure)
           down : cell closer to the outlet  (lower pressure, where p_out is imposed) */
        if (fs >= 0) {
            p_new[n_ax - 1] = p_out_val;
            for (int j = n_ax - 2; j >= 0; j--) {
                int down = j + 1;
                int up   = j;
                int face = down;
                double rho_down = rho_c[down] > 1e-10 ? rho_c[down] : 1e-10;
                double rho_up   = (j > 0) ? rho_c[up] : rho_down;
                double G_face = G_c[face] >= 0 ? G_c[face] : -G_c[face];
                int ii; int jj; double wp, wT;
                interp2d_indices(p_grid, n_p, T_grid, n_T, _ir_log, _ir_T, p_c[down], T_c[down], &ii, &jj, &wp, &wT);
                double mu_down = interp2d_at(mu_tab, n_T, ii, jj, wp, wT);

                double dp_fric;
                if (is_tp) {
                    double Re_lo = G_face * Dh_c_val / (mu_down > 1e-10 ? mu_down : 1e-10);
                    double f_lo = ff_scalar_c(Re_lo, Dh_c_val);
                    /* single index lookup for all saturation grids */
                    int i_sat; double w_sat;
                    i_sat = interp1d_idx(p_grid, n_p, p_c[down], &w_sat);
                    double hl = interp1d_at_w(hl_sat_grid, i_sat, w_sat);
                    double hg = interp1d_at_w(hg_sat_grid, i_sat, w_sat);
                    double x = quality_c(h_c[down], hl, hg);
                    if (x > 0.0) {
                        double rho_l_ = interp1d_at_w(rho_l_sat_grid, i_sat, w_sat);
                        double rho_g_ = interp1d_at_w(rho_g_sat_grid, i_sat, w_sat);
                        double mu_g = interp1d_at_w(mu_g_sat_grid, i_sat, w_sat);
                        double sigma_j = MAX(interp1d_at_w(sigma_tab, i_sat, w_sat), 0.001);
                        double phi2 = friedel_c(x, rho_l_, rho_g_, mu_down, mu_g, G_face, Dh_c_val, sigma_j);
                        dp_fric = f_lo * phi2 * fabs(G_c[face]) * G_face / (2.0 * Dh_c_val * rho_down) * dz;
                    } else {
                        dp_fric = f_lo * fabs(G_c[face]) * G_face / (2.0 * Dh_c_val * rho_down) * dz;
                    }
                } else {
                    double Re_j = G_face * Dh_c_val / (mu_down > 1e-10 ? mu_down : 1e-10);
                    double f = ff_scalar_c(Re_j, Dh_c_val);
                    dp_fric = f * fabs(G_c[face]) * G_face / (2.0 * Dh_c_val * rho_down) * dz;
                }
                double dp_grav = rho_down * 9.81 * dz;
                double dp_acc = G_c[face] * G_c[face] *
                                (1.0/fmax(rho_down, 1e-10) - 1.0/fmax(rho_up, 1e-10));
                double p_new_up = p_new[down] + dp_fric + dp_grav + dp_acc;
                p_new[up] = MAX(p_new_up, 1e4);
            }
        } else {
            p_new[0] = p_out_val;
            for (int j = 1; j < n_ax; j++) {
                int down = j - 1;
                int up   = j;
                int face = up;
                double rho_down = rho_c[down] > 1e-10 ? rho_c[down] : 1e-10;
                double rho_up   = rho_c[up] > 1e-10 ? rho_c[up] : 1e-10;
                double G_face = G_c[face] >= 0 ? G_c[face] : -G_c[face];
                int ii; int jj; double wp, wT;
                interp2d_indices(p_grid, n_p, T_grid, n_T, _ir_log, _ir_T, p_c[down], T_c[down], &ii, &jj, &wp, &wT);
                double mu_down = interp2d_at(mu_tab, n_T, ii, jj, wp, wT);

                double dp_fric;
                if (is_tp) {
                    double Re_lo = G_face * Dh_c_val / (mu_down > 1e-10 ? mu_down : 1e-10);
                    double f_lo = ff_scalar_c(Re_lo, Dh_c_val);
                    /* single index lookup for all saturation grids */
                    int i_sat; double w_sat;
                    i_sat = interp1d_idx(p_grid, n_p, p_c[down], &w_sat);
                    double hl = interp1d_at_w(hl_sat_grid, i_sat, w_sat);
                    double hg = interp1d_at_w(hg_sat_grid, i_sat, w_sat);
                    double x = quality_c(h_c[down], hl, hg);
                    if (x > 0.0) {
                        double rho_l_ = interp1d_at_w(rho_l_sat_grid, i_sat, w_sat);
                        double rho_g_ = interp1d_at_w(rho_g_sat_grid, i_sat, w_sat);
                        double mu_g = interp1d_at_w(mu_g_sat_grid, i_sat, w_sat);
                        double sigma_j = MAX(interp1d_at_w(sigma_tab, i_sat, w_sat), 0.001);
                        double phi2 = friedel_c(x, rho_l_, rho_g_, mu_down, mu_g, G_face, Dh_c_val, sigma_j);
                        dp_fric = f_lo * phi2 * fabs(G_c[face]) * G_face / (2.0 * Dh_c_val * rho_down) * dz;
                    } else {
                        dp_fric = f_lo * fabs(G_c[face]) * G_face / (2.0 * Dh_c_val * rho_down) * dz;
                    }
                } else {
                    double Re_j = G_face * Dh_c_val / (mu_down > 1e-10 ? mu_down : 1e-10);
                    double f = ff_scalar_c(Re_j, Dh_c_val);
                    dp_fric = f * fabs(G_c[face]) * G_face / (2.0 * Dh_c_val * rho_down) * dz;
                }
                /* flow_sign=-1: downward; hydrostatic head subtracts from pressure rise */
                double dp_grav = -rho_down * 9.81 * dz;
                double dp_acc = G_c[face] * G_c[face] *
                                (1.0/fmax(rho_down, 1e-10) - 1.0/fmax(rho_up, 1e-10));
                double p_new_up = p_new[down] + dp_fric + dp_grav + dp_acc;
                p_new[up] = MAX(p_new_up, 1e4);
            }
        }
        for (int j = 0; j < n_ax; j++) {
            p_c[j] += (p_new[j] - p_c[j]) * urf_p;
            if (p_c[j] < 1e4) p_c[j] = 1e4;
        }
    }
    PROBE_STOP(P_MOM);
}


/* ===================================================================
 * Exported: update_wall_temps_c — analytical wall temperature update
 * for coupled hot/cold channels. Uses current bulk T and htc_bulk.
 * =================================================================== */
__declspec(dllexport) void update_wall_temps_c(
    double *T_wall,
    const double *T, const double *htc_bulk,
    const int *wall_ch_i, const int *wall_ch_j,
    const double *wall_tw, const double *wall_kw, const double *wall_Pw,
    int n_wall, int n_ax
) {
    PROBE_START();
    for (int wi = 0; wi < n_wall; wi++) {
        int i = wall_ch_i[wi];
        int j = wall_ch_j[wi];
        double t_w = wall_tw[wi];
        double k_w = wall_kw[wi];
        double P_w = wall_Pw[wi];
        double R_cond = t_w / (k_w * P_w + 1e-30);

        for (int jj = 0; jj < n_ax; jj++) {
            double Ti = T[i * n_ax + jj];
            double Tj = T[j * n_ax + jj];
            double hi = htc_bulk[i * n_ax + jj];
            double hj = htc_bulk[j * n_ax + jj];
            double R_i = 1.0 / (hi * P_w + 1e-30);
            double R_j = 1.0 / (hj * P_w + 1e-30);
            double R_tot = R_i + R_cond + R_j;
            double q = (Ti - Tj) / R_tot;  /* positive: heat i -> j */
            double Tw_i = Ti - q * R_i;
            double Tw_j = Tj + q * R_j;
            T_wall[wi * 2 * n_ax + 0 * n_ax + jj] = Tw_i;
            T_wall[wi * 2 * n_ax + 1 * n_ax + jj] = Tw_j;
        }
    }
    PROBE_STOP(P_WALL);
}


/* ===================================================================
 * Exported: htc_bulk_c (unchanged, negligible time share)
 * =================================================================== */
__declspec(dllexport) void energy_loop_c(
    double *p, double *T, double *rho, double *h,
    double *G, double *alpha,
    double *T_wall,
    const double *Dh, const double *areas, const double *P_h,
    double dz, int n_ch, int n_ax,
    const int *ch_tp,
    const double *T_in,
    const int *flow_sign,
    double urf_T,
    const int *wall_ch_i, const int *wall_ch_j, int n_wall,
    const double *wall_tw, const double *wall_kw, const double *wall_Pw,
    const int *fluids,
    const double *mu_tab_w, const double *k_tab_w,
    const double *cp_tab_w, const double *rho_tab_w, const double *h_tab_w,
    const double *p_grid_w, int n_p_w,
    const double *T_grid_w, int n_T_w,
    const double *mu_tab_he, const double *k_tab_he,
    const double *cp_tab_he, const double *rho_tab_he, const double *h_tab_he,
    const double *p_grid_he, int n_p_he,
    const double *T_grid_he, int n_T_he,
    const double *Tsat_grid,
    const double *hl_grid, const double *hg_grid,
    const double *rho_l_sat_grid, const double *rho_g_sat_grid,
    const double *mu_l_sat_grid, const double *mu_g_sat_grid,
    const double *sigma_tab,
    double *htc_bulk,
    /* pre-allocated work arrays (caller provides, n_ch * n_ax each) */
    double *htc_film_all, double *cp_film_all, double *q_net,
    const double *T_wall_ext,
    const double *T_film
) {
    PROBE_START();
    if (n_ax > 40) return; /* fixed-size stack guard */
    double htc_film[40], cp_film[40], h_new[40], T_new[40];

    /* =================================================================
     * Pass 0: compute bulk-T-based htc (output in htc_bulk, replaces old htc_bulk_c)
     * ================================================================= */
    PROBE_START();
    for (int c = 0; c < n_ch; c++) {
        int is_water = (fluids[c] == 0);
        const double *mu_tab  = is_water ? mu_tab_w  : mu_tab_he;
        const double *k_tab   = is_water ? k_tab_w   : k_tab_he;
        const double *cp_tab  = is_water ? cp_tab_w  : cp_tab_he;
        const double *p_grid  = is_water ? p_grid_w  : p_grid_he;
        const double *T_grid  = is_water ? T_grid_w  : T_grid_he;
        int n_pv = is_water ? n_p_w : n_p_he;
        int n_Tv = is_water ? n_T_w : n_T_he;
        double _ir_log = 1.0 / (log(p_grid[n_pv-1]) - log(p_grid[0]));
        double _ir_T = 1.0 / (T_grid[n_Tv-1] - T_grid[0]);
        double Dh_c = Dh[c];
        for (int j = 0; j < n_ax; j++) {
            int idx = c * n_ax + j;
            int gj = c * (n_ax + 1) + j;
            int i, jg; double wp, wT;
            interp2d_indices(p_grid, n_pv, T_grid, n_Tv, _ir_log, _ir_T, p[idx], T[idx], &i, &jg, &wp, &wT);
            double mu = interp2d_at(mu_tab, n_Tv, i, jg, wp, wT);
            double k  = interp2d_at(k_tab,  n_Tv, i, jg, wp, wT);
            double cp = interp2d_at(cp_tab, n_Tv, i, jg, wp, wT);
            double Re = fabs(G[gj]) * Dh_c / (mu > 1e-10 ? mu : 1e-10);
            double Pr = mu * cp / (k > 1e-10 ? k : 1e-10);
            double Nu = nu_scalar_c(MAX(Re, 100), MAX(Pr, 0.1));
            htc_bulk[idx] = Nu * k / Dh_c;
        }
    }
    PROBE_STOP(P_HTC);

    /* =================================================================
     * Pass 1: compute film htc/cp for every channel/cell
     * ================================================================= */
    PROBE_START();
    for (int c = 0; c < n_ch; c++) {
        double *p_c   = p   + c * n_ax;
        const double *T_film_c = T_film + c * n_ax;
        double *G_c   = G   + c * (n_ax + 1);
        int is_he = fluids[c];
        const double *mu_tab  = is_he ? mu_tab_he  : mu_tab_w;
        const double *k_tab   = is_he ? k_tab_he   : k_tab_w;
        const double *cp_tab  = is_he ? cp_tab_he  : cp_tab_w;
        const double *p_grid  = is_he ? p_grid_he  : p_grid_w;
        const double *T_grid  = is_he ? T_grid_he  : T_grid_w;
        int n_p = is_he ? n_p_he : n_p_w;
        int n_T = is_he ? n_T_he : n_T_w;
        double _ir_log = 1.0 / (log(p_grid[n_p-1]) - log(p_grid[0]));
        double _ir_T = 1.0 / (T_grid[n_T-1] - T_grid[0]);
        double Dh_c_val = Dh[c];

        for (int j = 0; j < n_ax; j++) {
            double T_eff = T_film_c[j];
            /* for water, cap film temperature at saturation to avoid vapor props */
            if (!is_he) {
                double Tsat_j = interp1d(p_grid_w, Tsat_grid, n_p_w, p_c[j]);
                if (T_eff > Tsat_j) T_eff = Tsat_j;
            }
            int i, jg; double wp, wT;
            interp2d_indices(p_grid, n_p, T_grid, n_T, _ir_log, _ir_T, p_c[j], T_eff, &i, &jg, &wp, &wT);
            double mu_j = interp2d_at(mu_tab, n_T, i, jg, wp, wT);
            double k_j  = interp2d_at(k_tab,  n_T, i, jg, wp, wT);
            double cp   = interp2d_at(cp_tab, n_T, i, jg, wp, wT);
            double Re = fabs(G_c[j]) * Dh_c_val / (mu_j > 1e-10 ? mu_j : 1e-10);
            double Pr = mu_j * cp / (k_j > 1e-10 ? k_j : 1e-10);
            double Nu = nu_scalar_c(MAX(Re, 100), MAX(Pr, 0.1));
            htc_film_all[c * n_ax + j] = Nu * k_j / Dh_c_val;
            cp_film_all[c * n_ax + j]  = cp;
        }
    }

    PROBE_STOP(P_EN_P1);
    /* =================================================================
     * Pass 1b: two-phase heat-transfer enhancement (Chen + ONB)
     * Updates htc_film_all in place; htc_bulk is kept consistent so that
     * update_wall_temps_c sees the same htc used for q_wall.
     * ================================================================= */
    PROBE_START();
    double _w_ir_log = 1.0 / (log(p_grid_w[n_p_w-1]) - log(p_grid_w[0]));
    double _w_ir_T = 1.0 / (T_grid_w[n_T_w-1] - T_grid_w[0]);
    for (int c = 0; c < n_ch; c++) {
        if (!ch_tp[c]) continue;
        double *p_c = p + c * n_ax;
        double *h_c = h + c * n_ax;
        double *G_c = G + c * (n_ax + 1);
        for (int j = 0; j < n_ax; j++) {
            double Tsat_chen = interp1d(p_grid_w, Tsat_grid, n_p_w, p_c[j]);
            double hl_chen = interp1d(p_grid_w, hl_grid, n_p_w, p_c[j]);
            double hg_chen = interp1d(p_grid_w, hg_grid, n_p_w, p_c[j]);
            if (hl_chen >= hg_chen - 1.0) continue;
            double x_prev = quality_c(CLIP(h_c[j], hl_chen, hg_chen), hl_chen, hg_chen);
            double Tw_est = 2.0 * T_film[c * n_ax + j] - T[c * n_ax + j];
            double dT_sat = Tw_est - Tsat_chen;
            if (dT_sat <= 1.0) continue;
            double htc = htc_film_all[c * n_ax + j];
            if (x_prev <= 0.001 && dT_sat > 5.0) {
                double q_onb = 1.0e-3 * pow(dT_sat, 3.86);
                double htc_onb = q_onb / fmax(dT_sat, 1e-3);
                if (htc_onb > htc) htc = htc_onb;
            } else if (x_prev > 0.001 && x_prev < 0.999) {
                double rho_l_chen = interp1d(p_grid_w, rho_l_sat_grid, n_p_w, p_c[j]);
                double rho_g_chen = interp1d(p_grid_w, rho_g_sat_grid, n_p_w, p_c[j]);
                double mu_g_chen = interp1d(p_grid_w, mu_g_sat_grid, n_p_w, p_c[j]);
                /* liquid properties: cap film T at saturation */
                double T_chen = T_film[c * n_ax + j];
                if (T_chen > Tsat_chen) T_chen = Tsat_chen;
                int iF, jF; double wpF, wTF;
                interp2d_indices(p_grid_w, n_p_w, T_grid_w, n_T_w, _w_ir_log, _w_ir_T, p_c[j], T_chen, &iF, &jF, &wpF, &wTF);
                double mu_l_chen = interp2d_at(mu_tab_w, n_T_w, iF, jF, wpF, wTF);
                double k_liq = interp2d_at(k_tab_w, n_T_w, iF, jF, wpF, wTF);
                double cp_liq = interp2d_at(cp_tab_w, n_T_w, iF, jF, wpF, wTF);
                double sigma_val = interp1d(p_grid_w, sigma_tab, n_p_w, p_c[j]);
                double h_fg_chen = hg_chen - hl_chen;
                double htc_chen = chen_two_phase_c(x_prev, fabs(G_c[j]), Dh[c],
                                                   rho_l_chen, rho_g_chen,
                                                   mu_l_chen, mu_g_chen,
                                                   k_liq, cp_liq,
                                                   sigma_val, h_fg_chen,
                                                   dT_sat, Tsat_chen);
                if (htc_chen > htc) htc = htc_chen;
            }
            if (htc > htc_film_all[c * n_ax + j]) {
                htc_film_all[c * n_ax + j] = htc;
            }
            htc_bulk[c * n_ax + j] = htc_film_all[c * n_ax + j];
        }
    }
    PROBE_STOP(P_EN_P1B);

    /* =================================================================
     * Pass 2: compute net heat source + coupled wall heat fluxes
     * q_net[c][j] = external + sum of coupled wall contributions.
     * Coupled flux is directly accumulated (no temporary q_wall array).
     * ================================================================= */
    PROBE_START();
    /* Initialize q_net with external wall heating */
    for (int c = 0; c < n_ch; c++) {
        double P_h_c = P_h[c];
        for (int j = 0; j < n_ax; j++) {
            double htc = htc_film_all[c * n_ax + j];
            double val = 0.0;
            if (T_wall_ext) {
                val += htc * P_h_c * (T_wall_ext[c * n_ax + j] - T[c * n_ax + j]);
            }
            q_net[c * n_ax + j] = val;
        }
    }
    /* Add coupled wall fluxes directly into q_net */
    if (n_wall > 0) {
        for (int wi = 0; wi < n_wall; wi++) {
            int i = wall_ch_i[wi];
            int j = wall_ch_j[wi];
            double t_w = wall_tw[wi];
            double k_w = wall_kw[wi];
            double P_w = wall_Pw[wi];
            for (int jj = 0; jj < n_ax; jj++) {
                double htc_i = htc_film_all[i * n_ax + jj];
                double htc_j = htc_film_all[j * n_ax + jj];
                double U = 1.0 / (1.0/MAX(htc_i, 1e-10) + t_w/k_w + 1.0/MAX(htc_j, 1e-10));
                double q = U * P_w * (T[i * n_ax + jj] - T[j * n_ax + jj]);
                q_net[i * n_ax + jj] -= q;
                q_net[j * n_ax + jj] += q;
            }
        }
    }
    PROBE_STOP(P_EN_P2);

    /* =================================================================
     * Pass 3: march enthalpy/temperature with trapezoidal integration
     * of q_net so every cell's heat source is accounted for.
     * ================================================================= */
    PROBE_START();
    for (int c = 0; c < n_ch; c++) {
        double *p_c   = p   + c * n_ax;
        double *T_c   = T   + c * n_ax;
        double *h_c   = h   + c * n_ax;
        double *G_c   = G   + c * (n_ax + 1);
        double *alpha_c = alpha + c * n_ax;

        double Ac = areas[c];
        int is_tp = ch_tp[c];
        int is_he = fluids[c];

        /* select property table set per fluid (0=Water, 1=Helium) */
        const double *cp_tab  = is_he ? cp_tab_he  : cp_tab_w;
        const double *rho_tab = is_he ? rho_tab_he : rho_tab_w;
        const double *h_tab   = is_he ? h_tab_he   : h_tab_w;
        const double *p_grid  = is_he ? p_grid_he  : p_grid_w;
        const double *T_grid  = is_he ? T_grid_he  : T_grid_w;
        int n_p = is_he ? n_p_he : n_p_w;
        int n_T = is_he ? n_T_he : n_T_w;

        /* copy h, T; set inlet BC according to flow direction */
        for (int j = 0; j < n_ax; j++) { h_new[j] = h_c[j]; T_new[j] = T_c[j]; }
        int fs = flow_sign[c];
        int inlet_idx = (fs == 1) ? 0 : (n_ax - 1);
        T_new[inlet_idx] = T_in[c];
        h_new[inlet_idx] = interp2d(h_tab, p_grid, n_p, T_grid, n_T, p_c[inlet_idx], T_in[c]);
        if (is_tp) alpha_c[inlet_idx] = 0.0;

        double mdot = fabs(G_c[0]) * Ac;  /* mass flow rate is constant */

        if (fs == 1) {
            for (int j = 0; j < n_ax - 1; j++) {
                int j1 = j + 1;
                double dh = 0.5 * (q_net[c * n_ax + j] + q_net[c * n_ax + j1]) * dz / (mdot + 1e-10);
                h_new[j1] = h_new[j] + dh;
            }
        } else {
            for (int j = n_ax - 1; j > 0; j--) {
                int j1 = j - 1;
                double dh = 0.5 * (q_net[c * n_ax + j] + q_net[c * n_ax + j1]) * dz / (mdot + 1e-10);
                h_new[j1] = h_new[j] + dh;
            }
        }

        /* recover temperature from enthalpy, matching flow direction */
        if (fs == 1) {
            for (int j = 1; j < n_ax; j++) {
                if (j == inlet_idx) continue;
                int jprev = j - 1;
                double h_val = h_new[j], h_prev = h_new[jprev];
                if (is_tp) {
                    double Tsat = interp1d(p_grid_w, Tsat_grid, n_p_w, p_c[j]);
                    double hl = interp1d(p_grid_w, hl_grid, n_p_w, p_c[j]);
                    double hg = interp1d(p_grid_w, hg_grid, n_p_w, p_c[j]);
                    double cp_j = cp_film_all[c * n_ax + j];
                    double x_j = quality_c(CLIP(h_val, hl, hg), hl, hg);
                    double rho_l = interp1d(p_grid_w, rho_l_sat_grid, n_p_w, p_c[j]);
                    double rho_g = interp1d(p_grid_w, rho_g_sat_grid, n_p_w, p_c[j]);
                    double alpha_new = void_drift_flux_c(x_j, rho_l, rho_g, G_c[j], Dh[c]);
                    alpha_c[j] += (alpha_new - alpha_c[j]) * urf_T;
                    if (h_val < hl) {
                        double dh = h_val - h_prev;
                        T_new[j] = T_new[jprev] + dh / MAX(cp_j, 1e-10);
                        if (T_new[j] > Tsat) T_new[j] = Tsat;
                    } else if (h_val > hg) {
                        double cp_g = interp2d(cp_tab_w, p_grid_w, n_p_w, T_grid_w, n_T_w, p_c[j], Tsat + 0.5);
                        T_new[j] = Tsat + (h_val - hg) / MAX(cp_g, 1e-10);
                    } else {
                        T_new[j] = Tsat;
                    }
                } else {
                    double cp_j = cp_film_all[c * n_ax + j];
                    double dh = h_val - h_prev;
                    T_new[j] = T_new[jprev] + dh / MAX(cp_j, 1e-10);
                }
            }
        } else {
            for (int j = n_ax - 2; j >= 0; j--) {
                if (j == inlet_idx) continue;
                int jprev = j + 1;
                double h_val = h_new[j], h_prev = h_new[jprev];
                if (is_tp) {
                    double Tsat = interp1d(p_grid_w, Tsat_grid, n_p_w, p_c[j]);
                    double hl = interp1d(p_grid_w, hl_grid, n_p_w, p_c[j]);
                    double hg = interp1d(p_grid_w, hg_grid, n_p_w, p_c[j]);
                    double cp_j = cp_film_all[c * n_ax + j];
                    double x_j = quality_c(CLIP(h_val, hl, hg), hl, hg);
                    double rho_l = interp1d(p_grid_w, rho_l_sat_grid, n_p_w, p_c[j]);
                    double rho_g = interp1d(p_grid_w, rho_g_sat_grid, n_p_w, p_c[j]);
                    double alpha_new = void_drift_flux_c(x_j, rho_l, rho_g, G_c[j], Dh[c]);
                    alpha_c[j] += (alpha_new - alpha_c[j]) * urf_T;
                    if (h_val < hl) {
                        double dh = h_val - h_prev;
                        T_new[j] = T_new[jprev] + dh / MAX(cp_j, 1e-10);
                        if (T_new[j] > Tsat) T_new[j] = Tsat;
                    } else if (h_val > hg) {
                        double cp_g = interp2d(cp_tab_w, p_grid_w, n_p_w, T_grid_w, n_T_w, p_c[j], Tsat + 0.5);
                        T_new[j] = Tsat + (h_val - hg) / MAX(cp_g, 1e-10);
                    } else {
                        T_new[j] = Tsat;
                    }
                } else {
                    double cp_j = cp_film_all[c * n_ax + j];
                    double dh = h_val - h_prev;
                    T_new[j] = T_new[jprev] + dh / MAX(cp_j, 1e-10);
                }
            }
        }

        /* under-relax + rho update */
        for (int j = 0; j < n_ax; j++) {
            h_c[j] += (h_new[j] - h_c[j]) * urf_T;
            T_c[j] += (T_new[j] - T_c[j]) * urf_T;
            if (is_tp && alpha_c[j] > 0.0) {
                double Tsat_j = interp1d(p_grid_w, Tsat_grid, n_p_w, p_c[j]);
                double rho_l = interp1d(p_grid_w, rho_l_sat_grid, n_p_w, p_c[j]);
                double rho_g = interp1d(p_grid_w, rho_g_sat_grid, n_p_w, p_c[j]);
                rho[c * n_ax + j] = (1 - alpha_c[j]) * rho_l + alpha_c[j] * rho_g;
            } else {
                rho[c * n_ax + j] = interp2d(rho_tab, p_grid, n_p, T_grid, n_T, p_c[j], T_c[j]);
            }
        }
    }
    PROBE_STOP(P_EN_P3);

    /* htc_film_all, cp_film_all, q_net allocated by caller */
    PROBE_STOP(P_EN);
}


/* ===================================================================
 * Exported: htc_bulk_c (unchanged, negligible time share)
 * =================================================================== */
__declspec(dllexport) void htc_bulk_c(
    double *htc_out,
    const double *p, const double *T, const double *G,
    const double *Dh, int n_ch, int n_ax,
    const int *fluids,
    const double *mu_w, const double *k_w, const double *cp_w,
    const double *p_grid_w, int n_p_w,
    const double *T_grid_w, int n_T_w,
    const double *mu_he, const double *k_he, const double *cp_he,
    const double *p_grid_he, int n_p_he,
    const double *T_grid_he, int n_T_he
) {
    PROBE_START();
    for (int c = 0; c < n_ch; c++) {
        int is_water = (fluids[c] == 0);
        const double *mu_tab  = is_water ? mu_w  : mu_he;
        const double *k_tab   = is_water ? k_w   : k_he;
        const double *cp_tab  = is_water ? cp_w  : cp_he;
        const double *p_grid  = is_water ? p_grid_w : p_grid_he;
        const double *T_grid  = is_water ? T_grid_w : T_grid_he;
        int n_p = is_water ? n_p_w : n_p_he;
        int n_T = is_water ? n_T_w : n_T_he;
        double Dh_c = Dh[c];
        double _ir_log = 1.0 / (log(p_grid[n_p-1]) - log(p_grid[0]));
        double _ir_T = 1.0 / (T_grid[n_T-1] - T_grid[0]);

        for (int j = 0; j < n_ax; j++) {
            int idx = c * n_ax + j;
            int g_idx = c * (n_ax + 1) + j;
            int i, jg; double wp, wT;
            interp2d_indices(p_grid, n_p, T_grid, n_T, _ir_log, _ir_T, p[idx], T[idx], &i, &jg, &wp, &wT);
            double mu = interp2d_at(mu_tab, n_T, i, jg, wp, wT);
            double k  = interp2d_at(k_tab,  n_T, i, jg, wp, wT);
            double cp = interp2d_at(cp_tab, n_T, i, jg, wp, wT);
            double Re = fabs(G[g_idx]) * Dh_c / (mu > 1e-10 ? mu : 1e-10);
            double Pr = mu * cp / (k > 1e-10 ? k : 1e-10);
            double Nu = nu_scalar_c(MAX(Re, 100), MAX(Pr, 0.1));
            htc_out[idx] = Nu * k / Dh_c;
        }
    }
    PROBE_STOP(P_HTC);
}


/* ===================================================================
 * Exported: evaluate_post_c — one-pass post-processing (replaces Python loops)
 * Computes Qh/Qw/Q, dph/dpw, mean T/rho per fluid, max T_wall.
 * All in one C loop → no GIL contention.
 * =================================================================== */
__declspec(dllexport) void evaluate_post_c(
    const double *p, const double *T, const double *rho, const double *h,
    const double *T_wall,
    int n_ch, int n_ax, int n_wall,
    const int *flow_sign, const int *fluids, const double *m_dot_in,
    double *Qh, double *Qw,
    double *dph, double *dpw,
    double *Tha, double *Twa, double *rha, double *rwa,
    double *Twm
) {
    double Qh_acc = 0.0, Qw_acc = 0.0;
    double Th_sum = 0.0, Th_cnt = 0.0, Tw_sum = 0.0, Tw_cnt = 0.0;
    double rh_sum = 0.0, rh_cnt = 0.0, rw_sum = 0.0, rw_cnt = 0.0;
    double dph_max = 0.0, dpw_max = 0.0;
    double twm_val = 0.0;

    for (int c = 0; c < n_ch; c++) {
        int fs = flow_sign[c];
        int is_he = (fluids[c] == 1);
        int i_in = (fs == 1) ? 0 : (n_ax - 1);
        int i_out = (fs == 1) ? (n_ax - 1) : 0;
        double mdot = m_dot_in[c];

        double dh = h[c * n_ax + i_in] - h[c * n_ax + i_out];
        if (dh > 0) {
            Qh_acc += mdot * dh;
        } else {
            Qw_acc += mdot * (-dh);
        }

        double dp_c = p[c * n_ax + i_in] - p[c * n_ax + i_out];
        if (is_he) {
            if (dp_c > dph_max) dph_max = dp_c;
            for (int j = 0; j < n_ax; j++) {
                Th_sum += T[c * n_ax + j];
                Th_cnt += 1.0;
                rh_sum += rho[c * n_ax + j];
                rh_cnt += 1.0;
            }
        } else {
            if (dp_c > dpw_max) dpw_max = dp_c;
            for (int j = 0; j < n_ax; j++) {
                Tw_sum += T[c * n_ax + j];
                Tw_cnt += 1.0;
                rw_sum += rho[c * n_ax + j];
                rw_cnt += 1.0;
            }
        }
    }

    /* T_wall max */
    for (int wi = 0; wi < n_wall; wi++) {
        for (int j = 0; j < n_ax; j++) {
            double twi = T_wall[wi * 2 * n_ax + 0 * n_ax + j];
            double twj = T_wall[wi * 2 * n_ax + 1 * n_ax + j];
            if (twi > twm_val) twm_val = twi;
            if (twj > twm_val) twm_val = twj;
        }
    }

    *Qh  = Qh_acc;
    *Qw  = Qw_acc;
    *dph = dph_max;
    *dpw = dpw_max;
    *Tha = (Th_cnt > 0) ? (Th_sum / Th_cnt) : 823.15;
    *Twa = (Tw_cnt > 0) ? (Tw_sum / Tw_cnt) : 503.15;
    *rha = (rh_cnt > 0) ? (rh_sum / rh_cnt) : 3.0;
    *rwa = (rw_cnt > 0) ? (rw_sum / rw_cnt) : 800.0;
    *Twm = twm_val;
}
