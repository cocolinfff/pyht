   
                                                              
                                                                                    
                                                                                         
                                                                               
   

#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <stdio.h>

                                                                          
                                                                          
                                                                        
#define NAX_MAX 256

#ifdef PROFILE
#include <windows.h>
                                                        
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

                                                                            
  
                                                                            
                                                                           
                                                                      
                                                                           
                                                                            
                                                                   
  
                                                                            
                                                                 
  
                                                                     
                                                                           
   
typedef struct {
    const double *p_grid;
    int n_p;
    const double *u_grid;
    int n_T;
    const double *Tref;
    int is_norm;
    int k_sat;
} axis_t;
                                       
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

                                                                 
                                                                            
                                                                        
                                                                          
                                   
static inline double interp1d_logp(const double *p_grid, const double *y_grid,
                                   int n, double ir_log, double p) {
    if (p <= p_grid[0]) return y_grid[0];
    if (p >= p_grid[n-1]) return y_grid[n-1];
    (void)ir_log;
    int lo = 0, hi = n - 2;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (p_grid[mid] <= p) lo = mid; else hi = mid - 1;
    }
    double w = (p - p_grid[lo]) / (p_grid[lo+1] - p_grid[lo]);
    return y_grid[lo] + w * (y_grid[lo+1] - y_grid[lo]);
}

                                                                     
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

                                                                                        
                                                                           
                                  
                                                                                  
                                                                                    
                                                                                
                                                                          
                                                                                   
                                                                                  
                                                                                   
                                                                                    
                                           
                                                                              
                                                                              
                                                                               
static inline double h_isobar_at(const double *row_lo, const double *row_hi,
                                 int k, double wp) {
    return (1.0 - wp) * row_lo[k] + wp * row_hi[k];
}

                                               
  
                                                                     
                                                                              
static double T_invert(const double *h_tab, const axis_t *ax,
                       double p, double Tsat, double h_val) {
    const double *p_grid = ax->p_grid, *T_grid = ax->u_grid;
    const int n_p = ax->n_p, n_T = ax->n_T;
    if (n_T < 2) return Tsat;

                             
    p = CLIP(p, p_grid[0], p_grid[n_p-1]);
    double ir_log = 1.0 / (log(p_grid[n_p-1]) - log(p_grid[0]));
    double ratio = log(p / p_grid[0]) * ir_log;
    int ip = (int)(ratio * (n_p - 1));
    if (ip < 0) ip = 0; else if (ip > n_p - 2) ip = n_p - 2;
    double wp = (p - p_grid[ip]) / (p_grid[ip+1] - p_grid[ip]);
    double Tref = ax->is_norm
                ? (ax->Tref[ip] + wp * (ax->Tref[ip+1] - ax->Tref[ip]))
                : 1.0;
    double u_sat = (Tsat + 1.0) / Tref;
                                                                     
    int j0 = 0;
    while (j0 < n_T && T_grid[j0] <= u_sat) j0++;
    if (j0 > n_T - 2) j0 = n_T - 2;
    const double *row_lo = h_tab + (size_t)ip * n_T;
    const double *row_hi = row_lo + n_T;

    double h_top = h_isobar_at(row_lo, row_hi, n_T - 1, wp);
    if (h_val >= h_top) {
                                                                                
        double h_prev = h_isobar_at(row_lo, row_hi, n_T - 2, wp);
        double dh = h_top - h_prev;
        double dT = T_grid[n_T-1] - T_grid[n_T-2];
        double u_out = T_grid[n_T-1];
        if (dh > 1e-12 && dT > 1e-12)
            u_out = T_grid[n_T-1] + (h_val - h_top) * dT / dh;
        return u_out * Tref;
    }
                                                                                 
                                                                                    
                                                                       
    int lo = j0, hi = n_T - 2;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (h_isobar_at(row_lo, row_hi, mid, wp) <= h_val) lo = mid;
        else hi = mid - 1;
    }
    int j = lo;
    double h_lo = h_isobar_at(row_lo, row_hi, j, wp);
    if (h_val <= h_lo) return Tsat;                                          
    double h_hi = h_isobar_at(row_lo, row_hi, j + 1, wp);
    double dh = h_hi - h_lo;
    if (dh <= 1e-12) return T_grid[j] * Tref;                          
    double u_out = T_grid[j] + (h_val - h_lo) * (T_grid[j+1] - T_grid[j]) / dh;
    return u_out * Tref;
}



static inline void interp2d_indices(
    const axis_t *ax, double p, double T,
    int *i, int *j, double *wp, double *wT) {
    const double *pg = ax->p_grid;
    const double *ug = ax->u_grid;
    const int n_p = ax->n_p, n_T = ax->n_T;
    p = CLIP(p, pg[0], pg[n_p-1]);
    double ir_log = 1.0 / (log(pg[n_p-1]) - log(pg[0]));
    double ratio = log(p / pg[0]) * ir_log;                            
    *i = CLIP((int)(ratio * (n_p - 1)), 0, n_p - 2);
    *wp = (p - pg[*i]) / (pg[*i+1] - pg[*i]);
    if (ax->is_norm) {
        double Tref = ax->Tref[*i] + (*wp) * (ax->Tref[*i+1] - ax->Tref[*i]);
        double u = CLIP(T / Tref, ug[0], ug[n_T-1]);
        if (ax->k_sat >= 0 && u > 1.0 - 1.0e-9 && u <= 1.0 + 1.0e-3) {
            *j = ax->k_sat - 1;
            *wT = 1.0;
        } else {
            int lo = 0, hi = n_T - 1;
            while (lo + 1 < hi) {
                int mid = (lo + hi) >> 1;
                if (ug[mid] <= u) lo = mid; else hi = mid;
            }
            *j = CLIP(lo, 0, n_T - 2);
            double dT = ug[*j+1] - ug[*j];
            *wT = (dT > 0.0) ? (u - ug[*j]) / dT : 0.0;
        }
    } else {
        double u = CLIP(T, ug[0], ug[n_T-1]);
        double ir_T = 1.0 / (ug[n_T-1] - ug[0]);
        *j = CLIP((int)((u - ug[0]) * ir_T * (n_T - 1)), 0, n_T - 2);
        *wT = (u - ug[*j]) / (ug[*j+1] - ug[*j]);
    }
}

static inline double interp2d_at(const double *tab, int n_T,
                                 int i, int j, double wp, double wT) {
    return (1-wp)*(1-wT)*tab[(size_t)i*n_T + j] +
           wp*(1-wT)*tab[(size_t)(i+1)*n_T + j] +
           (1-wp)*wT*tab[(size_t)i*n_T + j+1] +
           wp*wT*tab[(size_t)(i+1)*n_T + j+1];
}

static inline double interp2d(const double *tab, const axis_t *ax,
                              double p, double T) {
    int i, j; double wp, wT;
    interp2d_indices(ax, p, T, &i, &j, &wp, &wT);
    return interp2d_at(tab, ax->n_T, i, j, wp, wT);
}

__declspec(dllexport) void table_lookup_bulk(
    const double *tab, const axis_t *ax,
    const double *p, const double *T, double *out, int n)
{
    for (int k = 0; k < n; k++)
        out[k] = interp2d(tab, ax, p[k], T[k]);
}

                                                                                                   
                                
  
                                                                               
                                                                              
                                                                              
                                                                            
               
  
                                                                             
                                                                           
                                                                              
              
static double _g_roughness_m = 10.0e-6;

                                                                             
                                                  
__declspec(dllexport) void pyht_set_roughness(double m) {
    _g_roughness_m = (m < 0.0) ? 10.0e-6 : m;
}

__declspec(dllexport) double pyht_get_roughness(void) {
    return _g_roughness_m;
}

static int _g_chen_nu_gnielinski = 1;

__declspec(dllexport) void pyht_set_chen_nu(int mode) {
    _g_chen_nu_gnielinski = (mode != 0);
}

__declspec(dllexport) int pyht_get_chen_nu(void) {
    return _g_chen_nu_gnielinski;
}

static double ff_scalar_eps(double Re, double Dh, double roughness);

static double ff_scalar_c(double Re, double Dh) {
    PROBE_INC(P_FFCNT);
    return ff_scalar_eps(Re, Dh, _g_roughness_m);
}

                                                                              
                                                                  
static double ff_scalar_eps(double Re, double Dh, double roughness) {
    if (Re < 1.0) Re = 1.0;
                                                                                
                                                                           
                                                                           
                                                                               
                                                                               
                                                                              
                                                                
    const double f_re_lam = 56.91;                                               
    double f_lam = f_re_lam / Re;
                                                                              
                                                                              
                                                                              
                                                                             
                                                                            
                                                                                   
    if (Re <= 2300.0) return f_lam;
                                                                        
    double t1 = pow(7.0 / Re, 0.9) + 0.27 * roughness / Dh;
    double t1l = log(1.0 / t1);
    double a0 = 2.457 * t1l;
    double a2 = a0 * a0, a4 = a2 * a2, a8 = a4 * a4, a16 = a8 * a8;
                           
    double b0 = 37530.0 / Re;
    double b2 = b0 * b0, b4 = b2 * b2, b8 = b4 * b4, b16 = b8 * b8;
                                                         
    double c0 = 8.0 / Re;
    double c2 = c0 * c0, c4 = c2 * c2, c8 = c4 * c4, c12 = c8 * c4;
    double d0 = a16 + b16;
    double d15 = d0 * sqrt(d0);                 
    double ft0 = c12 + 1.0 / d15;
    double ft1 = pow(ft0, 1.0 / 12.0);
    double f_turb = 8.0 * ft1;
    if (Re < 4000) return f_lam + (f_turb - f_lam) * (Re - 2300.0) / 1700.0;
    return f_turb;
}

                                              
static double nu_scalar_c(double Re, double Pr) {
                                                                       
                                                                             
                                       
      
                                                                              
                                                                       
                                                                           
                                                                              
                                                                             
                                                                              
                       
      
                                                                               
                                                                            
                                                                     
      
                                                                             
                                                                              
                                                                               
                                                                               
       
    if (Re < 2300) return 3.66;
    if (Re < 2301) Re = 2301;
    double t = 0.79 * log(Re) - 1.64;
    double f = 1.0 / (t * t);                                
    double Pr23 = exp(2.0/3.0 * log(Pr));                  
    double Nu = (f/8) * (Re - 1000) * Pr / (1 + 12.7 * sqrt(f/8) * (Pr23 - 1));
    Nu = MAX(Nu, 0.1);
                                                                             
                                                                            
                                                                           
                                                                          
                                                    
      
                                                               
      
                                                                        
                                                                          
                                                                               
                                                                           
                                                                       
    if (Re < 4000.0) {
        double w = (Re - 2300.0) / 1700.0;
        Nu = 3.66 + (Nu - 3.66) * w;
    }
    return Nu;
}

                                                                             
                                                                           
                                                                           
                                                                           
                                                                           
                                                                          
            
  
                                          
                                                                      
  
                                                                            
                                                                             
  
                                                                 
  
                                                                            
                                                                               
                                                                            
                                                       
  
                                                                               
  
                                                                  
  
                                                                             
                                                                               
                          
  
                                                                            
                                                                        
  
                                                                              
                                                                              
                                           
static double duct_nu_factor(double Re, int shape, double cut_ratio) {
                                                       
      
                                      
      
                                            
      
                                                                            
                                                                               
                                                                              
                                                                                
                                                                                
                                                                     
      
                                                                                
                                                                                
      
                                                   
                                                    
                                                    
                                                    
      
                                                                             
                                                                      
                                                                               
                                                           
      
                                                                              
                                                                                 
                                                                              
                                                                           
       
    if (shape == 1) {                            
        static const double Re_tab[3] = { 4000.0, 6000.0, 10000.0 };
        static const double f_tab[3]  = { 0.98590, 0.90679, 0.86070 };
        if (Re <= Re_tab[0]) {
                                                                               
                                                                                
                                                       
            double w = (Re - 2300.0) / (Re_tab[0] - 2300.0);
            if (w < 0.0) w = 0.0;
            return 0.98579265 + (f_tab[0] - 0.98579265) * w;
        }
        if (Re >= Re_tab[2]) return f_tab[2];
        {
            int i = (Re < Re_tab[1]) ? 0 : 1;
            double l0 = log(Re_tab[i]),   l1 = log(Re_tab[i + 1]);
            double w  = (log(Re) - l0) / (l1 - l0);
            return f_tab[i] + (f_tab[i + 1] - f_tab[i]) * w;
        }
    }
                                                                             
                                                                            
                                                                       
                                                                          
      
                                                                          
      
                                                                              
      
                                                       
                                                                       
      
                                                                  
                                                                  
                                                                              
                                                                 
      
                                                                         
                                                                            
                                                                          
                                                                      
                               
      
                                                                           
                                                                             
                                             
      
                                                                                
      
                                                                     
                                                                              
      
                                                     
                                                                           
                                                  
                                                  
                                                  
      
                                                                            
                                                                         
                                                                             
                                             
      
                                                                     
                                                                        
                                                                               
                                                                             
                                                                        
                                                                               
      
                                                                        
                                                                               
                                                                              
                                                                             
                                                           
       
    if (shape == 3) {
        static const double RE_TAB[4] = { 500.0, 4000.0, 8000.0, 12000.0 };
                                                                  
        static const double PHI_R1[4] = { 1.1241, 1.0167, 1.0178, 0.9796 };
        static const double PHI_R2[4] = { 1.1509, 1.0572, 1.0434, 1.0314 };
        const double R1 = 0.146446609, R2 = 0.292893218;
        double fsq = duct_nu_factor(Re, 1, 0.0);
        double r = cut_ratio;
        int i, k;
        double w, wl, p1, p2, phi;
        if (!(r > 0.0)) return fsq;                                     
        if (r > R2) r = R2;
                                                                           
        if (Re <= RE_TAB[0]) {
            i = 0; w = 0.0;
        } else if (Re >= RE_TAB[3]) {
            i = 2; w = 1.0;
        } else {
            i = 0;
            while (i < 2 && Re > RE_TAB[i + 1]) i++;
            w = (log(Re) - log(RE_TAB[i]))
                / (log(RE_TAB[i + 1]) - log(RE_TAB[i]));
        }
        k = (w < 0.0) ? 0 : ((w > 1.0) ? 1 : 1);
        (void) k;
        p1 = PHI_R1[i] + (PHI_R1[i + 1] - PHI_R1[i]) * w;
        p2 = PHI_R2[i] + (PHI_R2[i + 1] - PHI_R2[i]) * w;
                                           
        if (r <= R1) {
            wl = r / R1;
            phi = 1.0 + (p1 - 1.0) * wl;
        } else {
            wl = (r - R1) / (R2 - R1);
            phi = p1 + (p2 - p1) * wl;
        }
        if (phi < 0.5) phi = 0.5;
        return fsq * phi;
    }
                                                                                
                                                                                
                                                             
    {
        double x = Re / 5000.0;
        if (x <= 0.0) return 1.0;
        double c = 0.897 + 0.137 * pow(x, -1.63);
        return c > 1.0 ? 1.0 : c;
    }
}

                                                           
                                                                         
                                                                      
                                                                            
                                                                            
                                                                        
static double nu_rough_factor(double Re, double Dh) {
    if (Re <= 2300.0 || Dh <= 1e-12) return 1.0;
                                                               
      
                                                                           
                                                        
      
                                                                            
                                                                              
                                                                        
                                                                             
                                                                              
                                                                                
                            
    double f_s = ff_scalar_eps(Re, Dh, 0.0);
    double f_r = ff_scalar_eps(Re, Dh, _g_roughness_m);
    if (f_s <= 1e-12) return 1.0;
    return pow(f_r / f_s, 0.68);
}

static inline double sp_htc_c(double G_abs, double Dh, double mu, double cp,
                              double k, int shape, double cut) {
    double Re = MAX(G_abs * Dh / (mu > 1e-10 ? mu : 1e-10), 100.0);
    double Pr = MAX(mu * cp / (k > 1e-10 ? k : 1e-10), 0.1);
    double Nu = nu_scalar_c(Re, Pr);
    return Nu * k / Dh * nu_rough_factor(Re, Dh)
         * (shape ? duct_nu_factor(Re, shape, cut) : 1.0);
}

                                                                                   
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

                           
static inline double quality_c(double h, double hl, double hg) {
    if (hl >= hg - 1) return 0.0;
    return CLIP((h - hl) / (hg - hl), 0.0, 1.0);
}

                                                                               
  
                                                                          
                                                                           
                                                                             
                                                                         
                                                                         
                                                                             
                                                                           
                                                                     
  
                                                                            
                                                        
   
#define ALPHA_BLEND_EPS 0.02

static inline double _hermite01(double t) {
                                                                            
    if (t <= 0.0) return 0.0;
    if (t >= 1.0) return 1.0;
    return t * t * (3.0 - 2.0 * t);
}

static inline double void_drift_flux_c(double x, double rho_l, double rho_g,
                                        double G, double Dh) {
    double C0 = (Dh > 0.01) ? 1.2 : 1.13;
    double arg = 0.07 * 9.81 * (rho_l - rho_g) / fmax(rho_l * rho_l, 1e-10);
    double V_gj = 1.53 * sqrt(sqrt(arg));                                           
    double G_eff = fmax(fabs(G), 1.0);
                                                                            
                                                                  
    double x_c = CLIP(x, 1.0e-6, 1.0 - 1.0e-6);
    double alpha_raw;
    if (fabs(G) < 1e-6) {
        alpha_raw = 1.0 / (1.0 + (1.0 - x_c) / x_c * rho_g / rho_l);
    } else {
        alpha_raw = x_c / (rho_g * (C0 * (x_c/rho_g + (1.0 - x_c)/rho_l)
                                    + V_gj / G_eff));
    }
    alpha_raw = CLIP(alpha_raw, 0.0, 1.0);
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    if (x < ALPHA_BLEND_EPS) {
                                                                            
                                           
        double w = _hermite01(x / ALPHA_BLEND_EPS);
        return w * alpha_raw;
    }
    if (x > 1.0 - ALPHA_BLEND_EPS) {
        double w = _hermite01((x - (1.0 - ALPHA_BLEND_EPS)) / ALPHA_BLEND_EPS);
        return alpha_raw + w * (1.0 - alpha_raw);
    }
    return alpha_raw;
}

static double chen_F_c(double x, double rho_l, double rho_g,
                       double mu_l, double mu_g) {
    double xs = x > 0.0 ? x : 1.0e-10;
    double X_tt = pow((1.0-xs)/xs, 0.9) * sqrt(rho_g/rho_l)
                * pow(mu_l/MAX(mu_g,1.0e-10), 0.1);
    double X_inv = 1.0 / MAX(X_tt, 1.0e-10);
    double F = (X_inv <= 0.1) ? 1.0 : 2.35 * pow(X_inv + 0.213, 0.736);
    return (F > 70.0) ? 70.0 : F;
}

static double chen_Re_l_c(double x, double G, double Dh, double mu_l) {
    double xs = x > 0.0 ? x : 1.0e-10;
    return G * (1.0 - xs) * Dh / MAX(mu_l, 1e-10);
}

static double chen_two_phase_c(double x, double G, double Dh,
                                double rho_l, double rho_g,
                                double mu_l, double mu_g,
                                double k_l, double cp_l,
                                double sigma, double h_fg,
                                double dT_sat, double Tsat_K, double q_flux,
                                double htc_sp_l) {
    if (x <= 0.0 || x >= 1.0) return 0.0;



    if (q_flux <= 0.0 && dT_sat <= 1.0) return 0.0;
    double Pr_l = mu_l * cp_l / MAX(k_l, 1e-10);
    double F = chen_F_c(x, rho_l, rho_g, mu_l, mu_g);
    double Re_l = chen_Re_l_c(x, G, Dh, mu_l);
    double h_macro;
    if (_g_chen_nu_gnielinski && htc_sp_l > 0.0) {
        h_macro = F * htc_sp_l;
    } else {
        h_macro = 0.023 * pow(Re_l, 0.8) * pow(Pr_l, 0.4) * k_l / Dh * F;
    }
    double Re_tp = Re_l * pow(F, 1.25);
    double S = 1.0 / (1.0 + 2.53e-6 * pow(Re_tp, 1.17));
                                                                        
                                                                                 
                                                                               
                                                                                      
    double fz_liq = pow(k_l, 0.79) * pow(cp_l, 0.45) * pow(rho_l, 0.49);
                                                                             
    double C_nb = (0.00122 * fz_liq * pow(h_fg * rho_g / MAX(Tsat_K, 1.0), 0.75)) /
        (sqrt(sigma) * pow(mu_l,0.29) * pow(h_fg,0.24) * pow(rho_g,0.24)) * S;
    double h_micro;
    if (q_flux > 0.0) {
                                                                                 
                                                                             
                                                                                
                                                                             
                                                                                
                                                                             
                                                                               
                                                                       
                                                                            
        double q_nb = q_flux - h_macro * dT_sat;
        if (q_nb < 0.0) q_nb = 0.0;
        h_micro = pow(C_nb, 0.5025) * pow(q_nb, 0.4975);
    } else {
        double dP_sat = dT_sat * h_fg * rho_g / MAX(Tsat_K, 1.0);
        h_micro = C_nb * pow(dT_sat, 0.24) * pow(dP_sat, 0.75)
                / pow(h_fg * rho_g / MAX(Tsat_K, 1.0), 0.75);
    }
    return h_macro + (h_micro > 0 ? h_micro : 0);
}

__declspec(dllexport) double pyht_chen_macro_c(
    double x, double G, double Dh, double rho_l, double rho_g,
    double mu_l, double mu_g, double k_l, double cp_l, double htc_sp_l)
{
    if (x <= 0.0 || x >= 1.0) return 0.0;
    double Pr_l = mu_l * cp_l / MAX(k_l, 1.0e-10);
    double F = chen_F_c(x, rho_l, rho_g, mu_l, mu_g);
    double Re_l = chen_Re_l_c(x, G, Dh, mu_l);
    return (_g_chen_nu_gnielinski && htc_sp_l > 0.0)
        ? F * htc_sp_l
        : 0.023 * pow(Re_l, 0.8) * pow(Pr_l, 0.4) * k_l / Dh * F;
}


static inline double friction_face(
    int is_tp, double G_face, double h_down, double p_down,
    double mu_down, double Dh, double dz, double rho_down,
    const double *p_grid, int n_p,
    const double *hl_sat_grid, const double *hg_sat_grid,
    const double *rho_l_sat_grid, const double *rho_g_sat_grid,
    const double *mu_l_sat_grid, const double *mu_g_sat_grid,
    const double *sigma_tab,
    double *a_lin_inc)
{
    double dp_fric;
    if (is_tp) {
        int i_sat; double w_sat;
        i_sat = interp1d_idx(p_grid, n_p, p_down, &w_sat);
        double hl = interp1d_at_w(hl_sat_grid, i_sat, w_sat);
        double hg = interp1d_at_w(hg_sat_grid, i_sat, w_sat);
        double x = quality_c(h_down, hl, hg);
        double mu_liq = interp1d_at_w(mu_l_sat_grid, i_sat, w_sat);
        double mu_re = (x > 0.0) ? mu_liq : mu_down;
        double Re_lo = G_face * Dh / (mu_re > 1e-10 ? mu_re : 1e-10);
        double f_lo = ff_scalar_c(Re_lo, Dh);
        if (x > 0.0) {
            double rho_l_ = interp1d_at_w(rho_l_sat_grid, i_sat, w_sat);
            double rho_g_ = interp1d_at_w(rho_g_sat_grid, i_sat, w_sat);
            double mu_g = interp1d_at_w(mu_g_sat_grid, i_sat, w_sat);
            double sigma_j = MAX(interp1d_at_w(sigma_tab, i_sat, w_sat), 0.001);
            double phi2 = friedel_c(x, rho_l_, rho_g_, mu_re, mu_g,
                                    G_face, Dh, sigma_j);
            dp_fric = f_lo * phi2 * fabs(G_face) * G_face
                    / (2.0 * Dh * rho_down) * dz;
        } else {
            dp_fric = f_lo * fabs(G_face) * G_face
                    / (2.0 * Dh * rho_down) * dz;
        }
        if (a_lin_inc)
            *a_lin_inc = 32.0 * mu_re * dz / (Dh * Dh * rho_down);
    } else {
        double Re_j = G_face * Dh / (mu_down > 1e-10 ? mu_down : 1e-10);
        double f = ff_scalar_c(Re_j, Dh);
        dp_fric = f * fabs(G_face) * G_face / (2.0 * Dh * rho_down) * dz;
        if (a_lin_inc)
            *a_lin_inc = 32.0 * mu_down * dz / (Dh * Dh * rho_down);
    }
    return dp_fric;
}


                                                                      
                                                                
                                                                         
__declspec(dllexport) void momentum_loop_c(
    double *p, double *T, double *rho, double *h,
    double *G, const double *Dh, double dz,
    int n_ch, int n_ax,
    const double *k_in, const double *k_out,                                            
    double *a_lin,                                                                    
                                                                                               
    double *dp_fric_acc,                                                                    
                                                                                            
                                                                                                
                                                                                       
                                                                                          
                                                                                    
                                                                 
    const int *ch_tp,
    const double *p_out,
    const int *flow_sign,
    double urf_p,
    const int *fluids,
    const double *mu_tab_w, const double *k_tab_w,
    const double *cp_tab_w, const double *rho_tab_w, const double *h_tab_w,
    const double *p_grid_w, int n_p_w,
    const axis_t *ax_w,
    const double *mu_tab_he, const double *k_tab_he,
    const double *cp_tab_he, const double *rho_tab_he, const double *h_tab_he,
    const double *p_grid_he, int n_p_he,
    const axis_t *ax_he,
    const double *Tsat_grid,
    const double *hl_sat_grid, const double *hg_sat_grid,
    const double *rho_l_sat_grid, const double *rho_g_sat_grid,
    const double *mu_l_sat_grid, const double *mu_g_sat_grid,
    const double *sigma_tab
) {
    PROBE_START();
    if (n_ax > NAX_MAX) return;                             
    double p_new[NAX_MAX];
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

                                                 
        const double *mu_tab  = is_he ? mu_tab_he  : mu_tab_w;
        const double *k_tab   = is_he ? k_tab_he   : k_tab_w;
        const double *cp_tab  = is_he ? cp_tab_he  : cp_tab_w;
        const double *rho_tab = is_he ? rho_tab_he : rho_tab_w;
        const double *h_tab   = is_he ? h_tab_he   : h_tab_w;
        const axis_t *ax = is_he ? ax_he : ax_w;
        const double *p_grid = ax->p_grid;
        int n_p = ax->n_p;
        int n_T = ax->n_T;

        for (int j = 0; j < n_ax; j++) p_new[j] = p_c[j];
        int fs = flow_sign[c];

                                                                     
                                                          
                                                   
                                 
          
                                             
                                                  
                                      
          
                                                  
                              
        if (a_lin) a_lin[c] = 0.0;
        if (dp_fric_acc) dp_fric_acc[c] = 0.0;
        double dp_ports = 0.0;
        double G_mag = fabs(G_c[0]);
        if (G_mag < 1e-300) G_mag = fabs(G_c[n_ax]);
        if (k_in || k_out) {
            double rho_o = (fs >= 0) ? rho_c[n_ax-1] : rho_c[0];
            double rho_i = (fs >= 0) ? rho_c[0] : rho_c[n_ax-1];
            if (k_out && k_out[c] > 0.0)
                dp_ports += k_out[c] * G_mag * G_mag / (2.0 * MAX(rho_o, 1e-10));
            if (k_in && k_in[c] > 0.0)
                dp_ports += k_in[c] * G_mag * G_mag / (2.0 * MAX(rho_i, 1e-10));
        }

                                                                                        
                                                              
                                                                                        
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
                interp2d_indices(ax, p_c[down], T_c[down], &ii, &jj, &wp, &wT);
                double mu_down = interp2d_at(mu_tab, n_T, ii, jj, wp, wT);

                double a_inc = 0.0;
                double dp_fric = friction_face(
                    is_tp, G_face, h_c[down], p_c[down],
                    mu_down, Dh_c_val, dz, rho_down,
                    p_grid, n_p,
                    hl_sat_grid, hg_sat_grid,
                    rho_l_sat_grid, rho_g_sat_grid,
                    mu_l_sat_grid, mu_g_sat_grid,
                    sigma_tab,
                    a_lin ? &a_inc : NULL);
                                                                  
                                                     
                                  
                if (a_lin) a_lin[c] += a_inc;
                if (dp_fric_acc) dp_fric_acc[c] += fabs(dp_fric);
                double dp_grav = rho_down * 9.81 * dz;
                double dp_acc = G_c[face] * G_c[face] *
                                (1.0/fmax(rho_down, 1e-10) - 1.0/fmax(rho_up, 1e-10));
                double p_new_up = p_new[down] + dp_fric + dp_grav + dp_acc;
                p_new[up] = MAX(p_new_up, 1e4);
            }
                                                                        
                                                                             
                                                                               
                                                                      
                                                                        
                                                                            
                                                                   
            if (n_ax > 1) p_new[0] += p_new[0] - p_new[1];
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
                interp2d_indices(ax, p_c[down], T_c[down], &ii, &jj, &wp, &wT);
                double mu_down = interp2d_at(mu_tab, n_T, ii, jj, wp, wT);

                double a_inc = 0.0;
                double dp_fric = friction_face(
                    is_tp, G_face, h_c[down], p_c[down],
                    mu_down, Dh_c_val, dz, rho_down,
                    p_grid, n_p,
                    hl_sat_grid, hg_sat_grid,
                    rho_l_sat_grid, rho_g_sat_grid,
                    mu_l_sat_grid, mu_g_sat_grid,
                    sigma_tab,
                    a_lin ? &a_inc : NULL);
                if (a_lin) a_lin[c] += a_inc;
                                                                                           
                if (dp_fric_acc) dp_fric_acc[c] += fabs(dp_fric);
                double dp_grav = -rho_down * 9.81 * dz;
                double dp_acc = G_c[face] * G_c[face] *
                                (1.0/fmax(rho_down, 1e-10) - 1.0/fmax(rho_up, 1e-10));
                double p_new_up = p_new[down] + dp_fric + dp_grav + dp_acc;
                p_new[up] = MAX(p_new_up, 1e4);
            }
                                                                               
                                              
            if (n_ax > 1) p_new[n_ax - 1] += p_new[n_ax - 1] - p_new[n_ax - 2];
        }
        if (dp_ports > 0.0) {
            if (fs >= 0) p_new[0] += dp_ports;
            else         p_new[n_ax - 1] += dp_ports;
        }
        for (int j = 0; j < n_ax; j++) {
            p_c[j] += (p_new[j] - p_c[j]) * urf_p;
            if (p_c[j] < 1e4) p_c[j] = 1e4;
        }
    }
    PROBE_STOP(P_MOM);
}


                                                                      
                                                                     
                                                                   
                                                                         
__declspec(dllexport) void update_wall_temps_c(
    double *T_wall,
    const double *T, const double *htc_bulk,
    const int *wall_ch_i, const int *wall_ch_j,
    const double *wall_tw, const double *wall_kw,
    const double *wall_Pi, const double *wall_Pj, const double *wall_Pcond,
    int n_wall, int n_ax
) {
    PROBE_START();
    for (int wi = 0; wi < n_wall; wi++) {
        int i = wall_ch_i[wi];
        int j = wall_ch_j[wi];
        double t_w = wall_tw[wi];
        double k_w = wall_kw[wi];
                                                                              
                                                                             
                           
        double P_i = wall_Pi[wi];
        double P_j = wall_Pj[wi];
        double P_c = wall_Pcond[wi];
        double R_cond = t_w / (k_w * P_c + 1e-30);

        for (int jj = 0; jj < n_ax; jj++) {
            double Ti = T[i * n_ax + jj];
            double Tj = T[j * n_ax + jj];
            double hi = htc_bulk[i * n_ax + jj];
            double hj = htc_bulk[j * n_ax + jj];
            double R_i = 1.0 / (hi * P_i + 1e-30);
            double R_j = 1.0 / (hj * P_j + 1e-30);
            double R_tot = R_i + R_cond + R_j;
            double q = (Ti - Tj) / R_tot;                             
            double Tw_i = Ti - q * R_i;
            double Tw_j = Tj + q * R_j;
            T_wall[wi * 2 * n_ax + 0 * n_ax + jj] = Tw_i;
            T_wall[wi * 2 * n_ax + 1 * n_ax + jj] = Tw_j;
        }
    }
    PROBE_STOP(P_WALL);
}


                                                                      
                                                          
                                                                         
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
    double urf_htc,
    const int *wall_ch_i, const int *wall_ch_j, int n_wall,
    const double *wall_tw, const double *wall_kw,
    const double *wall_Pi, const double *wall_Pj, const double *wall_Pcond,
    const int *wall_one_sided,                                   
    const int *fluids,
    const int *ch_shape,                                                            
                                                               
    const double *ch_cut,                                                      
                                                                            
    const double *mu_tab_w, const double *k_tab_w,
    const double *cp_tab_w, const double *rho_tab_w, const double *h_tab_w,
    const double *p_grid_w, int n_p_w,
    const axis_t *ax_w,
    const double *mu_tab_he, const double *k_tab_he,
    const double *cp_tab_he, const double *rho_tab_he, const double *h_tab_he,
    const double *p_grid_he, int n_p_he,
    const axis_t *ax_he,
    const double *Tsat_grid,
    const double *hl_grid, const double *hg_grid,
    const double *rho_l_sat_grid, const double *rho_g_sat_grid,
    const double *mu_l_sat_grid, const double *mu_g_sat_grid,
    const double *k_l_sat_grid, const double *cp_l_sat_grid,
    const double *sigma_tab,
    double *htc_bulk,
                                                                       
    double *htc_film_all, double *cp_film_all, double *q_net,
    const double *T_wall_ext,
    const double *T_film
) {
    PROBE_START();
    if (n_ax > NAX_MAX) return;                             
                                                                            
                                                                           
                                                                      
                                                                         
                                                                               
                                                                          
                                                       
    double *tcap = (double *)malloc(sizeof(double) * 2 * (size_t)n_ch * (size_t)n_ax);
    if (!tcap) return;
    double *tcap_lo = tcap + (size_t)n_ch * (size_t)n_ax;
                                                                               
                                                            
    double *Pw_tot = (double *)calloc((size_t)n_ch, sizeof(double));
    double htc_film[NAX_MAX], cp_film[NAX_MAX], h_new[NAX_MAX], T_new[NAX_MAX];
                                                                                        
    double htc_sp_all[NAX_MAX];

                                                                        
                                                                                     
                                                                           
    PROBE_START();
    for (int c = 0; c < n_ch; c++) {
                                                                                 
                                                                                  
                                                                             
                                                                                  
                         
        if (ch_tp && ch_tp[c]) continue;
        int is_water = (fluids[c] == 0);
        const double *mu_tab  = is_water ? mu_tab_w  : mu_tab_he;
        const double *k_tab   = is_water ? k_tab_w   : k_tab_he;
        const double *cp_tab  = is_water ? cp_tab_w  : cp_tab_he;
        const axis_t *ax = is_water ? ax_w : ax_he;
        int n_Tv = ax->n_T;
        double Dh_c = Dh[c];
        for (int j = 0; j < n_ax; j++) {
            int idx = c * n_ax + j;
            int gj = c * (n_ax + 1) + j;
            int i, jg; double wp, wT;
            interp2d_indices(ax, p[idx], T[idx], &i, &jg, &wp, &wT);
            double mu = interp2d_at(mu_tab, n_Tv, i, jg, wp, wT);
            double k  = interp2d_at(k_tab,  n_Tv, i, jg, wp, wT);
            double cp = interp2d_at(cp_tab, n_Tv, i, jg, wp, wT);
            htc_bulk[idx] = sp_htc_c(fabs(G[gj]), Dh_c, mu, cp, k,
                                     ch_shape ? ch_shape[c] : 0,
                                     ch_cut ? ch_cut[c] : 0.0);
        }
    }
    PROBE_STOP(P_HTC);

                                                                        
                                                         
                                                                           
    PROBE_START();
    for (int c = 0; c < n_ch; c++) {
        double *p_c   = p   + c * n_ax;
        const double *T_film_c = T_film + c * n_ax;
        double *G_c   = G   + c * (n_ax + 1);
        int is_he = fluids[c];
        const double *mu_tab  = is_he ? mu_tab_he  : mu_tab_w;
        const double *k_tab   = is_he ? k_tab_he   : k_tab_w;
        const double *cp_tab  = is_he ? cp_tab_he  : cp_tab_w;
        const axis_t *ax = is_he ? ax_he : ax_w;
        int n_T = ax->n_T;
        double Dh_c_val = Dh[c];

        for (int j = 0; j < n_ax; j++) {
            double T_eff = T_film_c[j];
                                                                              
                                                                            
                                                                        
                                                                            
                                                                         
                                                                              
                                                                       
                                                             
            double mu_j, k_j, cp;
            int use_sat_liquid = 0;
            if (!is_he) {
                int i_sat; double w_sat;
                i_sat = interp1d_idx(p_grid_w, n_p_w,
                                     CLIP(p_c[j], p_grid_w[0], p_grid_w[n_p_w-1]),
                                     &w_sat);
                double Tsat_j = interp1d_at_w(Tsat_grid, i_sat, w_sat);
                double hg_j = interp1d_at_w(hg_grid, i_sat, w_sat);
                double h_bulk = h[c * n_ax + j];
                if (h_bulk < hg_j) {
                    if (T_eff >= Tsat_j - 0.5) {
                        mu_j = interp1d_at_w(mu_l_sat_grid, i_sat, w_sat);
                        k_j  = interp1d_at_w(k_l_sat_grid,  i_sat, w_sat);
                        cp   = interp1d_at_w(cp_l_sat_grid, i_sat, w_sat);
                        use_sat_liquid = 1;
                    }
                } else {
                    if (T_eff < Tsat_j + 1.0) T_eff = Tsat_j + 1.0;
                }
            }
            if (!use_sat_liquid) {
                int i, jg; double wp, wT;
                interp2d_indices(ax, p_c[j], T_eff, &i, &jg, &wp, &wT);
                mu_j = interp2d_at(mu_tab, n_T, i, jg, wp, wT);
                k_j  = interp2d_at(k_tab,  n_T, i, jg, wp, wT);
                cp   = interp2d_at(cp_tab, n_T, i, jg, wp, wT);
            }
            double htc_sp = sp_htc_c(fabs(G_c[j]), Dh_c_val, mu_j, cp, k_j,
                                     ch_shape ? ch_shape[c] : 0,
                                     ch_cut ? ch_cut[c] : 0.0);
            htc_sp_all[j] = htc_sp;
                                                                             
                                                                             
                                                                              
            if (!ch_tp[c]) htc_film_all[c * n_ax + j] = htc_sp;
            cp_film_all[c * n_ax + j]  = cp;
        }
    }

    PROBE_STOP(P_EN_P1);
                                                                        
                                                                
                                                                         
                                                             
                                                                           
    PROBE_START();
    double _w_ir_log = 1.0 / (log(p_grid_w[n_p_w-1]) - log(p_grid_w[0]));
    if (Pw_tot) {
        for (int wi = 0; wi < n_wall; wi++) {
            Pw_tot[wall_ch_i[wi]] += wall_Pi[wi];
            if (!wall_one_sided || !wall_one_sided[wi])
                Pw_tot[wall_ch_j[wi]] += wall_Pj[wi];
        }
    }
    for (int c = 0; c < n_ch; c++) {
        if (!ch_tp[c]) continue;
        double *p_c = p + c * n_ax;
        double *h_c = h + c * n_ax;
        double *G_c = G + c * (n_ax + 1);
        for (int j = 0; j < n_ax; j++) {
            double htc = htc_sp_all[j];                                     
                                                                                
                                                                          
                                                                               
                                                                      
            int i_sat; double w_sat;
            i_sat = interp1d_idx(p_grid_w, n_p_w,
                                 CLIP(p_c[j], p_grid_w[0], p_grid_w[n_p_w-1]), &w_sat);
            double Tsat_chen = interp1d_at_w(Tsat_grid, i_sat, w_sat);
            double hl_chen = interp1d_at_w(hl_grid, i_sat, w_sat);
            double hg_chen = interp1d_at_w(hg_grid, i_sat, w_sat);
            double x_prev = (hl_chen >= hg_chen - 1.0)
                          ? 0.0
                          : quality_c(CLIP(h_c[j], hl_chen, hg_chen), hl_chen, hg_chen);
                                                                                 
                                                                              
                                                                                 
                                                                               
                                                                                 
                                             
            double Tw_est = 2.0 * T_film[c * n_ax + j] - T[c * n_ax + j];
            double dT_sat = Tw_est - Tsat_chen;
                                                                             
                                                                              
                                                                                
                                                                    
            double q_flux = 0.0;
            if (Pw_tot && Pw_tot[c] > 0.0)
                q_flux = fabs(q_net[c * n_ax + j]) / (Pw_tot[c] * dz);
            if (hl_chen >= hg_chen - 1.0) {
                                                            
            } else if (x_prev <= 0.001 && dT_sat > 5.0) {
                double q_onb = 1.0e-3 * pow(dT_sat, 3.86);
                double htc_onb = q_onb / fmax(dT_sat, 1e-3);
                if (htc_onb > htc) htc = htc_onb;
            } else if (x_prev > 0.001 && x_prev < 0.999) {
                double rho_l_chen = interp1d_at_w(rho_l_sat_grid, i_sat, w_sat);
                double rho_g_chen = interp1d_at_w(rho_g_sat_grid, i_sat, w_sat);
                double mu_g_chen = interp1d_at_w(mu_g_sat_grid, i_sat, w_sat);
                                                                 
                double mu_l_chen = interp1d_at_w(mu_l_sat_grid, i_sat, w_sat);
                double k_liq = interp1d_at_w(k_l_sat_grid, i_sat, w_sat);
                double cp_liq = interp1d_at_w(cp_l_sat_grid, i_sat, w_sat);
                double sigma_val = interp1d_at_w(sigma_tab, i_sat, w_sat);
                double h_fg_chen = hg_chen - hl_chen;
                double htc_sp_l = htc_sp_all[j];
                if (_g_chen_nu_gnielinski) {
                    htc_sp_l = sp_htc_c(fabs(G_c[j]) * (1.0 - x_prev), Dh[c],
                                        mu_l_chen, cp_liq, k_liq,
                                        ch_shape ? ch_shape[c] : 0,
                                        ch_cut ? ch_cut[c] : 0.0);
                }
                double htc_chen = chen_two_phase_c(x_prev, fabs(G_c[j]), Dh[c],
                                                   rho_l_chen, rho_g_chen,
                                                   mu_l_chen, mu_g_chen,
                                                   k_liq, cp_liq,
                                                   sigma_val, h_fg_chen,
                                                   dT_sat, Tsat_chen, q_flux,
                                                   htc_sp_l);
                if (htc_chen > htc) htc = htc_chen;
            }
                                                                                
                                                                                
                                                                             
                                                                            
                                                                                  
            double prev_htc = htc_film_all[c * n_ax + j];
            htc_film_all[c * n_ax + j] = (prev_htc <= 0.0)
                ? htc
                : prev_htc + urf_htc * (htc - prev_htc);
            htc_bulk[c * n_ax + j] = htc_film_all[c * n_ax + j];
        }
    }
    PROBE_STOP(P_EN_P1B);

                                                                        
                                                                 
                                                                  
                                                                        
                                                                           
    PROBE_START();
                                                     
    for (int c = 0; c < n_ch; c++) {
        double P_h_c = P_h[c];
        for (int j = 0; j < n_ax; j++) {
            double htc = htc_film_all[c * n_ax + j];
            double val = 0.0;
            if (T_wall_ext) {
                val += htc * P_h_c * (T_wall_ext[c * n_ax + j] - T[c * n_ax + j]);
            }
            q_net[c * n_ax + j] = val;
                                                                                
                                                                               
            int has_ext = (T_wall_ext && P_h_c > 0.0);
            tcap[c * n_ax + j] = has_ext ? 1.0e30 : T_in[c];
            tcap_lo[c * n_ax + j] = has_ext ? -1.0e30 : T_in[c];
        }
    }
                                                     
    if (n_wall > 0) {
        for (int wi = 0; wi < n_wall; wi++) {
            int i = wall_ch_i[wi];
            int j = wall_ch_j[wi];
            double t_w = wall_tw[wi];
            double k_w = wall_kw[wi];
            double P_i = wall_Pi[wi];
            double P_j = wall_Pj[wi];
            double P_c = wall_Pcond[wi];
            for (int jj = 0; jj < n_ax; jj++) {
                double htc_i = htc_film_all[i * n_ax + jj];
                double htc_j = htc_film_all[j * n_ax + jj];
                double dT = T[i * n_ax + jj] - T[j * n_ax + jj];
                double q;
                if (P_i == P_j && P_j == P_c) {
                                                                              
                                                                               
                                    
                    double U = 1.0 / (1.0/MAX(htc_i, 1e-10) + t_w/k_w
                                      + 1.0/MAX(htc_j, 1e-10));
                    q = U * P_i * dT;
                } else {
                                                                            
                    double R_tot = 1.0/(MAX(htc_i, 1e-10) * P_i)
                                 + t_w/(k_w * P_c + 1e-30)
                                 + 1.0/(MAX(htc_j, 1e-10) * P_j);
                    q = dT / R_tot;
                }
                q_net[i * n_ax + jj] -= q;
                if (T[j * n_ax + jj] > tcap[i * n_ax + jj])
                    tcap[i * n_ax + jj] = T[j * n_ax + jj];
                if (T[j * n_ax + jj] < tcap_lo[i * n_ax + jj])
                    tcap_lo[i * n_ax + jj] = T[j * n_ax + jj];
                                                                                
                                                                                
                                                                    
                if (!wall_one_sided || !wall_one_sided[wi]) {
                    q_net[j * n_ax + jj] += q;
                    if (T[i * n_ax + jj] > tcap[j * n_ax + jj])
                        tcap[j * n_ax + jj] = T[i * n_ax + jj];
                    if (T[i * n_ax + jj] < tcap_lo[j * n_ax + jj])
                        tcap_lo[j * n_ax + jj] = T[i * n_ax + jj];
                }
            }
        }
    }
    PROBE_STOP(P_EN_P2);

                                                                        
                                                                      
                                                             
                                                                           
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

                                                                     
        const double *cp_tab  = is_he ? cp_tab_he  : cp_tab_w;
        const double *rho_tab = is_he ? rho_tab_he : rho_tab_w;
        const double *h_tab   = is_he ? h_tab_he   : h_tab_w;
        const axis_t *ax = is_he ? ax_he : ax_w;
        int n_T = ax->n_T;

                                                                 
        for (int j = 0; j < n_ax; j++) { h_new[j] = h_c[j]; T_new[j] = T_c[j]; }
        int fs = flow_sign[c];
        int inlet_idx = (fs == 1) ? 0 : (n_ax - 1);
        T_new[inlet_idx] = T_in[c];
        h_new[inlet_idx] = interp2d(h_tab, ax, p_c[inlet_idx], T_in[c]);
        if (is_tp) alpha_c[inlet_idx] = 0.0;

        double mdot = fabs(G_c[0]) * Ac;                                  

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

                                                                        
        if (fs == 1) {
            for (int j = 1; j < n_ax; j++) {
                if (j == inlet_idx) continue;
                int jprev = j - 1;
                double h_val = h_new[j], h_prev = h_new[jprev];
                if (is_tp) {
                    double Tsat = interp1d_logp(p_grid_w, Tsat_grid, n_p_w, _w_ir_log, p_c[j]);
                    double hl = interp1d_logp(p_grid_w, hl_grid, n_p_w, _w_ir_log, p_c[j]);
                    double hg = interp1d_logp(p_grid_w, hg_grid, n_p_w, _w_ir_log, p_c[j]);
                    double cp_j = cp_film_all[c * n_ax + j];
                    double x_j = quality_c(CLIP(h_val, hl, hg), hl, hg);
                    double rho_l = interp1d_logp(p_grid_w, rho_l_sat_grid, n_p_w, _w_ir_log, p_c[j]);
                    double rho_g = interp1d_logp(p_grid_w, rho_g_sat_grid, n_p_w, _w_ir_log, p_c[j]);
                                                                              
                                                                         
                                                                       
                                                                              
                                                                    
                                                                               
                                                                              
                                                                                
                                                                              
                                                                            
                                                                         
                    double alpha_new = (h_val <= hl)
                                     ? 0.0
                                     : void_drift_flux_c(x_j, rho_l, rho_g, G_c[j], Dh[c]);
                    alpha_c[j] += (alpha_new - alpha_c[j]) * urf_T;
                    if (h_val < hl) {
                        double dh = h_val - h_prev;
                        T_new[j] = T_new[jprev] + dh / MAX(cp_j, 1e-10);
                        if (T_new[j] > Tsat) T_new[j] = Tsat;
                    } else if (h_val > hg) {
                                                                                     
                                                                                   
                                                                                         
                        T_new[j] = T_invert(h_tab_w, ax_w,
                                            p_c[j], Tsat, h_val);
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
                    double Tsat = interp1d_logp(p_grid_w, Tsat_grid, n_p_w, _w_ir_log, p_c[j]);
                    double hl = interp1d_logp(p_grid_w, hl_grid, n_p_w, _w_ir_log, p_c[j]);
                    double hg = interp1d_logp(p_grid_w, hg_grid, n_p_w, _w_ir_log, p_c[j]);
                    double cp_j = cp_film_all[c * n_ax + j];
                    double x_j = quality_c(CLIP(h_val, hl, hg), hl, hg);
                    double rho_l = interp1d_logp(p_grid_w, rho_l_sat_grid, n_p_w, _w_ir_log, p_c[j]);
                    double rho_g = interp1d_logp(p_grid_w, rho_g_sat_grid, n_p_w, _w_ir_log, p_c[j]);
                                                                              
                                                                         
                                                                       
                                                                              
                                                                    
                                                                               
                                                                              
                                                                                
                                                                              
                                                                            
                                                                         
                    double alpha_new = (h_val <= hl)
                                     ? 0.0
                                     : void_drift_flux_c(x_j, rho_l, rho_g, G_c[j], Dh[c]);
                    alpha_c[j] += (alpha_new - alpha_c[j]) * urf_T;
                    if (h_val < hl) {
                        double dh = h_val - h_prev;
                        T_new[j] = T_new[jprev] + dh / MAX(cp_j, 1e-10);
                        if (T_new[j] > Tsat) T_new[j] = Tsat;
                    } else if (h_val > hg) {
                                                                                     
                                                                                   
                                                                                         
                        T_new[j] = T_invert(h_tab_w, ax_w,
                                            p_c[j], Tsat, h_val);
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

                                                                               
                                                                               
                                                                             
                                                                            
                                                                      
                                                                              
                                                                               
                                                                                 
                                                    
        for (int j = 0; j < n_ax; j++) {
            if (j == inlet_idx) continue;
            double hi_j = tcap[c * n_ax + j];
            double lo_j = tcap_lo[c * n_ax + j];
            double cap_j = T_new[j];
            if (hi_j < 1.0e29 && cap_j > hi_j) cap_j = hi_j;
            if (lo_j > -1.0e29 && cap_j < lo_j) cap_j = lo_j;
            if (cap_j != T_new[j]) {
                T_new[j] = cap_j;
                                                                         
                                                                            
                                                                         
                                                                             
                                                                              
                                                                              
                                                                          
                h_new[j] = interp2d(h_tab, ax, p_c[j], cap_j);
                if (is_tp) {
                    double Tsat_j = interp1d_logp(p_grid_w, Tsat_grid, n_p_w, _w_ir_log, p_c[j]);
                    double hl_j = interp1d_logp(p_grid_w, hl_grid, n_p_w, _w_ir_log, p_c[j]);
                    double hg_j = interp1d_logp(p_grid_w, hg_grid, n_p_w, _w_ir_log, p_c[j]);
                    double rho_l_j = interp1d_logp(p_grid_w, rho_l_sat_grid, n_p_w, _w_ir_log, p_c[j]);
                    double rho_g_j = interp1d_logp(p_grid_w, rho_g_sat_grid, n_p_w, _w_ir_log, p_c[j]);
                    double x_j = quality_c(CLIP(h_new[j], hl_j, hg_j), hl_j, hg_j);
                    alpha_c[j] = (h_new[j] <= hl_j || cap_j <= Tsat_j)
                               ? 0.0
                               : void_drift_flux_c(x_j, rho_l_j, rho_g_j, G_c[j], Dh[c]);
                }
            }
        }

                                      
        for (int j = 0; j < n_ax; j++) {
            h_c[j] += (h_new[j] - h_c[j]) * urf_T;
            T_c[j] += (T_new[j] - T_c[j]) * urf_T;
            if (is_tp && alpha_c[j] > 0.0) {
                double Tsat_j = interp1d_logp(p_grid_w, Tsat_grid, n_p_w, _w_ir_log, p_c[j]);
                double rho_l = interp1d_logp(p_grid_w, rho_l_sat_grid, n_p_w, _w_ir_log, p_c[j]);
                double rho_g = interp1d_logp(p_grid_w, rho_g_sat_grid, n_p_w, _w_ir_log, p_c[j]);
                rho[c * n_ax + j] = (1 - alpha_c[j]) * rho_l + alpha_c[j] * rho_g;
            } else {
                int ri, rj; double rwp, rwT;
                interp2d_indices(ax,
                                 p_c[j], T_c[j], &ri, &rj, &rwp, &rwT);
                rho[c * n_ax + j] = interp2d_at(rho_tab, n_T, ri, rj, rwp, rwT);
            }
        }
    }
    PROBE_STOP(P_EN_P3);

    free(tcap);
    free(Pw_tot);

                                                              
    PROBE_STOP(P_EN);
}


                                                                      
                                                                               
                                                               
                                         
                                                                         
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
