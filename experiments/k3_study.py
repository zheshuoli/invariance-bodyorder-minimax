"""
Supplementary K=3 study (Appendix N): pure-S pullback at the first nontrivial order
=====================================================================================
Setting: n=5, K=3, s=5. Compares the F_{K,inv} estimator (pair + pure-S triplet
modes in relative coordinates) with the F_K estimator (absolute coordinates).

Pure-S projection P_Delta^perp (Theorem on the pure-S pullback):
  Within each matched (l, l) block of the SO(3)^2 triplet basis, for each pair of
  right indices (n_1, n_2), the diagonal-invariant direction is
      I^l_{n1,n2} = (2l+1)^{-1/2} sum_m (-1)^(l-m) e_{m,n1,-m,n2}.
  Its coefficient beta_{n1,n2} = (2l+1)^{-1/2} sum_m (-1)^(l-m) c_{m,n1,-m,n2}
  is subtracted, removing the lower-order (pair) leakage. Applied to both the
  ground truth (so the triplet signal is pure V_S content) and the estimator
  (so pair signal is not double-counted).

Config: GT ell_pair=5, ell_trip=2; estimator ell_pair=3, ell_trip=1.
Theoretical slopes at n=5, K=3, s=5: F_Kinv (D=6): -10/16 = -0.625;
F_K (D=9): -10/19 = -0.526.

Usage (run from the results/ directory):
  python ../experiments/k3_study.py --quick   # 3 reps
  python ../experiments/k3_study.py           # 20 reps
Outputs: k3_study_quick.json / k3_study_data.json.
Generate the paper figure with plots/make_figures.py.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import linregress
from math import factorial, sqrt, comb
from itertools import combinations, product as iterproduct
import json, time, sys, os

# ================================================================
# Wigner-D (unchanged)
# ================================================================
def wigner_d_small(j, mp, m, beta):
    j,mp,m = int(j),int(mp),int(m)
    cos_h, sin_h = np.cos(beta/2.0), np.sin(beta/2.0)
    pre = sqrt(factorial(j+mp)*factorial(j-mp)*factorial(j+m)*factorial(j-m))
    result = np.zeros_like(beta, dtype=np.float64)
    for s in range(max(0,m-mp), min(j+m,j-mp)+1):
        denom = factorial(j+m-s)*factorial(s)*factorial(mp-m+s)*factorial(j-mp-s)
        pc, ps = 2*j+m-mp-2*s, mp-m+2*s
        ct = np.where(pc==0,1.0,cos_h**pc)
        st = np.where(ps==0,1.0,sin_h**ps)
        result += ((-1)**(mp-m+s)/denom)*ct*st
    return pre*result

def wigner_D(j, mp, m, alpha, beta, gamma):
    return np.exp(-1j*mp*alpha)*wigner_d_small(j,mp,m,beta)*np.exp(-1j*m*gamma)

def precompute_D_cache(euler, ell_max):
    a,b,g = euler[:,0],euler[:,1],euler[:,2]
    cache = {}
    for ell in range(0, ell_max+1):
        for m in range(-ell,ell+1):
            for nn in range(-ell,ell+1):
                cache[(ell,m,nn)] = wigner_D(ell,m,nn,a,b,g)
    return cache

def enumerate_modes_so3(ell_max):
    modes, casimirs = [], []
    for ell in range(1, ell_max+1):
        for m in range(-ell,ell+1):
            for nn in range(-ell,ell+1):
                modes.append((ell,m,nn))
                casimirs.append(float(ell*(ell+1)))
    return modes, np.array(casimirs)


# ================================================================
# GT helpers (unchanged)
# ================================================================
def make_random_so3_func(ell_max, s, rng):
    coeffs = {}
    for ell in range(1, ell_max+1):
        w = (1+ell*(ell+1))**(-(s+1)/2.0)
        for m in range(-ell,ell+1):
            for nn in range(-ell,ell+1):
                coeffs[(ell,m,nn)] = w*(rng.randn()+1j*rng.randn())/sqrt(2)
    return coeffs

def eval_so3_func(coeffs, euler):
    a,b,g = euler[:,0],euler[:,1],euler[:,2]
    val = np.zeros(len(a), dtype=complex)
    for (ell,m,nn),c in coeffs.items():
        if ell > 0:
            val += c*sqrt(2*ell+1)*wigner_D(ell,m,nn,a,b,g)
    return val.real

def make_random_so3x2_func(ell_max, s, rng):
    coeffs = {}
    for l1 in range(1, ell_max+1):
        for l2 in range(1, ell_max+1):
            cas = l1*(l1+1) + l2*(l2+1)
            w = (1+cas)**(-(s+1)/2.0)
            for m1 in range(-l1,l1+1):
                for n1 in range(-l1,l1+1):
                    for m2 in range(-l2,l2+1):
                        for n2 in range(-l2,l2+1):
                            coeffs[(l1,m1,n1,l2,m2,n2)] = w*(rng.randn()+1j*rng.randn())/sqrt(2)
    return coeffs

def eval_so3x2_func(coeffs, euler1, euler2):
    a1,b1,g1 = euler1[:,0],euler1[:,1],euler1[:,2]
    a2,b2,g2 = euler2[:,0],euler2[:,1],euler2[:,2]
    val = np.zeros(len(a1), dtype=complex)
    for (l1,m1,n1,l2,m2,n2),c in coeffs.items():
        D1 = sqrt(2*l1+1)*wigner_D(l1,m1,n1,a1,b1,g1)
        D2 = sqrt(2*l2+1)*wigner_D(l2,m2,n2,a2,b2,g2)
        val += c * D1 * D2
    return val.real


# ================================================================
# Pure-S projection (P_Delta^perp): remove 2-body leakage
# ================================================================
def apply_pure_S_projection(coeffs, ell_trip):
    """
    Apply P_Delta^perp to coefficients on SO(3)^2 basis.
    
    The diagonal-invariant subspace at matched (l,l) block has
    dimension (2l+1)^2 (one invariant per (n1, n2) pair of right-indices).
    
    The invariant vectors are:
      I^l_{n1, n2} = (1/sqrt(2l+1)) sum_m (-1)^(l-m) e_{m, n1, -m, n2}
    where e_{m1, n1, m2, n2} = sqrt(2l+1) D^l_{m1 n1}(r_2) · sqrt(2l+1) D^l_{m2 n2}(r_3)
    is the orthonormal standard basis.
    
    For each matched (l,l) block and each (n1, n2):
      1. Compute leakage coeff: beta = (1/sqrt(2l+1)) sum_m (-1)^(l-m) c_{m,n1,-m,n2}
      2. Subtract: c_{m,n1,-m,n2} -= (-1)^(l-m) / sqrt(2l+1) * beta
    
    Modes with m2 != -m1 are NOT affected (they are not in leakage subspace).
    Modes with l1 != l2 are NOT affected (no invariant in mixed blocks).
    
    Modifies coeffs in place AND returns it.
    """
    for l in range(1, ell_trip + 1):
        norm = 1.0 / sqrt(2*l + 1)
        for n1 in range(-l, l+1):
            for n2 in range(-l, l+1):
                # Step 1: compute leakage coefficient beta_{n1, n2}
                beta = 0.0 + 0.0j
                for m in range(-l, l+1):
                    key = (l, m, n1, l, -m, n2)
                    if key in coeffs:
                        beta += ((-1)**(l - m)) * coeffs[key]
                beta *= norm  # beta = (1/sqrt(2l+1)) * sum
                
                # Step 2: subtract leakage from affected modes
                for m in range(-l, l+1):
                    key = (l, m, n1, l, -m, n2)
                    if key in coeffs:
                        coeffs[key] -= ((-1)**(l - m)) * norm * beta
    return coeffs


# ================================================================
# Ground-truth constructors (pure-S projection applied to triplet modes)
# ================================================================
def make_gt_Kinv_K3(ell_pair, ell_trip, s, n, seed=42):
    """
    Apply the pure-S projection to the triplet GT coefficients.
    Ensures triplet signal lies strictly in V_{S} (pure 3-body subspace),
    with no 2-body leakage that would overlap with pair terms.
    """
    rng = np.random.RandomState(seed)
    pairs = list(combinations(range(n), 2))
    triplets = list(combinations(range(n), 3))
    pair_coeffs = {p: make_random_so3_func(ell_pair, s, rng) for p in pairs}
    trip_coeffs = {}
    for t in triplets:
        c = make_random_so3x2_func(ell_trip, s, rng)
        c = apply_pure_S_projection(c, ell_trip)  # pure-S content only
        trip_coeffs[t] = c
    return pair_coeffs, trip_coeffs, pairs, triplets

def eval_gt_Kinv_K3(pair_coeffs, trip_coeffs, pairs, triplets, g_list, idx=None):
    gl = [g[idx] for g in g_list] if idx is not None else g_list
    N = len(gl[0])
    val = np.zeros(N)
    for (i,j) in pairs:
        rel = (gl[i].inv() * gl[j]).as_euler('ZYZ')
        val += eval_so3_func(pair_coeffs[(i,j)], rel)
    for (i,j,k) in triplets:
        rel1 = (gl[i].inv() * gl[j]).as_euler('ZYZ')
        rel2 = (gl[i].inv() * gl[k]).as_euler('ZYZ')
        val += eval_so3x2_func(trip_coeffs[(i,j,k)], rel1, rel2)
    return val

def make_gt_FK_K3(ell_1b, ell_2b, ell_3b, s, n, seed=43):
    rng = np.random.RandomState(seed)
    coeffs_1b = [make_random_so3_func(ell_1b, s, rng) for _ in range(n)]
    pairs = list(combinations(range(n), 2))
    coeffs_2b = {p: make_random_so3x2_func(ell_2b, s, rng) for p in pairs}
    triplets = list(combinations(range(n), 3))
    coeffs_3b = {}
    for t in triplets:
        c3 = {}
        for l1 in range(1, ell_3b+1):
            for l2 in range(1, ell_3b+1):
                for l3 in range(1, ell_3b+1):
                    cas = l1*(l1+1)+l2*(l2+1)+l3*(l3+1)
                    w = (1+cas)**(-(s+1)/2.0)
                    for m1 in range(-l1,l1+1):
                        for n1 in range(-l1,l1+1):
                            for m2 in range(-l2,l2+1):
                                for n2 in range(-l2,l2+1):
                                    for m3 in range(-l3,l3+1):
                                        for n3 in range(-l3,l3+1):
                                            c3[(l1,m1,n1,l2,m2,n2,l3,m3,n3)] = \
                                                w*(rng.randn()+1j*rng.randn())/sqrt(2)
        coeffs_3b[t] = c3
    return coeffs_1b, coeffs_2b, coeffs_3b, pairs, triplets

def eval_gt_FK_K3(coeffs_1b, coeffs_2b, coeffs_3b, pairs, triplets, g_list, idx=None):
    gl = [g[idx] for g in g_list] if idx is not None else g_list
    N = len(gl[0]); n = len(gl)
    val = np.zeros(N)
    for i in range(n):
        euler = gl[i].as_euler('ZYZ')
        val += eval_so3_func(coeffs_1b[i], euler)
    for (i,j) in pairs:
        ei = gl[i].as_euler('ZYZ')
        ej = gl[j].as_euler('ZYZ')
        val += eval_so3x2_func(coeffs_2b[(i,j)], ei, ej)
    for (i,j,k) in triplets:
        ei = gl[i].as_euler('ZYZ')
        ej = gl[j].as_euler('ZYZ')
        ek = gl[k].as_euler('ZYZ')
        ai,bi,gi_ = ei[:,0],ei[:,1],ei[:,2]
        aj,bj,gj = ej[:,0],ej[:,1],ej[:,2]
        ak,bk,gk = ek[:,0],ek[:,1],ek[:,2]
        v = np.zeros(N, dtype=complex)
        for (l1,m1,n1,l2,m2,n2,l3,m3,n3),c in coeffs_3b[(i,j,k)].items():
            D1 = sqrt(2*l1+1)*wigner_D(l1,m1,n1,ai,bi,gi_)
            D2 = sqrt(2*l2+1)*wigner_D(l2,m2,n2,aj,bj,gj)
            D3 = sqrt(2*l3+1)*wigner_D(l3,m3,n3,ak,bk,gk)
            v += c * D1 * D2 * D3
        val += v.real
    return val


# ================================================================
# Estimators: per-level regularization with sqrt(M) correction
# ================================================================
def est_Kinv_K3_fixed(Y, g_tr, g_te, pairs, triplets, ell_pair, ell_trip,
                      N, M_pairs, M_trips, s, idx):
    """
    Per-level regularization:
    - Pair terms on SO(3)^1: Lam_pair based on D_pair=3, mu_pair /= sqrt(M_pairs)
    - Triplet terms on SO(3)^2: Lam_trip based on D_trip=6, mu_trip /= sqrt(M_trips)
    """
    # Pair level (SO(3)^1, D=3)
    Lam_pair = (N/M_pairs)**(2.0/(2*s+3))
    mu_pair = 1.0/(1.0+Lam_pair)**s
    mu_pair /= sqrt(M_pairs)
    
    # Triplet level (SO(3)^2, D=6)
    Lam_trip = (N/M_trips)**(2.0/(2*s+6))
    mu_trip = 1.0/(1.0+Lam_trip)**s
    mu_trip /= sqrt(M_trips)

    mean_Y = float(np.mean(Y))
    f_hat = np.full(len(g_te[0]), mean_Y)

    # Pair terms with per-level mu_pair
    modes_1, cas_1 = enumerate_modes_so3(ell_pair)
    w_1 = 1.0/(1.0+mu_pair*(1.0+cas_1)**s)
    for (i,j) in pairs:
        D_tr = precompute_D_cache((g_tr[i][idx].inv()*g_tr[j][idx]).as_euler('ZYZ'), ell_pair)
        D_te = precompute_D_cache((g_te[i].inv()*g_te[j]).as_euler('ZYZ'), ell_pair)
        for mi,(l,m,nn) in enumerate(modes_1):
            c = np.mean(Y*sqrt(2*l+1)*D_tr[(l,m,nn)].conj())
            a = c*w_1[mi]
            if abs(a)>1e-15:
                f_hat += (a*sqrt(2*l+1)*D_te[(l,m,nn)]).real

    # Triplet terms with per-level mu_trip
    # Three phases: collect coefficients, apply the pure-S projection, reconstruct
    for (i,j,k) in triplets:
        D_tr_1 = precompute_D_cache((g_tr[i][idx].inv()*g_tr[j][idx]).as_euler('ZYZ'), ell_trip)
        D_tr_2 = precompute_D_cache((g_tr[i][idx].inv()*g_tr[k][idx]).as_euler('ZYZ'), ell_trip)
        D_te_1 = precompute_D_cache((g_te[i].inv()*g_te[j]).as_euler('ZYZ'), ell_trip)
        D_te_2 = precompute_D_cache((g_te[i].inv()*g_te[k]).as_euler('ZYZ'), ell_trip)
        
        # Phase 1: collect shrinkage-adjusted coefficients in dict
        a_dict = {}
        for l1 in range(1, ell_trip+1):
            for l2 in range(1, ell_trip+1):
                cas = l1*(l1+1)+l2*(l2+1)
                wt = 1.0/(1.0+mu_trip*(1.0+cas)**s)
                for m1 in range(-l1,l1+1):
                    for n1 in range(-l1,l1+1):
                        d1_tr = sqrt(2*l1+1)*D_tr_1[(l1,m1,n1)]
                        for m2 in range(-l2,l2+1):
                            for n2 in range(-l2,l2+1):
                                e_tr = d1_tr * sqrt(2*l2+1)*D_tr_2[(l2,m2,n2)]
                                c = np.mean(Y*e_tr.conj())
                                a_dict[(l1,m1,n1,l2,m2,n2)] = c * wt
        
        # Phase 2: apply pure-S projection (remove 2-body leakage)
        a_dict = apply_pure_S_projection(a_dict, ell_trip)
        
        # Phase 3: reconstruct f_hat from pure-S coefficients
        for (l1,m1,n1,l2,m2,n2), a in a_dict.items():
            if abs(a) > 1e-15:
                d1_te = sqrt(2*l1+1)*D_te_1[(l1,m1,n1)]
                e_te = d1_te * sqrt(2*l2+1)*D_te_2[(l2,m2,n2)]
                f_hat += (a * e_te).real
    return f_hat


def est_K_K3_fixed(Y, g_tr, g_te, pairs, triplets, ell_1b, ell_2b, ell_3b,
                   N, n, M_1b, M_2b, M_3b, s, idx):
    """
    F_K estimator with per-level regularization.
    - 1-body terms on SO(3)^1: D=3
    - 2-body terms on SO(3)^2: D=6
    - 3-body terms on SO(3)^3: D=9 (this is the top order for F_K)
    """
    # Per-level regularization
    Lam_1b = (N/M_1b)**(2.0/(2*s+3))
    mu_1b = 1.0/(1.0+Lam_1b)**s / sqrt(M_1b)
    
    Lam_2b = (N/M_2b)**(2.0/(2*s+6))
    mu_2b = 1.0/(1.0+Lam_2b)**s / sqrt(M_2b)
    
    Lam_3b = (N/M_3b)**(2.0/(2*s+9))
    mu_3b = 1.0/(1.0+Lam_3b)**s / sqrt(M_3b)

    mean_Y = float(np.mean(Y))
    max_ell = max(ell_1b, ell_2b, ell_3b)
    D_tr_c, D_te_c = {}, {}
    for f in range(n):
        D_tr_c[f] = precompute_D_cache(g_tr[f][idx].as_euler('ZYZ'), max_ell)
        D_te_c[f] = precompute_D_cache(g_te[f].as_euler('ZYZ'), max_ell)
    f_hat = np.full(len(g_te[0]), mean_Y)

    # 1-body with mu_1b
    for f in range(n):
        for l in range(1, ell_1b+1):
            wt = 1.0/(1.0+mu_1b*(1.0+l*(l+1))**s)
            for m in range(-l,l+1):
                for nn in range(-l,l+1):
                    c = np.mean(Y*sqrt(2*l+1)*D_tr_c[f][(l,m,nn)].conj())
                    a = c*wt
                    if abs(a)>1e-15:
                        f_hat += (a*sqrt(2*l+1)*D_te_c[f][(l,m,nn)]).real
    # 2-body with mu_2b
    for (fi,fj) in pairs:
        for l1 in range(1, ell_2b+1):
            for l2 in range(1, ell_2b+1):
                cas = l1*(l1+1)+l2*(l2+1)
                wt = 1.0/(1.0+mu_2b*(1.0+cas)**s)
                for m1 in range(-l1,l1+1):
                    for n1 in range(-l1,l1+1):
                        d1_tr = sqrt(2*l1+1)*D_tr_c[fi][(l1,m1,n1)]
                        d1_te = sqrt(2*l1+1)*D_te_c[fi][(l1,m1,n1)]
                        for m2 in range(-l2,l2+1):
                            for n2 in range(-l2,l2+1):
                                e_tr = d1_tr*sqrt(2*l2+1)*D_tr_c[fj][(l2,m2,n2)]
                                c = np.mean(Y*e_tr.conj())
                                a = c*wt
                                if abs(a)>1e-15:
                                    e_te = d1_te*sqrt(2*l2+1)*D_te_c[fj][(l2,m2,n2)]
                                    f_hat += (a*e_te).real
    # 3-body with mu_3b
    for (fi,fj,fk) in triplets:
        for l1 in range(1, ell_3b+1):
            for l2 in range(1, ell_3b+1):
                for l3 in range(1, ell_3b+1):
                    cas = l1*(l1+1)+l2*(l2+1)+l3*(l3+1)
                    wt = 1.0/(1.0+mu_3b*(1.0+cas)**s)
                    for m1 in range(-l1,l1+1):
                        for n1 in range(-l1,l1+1):
                            d1_tr = sqrt(2*l1+1)*D_tr_c[fi][(l1,m1,n1)]
                            d1_te = sqrt(2*l1+1)*D_te_c[fi][(l1,m1,n1)]
                            for m2 in range(-l2,l2+1):
                                for n2 in range(-l2,l2+1):
                                    d12_tr = d1_tr*sqrt(2*l2+1)*D_tr_c[fj][(l2,m2,n2)]
                                    d12_te = d1_te*sqrt(2*l2+1)*D_te_c[fj][(l2,m2,n2)]
                                    for m3 in range(-l3,l3+1):
                                        for n3 in range(-l3,l3+1):
                                            e_tr = d12_tr*sqrt(2*l3+1)*D_tr_c[fk][(l3,m3,n3)]
                                            c = np.mean(Y*e_tr.conj())
                                            a = c*wt
                                            if abs(a)>1e-15:
                                                e_te = d12_te*sqrt(2*l3+1)*D_te_c[fk][(l3,m3,n3)]
                                                f_hat += (a*e_te).real
    return f_hat


# ================================================================
# Run
# ================================================================
def run(cfg):
    n=cfg['n']; K=3; s=cfg['s']; sigma=cfg['sigma']
    N_values=cfg['N_values']; N_max=max(N_values)
    N_test=cfg['N_test']; M_rep=cfg['M_rep']

    pairs = list(combinations(range(n), 2))
    triplets = list(combinations(range(n), 3))
    Mp = len(pairs); Mt = len(triplets); M_1b = n

    EL_KINV_PAIR = cfg.get('ell_kinv_pair', 3)      # Estimator cutoff
    EL_KINV_TRIP = cfg.get('ell_kinv_trip', 1)      # Estimator cutoff
    # GT uses higher ell cutoffs than the estimator (target carries energy beyond the dictionary)
    EL_KINV_PAIR_GT = cfg.get('ell_kinv_pair_gt', EL_KINV_PAIR)  # GT cutoff (default: same as est)
    EL_KINV_TRIP_GT = cfg.get('ell_kinv_trip_gt', EL_KINV_TRIP)  # GT cutoff (default: same as est)
    EL_K1 = cfg.get('ell_k1', 2)
    EL_K2 = cfg.get('ell_k2', 1)
    EL_K3 = cfg.get('ell_k3', 1)

    info = {
        'F_Kinv': {'D': 3*(K-1), 'slope': -2*s/(2*s+3*(K-1))},
        'F_K':    {'D': 3*K,     'slope': -2*s/(2*s+3*K)},
    }

    print(f"\n{'='*70}")
    print(f"K=3 STUDY (pure-S projection P_Delta^perp applied)")
    print(f"{'='*70}")
    print(f"Config: n={n} K={K} s={s} sigma={sigma:.3f}, noise_var={sigma**2:.3f}")
    print(f"  1-body={M_1b}, Pairs={Mp}, Triplets={Mt}")
    print(f"  F_Kinv GT ell: pair={EL_KINV_PAIR_GT}, trip={EL_KINV_TRIP_GT}")
    print(f"  F_Kinv Est ell: pair={EL_KINV_PAIR}, trip={EL_KINV_TRIP}")
    print(f"  F_K ell (GT=Est): 1b={EL_K1}, 2b={EL_K2}, 3b={EL_K3}")
    print(f"\nRegime check (Thm 1: s > D_eff/2):")
    for nm, d in info.items():
        req = d['D'] / 2.0
        in_regime = "[IN REGIME ✓]" if s > req else "[BOUNDARY/OUT ✗]"
        print(f"  {nm}: D={d['D']}, needs s>{req}, s={s} {in_regime}, "
              f"theory_slope={d['slope']:.4f}")

    print("\nConstructing ground truths...")
    # GT built with the GT cutoffs
    pair_c, trip_c, _, _ = make_gt_Kinv_K3(EL_KINV_PAIR_GT, EL_KINV_TRIP_GT, s, n, seed=42)
    coeffs_1b, coeffs_2b, coeffs_3b, _, _ = make_gt_FK_K3(EL_K1, EL_K2, EL_K3, s, n, seed=43)

    ng = [Rotation.random(5000, random_state=777+i) for i in range(n)]
    gt_fns = {
        'F_Kinv': lambda gl: eval_gt_Kinv_K3(pair_c, trip_c, pairs, triplets, gl),
        'F_K': lambda gl: eval_gt_FK_K3(coeffs_1b, coeffs_2b, coeffs_3b, pairs, triplets, gl),
    }
    norms = {}
    for nm, fn in gt_fns.items():
        v = fn(ng)
        norms[nm] = float(np.std(v)) / 0.1 if np.std(v) > 0 else 1.0
        print(f"  {nm}: std={np.std(v):.4f}, norm={norms[nm]:.2f}")

    results = {nm: {N: [] for N in N_values} for nm in info}

    print(f"\nRunning {M_rep} reps x {len(N_values)} N values...")
    for rep in range(M_rep):
        t0 = time.time()
        rng = np.random.RandomState(cfg.get('seed_base',0) + rep)
        g_tr = [Rotation.random(N_max, random_state=rng.randint(2**31)) for _ in range(n)]
        g_te = [Rotation.random(N_test, random_state=rng.randint(2**31)) for _ in range(n)]

        f_tr, f_te, Y_all = {}, {}, {}
        for nm, fn in gt_fns.items():
            f_tr[nm] = fn(g_tr) / norms[nm]
            f_te[nm] = fn(g_te) / norms[nm]
            Y_all[nm] = f_tr[nm] + rng.randn(N_max)*sigma

        for N in N_values:
            idx = np.arange(N)

            # F_Kinv
            Y = Y_all['F_Kinv'][:N]
            fh = est_Kinv_K3_fixed(Y, g_tr, g_te, pairs, triplets,
                                   EL_KINV_PAIR, EL_KINV_TRIP, N, Mp, Mt, s, idx)
            results['F_Kinv'][N].append(float(np.mean((fh-f_te['F_Kinv'])**2)))

            # F_K
            Y = Y_all['F_K'][:N]
            fh = est_K_K3_fixed(Y, g_tr, g_te, pairs, triplets,
                                EL_K1, EL_K2, EL_K3, N, n, M_1b, Mp, Mt, s, idx)
            results['F_K'][N].append(float(np.mean((fh-f_te['F_K'])**2)))

        dt = time.time()-t0
        if (rep+1) % max(1, M_rep//5) == 0 or rep == 0:
            Ns = N_values[-1]
            print(f"  Rep {rep+1}/{M_rep} ({dt:.1f}s) " +
                  " ".join(f"{k}={results[k][Ns][-1]:.3e}" for k in info))

    return results, info


# ================================================================
# Save & Plot
# ================================================================
def save_plot(results, info, cfg, path):
    N_values = cfg['N_values']

    analysis = {}
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    # Print MSE table for diagnosis
    print("\nMean MSE per N:")
    print(f"  {'N':>8} | " + " | ".join(f"{nm:>12}" for nm in ['F_Kinv','F_K']))
    print(f"  {'-'*8}-+-" + "-+-".join("-"*12 for _ in ['F_Kinv','F_K']))
    for N in N_values:
        row = f"  {N:>8} | "
        row += " | ".join(f"{np.mean(results[nm][N]):>12.3e}" for nm in ['F_Kinv','F_K'])
        print(row)

    print("\nSlope analysis:")
    for nm in ['F_Kinv','F_K']:
        means = [float(np.mean(results[nm][N])) for N in N_values]
        stds = [float(np.std(results[nm][N])) for N in N_values]
        logN = [np.log10(N) for N in N_values]
        logR = [np.log10(m) for m in means if m > 0]
        npt = min(len(logN),len(logR))
        sl = linregress(logN[:npt], logR[:npt])
        th = info[nm]['slope']
        gap = abs(sl.slope) - abs(th)
        analysis[nm] = {'D':info[nm]['D'],'theory':th,
                       'slope_full':float(sl.slope),'R2_full':float(sl.rvalue**2),
                       'abs_gap':float(abs(gap)),
                       'within_0.05':bool(abs(gap) <= 0.05),
                       'N':list(N_values),'means':means,'stds':stds}
        status = "✓ within 0.05" if abs(gap) <= 0.05 else "✗ outside 0.05"
        print(f"  {nm}: D={info[nm]['D']}, empirical={sl.slope:.4f}, "
              f"theory={th:.4f}, |gap|={abs(gap):.4f} [{status}]")

    kinv_gap = analysis['F_Kinv']['abs_gap']
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  F_Kinv at K=3: |empirical - theory| slope gap = {kinv_gap:.4f}")
    ratio = [k / i for k, i in zip(analysis['F_K']['means'], analysis['F_Kinv']['means'])]
    print("  Risk ratio F_K / F_Kinv: " + " ".join(f"{N}:{r:.2f}" for N, r in zip(N_values, ratio)))

    data = {'config':{k:v for k,v in cfg.items()},
            'theory':{nm:{'D':info[nm]['D'],'slope':info[nm]['slope']} for nm in info},
            'analysis':analysis,
            'raw':{nm:{str(N):v for N,v in results[nm].items()} for nm in results}}
    
    with open(path,'w') as f:
        json.dump(data,f,indent=2)
    print(f"\n  Data saved: {path}")


# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument('--quick', action='store_true',
                    help='reduced configuration (fewer repetitions, smaller N) to check the environment')
    args = ap.parse_args()
    quick = args.quick

    print("="*70)
    print("K=3 STUDY (n=5, K=3, s=5): pure-S pullback validation")
    print("Pure-S projection implemented in both GT and estimator")
    print("="*70)

    beta=np.array([.5,1.,2.])
    assert np.allclose(wigner_d_small(1,1,1,beta),np.cos(beta/2)**2)
    print("  Wigner-D check passed")

    script_dir = '.'
    
    if quick:
        # Quick: 3 reps
        cfg = {'n':5, 'K':3, 's':5.0, 'sigma':sqrt(0.1),
               'N_values':[500,1000,2000,5000,10000,20000,50000],
               'N_test':2000, 'M_rep':3, 'seed_base':3000,
               # Estimator ell cutoffs
               'ell_kinv_pair':3, 'ell_kinv_trip':1,
               # GT ell cutoffs (higher than the estimator)
               'ell_kinv_pair_gt':5, 'ell_kinv_trip_gt':2,
               # F_K (absolute coordinates; no pure-S projection)
               'ell_k1':2, 'ell_k2':1, 'ell_k3':1}
        path = os.path.join(script_dir, 'k3_study_quick.json')
    else:
        # Full: 20 reps
        cfg = {'n':5, 'K':3, 's':5.0, 'sigma':sqrt(0.1),
               'N_values':[500,1000,2000,5000,10000,20000,50000],
               'N_test':5000, 'M_rep':20, 'seed_base':3000,
               'ell_kinv_pair':3, 'ell_kinv_trip':1,
               'ell_kinv_pair_gt':5, 'ell_kinv_trip_gt':2,
               'ell_k1':2, 'ell_k2':1, 'ell_k3':1}
        path = os.path.join(script_dir, 'k3_study_data.json')

    t0 = time.time()
    res, info = run(cfg)
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    save_plot(res, info, cfg, path)
    print("\nDone!")
