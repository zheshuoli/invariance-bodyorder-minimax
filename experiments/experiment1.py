"""
Experiment 1: per-class rate exponents (Figure 2)
=================================================
Each class uses its own class-matched target with Sobolev-decaying coefficients.

F_Kinv: f* = Σ_{i<j} φ(gᵢ⁻¹gⱼ)                     [invariant 2-body]
F_K:    f* = Σ_i φ_i(gᵢ) + Σ_{i<j,r} ψ_r^i(gᵢ)ψ_r^j(gⱼ)  [non-invariant 2-body]
F_inv:  f* = Σ_r Π_j φ_r^j(g₁⁻¹g_{j+1})              [invariant, all body orders]
F_0:    f* = Σ_r Π_j φ_r^j(gⱼ)                       [non-invariant, all body orders]

Tikhonov shrinkage with μ = (1+Λ*)^(-s); for the F_Kinv estimator μ is additionally
divided by sqrt(M), M = number of pairs (see est_Kinv below and the README).

Usage (from results/):
  python ../experiments/experiment1.py --quick   # 5 reps, N to 10000
  python ../experiments/experiment1.py           # 30 reps, N to 50000 (Figure 2)
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import linregress
from math import factorial, sqrt, comb
from itertools import combinations, product as iterproduct
import json, time, sys

# ================================================================
# Wigner-D
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
            for n in range(-ell,ell+1):
                cache[(ell,m,n)] = wigner_D(ell,m,n,a,b,g)
    return cache

# ================================================================
# Per-factor random function
# ================================================================
def make_random_so3_func(ell_max, s, rng, include_const=False):
    """Random function on SO(3). If include_const, add ℓ=0 (nonzero mean)."""
    coeffs = {}
    if include_const:
        coeffs[(0,0,0)] = 0.5*(rng.randn()+1j*rng.randn())/sqrt(2)
    for ell in range(1, ell_max+1):
        w = (1+ell*(ell+1))**(-(s+1)/2.0)
        for m in range(-ell,ell+1):
            for nn in range(-ell,ell+1):
                coeffs[(ell,m,nn)] = w*(rng.randn()+1j*rng.randn())/sqrt(2)
    return coeffs

def eval_so3_func(coeffs, euler):
    """Evaluate SO(3) function at points given by Euler angles."""
    a,b,g = euler[:,0],euler[:,1],euler[:,2]
    N = len(a)
    # Start with constant if present
    c0 = coeffs.get((0,0,0), 0.0)
    val = np.full(N, c0, dtype=complex)
    for (ell,m,nn),c in coeffs.items():
        if ell > 0:
            val += c*(2*ell+1)*wigner_D(ell,m,nn,a,b,g)
    return val

# ================================================================
# Ground Truth Constructors
# ================================================================

def make_gt_Kinv(ell_max, s, n, seed=42):
    """f* ∈ F_{K,inv}: Σ_{i<j} φ(gᵢ⁻¹gⱼ)."""
    rng = np.random.RandomState(seed)
    pairs = list(combinations(range(n), 2))
    coeffs = {pair: make_random_so3_func(ell_max, s, rng) for pair in pairs}
    return coeffs

def eval_gt_Kinv(coeffs, g_list, idx=None):
    gl = [g[idx] for g in g_list] if idx is not None else g_list
    N = len(gl[0]); fv = np.zeros(N)
    for (i,j), pc in coeffs.items():
        eu = (gl[i].inv()*gl[j]).as_euler('ZYZ')
        fv += eval_so3_func(pc, eu).real
    return fv


def make_gt_FK(ell_max_1b, ell_max_2b, s, n, R_2b=5, seed=43):
    """f* ∈ F_K: 1-body + non-invariant 2-body.
    2-body via product: Σ_r ψ_r^i(gᵢ) × ψ_r^j(gⱼ) per pair.
    """
    rng = np.random.RandomState(seed)
    pairs = list(combinations(range(n), 2))
    # 1-body
    coeffs_1b = {i: make_random_so3_func(ell_max_1b, s, rng) for i in range(n)}
    # 2-body: per pair, R product terms
    coeffs_2b = {}
    for pair in pairs:
        terms = []
        for r in range(R_2b):
            ci = make_random_so3_func(ell_max_2b, s, rng)
            cj = make_random_so3_func(ell_max_2b, s, rng)
            terms.append((ci, cj))
        coeffs_2b[pair] = terms
    return coeffs_1b, coeffs_2b

def eval_gt_FK(coeffs_1b, coeffs_2b, g_list, idx=None):
    gl = [g[idx] for g in g_list] if idx is not None else g_list
    N = len(gl[0]); fv = np.zeros(N)
    # 1-body
    for i, pc in coeffs_1b.items():
        eu = gl[i].as_euler('ZYZ')
        fv += eval_so3_func(pc, eu).real
    # 2-body products
    for (i,j), terms in coeffs_2b.items():
        eu_i, eu_j = gl[i].as_euler('ZYZ'), gl[j].as_euler('ZYZ')
        for ci, cj in terms:
            vi = eval_so3_func(ci, eu_i)
            vj = eval_so3_func(cj, eu_j)
            fv += (vi * vj).real
    return fv


def make_gt_fourier(n_factors, ell_max, s, N_max, D_eff, M=1, seed=44):
    """Fourier-domain dense GT on SO(3)^{n_factors}.
    
    Enumerates all product modes within the estimator's reach at N_max.
    Coefficients: c_α = (1+λ_α)^{-(s+1)/2} × random_complex.
    Dense support → hard target for the corresponding estimator.
    """
    rng = np.random.RandomState(seed)
    
    # Mode set aligned with estimator at N_max
    Lam = (N_max / M) ** (2.0 / (2*s + D_eff))
    Lam_cut = max(Lam * 3, 6.0)
    ell_mp = 0
    while (ell_mp+1)*(ell_mp+2) <= Lam_cut:
        ell_mp += 1
    ell_mp = min(ell_mp, ell_max)
    
    # Enumerate ℓ-tuples with Casimir ≤ Lam_cut
    def enum(d, lr):
        if d == 0: return [()]
        r = []
        for l in range(0, ell_mp+1):
            lam = l*(l+1)
            if lam > lr: break
            for sub in enum(d-1, lr-lam):
                r.append((l,) + sub)
        return r
    
    ell_tuples = [t for t in enum(n_factors, Lam_cut) if any(l > 0 for l in t)]
    
    # Expand to full (ℓ,m,n) modes with Casimir-decaying random coefficients
    modes = []
    for ells in ell_tuples:
        cas = sum(l*(l+1) for l in ells)
        weight = (1.0 + cas) ** (-(s+1) / 2.0)
        ranges = [[(0,0)] if l==0 
                  else [(m,nn) for m in range(-l,l+1) for nn in range(-l,l+1)]
                  for l in ells]
        for mn in iterproduct(*ranges):
            spec = tuple((ells[j], mn[j][0], mn[j][1]) for j in range(n_factors))
            c = weight * (rng.randn() + 1j * rng.randn()) / sqrt(2)
            modes.append((spec, c))
    
    # Normalize ‖f*‖_{H^s} = 1
    Hs_sq = sum((1 + sum(l*(l+1) for l,_,_ in spec))**s * abs(c)**2
                for spec, c in modes)
    norm = 1.0 / sqrt(Hs_sq) if Hs_sq > 0 else 1.0
    modes = [(spec, c * norm) for spec, c in modes]
    
    n_modes = len(modes)
    n_tuples = len(ell_tuples)
    print(f"    Fourier GT: {n_factors} factors, ℓ_max={ell_mp}, "
          f"Λ_cut={Lam_cut:.1f}, {n_tuples} ℓ-tuples, {n_modes} modes")
    
    return modes, ell_mp


def eval_gt_fourier(modes, ell_max, g_list, idx=None, relative=False):
    """Evaluate Fourier-domain GT. Complex accumulation, .real at end.
    
    If relative=True: uses g₀⁻¹gⱼ (j=1..n-1) as coordinates (for F_inv).
    If relative=False: uses gⱼ directly (for F_0).
    """
    gl = [g[idx] for g in g_list] if idx is not None else g_list
    
    if relative:
        n_factors = len(gl) - 1
        euler = [(gl[0].inv() * gl[j+1]).as_euler('ZYZ') for j in range(n_factors)]
    else:
        n_factors = len(gl)
        euler = [gl[j].as_euler('ZYZ') for j in range(n_factors)]
    
    N = len(euler[0])
    
    # Precompute D caches per factor
    D_caches = [precompute_D_cache(euler[j], ell_max) for j in range(n_factors)]
    
    # Complex accumulation; take the real part at the end
    f_vals = np.zeros(N, dtype=complex)
    for spec, c in modes:
        val = np.ones(N, dtype=complex)
        for j, (l, m, nn) in enumerate(spec):
            if l > 0:
                val *= sqrt(2*l+1) * D_caches[j][(l, m, nn)]
        f_vals += c * val
    
    return f_vals.real


def normalize_gt(eval_fn, g_list, target_std=0.1):
    """Normalize ground truth to have std ≈ target_std."""
    test_g = [Rotation.random(5000, random_state=999) for _ in range(len(g_list))]
    vals = eval_fn(test_g)
    current_std = np.std(vals)
    return current_std / target_std if current_std > 0 else 1.0

# ================================================================
# Estimators (Tikhonov, theoretical μ)
# ================================================================

def enumerate_modes_so3(ell_max):
    modes, casimirs = [], []
    for ell in range(1, ell_max+1):
        for m in range(-ell,ell+1):
            for nn in range(-ell,ell+1):
                modes.append((ell,m,nn))
                casimirs.append(float(ell*(ell+1)))
    return modes, np.array(casimirs)

def est_Kinv(Y, g_tr, g_te, pairs, ell_max, N, D, M, s, idx):
    Lam = (N/M)**(2.0/(2*s+D)); mu = 1.0/(1.0+Lam)**s
    mu /= sqrt(M)  # Experiment 1 convention: mu = (1+Lam)^(-s) / sqrt(M), M = number of pairs (see README)
    modes, cas = enumerate_modes_so3(ell_max)
    w = 1.0/(1.0+mu*(1.0+cas)**s)
    mean_Y = float(np.mean(Y))
    f_hat = np.full(len(g_te[0]), mean_Y)
    for pair in pairs:
        i,j = pair
        D_tr = precompute_D_cache((g_tr[i][idx].inv()*g_tr[j][idx]).as_euler('ZYZ'), ell_max)
        D_te = precompute_D_cache((g_te[i].inv()*g_te[j]).as_euler('ZYZ'), ell_max)
        for mi,(l,m,nn) in enumerate(modes):
            c = np.mean(Y*sqrt(2*l+1)*D_tr[(l,m,nn)].conj())
            a = c*w[mi]
            if abs(a)>1e-15:
                f_hat += (a*sqrt(2*l+1)*D_te[(l,m,nn)]).real
    return f_hat

def est_K(Y, g_tr, g_te, pairs, ell_1b, ell_2b, N, D, M, s, n, idx):
    Lam = (N/M)**(2.0/(2*s+D)); mu = 1.0/(1.0+Lam)**s
    mean_Y = float(np.mean(Y))
    max_ell = max(ell_1b, ell_2b)
    D_tr, D_te = {}, {}
    for f in range(n):
        D_tr[f] = precompute_D_cache(g_tr[f][idx].as_euler('ZYZ'), max_ell)
        D_te[f] = precompute_D_cache(g_te[f].as_euler('ZYZ'), max_ell)
    f_hat = np.full(len(g_te[0]), mean_Y)
    # 1-body
    for f in range(n):
        for l in range(1,ell_1b+1):
            wt = 1.0/(1.0+mu*(1.0+l*(l+1))**s)
            for m in range(-l,l+1):
                for nn in range(-l,l+1):
                    c = np.mean(Y*sqrt(2*l+1)*D_tr[f][(l,m,nn)].conj())
                    a = c*wt
                    if abs(a)>1e-15:
                        f_hat += (a*sqrt(2*l+1)*D_te[f][(l,m,nn)]).real
    # 2-body
    for (fi,fj) in pairs:
        for l1 in range(1,ell_2b+1):
            for l2 in range(1,ell_2b+1):
                cas = l1*(l1+1)+l2*(l2+1)
                wt = 1.0/(1.0+mu*(1.0+cas)**s)
                for m1 in range(-l1,l1+1):
                    for n1 in range(-l1,l1+1):
                        d1_tr = sqrt(2*l1+1)*D_tr[fi][(l1,m1,n1)]
                        d1_te = sqrt(2*l1+1)*D_te[fi][(l1,m1,n1)]
                        for m2 in range(-l2,l2+1):
                            for n2 in range(-l2,l2+1):
                                e_tr = d1_tr*sqrt(2*l2+1)*D_tr[fj][(l2,m2,n2)]
                                c = np.mean(Y*e_tr.conj())
                                a = c*wt
                                if abs(a)>1e-15:
                                    e_te = d1_te*sqrt(2*l2+1)*D_te[fj][(l2,m2,n2)]
                                    f_hat += (a*e_te).real
    return f_hat

def est_inv(Y, g_tr, g_te, ell_max, N, D, M, s, n, idx):
    Lam = (N/M)**(2.0/(2*s+D)); mu = 1.0/(1.0+Lam)**s
    n_rel = n-1
    Lam_cut = max(Lam*3, 6.0)
    ell_mp = 0
    while (ell_mp+1)*(ell_mp+2) <= Lam_cut: ell_mp += 1
    ell_mp = min(ell_mp, ell_max)
    def enum(d, lr):
        if d==0: return [()]
        r = []
        for l in range(0,ell_mp+1):
            lam=l*(l+1)
            if lam>lr: break
            for sub in enum(d-1,lr-lam): r.append((l,)+sub)
        return r
    ells = [t for t in enum(n_rel, Lam_cut) if any(l>0 for l in t)]
    
    D_tr, D_te = [], []
    for j in range(n_rel):
        D_tr.append(precompute_D_cache((g_tr[0][idx].inv()*g_tr[j+1][idx]).as_euler('ZYZ'), ell_mp))
        D_te.append(precompute_D_cache((g_te[0].inv()*g_te[j+1]).as_euler('ZYZ'), ell_mp))
    
    mean_Y = float(np.mean(Y))
    f_hat = np.full(len(g_te[0]), mean_Y)
    for et in ells:
        cas = float(sum(l*(l+1) for l in et))
        wt = 1.0/(1.0+mu*(1.0+cas)**s)
        ranges = [[(0,0)] if l==0 else [(m,nn) for m in range(-l,l+1) for nn in range(-l,l+1)] for l in et]
        for mn in iterproduct(*ranges):
            e_tr = np.ones(len(Y), dtype=complex)
            e_te_v = np.ones(len(g_te[0]), dtype=complex)
            for j,(l,(m,nn)) in enumerate(zip(et,mn)):
                if l>0:
                    e_tr *= sqrt(2*l+1)*D_tr[j][(l,m,nn)]
                    e_te_v *= sqrt(2*l+1)*D_te[j][(l,m,nn)]
            c = np.mean(Y*e_tr.conj())
            a = c*wt
            if abs(a)>1e-15: f_hat += (a*e_te_v).real
    return f_hat

def est_F0(Y, g_tr, g_te, ell_max, N, D, M, s, n, idx):
    Lam = (N/M)**(2.0/(2*s+D)); mu = 1.0/(1.0+Lam)**s
    Lam_cut = max(Lam*3, 6.0)
    ell_mp = 0
    while (ell_mp+1)*(ell_mp+2) <= Lam_cut: ell_mp += 1
    ell_mp = min(ell_mp, ell_max)
    def enum(d, lr):
        if d==0: return [()]
        r = []
        for l in range(0,ell_mp+1):
            lam=l*(l+1)
            if lam>lr: break
            for sub in enum(d-1,lr-lam): r.append((l,)+sub)
        return r
    ells = [t for t in enum(n, Lam_cut) if any(l>0 for l in t)]
    
    D_tr, D_te = [], []
    for j in range(n):
        D_tr.append(precompute_D_cache(g_tr[j][idx].as_euler('ZYZ'), ell_mp))
        D_te.append(precompute_D_cache(g_te[j].as_euler('ZYZ'), ell_mp))
    
    mean_Y = float(np.mean(Y))
    f_hat = np.full(len(g_te[0]), mean_Y)
    for et in ells:
        cas = float(sum(l*(l+1) for l in et))
        wt = 1.0/(1.0+mu*(1.0+cas)**s)
        ranges = [[(0,0)] if l==0 else [(m,nn) for m in range(-l,l+1) for nn in range(-l,l+1)] for l in et]
        for mn in iterproduct(*ranges):
            e_tr = np.ones(len(Y), dtype=complex)
            e_te_v = np.ones(len(g_te[0]), dtype=complex)
            for j,(l,(m,nn)) in enumerate(zip(et,mn)):
                if l>0:
                    e_tr *= sqrt(2*l+1)*D_tr[j][(l,m,nn)]
                    e_te_v *= sqrt(2*l+1)*D_te[j][(l,m,nn)]
            c = np.mean(Y*e_tr.conj())
            a = c*wt
            if abs(a)>1e-15: f_hat += (a*e_te_v).real
    return f_hat

# ================================================================
# Experiment
# ================================================================
def run(config):
    n,K,s = config['n'],config['K'],config['s']
    sigma = config['sigma']; sigma2 = sigma**2
    N_values = sorted(config['N_values'])
    N_test,M_rep = config['N_test'],config['M_rep']
    N_max = max(N_values)
    pairs = list(combinations(range(n),2)); Mp = len(pairs)
    
    EL_KINV = config.get('ell_kinv',5)
    EL_K1 = config.get('ell_k1',3); EL_K2 = config.get('ell_k2',2)
    EL_INV = config.get('ell_inv',2); EL_F0 = config.get('ell_f0',1)
    
    info = {
        'F_Kinv':{'D':3*(K-1),'M':Mp}, 'F_K':{'D':3*K,'M':Mp},
        'F_inv':{'D':3*(n-1),'M':1}, 'F_0':{'D':3*n,'M':1}
    }
    for nm in info: info[nm]['slope']=-2*s/(2*s+info[nm]['D'])
    
    print(f"Config: n={n} K={K} s={s} σ={sigma:.3f}")
    for nm,d in info.items(): print(f"  {nm}: D={d['D']}, slope={d['slope']:.4f}")
    
    # Class-specific ground truths
    # F_Kinv, F_K: keep existing construction (already working well)
    # F_inv, F_0: dense Fourier-domain construction
    print("\nConstructing ground truths...")
    print(f"  Estimator ℓ_max: Kinv={EL_KINV}, K_1b={EL_K1}, K_2b={EL_K2}, "
          f"inv={EL_INV}, F0={EL_F0}")
    gt_kinv = make_gt_Kinv(EL_KINV, s, n, seed=42)
    gt_fk_1b, gt_fk_2b = make_gt_FK(EL_K1, EL_K2, s, n, R_2b=5, seed=43)
    print("  F_inv (Fourier):")
    gt_finv, gt_finv_ell = make_gt_fourier(
        n-1, EL_INV, s, N_max, 3*(n-1), M=1, seed=44)
    print("  F_0 (Fourier):")
    gt_f0, gt_f0_ell = make_gt_fourier(
        n, EL_F0, s, N_max, 3*n, M=1, seed=45)
    
    # Normalization factors
    ng = [Rotation.random(5000, random_state=999+i) for i in range(n)]
    norms = {}
    for nm, ef in [
            ('F_Kinv', lambda g: eval_gt_Kinv(gt_kinv, g)),
            ('F_K',    lambda g: eval_gt_FK(gt_fk_1b, gt_fk_2b, g)),
            ('F_inv',  lambda g: eval_gt_fourier(gt_finv, gt_finv_ell, g, relative=True)),
            ('F_0',    lambda g: eval_gt_fourier(gt_f0, gt_f0_ell, g, relative=False))]:
        v = ef(ng); norms[nm] = float(np.std(v))/0.1 if np.std(v)>0 else 1.0
        print(f"  {nm}: std={np.std(v):.4f}, norm_factor={norms[nm]:.2f}")
    
    results = {nm:{N:[] for N in N_values} for nm in info}
    
    for rep in range(M_rep):
        t0 = time.time()
        rng = np.random.RandomState(config.get('seed_base',0)+rep)
        g_tr = [Rotation.random(N_max, random_state=rng.randint(2**31)) for _ in range(n)]
        g_te = [Rotation.random(N_test, random_state=rng.randint(2**31)) for _ in range(n)]
        
        # Evaluate each class's ground truth
        f_kinv_tr = eval_gt_Kinv(gt_kinv, g_tr)/norms['F_Kinv']
        f_kinv_te = eval_gt_Kinv(gt_kinv, g_te)/norms['F_Kinv']
        f_fk_tr = eval_gt_FK(gt_fk_1b,gt_fk_2b, g_tr)/norms['F_K']
        f_fk_te = eval_gt_FK(gt_fk_1b,gt_fk_2b, g_te)/norms['F_K']
        f_inv_tr = eval_gt_fourier(gt_finv, gt_finv_ell, g_tr, relative=True)/norms['F_inv']
        f_inv_te = eval_gt_fourier(gt_finv, gt_finv_ell, g_te, relative=True)/norms['F_inv']
        f_f0_tr = eval_gt_fourier(gt_f0, gt_f0_ell, g_tr, relative=False)/norms['F_0']
        f_f0_te = eval_gt_fourier(gt_f0, gt_f0_ell, g_te, relative=False)/norms['F_0']
        
        noise = rng.randn(N_max)*sigma
        
        for N in N_values:
            idx = np.arange(N)
            
            Y = f_kinv_tr[:N]+noise[:N]
            fh = est_Kinv(Y,g_tr,g_te,pairs,EL_KINV,N,info['F_Kinv']['D'],Mp,s,idx)
            results['F_Kinv'][N].append(float(np.mean((fh-f_kinv_te)**2)))
            
            Y = f_fk_tr[:N]+noise[:N]
            fh = est_K(Y,g_tr,g_te,pairs,EL_K1,EL_K2,N,info['F_K']['D'],Mp,s,n,idx)
            results['F_K'][N].append(float(np.mean((fh-f_fk_te)**2)))
            
            Y = f_inv_tr[:N]+noise[:N]
            fh = est_inv(Y,g_tr,g_te,EL_INV,N,info['F_inv']['D'],1,s,n,idx)
            results['F_inv'][N].append(float(np.mean((fh-f_inv_te)**2)))
            
            Y = f_f0_tr[:N]+noise[:N]
            fh = est_F0(Y,g_tr,g_te,EL_F0,N,info['F_0']['D'],1,s,n,idx)
            results['F_0'][N].append(float(np.mean((fh-f_f0_te)**2)))
        
        dt = time.time()-t0
        if rep%max(1,M_rep//10)==0 or rep==M_rep-1:
            Ns = N_values[-1]
            print(f"  Rep {rep+1}/{M_rep} ({dt:.1f}s) " +
                  " ".join(f"{k}={results[k][Ns][-1]:.3e}" for k in info))
    
    return results, info

# ================================================================
# Save & Plot
# ================================================================
def save_plot(results, info, config, path):
    N_values = sorted(config['N_values'])
    analysis = {}
    print("\n"+"="*70+"\nRESULTS\n"+"="*70)
    for nm in ['F_Kinv','F_K','F_inv','F_0']:
        means = [float(np.mean(results[nm][N])) for N in N_values]
        stds = [float(np.std(results[nm][N])) for N in N_values]
        logN = [np.log10(N) for N in N_values]
        logR = [np.log10(m) for m in means if m>0]
        npt = min(len(logN),len(logR)); nlg = max(3,npt//2)
        sl = linregress(logN[-nlg:],logR[-nlg:]) if npt>=3 else None
        sl_f = linregress(logN[:npt],logR[:npt]) if npt>=3 else None
        analysis[nm] = {'D':info[nm]['D'],'theory':info[nm]['slope'],
            'slope_full':float(sl_f.slope) if sl_f else None,
            'R2_full':float(sl_f.rvalue**2) if sl_f else None,
            'slope_large':float(sl.slope) if sl else None,
            'R2_large':float(sl.rvalue**2) if sl else None,
            'N':N_values,'means':means,'stds':stds}
        print(f"\n  {nm} (D={info[nm]['D']}): theory={info[nm]['slope']:.4f}")
        if sl_f: print(f"    full:  {sl_f.slope:.4f} (R²={sl_f.rvalue**2:.4f})")
        if sl: print(f"    large: {sl.slope:.4f} (R²={sl.rvalue**2:.4f})")
        for i,N in enumerate(N_values): print(f"    N={N:>6d}: {means[i]:.4e} ± {stds[i]:.4e}")
    
    data = {'config':config,'theory':{n:{'D':i['D'],'slope':i['slope']} for n,i in info.items()},
            'analysis':analysis,'raw':{n:{str(N):v for N,v in d.items()} for n,d in results.items()}}
    with open(path,'w') as f: json.dump(data,f,indent=2)
    print(f"\n  Saved: {path}")
    
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        C = {'F_0':'#E69F00','F_inv':'#56B4E9','F_K':'#009E73','F_Kinv':'#CC79A7'}
        L = {'F_0':r'$\mathcal{F}_0$','F_inv':r'$\mathcal{F}_{\rm inv}$',
             'F_K':r'$\mathcal{F}_K$','F_Kinv':r'$\mathcal{F}_{K,{\rm inv}}$'}
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(11,4.5))
        for nm in ['F_0','F_inv','F_K','F_Kinv']:
            a = analysis[nm]
            ax1.errorbar(a['N'],a['means'],yerr=a['stds'],fmt='o-',color=C[nm],
                        label=L[nm],markersize=4,capsize=2,linewidth=1.5)
            ts = info[nm]['slope']; lN = np.array([np.log10(x) for x in a['N']])
            ax1.plot(a['N'],10**(ts*(lN-lN[-1])+np.log10(a['means'][-1])),'--',color=C[nm],alpha=.4)
        ax1.set_xscale('log');ax1.set_yscale('log')
        ax1.set_xlabel(r'$N$');ax1.set_ylabel(r'$L^2$ risk')
        ax1.legend(fontsize=8);ax1.set_title('(a) Risk vs $N$');ax1.grid(True,alpha=.3)
        nms = list(info.keys()); x = range(len(nms))
        ax2.bar([i-.15 for i in x],[-info[n]['slope'] for n in nms],.3,label='Theory',color='gray',alpha=.6)
        emp = [-(analysis[n]['slope_full'] or 0) for n in nms]
        ax2.bar([i+.15 for i in x],emp,.3,label='Empirical',color=[C[n] for n in nms])
        ax2.set_xticks(list(x));ax2.set_xticklabels([L[n] for n in nms],fontsize=9)
        ax2.set_ylabel(r'$|\mathrm{slope}|$');ax2.legend()
        ax2.set_title('(b) Theory vs empirical');ax2.grid(True,alpha=.3,axis='y')
        plt.tight_layout()
        fp = path.replace('.json','_fig.pdf')
        plt.savefig(fp,dpi=300,bbox_inches='tight'); print(f"  Figure: {fp}")
    except Exception as e: print(f"  Plot: {e}")

# ================================================================
# Main
# ================================================================
if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument('--quick', action='store_true',
                    help='reduced configuration (fewer repetitions, smaller N) to check the environment')
    args = ap.parse_args()
    quick = args.quick
    print("="*70+"\nEXP 1: per-class rate exponents\n"+"="*70)
    beta=np.array([.5,1.,2.])
    assert np.allclose(wigner_d_small(1,1,1,beta),np.cos(beta/2)**2)
    print("  Wigner-D ✓")
    
    if quick:
        cfg = {'n':5,'K':2,'s':2.0,'sigma':sqrt(0.1),
               'N_values':[500,1000,2000,5000,10000],
               'N_test':2000,'M_rep':5,'ell_max_gt':8,'seed_base':0,
               'ell_kinv':5,'ell_k1':3,'ell_k2':2,'ell_inv':2,'ell_f0':1}
        path = 'exp1_quick.json'
    else:
        cfg = {'n':5,'K':2,'s':2.0,'sigma':sqrt(0.1),
               'N_values':[500,1000,2000,5000,10000,20000,50000],
               'N_test':5000,'M_rep':30,'ell_max_gt':8,'seed_base':1000,
               'ell_kinv':5,'ell_k1':3,'ell_k2':2,'ell_inv':2,'ell_f0':1}
        path = 'exp1_data.json'
    
    t0 = time.time()
    res,info = run(cfg)
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
    save_plot(res,info,cfg,path)
    print("Done!")
