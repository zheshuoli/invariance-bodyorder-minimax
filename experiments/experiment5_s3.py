"""
Experiment 5: Physically Grounded Pair-Invariant Benchmark
==========================================================
Target: Maier-Saupe molecular orientation energy on SO(3)^n.
  E(g₁,...,gₙ) = Σ_{i<j} V(gᵢ⁻¹gⱼ)
  V(g) = -ε₂·P₂(cosθ) - ε₄·P₄(cosθ)
where θ is the rotation angle of g.

A Maier-Saupe-type pair potential (the cosine of the rotation angle replaces
the director angle of classical Maier-Saupe theory); it lies in F_{K,inv} with K=2.

Estimator settings match Experiment 3 (shared-target comparison):
  - ell_kinv=5, ell_k1=1, ell_k2=1, ell_inv=2, ell_f0=1
  - fixed joint Casimir cutoff Lam_cut=6.0 for F_inv and F_0

Usage:
  python ../experiments/experiment5_s3.py --quick
  python ../experiments/experiment5_s3.py          # s=3 variant (Appendix L)
  python ../experiments/experiment5_s3.py --robust
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import linregress
from math import factorial, sqrt, comb
from itertools import combinations, product as iterproduct
import json, time, sys, os

# ================================================================
# Wigner-D (same as experiments 1-4)
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


# ================================================================
# Maier-Saupe Ground Truth
# ================================================================
def rotation_angle(g):
    """Compute rotation angle θ ∈ [0,π] for each rotation in g.
    Uses rotvec: ||rotvec|| = θ."""
    rv = g.as_rotvec()  # shape (N, 3)
    return np.linalg.norm(rv, axis=1)  # shape (N,)

def legendre_P2(x):
    """P₂(x) = (3x²-1)/2"""
    return (3*x**2 - 1) / 2

def legendre_P4(x):
    """P₄(x) = (35x⁴-30x²+3)/8"""
    return (35*x**4 - 30*x**2 + 3) / 8

def maier_saupe_pair(g_i, g_j, eps2=1.0, eps4=0.3):
    """Compute pairwise Maier-Saupe potential V(gᵢ⁻¹gⱼ) for arrays of rotations."""
    g_rel = g_i.inv() * g_j
    theta = rotation_angle(g_rel)
    cos_theta = np.cos(theta)
    return -eps2 * legendre_P2(cos_theta) - eps4 * legendre_P4(cos_theta)

def eval_maier_saupe(g_list, eps2=1.0, eps4=0.3, lambda_3body=0.0, idx=None):
    """Compute Maier-Saupe energy E = Σ_{i<j} V(gᵢ⁻¹gⱼ) + λ·Σ_{i<j<k} W.
    
    Args:
        g_list: list of n Rotation arrays
        eps2, eps4: Maier-Saupe coefficients
        lambda_3body: strength of 3-body misspecification term
        idx: optional index array for subselecting samples
    Returns:
        energy array of shape (N,)
    """
    gl = [g[idx] for g in g_list] if idx is not None else g_list
    n = len(gl)
    N = len(gl[0])
    
    # Pairwise energy
    energy = np.zeros(N)
    pairs = list(combinations(range(n), 2))
    for (i, j) in pairs:
        energy += maier_saupe_pair(gl[i], gl[j], eps2, eps4)
    
    # Optional 3-body misspecification term
    # W(gᵢ,gⱼ,gₖ) = P₂(cos angle(gᵢ⁻¹gⱼ)) · P₂(cos angle(gⱼ⁻¹gₖ))
    # This is NOT pair-decomposable (product of two pair terms through shared index)
    if lambda_3body > 0:
        triplets = list(combinations(range(n), 3))
        for (i, j, k) in triplets:
            theta_ij = rotation_angle(gl[i].inv() * gl[j])
            theta_jk = rotation_angle(gl[j].inv() * gl[k])
            energy += lambda_3body * legendre_P2(np.cos(theta_ij)) * legendre_P2(np.cos(theta_jk))
    
    return energy


# ================================================================
# Estimators (as in experiment3.py)
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
    """Pair-invariant estimator; mu = (1+Lam)^(-s) with no 1/sqrt(M) factor (as in experiment3.py)."""
    Lam = (N/M)**(2.0/(2*s+D)); mu = 1.0/(1.0+Lam)**s
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
    """Body-order K estimator on absolute coords."""
    Lam = (N/M)**(2.0/(2*s+D)); mu = 1.0/(1.0+Lam)**s
    mean_Y = float(np.mean(Y))
    max_ell = max(ell_1b, ell_2b)
    D_tr_c, D_te_c = {}, {}
    for f in range(n):
        D_tr_c[f] = precompute_D_cache(g_tr[f][idx].as_euler('ZYZ'), max_ell)
        D_te_c[f] = precompute_D_cache(g_te[f].as_euler('ZYZ'), max_ell)
    f_hat = np.full(len(g_te[0]), mean_Y)
    for f in range(n):
        for l in range(1,ell_1b+1):
            wt = 1.0/(1.0+mu*(1.0+l*(l+1))**s)
            for m in range(-l,l+1):
                for nn in range(-l,l+1):
                    c = np.mean(Y*sqrt(2*l+1)*D_tr_c[f][(l,m,nn)].conj())
                    a = c*wt
                    if abs(a)>1e-15:
                        f_hat += (a*sqrt(2*l+1)*D_te_c[f][(l,m,nn)]).real
    for (fi,fj) in pairs:
        for l1 in range(1,ell_2b+1):
            for l2 in range(1,ell_2b+1):
                cas_v = l1*(l1+1)+l2*(l2+1)
                wt = 1.0/(1.0+mu*(1.0+cas_v)**s)
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
    return f_hat

def est_inv(Y, g_tr, g_te, ell_max, N, D, M, s, n, idx, fixed_lam_cut=6.0):
    """Invariant estimator on relative coords. Fixed mode set."""
    Lam = (N/M)**(2.0/(2*s+D)); mu = 1.0/(1.0+Lam)**s
    n_rel = n-1
    Lam_cut = fixed_lam_cut
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
    D_tr_c, D_te_c = [], []
    for j in range(n_rel):
        D_tr_c.append(precompute_D_cache((g_tr[0][idx].inv()*g_tr[j+1][idx]).as_euler('ZYZ'), ell_mp))
        D_te_c.append(precompute_D_cache((g_te[0].inv()*g_te[j+1]).as_euler('ZYZ'), ell_mp))
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
                    e_tr *= sqrt(2*l+1)*D_tr_c[j][(l,m,nn)]
                    e_te_v *= sqrt(2*l+1)*D_te_c[j][(l,m,nn)]
            c = np.mean(Y*e_tr.conj())
            a = c*wt
            if abs(a)>1e-15: f_hat += (a*e_te_v).real
    return f_hat

def est_F0(Y, g_tr, g_te, ell_max, N, D, M, s, n, idx, fixed_lam_cut=6.0):
    """Unstructured estimator on absolute coords. Fixed mode set."""
    Lam = (N/M)**(2.0/(2*s+D)); mu = 1.0/(1.0+Lam)**s
    Lam_cut = fixed_lam_cut
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
    D_tr_c, D_te_c = [], []
    for j in range(n):
        D_tr_c.append(precompute_D_cache(g_tr[j][idx].as_euler('ZYZ'), ell_mp))
        D_te_c.append(precompute_D_cache(g_te[j].as_euler('ZYZ'), ell_mp))
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
                    e_tr *= sqrt(2*l+1)*D_tr_c[j][(l,m,nn)]
                    e_te_v *= sqrt(2*l+1)*D_te_c[j][(l,m,nn)]
            c = np.mean(Y*e_tr.conj())
            a = c*wt
            if abs(a)>1e-15: f_hat += (a*e_te_v).real
    return f_hat


# ================================================================
# Checkpoint Manager (same as experiment 2)
# ================================================================
class CheckpointManager:
    def __init__(self, path):
        self.path = path
        self.data = {'completed': {}, 'results': {}}
        if os.path.exists(path):
            with open(path) as f:
                self.data = json.load(f)
            print(f"  Loaded checkpoint: {len(self.data['completed'])} entries")
    
    def is_done(self, key):
        return key in self.data['completed']
    
    def save(self, key, result_dict):
        self.data['completed'][key] = True
        for cls, N_risks in result_dict.items():
            if cls not in self.data['results']:
                self.data['results'][cls] = {}
            for N_str, risk_val in N_risks.items():
                if N_str not in self.data['results'][cls]:
                    self.data['results'][cls][N_str] = []
                self.data['results'][cls][N_str].append(risk_val)
        with open(self.path, 'w') as f:
            json.dump(self.data, f)
    
    def get_results(self):
        return self.data['results']


# ================================================================
# Run
# ================================================================
def run(config):
    n = config['n']; K = config['K']; s = config['s']
    sigma = config['sigma']
    N_values = config['N_values']
    N_max = max(N_values)
    N_test = config['N_test']
    M_rep = config['M_rep']
    eps2 = config.get('eps2', 1.0)
    eps4 = config.get('eps4', 0.3)
    lambda_3body = config.get('lambda_3body', 0.0)
    
    pairs = list(combinations(range(n), 2)); Mp = len(pairs)
    EL_KINV = config.get('ell_kinv', 5)
    EL_K1 = config.get('ell_k1', 1)
    EL_K2 = config.get('ell_k2', 1)
    EL_INV = config.get('ell_inv', 2)
    EL_F0 = config.get('ell_f0', 1)
    
    info = {
        'F_Kinv': {'D': 3*(K-1), 'M': Mp},
        'F_K':    {'D': 3*K,     'M': Mp},
        'F_inv':  {'D': 3*(n-1), 'M': 1},
        'F_0':    {'D': 3*n,     'M': 1},
    }
    for nm in info:
        info[nm]['slope'] = -2*s / (2*s + info[nm]['D'])
    
    print(f"Config: n={n} K={K} s={s} σ={sigma:.3f}")
    print(f"  Maier-Saupe: ε₂={eps2}, ε₄={eps4}, λ_3body={lambda_3body}")
    print(f"  N_values={N_values}, N_test={N_test}, M_rep={M_rep}")
    print(f"  ℓ_max: Kinv={EL_KINV}, K_1b={EL_K1}, K_2b={EL_K2}, inv={EL_INV}, F0={EL_F0}")
    for nm, d in info.items():
        print(f"  {nm}: D={d['D']}, slope={d['slope']:.4f}")
    
    # Normalize: evaluate on random sample to get std
    ng = [Rotation.random(5000, random_state=777+i) for i in range(n)]
    v_test = eval_maier_saupe(ng, eps2, eps4, lambda_3body)
    norm_factor = float(np.std(v_test)) / 0.1 if np.std(v_test) > 0 else 1.0
    print(f"  GT std={np.std(v_test):.4f}, norm_factor={norm_factor:.2f}")
    
    results = {nm: {N: [] for N in N_values} for nm in info}
    
    for rep in range(M_rep):
        t0 = time.time()
        rng = np.random.RandomState(config.get('seed_base', 0) + rep)
        
        g_tr = [Rotation.random(N_max, random_state=rng.randint(2**31)) for _ in range(n)]
        g_te = [Rotation.random(N_test, random_state=rng.randint(2**31)) for _ in range(n)]
        
        # Evaluate Maier-Saupe energy (shared GT for all estimators)
        f_tr = eval_maier_saupe(g_tr, eps2, eps4, lambda_3body) / norm_factor
        f_te = eval_maier_saupe(g_te, eps2, eps4, lambda_3body) / norm_factor
        
        noise = rng.randn(N_max) * sigma
        Y_all = f_tr + noise
        
        for N in N_values:
            idx = np.arange(N)
            Y = Y_all[:N]
            
            fh = est_Kinv(Y, g_tr, g_te, pairs, EL_KINV, N, info['F_Kinv']['D'], Mp, s, idx)
            results['F_Kinv'][N].append(float(np.mean((fh-f_te)**2)))
            
            fh = est_K(Y, g_tr, g_te, pairs, EL_K1, EL_K2, N, info['F_K']['D'], Mp, s, n, idx)
            results['F_K'][N].append(float(np.mean((fh-f_te)**2)))
            
            fh = est_inv(Y, g_tr, g_te, EL_INV, N, info['F_inv']['D'], 1, s, n, idx,
                         fixed_lam_cut=6.0)
            results['F_inv'][N].append(float(np.mean((fh-f_te)**2)))
            
            fh = est_F0(Y, g_tr, g_te, EL_F0, N, info['F_0']['D'], 1, s, n, idx,
                        fixed_lam_cut=6.0)
            results['F_0'][N].append(float(np.mean((fh-f_te)**2)))
        
        dt = time.time()-t0
        if (rep+1) % max(1, M_rep//10) == 0 or rep+1 == M_rep:
            Ns = N_values[-1]
            print(f"  Rep {rep+1}/{M_rep} ({dt:.1f}s) " +
                  " ".join(f"{k}={results[k][Ns][-1]:.3e}" for k in info))
    
    return results, info


# ================================================================
# Analysis
# ================================================================
def analyze(results, info, config):
    N_values = config['N_values']
    s = config['s']
    analysis = {}
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    for nm in ['F_Kinv', 'F_K', 'F_inv', 'F_0']:
        means = [float(np.mean(results[nm][N])) for N in N_values]
        stds = [float(np.std(results[nm][N])) for N in N_values]
        logN = [np.log10(N) for N in N_values]
        logR = [np.log10(m) for m in means if m > 0]
        npt = min(len(logN), len(logR))
        sl_f = linregress(logN[:npt], logR[:npt]) if npt >= 3 else None
        
        analysis[nm] = {
            'D_theory': info[nm]['D'],
            'slope_theory': info[nm]['slope'],
            'slope_emp': float(sl_f.slope) if sl_f else None,
            'R2': float(sl_f.rvalue**2) if sl_f else None,
            'N_values': N_values,
            'risk_means': means,
            'risk_stds': stds,
        }
        
        print(f"\n  {nm} (D_theory={info[nm]['D']})")
        if sl_f:
            gap = abs(sl_f.slope) - abs(info[nm]['slope'])
            print(f"    slope: {sl_f.slope:.4f} (theory={info[nm]['slope']:.4f}, gap={gap:+.3f}, R²={sl_f.rvalue**2:.4f})")
        for i, N in enumerate(N_values):
            print(f"      N={N:>6d}: {means[i]:.4e} ± {stds[i]:.4e}")
    
    # Ordering check
    print("\n  === ORDERING CHECK ===")
    for i, N in enumerate(N_values):
        r = {nm: analysis[nm]['risk_means'][i] for nm in info}
        ordered = sorted(r.items(), key=lambda x: x[1])
        best = ordered[0][0]
        mark = '✓' if best == 'F_Kinv' else '✗'
        print(f"    N={N:>6d}: {' < '.join(f'{k}' for k,_ in ordered)} {mark}")
    
    # Sample efficiency: N needed to reach threshold
    print("\n  === SAMPLE EFFICIENCY ===")
    # Use F_Kinv's risk at largest N as threshold
    threshold = analysis['F_Kinv']['risk_means'][-1] * 2  # 2× F_Kinv's best
    print(f"  Threshold: {threshold:.4e} (2× F_Kinv at N={N_values[-1]})")
    for nm in ['F_Kinv', 'F_K', 'F_inv', 'F_0']:
        means = analysis[nm]['risk_means']
        # Find first N where risk < threshold
        found = None
        for i, (N, m) in enumerate(zip(N_values, means)):
            if m < threshold:
                found = N
                break
        if found:
            print(f"    {nm}: reaches threshold at N={found}")
        else:
            print(f"    {nm}: does NOT reach threshold (risk={means[-1]:.4e})")
    
    # Risk ratios
    print("\n  === RISK RATIOS vs F_Kinv ===")
    for i, N in enumerate(N_values):
        rk = analysis['F_Kinv']['risk_means'][i]
        line = f"    N={N:>6d}:"
        for nm in ['F_K', 'F_inv', 'F_0']:
            r = analysis[nm]['risk_means'][i]
            line += f"  {nm}={r/rk:.2f}×"
        print(line)
    
    return analysis


# ================================================================
# Save & Plot
# ================================================================
def save_and_plot(results, info, analysis, config, path):
    data = {
        'config': {k: v for k, v in config.items() if not callable(v)},
        'theory': {nm: {'D': d['D'], 'slope': d['slope']} for nm, d in info.items()},
        'analysis': analysis,
        'raw': {nm: {str(N): vals for N, vals in results[nm].items()} for nm in results},
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n  Saved: {path}")
    
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        C = {'F_0':'#E69F00', 'F_inv':'#56B4E9', 'F_K':'#009E73', 'F_Kinv':'#CC79A7'}
        L = {'F_0':r'$\mathcal{F}_0$ (no structure)',
             'F_inv':r'$\mathcal{F}_{\rm inv}$ (inv. only)',
             'F_K':r'$\mathcal{F}_K$ (body-order only)',
             'F_Kinv':r'$\mathcal{F}_{K,{\rm inv}}$ (both priors)'}
        MK = {'F_Kinv': 'o', 'F_K': 's', 'F_inv': '^', 'F_0': 'D'}
        
        N_values = config['N_values']
        s = config['s']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        
        # Panel (a): Learning curves
        for nm in ['F_Kinv', 'F_K', 'F_inv', 'F_0']:
            means = analysis[nm]['risk_means']
            stds = analysis[nm]['risk_stds']
            ax1.errorbar(N_values, means, yerr=stds, fmt=MK[nm]+'-',
                        color=C[nm], label=L[nm], markersize=5, linewidth=1.5,
                        capsize=2, elinewidth=0.6)
            # Theory slope reference
            D = info[nm]['D']
            th_slope = -2*s/(2*s+D)
            Nm, Rm = N_values[-1], means[-1]
            N_line = np.array([N_values[0], N_values[-1]])
            R_line = Rm * (N_line / Nm) ** th_slope
            ax1.plot(N_line, R_line, '--', color=C[nm], alpha=0.3, linewidth=0.8)
        
        ax1.set_xscale('log'); ax1.set_yscale('log')
        ax1.set_xlabel(r'Sample size $N$')
        ax1.set_ylabel(r'$L^2$ risk')
        n_val = config['n']
        ax1.set_title(f'(a) Learning curves ($n={n_val}$, Maier-Saupe)')
        ax1.legend(fontsize=7, loc='lower left')
        ax1.grid(True, alpha=0.3)
        
        # Panel (b): Risk ratio vs F_Kinv
        for nm in ['F_K', 'F_inv', 'F_0']:
            ratios = [analysis[nm]['risk_means'][i] / analysis['F_Kinv']['risk_means'][i]
                     for i in range(len(N_values))]
            ax2.plot(N_values, ratios, MK[nm]+'-', color=C[nm],
                    label=L[nm], markersize=5, linewidth=1.5)
        
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xscale('log')
        ax2.set_xlabel(r'Sample size $N$')
        ax2.set_ylabel(r'Risk ratio vs $\mathcal{F}_{K,\mathrm{inv}}$')
        ax2.set_title('(b) Relative inefficiency without correct prior')
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fp = path.replace('.json', '_fig.pdf')
        plt.savefig(fp, dpi=300, bbox_inches='tight')
        print(f"  Figure: {fp}")
    except Exception as e:
        print(f"  Plot error: {e}")


# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument('--quick', action='store_true',
                    help='reduced configuration (fewer repetitions, smaller N) to check the environment')
    ap.add_argument('--robust', action='store_true',
                    help='lambda=0.3 three-body misspecification (Appendix M)')
    args = ap.parse_args()
    quick = args.quick
    robust = args.robust
    
    print("="*70)
    print("EXP 5: Maier-Saupe Pair-Invariant Benchmark")
    print("="*70)
    
    # Sanity check: Wigner-D
    beta = np.array([0.5, 1.0, 2.0])
    assert np.allclose(wigner_d_small(1,1,1,beta), np.cos(beta/2)**2)
    print("  Wigner-D ✓")
    
    # Sanity check: Maier-Saupe is rotation-invariant
    g_test = [Rotation.random(100, random_state=42) for _ in range(3)]
    e1 = eval_maier_saupe(g_test, 1.0, 0.3)
    g_rot = Rotation.random(1, random_state=99)
    g_rotated = [g_rot * g for g in g_test]
    e2 = eval_maier_saupe(g_rotated, 1.0, 0.3)
    assert np.allclose(e1, e2, atol=1e-10), f"Rotation invariance failed! max diff={np.max(np.abs(e1-e2))}"
    print("  Rotation invariance ✓")
    
    lambda_3body = 0.3 if robust else 0.0
    suffix = '_robust' if robust else ''
    
    if quick:
        configs = [{
            'n': 6, 'K': 2, 's': 3.0, 'sigma': sqrt(0.1),
            'N_values': [100, 200, 500, 1000, 2000, 5000],
            'N_test': 2000, 'M_rep': 10,
            'seed_base': 8000,
            'ell_kinv': 5, 'ell_k1': 1, 'ell_k2': 1, 'ell_inv': 2, 'ell_f0': 1,
            'eps2': 1.0, 'eps4': 0.3, 'lambda_3body': lambda_3body,
        }]
        paths = [f'exp5_s3_quick_n6{suffix}.json']
    else:
        configs = [
            {
                'n': 6, 'K': 2, 's': 3.0, 'sigma': sqrt(0.1),
                'N_values': [100, 200, 500, 1000, 2000, 5000, 10000],
                'N_test': 5000, 'M_rep': 30,
                'seed_base': 8000,
                'ell_kinv': 5, 'ell_k1': 1, 'ell_k2': 1, 'ell_inv': 2, 'ell_f0': 1,
                'eps2': 1.0, 'eps4': 0.3, 'lambda_3body': lambda_3body,
            },
            {
                'n': 10, 'K': 2, 's': 3.0, 'sigma': sqrt(0.1),
                'N_values': [100, 200, 500, 1000, 2000, 5000, 10000],
                'N_test': 5000, 'M_rep': 30,
                'seed_base': 9000,
                'ell_kinv': 5, 'ell_k1': 1, 'ell_k2': 1, 'ell_inv': 2, 'ell_f0': 1,
                'eps2': 1.0, 'eps4': 0.3, 'lambda_3body': lambda_3body,
            },
        ]
        paths = [f'exp5_s3_data_n6{suffix}.json', f'exp5_s3_data_n10{suffix}.json']
    
    for cfg, path in zip(configs, paths):
        print(f"\n{'='*70}")
        print(f"Running n={cfg['n']}...")
        print(f"{'='*70}")
        
        t0 = time.time()
        res, info = run(cfg)
        analysis = analyze(res, info, cfg)
        save_and_plot(res, info, analysis, cfg, path)
        print(f"\n  Total time: {(time.time()-t0)/60:.1f} min")
    
    print("\nDone!")
