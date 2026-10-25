"""
Experiment 2: Dimension Decomposition — Sweep n=3 to 8
======================================================
For each n, fit log-log risk slopes of the 4 estimators (class-specific targets).
Theory predicts a slope magnitude 2s/(2s+D) with D = 3n for F_0 and D = 3(K-1) = 3
for F_Kinv, so the predicted F_0 exponent degrades with n while the F_Kinv exponent
does not depend on n; the script reports the fitted slopes for comparison.

Checkpointing: the script saves after every completed (n, rep). A checkpoint
found in the working directory is resumed automatically; --fresh ignores it.
Each (n, rep) is seeded independently, so a resumed run computes the same
values as an uninterrupted one.

Usage:
  python ../experiments/experiment2.py --quick    # n=3,4,5, 5 reps, N to 10000
  python ../experiments/experiment2.py            # n=3..8, 20 reps, N to 50000 (Figure 3a,b)
  python ../experiments/experiment2.py --fresh    # ignore an existing checkpoint
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import linregress
from math import factorial, sqrt, comb
from itertools import combinations, product as iterproduct
import json, time, sys, os

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
# Ground-truth constructors (as in experiment1.py)
# ================================================================
def make_random_so3_func(ell_max, s, rng, include_const=False):
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
    a,b,g = euler[:,0],euler[:,1],euler[:,2]
    N = len(a)
    c0 = coeffs.get((0,0,0), 0.0)
    val = np.full(N, c0, dtype=complex)
    for (ell,m,nn),c in coeffs.items():
        if ell > 0:
            val += c*(2*ell+1)*wigner_D(ell,m,nn,a,b,g)
    return val

# F_Kinv GT
def make_gt_Kinv(ell_max, s, n, seed=42):
    rng = np.random.RandomState(seed)
    pairs = list(combinations(range(n), 2))
    coeffs = {}
    for pair in pairs:
        pc = {}
        for ell in range(1, ell_max+1):
            w = (1+ell*(ell+1))**(-(s+1)/2.0)
            for m in range(-ell,ell+1):
                for nn in range(-ell,ell+1):
                    pc[(ell,m,nn)] = w*(rng.randn()+1j*rng.randn())/sqrt(2)
        coeffs[pair] = pc
    return coeffs

def eval_gt_Kinv(coeffs, g_list, idx=None):
    gl = [g[idx] for g in g_list] if idx is not None else g_list
    N = len(gl[0]); fv = np.zeros(N)
    for (i,j), pc in coeffs.items():
        eu = (gl[i].inv()*gl[j]).as_euler('ZYZ')
        a,b,g = eu[:,0],eu[:,1],eu[:,2]
        for (l,m,nn),c in pc.items():
            fv += (c*(2*l+1)*wigner_D(l,m,nn,a,b,g)).real
    return fv

# F_K GT (1-body + non-invariant 2-body products)
def make_gt_FK(ell_max_1b, ell_max_2b, s, n, R_2b=5, seed=43):
    rng = np.random.RandomState(seed)
    pairs = list(combinations(range(n), 2))
    coeffs_1b = {i: make_random_so3_func(ell_max_1b, s, rng) for i in range(n)}
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
    for i, pc in coeffs_1b.items():
        eu = gl[i].as_euler('ZYZ')
        fv += eval_so3_func(pc, eu).real
    for (i,j), terms in coeffs_2b.items():
        eu_i, eu_j = gl[i].as_euler('ZYZ'), gl[j].as_euler('ZYZ')
        for ci, cj in terms:
            vi = eval_so3_func(ci, eu_i)
            vj = eval_so3_func(cj, eu_j)
            fv += (vi * vj).real
    return fv

# F_inv and F_0 GT: dense Fourier-domain construction (as in experiment1.py)
def make_gt_fourier(n_factors, ell_max, s, N_max, D_eff, M=1, seed=44, fixed_lam_cut=None):
    """Dense Fourier-domain GT. If fixed_lam_cut=None, uses the N_max-dependent cutoff (as in experiment1.py)."""
    rng = np.random.RandomState(seed)
    if fixed_lam_cut is not None:
        Lam_cut = fixed_lam_cut
    else:
        Lam = (N_max / M) ** (2.0 / (2*s + D_eff))
        Lam_cut = max(Lam * 3, 6.0)
    ell_mp = 0
    while (ell_mp+1)*(ell_mp+2) <= Lam_cut:
        ell_mp += 1
    ell_mp = min(ell_mp, ell_max)
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
    Hs_sq = sum((1 + sum(l*(l+1) for l,_,_ in spec))**s * abs(c)**2
                for spec, c in modes)
    norm = 1.0 / sqrt(Hs_sq) if Hs_sq > 0 else 1.0
    modes = [(spec, c * norm) for spec, c in modes]
    return modes, ell_mp

def eval_gt_fourier(modes, ell_max, g_list, idx=None, relative=False):
    gl = [g[idx] for g in g_list] if idx is not None else g_list
    if relative:
        n_factors = len(gl) - 1
        euler = [(gl[0].inv() * gl[j+1]).as_euler('ZYZ') for j in range(n_factors)]
    else:
        n_factors = len(gl)
        euler = [gl[j].as_euler('ZYZ') for j in range(n_factors)]
    N = len(euler[0])
    D_caches = [precompute_D_cache(euler[j], ell_max) for j in range(n_factors)]
    f_vals = np.zeros(N, dtype=complex)
    for spec, c in modes:
        val = np.ones(N, dtype=complex)
        for j, (l, m, nn) in enumerate(spec):
            if l > 0:
                val *= sqrt(2*l+1) * D_caches[j][(l, m, nn)]
        f_vals += c * val
    return f_vals.real

# ================================================================
# Estimators (as in experiment1.py / experiment3.py)
# ================================================================
def est_Kinv(Y, g_tr, g_te, pairs, ell_max, N, D, M, s, idx):
    Lam = (N/M)**(2.0/(2*s+D)); mu = 1.0/(1.0+Lam)**s
    # Same shrinkage rule for all n, with M = C(n, 2); no additional 1/sqrt(M) factor (see README)
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

def est_inv(Y, g_tr, g_te, ell_max, N, D, M, s, n, idx, fixed_lam_cut=None):
    Lam = (N/M)**(2.0/(2*s+D)); mu = 1.0/(1.0+Lam)**s
    n_rel = n-1
    if fixed_lam_cut is not None:
        Lam_cut = fixed_lam_cut
    else:
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

def est_F0(Y, g_tr, g_te, ell_max, N, D, M, s, n, idx, fixed_lam_cut=None):
    Lam = (N/M)**(2.0/(2*s+D)); mu = 1.0/(1.0+Lam)**s
    if fixed_lam_cut is not None:
        Lam_cut = fixed_lam_cut
    else:
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
# Checkpoint manager
# ================================================================
class CheckpointManager:
    """Saves after every completed (n, rep). A checkpoint is reused only if it was
    written with an identical experiment configuration."""
    def __init__(self, path, config, fresh=False):
        self.path = path
        self.config_json = json.loads(json.dumps(config))
        self.data = {'experiment_config': self.config_json, 'completed': {}, 'results': {}}
        if os.path.exists(path) and not fresh:
            with open(path) as f:
                loaded = json.load(f)
            if loaded.get('experiment_config') == self.config_json:
                self.data = loaded
                print(f"  Loaded checkpoint: {len(self.data['completed'])} entries")
            else:
                print("  Checkpoint configuration differs from the current one; starting over (the file will be overwritten)")
    
    def is_done(self, n_val, rep):
        key = f"{n_val}_{rep}"
        return key in self.data['completed']
    
    def save_result(self, n_val, rep, result_dict):
        key = f"{n_val}_{rep}"
        self.data['completed'][key] = True
        # Merge results
        for cls_name, N_risks in result_dict.items():
            if cls_name not in self.data['results']:
                self.data['results'][cls_name] = {}
            if str(n_val) not in self.data['results'][cls_name]:
                self.data['results'][cls_name][str(n_val)] = {}
            for N_str, risk_val in N_risks.items():
                if N_str not in self.data['results'][cls_name][str(n_val)]:
                    self.data['results'][cls_name][str(n_val)][N_str] = []
                self.data['results'][cls_name][str(n_val)][N_str].append(risk_val)
        # Atomic write: an interrupted write never corrupts the previous checkpoint
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.data, f)
        os.replace(tmp, self.path)
    
    def get_results(self):
        return self.data['results']


# ================================================================
# Run one (n, rep) trial
# ================================================================
def run_one_trial(n, K, s, sigma, N_values, N_test, N_max, ell_cfg, rep, seed_base):
    """Run all 4 estimators for one (n, rep). Returns dict of risks.
    
    Key design: est_inv and est_F0 use FIXED mode sets (computed from N_max),
    just like est_Kinv and est_K use fixed mode sets (from ell_max config).
    This ensures the slope reflects only the sample-size effect, not mode-set growth.
    """
    pairs = list(combinations(range(n), 2)); Mp = len(pairs)
    EL_KINV = ell_cfg['kinv']; EL_K1 = ell_cfg['k1']; EL_K2 = ell_cfg['k2']
    EL_INV = ell_cfg['inv']; EL_F0 = ell_cfg['f0']
    
    info = {
        'F_Kinv': {'D': 3*(K-1), 'M': Mp},
        'F_K':    {'D': 3*K,     'M': Mp},
        'F_inv':  {'D': 3*(n-1), 'M': 1},
        'F_0':    {'D': 3*n,     'M': 1},
    }
    
    # Fixed Lam_cut from N_max — mode set identical for all N within this n
    # Same principle as est_Kinv using fixed ell_max for all N
    Lam_inv = (N_max / 1) ** (2.0 / (2*s + info['F_inv']['D']))
    fixed_lc_inv = max(Lam_inv * 3, 6.0)
    Lam_f0 = (N_max / 1) ** (2.0 / (2*s + info['F_0']['D']))
    fixed_lc_f0 = max(Lam_f0 * 3, 6.0)
    
    # Class-specific ground truths (deterministic per n)
    # GT and estimator use the same fixed cutoff
    gt_kinv = make_gt_Kinv(EL_KINV, s, n, seed=42)
    gt_fk_1b, gt_fk_2b = make_gt_FK(EL_K1, EL_K2, s, n, R_2b=5, seed=43)
    gt_finv, gt_finv_ell = make_gt_fourier(
        n-1, EL_INV, s, N_max, 3*(n-1), M=1, seed=44, fixed_lam_cut=fixed_lc_inv)
    gt_f0, gt_f0_ell = make_gt_fourier(
        n, EL_F0, s, N_max, 3*n, M=1, seed=45, fixed_lam_cut=fixed_lc_f0)
    
    # Normalize
    ng = [Rotation.random(5000, random_state=999+i) for i in range(n)]
    norms = {}
    for nm, ef in [
            ('F_Kinv', lambda g: eval_gt_Kinv(gt_kinv, g)),
            ('F_K',    lambda g: eval_gt_FK(gt_fk_1b, gt_fk_2b, g)),
            ('F_inv',  lambda g: eval_gt_fourier(gt_finv, gt_finv_ell, g, relative=True)),
            ('F_0',    lambda g: eval_gt_fourier(gt_f0, gt_f0_ell, g, relative=False))]:
        v = ef(ng)
        norms[nm] = float(np.std(v))/0.1 if np.std(v)>0 else 1.0
    
    rng = np.random.RandomState(seed_base + n*1000 + rep)
    g_tr = [Rotation.random(N_max, random_state=rng.randint(2**31)) for _ in range(n)]
    g_te = [Rotation.random(N_test, random_state=rng.randint(2**31)) for _ in range(n)]
    
    # Evaluate GTs
    f_kinv_tr = eval_gt_Kinv(gt_kinv, g_tr)/norms['F_Kinv']
    f_kinv_te = eval_gt_Kinv(gt_kinv, g_te)/norms['F_Kinv']
    f_fk_tr = eval_gt_FK(gt_fk_1b, gt_fk_2b, g_tr)/norms['F_K']
    f_fk_te = eval_gt_FK(gt_fk_1b, gt_fk_2b, g_te)/norms['F_K']
    f_inv_tr = eval_gt_fourier(gt_finv, gt_finv_ell, g_tr, relative=True)/norms['F_inv']
    f_inv_te = eval_gt_fourier(gt_finv, gt_finv_ell, g_te, relative=True)/norms['F_inv']
    f_f0_tr = eval_gt_fourier(gt_f0, gt_f0_ell, g_tr, relative=False)/norms['F_0']
    f_f0_te = eval_gt_fourier(gt_f0, gt_f0_ell, g_te, relative=False)/norms['F_0']
    
    noise = rng.randn(N_max) * sigma
    
    result = {}
    for N in N_values:
        idx = np.arange(N)
        
        Y = f_kinv_tr[:N]+noise[:N]
        fh = est_Kinv(Y, g_tr, g_te, pairs, EL_KINV, N, info['F_Kinv']['D'], Mp, s, idx)
        result.setdefault('F_Kinv', {})[str(N)] = float(np.mean((fh-f_kinv_te)**2))
        
        Y = f_fk_tr[:N]+noise[:N]
        fh = est_K(Y, g_tr, g_te, pairs, EL_K1, EL_K2, N, info['F_K']['D'], Mp, s, n, idx)
        result.setdefault('F_K', {})[str(N)] = float(np.mean((fh-f_fk_te)**2))
        
        Y = f_inv_tr[:N]+noise[:N]
        fh = est_inv(Y, g_tr, g_te, EL_INV, N, info['F_inv']['D'], 1, s, n, idx,
                     fixed_lam_cut=fixed_lc_inv)
        result.setdefault('F_inv', {})[str(N)] = float(np.mean((fh-f_inv_te)**2))
        
        Y = f_f0_tr[:N]+noise[:N]
        fh = est_F0(Y, g_tr, g_te, EL_F0, N, info['F_0']['D'], 1, s, n, idx,
                    fixed_lam_cut=fixed_lc_f0)
        result.setdefault('F_0', {})[str(N)] = float(np.mean((fh-f_f0_te)**2))
    
    return result


# ================================================================
# Analysis & Plot
# ================================================================
def analyze_and_plot(ckpt_results, config, output_path):
    n_values = config['n_values']
    N_values = config['N_values']
    K = config['K']; s = config['s']
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    analysis = {}
    for cls in ['F_Kinv', 'F_K', 'F_inv', 'F_0']:
        analysis[cls] = {}
        for n_val in n_values:
            if cls not in ckpt_results or str(n_val) not in ckpt_results[cls]:
                continue
            N_data = ckpt_results[cls][str(n_val)]
            means = []
            stds = []
            for N in N_values:
                vals = N_data.get(str(N), [])
                if vals:
                    means.append(float(np.mean(vals)))
                    stds.append(float(np.std(vals)))
                else:
                    means.append(None)
                    stds.append(None)
            
            # Fit slope: full range
            valid = [(np.log10(N), np.log10(m)) for N, m in zip(N_values, means)
                     if m is not None and m > 0]
            if len(valid) >= 3:
                logN, logR = zip(*valid)
                sl = linregress(logN, logR)
                slope, r2 = float(sl.slope), float(sl.rvalue**2)
            else:
                slope, r2 = None, None
            
            # Fit slope: large-N (last half)
            nlg = max(3, len(valid)//2)
            if len(valid) >= 3:
                sl_l = linregress(logN[-nlg:], logR[-nlg:])
                slope_large, r2_large = float(sl_l.slope), float(sl_l.rvalue**2)
            else:
                slope_large, r2_large = None, None
            
            # Theory
            Mp = comb(n_val, K)
            D_theory = {'F_Kinv': 3*(K-1), 'F_K': 3*K,
                        'F_inv': 3*(n_val-1), 'F_0': 3*n_val}[cls]
            slope_theory = -2*s/(2*s+D_theory)
            
            analysis[cls][n_val] = {
                'D_theory': D_theory, 'slope_theory': slope_theory,
                'slope_emp': slope, 'R2': r2,
                'slope_large': slope_large, 'R2_large': r2_large,
                'means': means, 'stds': stds,
                'n_reps': len(N_data.get(str(N_values[0]), [])),
            }
            
            sl_str = f"full={slope:.4f}" if slope else "full=N/A"
            sl_l_str = f"large={slope_large:.4f}" if slope_large else "large=N/A"
            print(f"  {cls} n={n_val}: D={D_theory}, "
                  f"th={slope_theory:.4f}, {sl_str}, {sl_l_str}, "
                  f"reps={analysis[cls][n_val]['n_reps']}")
    
    # Save
    out = {'config': config, 'analysis': {}, 'raw': ckpt_results}
    for cls in analysis:
        out['analysis'][cls] = {}
        for n_val, a in analysis[cls].items():
            out['analysis'][cls][str(n_val)] = a
    with open(output_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {output_path}")
    
    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        
        C = {'F_0':'#E69F00', 'F_inv':'#56B4E9', 'F_K':'#009E73', 'F_Kinv':'#CC79A7'}
        L = {'F_0':r'$\mathcal{F}_0$', 'F_inv':r'$\mathcal{F}_{\rm inv}$',
             'F_K':r'$\mathcal{F}_K$', 'F_Kinv':r'$\mathcal{F}_{K,{\rm inv}}$'}
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        
        # Fig 2a: |slope| vs n for each class
        for cls in ['F_Kinv', 'F_K', 'F_inv', 'F_0']:
            ns_plot, slopes_emp, slopes_th = [], [], []
            for n_val in n_values:
                if n_val in analysis[cls] and analysis[cls][n_val]['slope_emp'] is not None:
                    ns_plot.append(n_val)
                    slopes_emp.append(abs(analysis[cls][n_val]['slope_emp']))
                    slopes_th.append(abs(analysis[cls][n_val]['slope_theory']))
            if ns_plot:
                ax1.plot(ns_plot, slopes_emp, 'o-', color=C[cls], label=L[cls]+' (emp)',
                        markersize=5, linewidth=1.5)
                ax1.plot(ns_plot, slopes_th, '--', color=C[cls], alpha=0.4,
                        label=L[cls]+' (theory)')
        ax1.set_xlabel(r'Number of factors $n$')
        ax1.set_ylabel(r'$|\mathrm{slope}|$')
        ax1.set_title(r'(a) Rate exponent vs $n$')
        ax1.legend(fontsize=6, ncol=2)
        ax1.grid(True, alpha=0.3)
        
        # Fig 2b: theory - empirical slope gap vs n
        for cls in ['F_Kinv', 'F_K', 'F_inv', 'F_0']:
            ns_plot, gaps = [], []
            for n_val in n_values:
                if n_val in analysis[cls] and analysis[cls][n_val]['slope_emp'] is not None:
                    ns_plot.append(n_val)
                    gaps.append(abs(analysis[cls][n_val]['slope_emp']) -
                               abs(analysis[cls][n_val]['slope_theory']))
            if ns_plot:
                ax2.plot(ns_plot, gaps, 'o-', color=C[cls], label=L[cls],
                        markersize=5, linewidth=1.5)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel(r'Number of factors $n$')
        ax2.set_ylabel(r'$|\hat\alpha| - |\alpha_{\rm theory}|$')
        ax2.set_title('(b) Slope deviation from theory')
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fp = output_path.replace('.json', '_fig.pdf')
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
    ap.add_argument('--fresh', action='store_true',
                    help='ignore an existing checkpoint in the working directory and start over')
    args = ap.parse_args()
    quick = args.quick
    
    print("="*70)
    print("EXP 2: Dimension Decomposition — n sweep")
    print("="*70)
    
    beta = np.array([0.5, 1.0, 2.0])
    assert np.allclose(wigner_d_small(1,1,1,beta), np.cos(beta/2)**2)
    print("  Wigner-D ✓")
    
    if quick:
        config = {
            'n_values': [3, 4, 5],
            'K': 2, 's': 2.0, 'sigma': sqrt(0.1),
            'N_values': [500, 1000, 2000, 5000, 10000],
            'N_test': 2000, 'N_max': 10000, 'M_rep': 5,
            'seed_base': 2000,
            'ell': {'kinv': 5, 'k1': 3, 'k2': 2, 'inv': 2, 'f0': 1},
        }
        ckpt_path = 'exp2_quick_ckpt.json'
        output_path = 'exp2_quick.json'
    else:
        config = {
            'n_values': [3, 4, 5, 6, 7, 8],
            'K': 2, 's': 2.0, 'sigma': sqrt(0.1),
            'N_values': [500, 1000, 2000, 5000, 10000, 20000, 50000],
            'N_test': 5000, 'N_max': 50000, 'M_rep': 20,
            'seed_base': 2000,
            'ell': {'kinv': 5, 'k1': 3, 'k2': 2, 'inv': 2, 'f0': 1},
        }
        ckpt_path = 'exp2_data_ckpt.json'
        output_path = 'exp2_data.json'
    
    ckpt = CheckpointManager(ckpt_path, config, fresh=args.fresh)
    
    n_values = config['n_values']
    M_rep = config['M_rep']
    total = len(n_values) * M_rep
    done = sum(1 for n_val in n_values for rep in range(M_rep) 
               if ckpt.is_done(n_val, rep))
    print(f"  Progress: {done}/{total} trials completed")
    
    t0_global = time.time()
    trial_num = 0
    
    for n_val in n_values:
        print(f"\n--- n={n_val} (D_Kinv={3*(config['K']-1)}, D_K={3*config['K']}, "
              f"D_inv={3*(n_val-1)}, D_0={3*n_val}) ---")
        
        for rep in range(M_rep):
            trial_num += 1
            
            if ckpt.is_done(n_val, rep):
                continue
            
            t0 = time.time()
            result = run_one_trial(
                n=n_val, K=config['K'], s=config['s'], sigma=config['sigma'],
                N_values=config['N_values'], N_test=config['N_test'],
                N_max=config['N_max'], ell_cfg=config['ell'],
                rep=rep, seed_base=config['seed_base']
            )
            
            ckpt.save_result(n_val, rep, result)
            dt = time.time() - t0
            
            done += 1
            elapsed = time.time() - t0_global
            eta = elapsed / done * (total - done) / 60 if done > 0 else 0
            
            if done % max(1, total//20) == 0 or done == total:
                Ns = str(config['N_values'][-1])
                print(f"    [{done}/{total}] n={n_val} rep={rep+1} ({dt:.1f}s) "
                      f"ETA={eta:.0f}min "
                      f"Kinv={result['F_Kinv'][Ns]:.3e} "
                      f"F0={result['F_0'][Ns]:.3e}")
    
    print(f"\nTotal: {(time.time()-t0_global)/60:.1f} min")
    analyze_and_plot(ckpt.get_results(), config, output_path)
    print("Done!")
