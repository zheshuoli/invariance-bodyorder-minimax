"""
Experiment 4: Adaptive body-order selection (sample splitting)
==============================================================
Empirical evaluation of fixed-family body-order selection (Section 4, Figure 4).
Candidates K' = 1, 2, 3 use Sobolev-weighted coefficient shrinkage; the K'=3
candidate uses pure-S triplet modes (the projection P_Delta^perp of Appendix C); the
candidate family is the fixed-cutoff sample-splitting estimator of Appendix J.
The selected candidate is refit on all training samples.

Setup: n=6, K*=2, K_max=3, f* in F_{K*,inv}
Estimators reported:
  - results['oracle']  : true-order candidate K'=2, fit on all N
  - results['adaptive']: sample split, K' selected on the validation half, refit on all N
  - results['over']    : K'=3, fit on all N
  - results['under']   : K'=1 (constant), fit on all N
The paper's "empirical oracle" is the pointwise minimum of the mean risks of
the three fixed candidates (K'=1,2,3); analyze() reports the adaptive risk
relative to both the true-order candidate and this empirical oracle.

Usage:
  python ../experiments/experiment4.py --quick    # 10 reps, N to 10000
  python ../experiments/experiment4.py            # 30 reps, N to 50000 (Figure 4)
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import linregress
from math import factorial, sqrt, comb
from itertools import combinations, product as iterproduct
import json, time, sys

# ================================================================
# Wigner-D (shared with exp1/3)
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
# Ground Truth: f* ∈ F_{K*=2,inv}
# ================================================================
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
    Hs = sum((1+l*(l+1))**s*(2*l+1)*abs(c)**2
             for p in pairs for (l,_,_),c in coeffs[p].items())
    norm = 1.0/sqrt(Hs) if Hs>0 else 1.0
    for p in pairs:
        for k in coeffs[p]: coeffs[p][k] *= norm
    return coeffs

def eval_gt(coeffs, g_list, idx=None):
    gl = [g[idx] for g in g_list] if idx is not None else g_list
    N = len(gl[0]); fv = np.zeros(N)
    for (i,j), pc in coeffs.items():
        eu = (gl[i].inv()*gl[j]).as_euler('ZYZ')
        a,b,g = eu[:,0],eu[:,1],eu[:,2]
        for (l,m,nn),c in pc.items():
            fv += (c*(2*l+1)*wigner_D(l,m,nn,a,b,g)).real
    return fv

# ================================================================
# General Invariant Estimator for body-order ≤ K'
# ================================================================
def apply_pure_S_projection_k3(coeffs, ell_max_k):
    """P_Delta^perp on SO(3)^2 tensor coefficients (k=3 triplets).

    Removes the lower-order leakage subspace: functions of (r_1, r_2) that
    depend only on r_1^{-1} r_2, i.e. pair-(s_1,s_2) invariants.  Within each
    matched (l,l) block and for each pair of right indices (n1,n2) the leakage
    direction is  I = (2l+1)^{-1/2} sum_m (-1)^(l-m) e_{m,n1,-m,n2}.
    Mixed blocks (l1 != l2) contain no leakage.  Same formula as
    k3_study.apply_pure_S_projection.
    coeffs: dict keyed (l1,m1,n1,l2,m2,n2); modified in place and returned.
    """
    for l in range(1, ell_max_k + 1):
        norm = 1.0 / sqrt(2*l + 1)
        for n1 in range(-l, l+1):
            for n2 in range(-l, l+1):
                beta = 0.0 + 0.0j
                for m in range(-l, l+1):
                    key = (l, m, n1, l, -m, n2)
                    if key in coeffs:
                        beta += ((-1)**(l - m)) * coeffs[key]
                beta *= norm
                for m in range(-l, l+1):
                    key = (l, m, n1, l, -m, n2)
                    if key in coeffs:
                        coeffs[key] -= ((-1)**(l - m)) * norm * beta
    return coeffs


def est_Kinv_general(Y, g_tr, g_eval, n, K_prime, ell_maxes, s, idx_tr,
                     project=True):
    """Invariant estimator using body-orders 2 through K'.

    K'=1: constant (mean(Y)); K'=2: pair modes on SO(3); K'=3: pair modes +
    pure-S triplet modes on SO(3)^2 (P_Delta^perp applied, Definition of the
    adaptive estimator / Theorem on the pure-S pullback).
    ell_maxes: dict {k: ell_max_k} of per-body-order degree cutoffs, e.g. {2:5, 3:1}.
    Coefficients of each k>=3 block are collected, projected by P_Delta^perp,
    then reconstructed; without the projection the matched-(l,l) invariant modes
    would duplicate pair signal already captured at k=2.
    """
    N_train = len(Y)
    N_eval = len(g_eval[0])
    mean_Y = float(np.mean(Y))
    f_hat = np.full(N_eval, mean_Y)

    if K_prime < 2:
        return f_hat

    for k in range(2, K_prime + 1):
        if k not in ell_maxes:
            continue
        ell_max_k = ell_maxes[k]
        subsets = list(combinations(range(n), k))
        M_k = len(subsets)
        n_rel = k - 1
        if project and k > 3:
            raise NotImplementedError("P_Delta^perp implemented for k<=3 only")

        D_k = 3 * n_rel
        Lam_k = (N_train / M_k) ** (2.0 / (2*s + D_k))
        mu_k = 1.0 / (1.0 + Lam_k) ** s

        def enum_fs(depth):
            if depth == 0:
                return [()]
            result = []
            for l in range(1, ell_max_k + 1):
                for sub in enum_fs(depth - 1):
                    result.append((l,) + sub)
            return result
        ell_tuples = enum_fs(n_rel)

        for subset in subsets:
            s0 = subset[0]
            D_tr_caches, D_ev_caches = [], []
            for j in range(n_rel):
                sj = subset[j + 1]
                rel_tr = g_tr[s0][idx_tr].inv() * g_tr[sj][idx_tr]
                rel_ev = g_eval[s0].inv() * g_eval[sj]
                D_tr_caches.append(precompute_D_cache(rel_tr.as_euler('ZYZ'), ell_max_k))
                D_ev_caches.append(precompute_D_cache(rel_ev.as_euler('ZYZ'), ell_max_k))

            # Phase 1: raw empirical coefficients, keyed (l1,m1,n1,l2,m2,n2,...)
            coeffs = {}
            for ells in ell_tuples:
                ranges = [[(m, nn) for m in range(-l, l+1) for nn in range(-l, l+1)] for l in ells]
                for mn_combo in iterproduct(*ranges):
                    e_tr = np.ones(N_train, dtype=complex)
                    key = ()
                    for j, (l, (m, nn)) in enumerate(zip(ells, mn_combo)):
                        e_tr *= sqrt(2*l+1) * D_tr_caches[j][(l, m, nn)]
                        key += (l, m, nn)
                    coeffs[key] = np.mean(Y * e_tr.conj())

            # Phase 2: pure-S projection (k=3 only; k=2 has no leakage)
            if project and k == 3:
                coeffs = apply_pure_S_projection_k3(coeffs, ell_max_k)

            # Phase 3: shrink and reconstruct
            for key, c_hat in coeffs.items():
                ells = key[0::3]
                cas = sum(l*(l+1) for l in ells)
                wt = 1.0 / (1.0 + mu_k * (1.0 + cas)**s)
                a = c_hat * wt
                if abs(a) > 1e-15:
                    e_ev = np.ones(N_eval, dtype=complex)
                    for j in range(n_rel):
                        l, m, nn = key[3*j], key[3*j+1], key[3*j+2]
                        e_ev *= sqrt(2*l+1) * D_ev_caches[j][(l, m, nn)]
                    f_hat += (a * e_ev).real

    return f_hat


# ================================================================
# Adaptive Estimator (sample splitting)
# ================================================================
def adaptive_estimator(Y_all, g_tr, g_te, n, K_max, ell_maxes, s):
    """Adaptive estimator with refit.
    
    Selection: sample-splitting protocol
      1. Split D = D1 ∪ D2, |D1|=|D2|=N/2
      2. For each K' ∈ {1,...,K_max}: build est on D1
      3. Validate on D2: R̂(K') = mean((Y_i - f̂^(K')(X_i))²)
      4. Select K̂ = argmin R̂(K')
    
    Refit: after selecting K̂, refit on all N so that the adaptive estimator and the
    fixed-K' candidates use the same amount of data.
    
    Returns: (f_hat_test, K_selected)
    """
    N = len(Y_all)
    N1 = N // 2
    
    # Split for selection
    idx1 = np.arange(N1)
    idx2 = np.arange(N1, N)
    Y1 = Y_all[:N1]
    Y2 = Y_all[N1:N]
    g_val = [g[idx2] for g in g_tr]
    
    # Try each candidate K'
    best_K = 1
    best_val_risk = np.inf
    
    for K_prime in range(1, K_max + 1):
        f_hat_val = est_Kinv_general(Y1, g_tr, g_val, n, K_prime, 
                                      ell_maxes, s, idx1)
        val_risk = np.mean((Y2 - f_hat_val) ** 2)
        
        if val_risk < best_val_risk:
            best_val_risk = val_risk
            best_K = K_prime
    
    # REFIT: use selected K̂ with ALL N data, then predict on test
    idx_all = np.arange(N)
    f_hat_test = est_Kinv_general(Y_all, g_tr, g_te, n, best_K, 
                                   ell_maxes, s, idx_all)
    
    return f_hat_test, best_K


# ================================================================
# Experiment Loop
# ================================================================
def run(config):
    n, K_star, s = config['n'], config['K_star'], config['s']
    sigma = config['sigma']
    K_max = config['K_max']
    N_values = sorted(config['N_values'])
    N_test = config['N_test']
    M_rep = config['M_rep']
    N_max = max(N_values)
    ell_maxes = config['ell_maxes']  # e.g. {2:5, 3:1}
    
    print(f"Config: n={n}, K*={K_star}, K_max={K_max}, s={s}, σ={sigma:.3f}")
    print(f"  N_values={N_values}, M_rep={M_rep}")
    print(f"  ell_maxes={ell_maxes}")
    
    # Ground truth: f* ∈ F_{K*,inv}
    ell_max_gt = ell_maxes[K_star]
    gt = make_gt_Kinv(ell_max_gt, s, n, seed=42)
    
    # Normalize
    ng = [Rotation.random(5000, random_state=7777+i) for i in range(n)]
    v = eval_gt(gt, ng)
    norm = float(np.std(v)) / 0.1 if np.std(v) > 0 else 1.0
    print(f"  GT: std={np.std(v):.4f}, norm={norm:.2f}")
    
    # Results: oracle, adaptive, over-specified, under-specified
    estimators = ['oracle', 'adaptive', 'over', 'under']
    results = {est: {N: [] for N in N_values} for est in estimators}
    K_selected = {N: [] for N in N_values}  # track adaptive's K̂
    
    for rep in range(M_rep):
        t0 = time.time()
        rng = np.random.RandomState(config.get('seed_base', 0) + rep)
        
        g_tr = [Rotation.random(N_max, random_state=rng.randint(2**31)) 
                for _ in range(n)]
        g_te = [Rotation.random(N_test, random_state=rng.randint(2**31)) 
                for _ in range(n)]
        
        f_tr = eval_gt(gt, g_tr) / norm
        f_te = eval_gt(gt, g_te) / norm
        noise = rng.randn(N_max) * sigma
        Y_all = f_tr + noise
        
        for N in N_values:
            idx = np.arange(N)
            Y = Y_all[:N]
            
            # 1. True-order candidate K'=K*=2, full N
            fh = est_Kinv_general(Y, g_tr, g_te, n, K_star, 
                                   ell_maxes, s, idx)
            results['oracle'][N].append(float(np.mean((fh - f_te)**2)))
            
            # 2. Adaptive: sample split, select K'
            fh, K_hat = adaptive_estimator(Y, g_tr, g_te, n, K_max, 
                                            ell_maxes, s)
            results['adaptive'][N].append(float(np.mean((fh - f_te)**2)))
            K_selected[N].append(K_hat)
            
            # 3. Over-specified: K'=K_max, uses full N
            fh = est_Kinv_general(Y, g_tr, g_te, n, K_max, 
                                   ell_maxes, s, idx)
            results['over'][N].append(float(np.mean((fh - f_te)**2)))
            
            # 4. Under-specified: K'=1, uses full N (just mean)
            fh = est_Kinv_general(Y, g_tr, g_te, n, 1, 
                                   ell_maxes, s, idx)
            results['under'][N].append(float(np.mean((fh - f_te)**2)))
        
        dt = time.time() - t0
        if rep % max(1, M_rep//10) == 0 or rep == M_rep-1:
            Ns = N_values[-1]
            K_dist = K_selected[Ns][-5:] if len(K_selected[Ns]) >= 5 else K_selected[Ns]
            print(f"  Rep {rep+1}/{M_rep} ({dt:.1f}s) "
                  f"oracle={results['oracle'][Ns][-1]:.3e} "
                  f"adaptive={results['adaptive'][Ns][-1]:.3e} "
                  f"over={results['over'][Ns][-1]:.3e} "
                  f"K̂={K_dist}")
    
    return results, K_selected


# ================================================================
# Analysis
# ================================================================
def analyze(results, K_selected, config):
    N_values = sorted(config['N_values'])
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    analysis = {}
    for est in ['oracle', 'adaptive', 'over', 'under']:
        means = [float(np.mean(results[est][N])) for N in N_values]
        stds = [float(np.std(results[est][N])) for N in N_values]
        logN = [np.log10(N) for N in N_values]
        logR = [np.log10(m) for m in means if m > 0]
        npt = min(len(logN), len(logR))
        sl = linregress(logN[:npt], logR[:npt]) if npt >= 3 else None
        
        analysis[est] = {
            'N_values': N_values, 'risk_means': means, 'risk_stds': stds,
            'slope': float(sl.slope) if sl else None,
            'R2': float(sl.rvalue**2) if sl else None,
        }
        
        label_est = {'oracle': "oracle (true-order candidate, K'=2)"}.get(est, est)
        print(f"\n  {label_est}:")
        if sl:
            print(f"    slope={sl.slope:.4f}, R²={sl.rvalue**2:.4f}")
        for i, N in enumerate(N_values):
            print(f"      N={N:>6d}: {means[i]:.4e} ± {stds[i]:.4e}")
    
    # Adaptive risk relative to (i) the true-order candidate K'=2 and
    # (ii) the empirical oracle min_{K'} (mean risk of candidate K'), as in the paper
    print("\n  === ADAPTIVE vs TRUE-ORDER CANDIDATE (K'=2) AND EMPIRICAL ORACLE ===")
    ratios = []
    ratios_emp = []
    for i, N in enumerate(N_values):
        r_true = analysis['oracle']['risk_means'][i]
        r_emp = min(analysis[e]['risk_means'][i] for e in ['under', 'oracle', 'over'])
        r_adapt = analysis['adaptive']['risk_means'][i]
        ratio = r_adapt / r_true if r_true > 0 else float('inf')
        ratio_emp = r_adapt / r_emp if r_emp > 0 else float('inf')
        ratios.append(ratio); ratios_emp.append(ratio_emp)
        print(f"    N={N:>6d}: adaptive/true-order(K'=2) = {ratio:.2f}×   adaptive/empirical-oracle = {ratio_emp:.2f}×")
    
    # Ratio trend summary
    print(f"\n    Ratio trend (relative to K'=2): {ratios[0]:.2f}× (N={N_values[0]}) → "
          f"{ratios[-1]:.2f}× (N={N_values[-1]})")
    if len(ratios) >= 3:
        last3 = ratios[-3:]
        print(f"    Last 3 N values: mean={np.mean(last3):.2f}×, "
              f"stable={'yes' if max(last3)-min(last3)<0.3 else 'no'}")
    analysis['adaptive_oracle_ratios'] = ratios            # relative to the true-order candidate K'=2
    analysis['adaptive_empirical_oracle_ratios'] = ratios_emp
    
    # K̂ distribution
    print("\n  === K̂ SELECTION DISTRIBUTION ===")
    K_max = config['K_max']
    for N in N_values:
        counts = {k: 0 for k in range(1, K_max+1)}
        for k in K_selected[N]:
            counts[k] += 1
        total = len(K_selected[N])
        dist_str = " ".join(f"K={k}:{counts[k]/total*100:.0f}%" for k in range(1, K_max+1))
        print(f"    N={N:>6d}: {dist_str}")
    
    analysis['K_selected'] = {str(N): K_selected[N] for N in N_values}
    return analysis


# ================================================================
# Save & Plot
# ================================================================
def save_and_plot(results, analysis, config, path):
    data = {
        'config': config,
        'analysis': analysis,
        'raw': {est: {str(N): v for N, v in d.items()} 
                for est, d in results.items()},
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n  Saved: {path}")
    
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        C = {'oracle': '#009E73', 'adaptive': '#0072B2', 
             'over': '#E69F00', 'under': '#D55E00'}
        L = {'oracle': f'True-order candidate ($K\'={config["K_star"]}$)',
             'adaptive': f'Adaptive ($K_{{\\max}}={config["K_max"]}$)',
             'over': f'Over-specified ($K\'={config["K_max"]}$)',
             'under': r"Under-specified ($K'=1$)"}
        LS = {'oracle': '-', 'adaptive': '--', 'over': '-.', 'under': ':'}
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        
        # Fig 4a: Risk curves
        for est in ['oracle', 'adaptive', 'over', 'under']:
            a = analysis[est]
            ax1.errorbar(a['N_values'], a['risk_means'], yerr=a['risk_stds'],
                        fmt='o'+LS[est], color=C[est], label=L[est],
                        markersize=4, capsize=2, linewidth=1.5)
        ax1.set_xscale('log'); ax1.set_yscale('log')
        ax1.set_xlabel(r'Sample size $N$')
        ax1.set_ylabel(r'$L^2$ risk')
        ax1.legend(fontsize=7)
        ax1.set_title(r'(a) Adaptive vs fixed candidates ($K^*=2$)')
        ax1.grid(True, alpha=0.3)
        
        # Fig 4b: K̂ selection frequency vs N
        N_vals = analysis['oracle']['N_values']
        K_max = config['K_max']
        K_selected_data = analysis['K_selected']
        
        x = np.arange(len(N_vals))
        bottom = np.zeros(len(N_vals))
        colors_k = ['#D55E00', '#009E73', '#56B4E9', '#CC79A7', '#E69F00']
        
        for k in range(1, K_max+1):
            freqs = []
            for N in N_vals:
                ks = K_selected_data[str(N)]
                freqs.append(sum(1 for kk in ks if kk == k) / len(ks))
            ax2.bar(x, freqs, bottom=bottom, width=0.6,
                   label=f"$\\hat{{K}}={k}$", color=colors_k[k-1])
            bottom += freqs
        
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{N}' for N in N_vals], fontsize=8)
        ax2.set_xlabel(r'Sample size $N$')
        ax2.set_ylabel('Selection frequency')
        ax2.legend(fontsize=7, loc='upper left')
        ax2.set_title(r'(b) Adaptive $\hat{K}$ distribution')
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3, axis='y')
        
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
    args = ap.parse_args()
    quick = args.quick
    
    print("="*70)
    print("EXP 4: adaptive body-order selection")
    print("="*70)
    
    beta = np.array([0.5, 1.0, 2.0])
    assert np.allclose(wigner_d_small(1,1,1,beta), np.cos(beta/2)**2)
    print("  Wigner-D ✓")
    
    if quick:
        cfg = {
            'n': 6, 'K_star': 2, 'K_max': 3, 's': 2.0, 'sigma': sqrt(0.1),
            'N_values': [500, 1000, 2000, 5000, 10000],
            'N_test': 2000, 'M_rep': 10,
            'seed_base': 8000,
            'ell_maxes': {2: 5, 3: 1},
        }
        path = 'exp4_quick.json'
    else:
        cfg = {
            'n': 6, 'K_star': 2, 'K_max': 3, 's': 2.0, 'sigma': sqrt(0.1),
            'N_values': [500, 1000, 2000, 5000, 10000, 20000, 50000],
            'N_test': 5000, 'M_rep': 30,
            'seed_base': 8000,
            'ell_maxes': {2: 5, 3: 1},
        }
        path = 'exp4_data.json'
    
    t0 = time.time()
    res, K_sel = run(cfg)
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
    
    analysis = analyze(res, K_sel, cfg)
    save_and_plot(res, analysis, cfg, path)
    print("Done!")
