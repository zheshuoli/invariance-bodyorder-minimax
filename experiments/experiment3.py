"""
Experiment 3: Four-way risk comparison on shared targets
========================================================
All four estimators use the same target f* in F_{K,inv} and the same noisy
observations within each trial. The experiment reports mean risks and risk
ratios under the specified finite dictionaries (Figure 3c,d).

Config: n=6, K=2, s=2; effective dimensions D_0=18, D_inv=15, D_K=6, D_Kinv=3.
3 ground-truth seeds × 30 reps = 90 trials.

Usage:
  python ../experiments/experiment3.py --quick    # 3 seeds x 3 reps
  python ../experiments/experiment3.py            # 3 seeds x 30 reps (Figure 3c,d)
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import linregress
from math import factorial, sqrt, comb
from itertools import combinations, product as iterproduct
import json, time, sys

# ================================================================
# Wigner-D (identical to experiment 1)
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
# Ground Truth: f* ∈ F_{K,inv} (pair-sum of invariant functions)
# ================================================================
def make_gt_Kinv(ell_max, s, n, seed=42):
    """f* = Σ_{i<j} φ_{ij}(gᵢ⁻¹gⱼ), each φ with Sobolev-decaying coefficients."""
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
    # Normalize ‖f*‖_{H^s} = 1
    Hs = sum((1+l*(l+1))**s*(2*l+1)*abs(c)**2
             for p in pairs for (l,_,_),c in coeffs[p].items())
    norm = 1.0/sqrt(Hs) if Hs>0 else 1.0
    for p in pairs:
        for k in coeffs[p]: coeffs[p][k] *= norm
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

# ================================================================
# Mode enumeration helper
# ================================================================
def enumerate_modes_so3(ell_max):
    modes, casimirs = [], []
    for ell in range(1, ell_max+1):
        for m in range(-ell,ell+1):
            for nn in range(-ell,ell+1):
                modes.append((ell,m,nn))
                casimirs.append(float(ell*(ell+1)))
    return modes, np.array(casimirs)

# ================================================================
# 4 Estimators (Tikhonov, theoretical μ)
# All estimate the SAME f*, using different structural assumptions.
# ================================================================

def est_Kinv(Y, g_tr, g_te, pairs, ell_max, N, D, M, s, idx):
    """Uses invariance + body-order K: modes on SO(3) per pair."""
    Lam = (N/M)**(2.0/(2*s+D)); mu = 1.0/(1.0+Lam)**s
    # mu = (1+Lam)^(-s) with no 1/sqrt(M) factor (shared-target comparison; see README)
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
    """Uses body-order K only: 1-body + 2-body on absolute coords."""
    Lam = (N/M)**(2.0/(2*s+D)); mu = 1.0/(1.0+Lam)**s
    mean_Y = float(np.mean(Y))
    max_ell = max(ell_1b, ell_2b)
    D_tr_c, D_te_c = {}, {}
    for f in range(n):
        D_tr_c[f] = precompute_D_cache(g_tr[f][idx].as_euler('ZYZ'), max_ell)
        D_te_c[f] = precompute_D_cache(g_te[f].as_euler('ZYZ'), max_ell)
    f_hat = np.full(len(g_te[0]), mean_Y)
    # 1-body
    for f in range(n):
        for l in range(1,ell_1b+1):
            wt = 1.0/(1.0+mu*(1.0+l*(l+1))**s)
            for m in range(-l,l+1):
                for nn in range(-l,l+1):
                    c = np.mean(Y*sqrt(2*l+1)*D_tr_c[f][(l,m,nn)].conj())
                    a = c*wt
                    if abs(a)>1e-15:
                        f_hat += (a*sqrt(2*l+1)*D_te_c[f][(l,m,nn)]).real
    # 2-body full-support
    for (fi,fj) in pairs:
        for l1 in range(1,ell_2b+1):
            for l2 in range(1,ell_2b+1):
                cas = l1*(l1+1)+l2*(l2+1)
                wt = 1.0/(1.0+mu*(1.0+cas)**s)
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
    """Uses invariance only: product modes on SO(3)^{n-1} relative coords.
    If fixed_lam_cut is given, it replaces the N-dependent cutoff, so the mode set does not change with N.
    """
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
        for l in range(0, ell_mp+1):
            lam=l*(l+1)
            if lam>lr: break
            for sub in enum(d-1, lr-lam): r.append((l,)+sub)
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
    """Uses no structure: product modes on SO(3)^n absolute coords.
    If fixed_lam_cut is given, it replaces the N-dependent cutoff, so the mode set does not change with N.
    """
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
        for l in range(0, ell_mp+1):
            lam=l*(l+1)
            if lam>lr: break
            for sub in enum(d-1, lr-lam): r.append((l,)+sub)
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
# Experiment Loop
# ================================================================
def run(config):
    n, K, s = config['n'], config['K'], config['s']
    sigma = config['sigma']; sigma2 = sigma**2
    N_values = sorted(config['N_values'])
    N_test = config['N_test']
    M_rep = config['M_rep']
    N_max = max(N_values)
    gt_seeds = config['gt_seeds']
    n_gt = len(gt_seeds)
    pairs = list(combinations(range(n), 2)); Mp = len(pairs)

    EL_KINV = config.get('ell_kinv', 5)
    EL_K1 = config.get('ell_k1', 3); EL_K2 = config.get('ell_k2', 2)
    EL_INV = config.get('ell_inv', 2); EL_F0 = config.get('ell_f0', 1)

    info = {
        'F_Kinv': {'D': 3*(K-1), 'M': Mp},
        'F_K':    {'D': 3*K,     'M': Mp},
        'F_inv':  {'D': 3*(n-1), 'M': 1},
        'F_0':    {'D': 3*n,     'M': 1},
    }
    for nm in info:
        info[nm]['slope'] = -2*s / (2*s + info[nm]['D'])

    print(f"Config: n={n} K={K} s={s} σ={sigma:.3f}")
    print(f"  N_values={N_values}, N_test={N_test}")
    print(f"  {n_gt} GT seeds × {M_rep} reps = {n_gt*M_rep} total trials")
    print(f"  ℓ_max: Kinv={EL_KINV}, K_1b={EL_K1}, K_2b={EL_K2}, inv={EL_INV}, F0={EL_F0}")
    for nm, d in info.items():
        print(f"  {nm}: D={d['D']}, slope={d['slope']:.4f}")

    # Fixed Casimir cutoff for est_inv and est_F0
    # max_body=3 → Casimir cutoff=6 (three ℓ=1 factors)
    # Independent of N_max → quick and full use identical mode sets
    fixed_lam_cut_inv = 6.0
    fixed_lam_cut_f0 = 6.0
    print(f"  Fixed Lam_cut: inv={fixed_lam_cut_inv:.0f}, F0={fixed_lam_cut_f0:.0f} (max 3-body)")

    results = {nm: {N: [] for N in N_values} for nm in info}

    for gt_idx, gt_seed in enumerate(gt_seeds):
        # Construct shared ground truth
        gt_coeffs = make_gt_Kinv(EL_KINV, s, n, seed=gt_seed)

        # Normalize
        ng = [Rotation.random(5000, random_state=7777+i) for i in range(n)]
        v = eval_gt_Kinv(gt_coeffs, ng)
        norm_factor = float(np.std(v)) / 0.1 if np.std(v) > 0 else 1.0
        print(f"\n  GT seed={gt_seed}: std={np.std(v):.4f}, norm={norm_factor:.2f}")

        for rep in range(M_rep):
            t0 = time.time()
            rng = np.random.RandomState(config.get('seed_base', 0) + gt_idx*1000 + rep)

            g_tr = [Rotation.random(N_max, random_state=rng.randint(2**31)) for _ in range(n)]
            g_te = [Rotation.random(N_test, random_state=rng.randint(2**31)) for _ in range(n)]

            f_tr = eval_gt_Kinv(gt_coeffs, g_tr) / norm_factor
            f_te = eval_gt_Kinv(gt_coeffs, g_te) / norm_factor
            noise = rng.randn(N_max) * sigma
            Y_all = f_tr + noise

            for N in N_values:
                idx = np.arange(N)
                Y = Y_all[:N]

                fh = est_Kinv(Y,g_tr,g_te,pairs,EL_KINV,N,info['F_Kinv']['D'],Mp,s,idx)
                results['F_Kinv'][N].append(float(np.mean((fh-f_te)**2)))

                fh = est_K(Y,g_tr,g_te,pairs,EL_K1,EL_K2,N,info['F_K']['D'],Mp,s,n,idx)
                results['F_K'][N].append(float(np.mean((fh-f_te)**2)))

                fh = est_inv(Y,g_tr,g_te,EL_INV,N,info['F_inv']['D'],1,s,n,idx,
                             fixed_lam_cut=fixed_lam_cut_inv)
                results['F_inv'][N].append(float(np.mean((fh-f_te)**2)))

                fh = est_F0(Y,g_tr,g_te,EL_F0,N,info['F_0']['D'],1,s,n,idx,
                            fixed_lam_cut=fixed_lam_cut_f0)
                results['F_0'][N].append(float(np.mean((fh-f_te)**2)))

            dt = time.time()-t0
            trial_num = gt_idx*M_rep + rep + 1
            total_trials = n_gt * M_rep
            if trial_num % max(1, total_trials//10) == 0 or trial_num == total_trials:
                Ns = N_values[-1]
                print(f"    Trial {trial_num}/{total_trials} ({dt:.1f}s) " +
                      " ".join(f"{k}={results[k][Ns][-1]:.3e}" for k in info))

    return results, info


# ================================================================
# Analysis
# ================================================================
def analyze(results, info, config):
    N_values = sorted(config['N_values'])
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
            'slope_full': float(sl_f.slope) if sl_f else None,
            'R2_full': float(sl_f.rvalue**2) if sl_f else None,
            'N_values': N_values,
            'risk_means': means,
            'risk_stds': stds,
        }

        print(f"\n  {nm} (D_theory={info[nm]['D']})")
        print(f"    slope: {sl_f.slope:.4f} (R²={sl_f.rvalue**2:.4f})" if sl_f else "")
        for i, N in enumerate(N_values):
            print(f"      N={N:>6d}: {means[i]:.4e} ± {stds[i]:.4e}")

    # Monotonicity check
    print("\n  === MONOTONICITY CHECK ===")
    for nm in ['F_Kinv', 'F_K', 'F_inv', 'F_0']:
        means = analysis[nm]['risk_means']
        mono = all(means[i] > means[i+1] for i in range(len(means)-1))
        print(f"    {nm}: {'✓ monotone' if mono else '✗ NON-MONOTONE!'}")

    # Risk ratios relative to F_Kinv
    print(f"\n  === RISK RATIOS vs F_Kinv ===")
    for i, N in enumerate(N_values):
        rk = analysis['F_Kinv']['risk_means'][i]
        line = f"    N={N:>6d}:"
        for nm in ['F_K', 'F_inv', 'F_0']:
            r = analysis[nm]['risk_means'][i]
            line += f"  {nm}={r/rk:.2f}×"
        print(line)

    # Ordering check
    print("\n  === ORDERING CHECK ===")
    print("  (observed mean-risk ordering; F_inv vs F_K depends on the target)")
    for i, N in enumerate(N_values):
        r = {nm: analysis[nm]['risk_means'][i] for nm in info}
        ordered = sorted(r.items(), key=lambda x: x[1])
        best_is_kinv = ordered[0][0] == 'F_Kinv'
        worst_is_f0 = ordered[-1][0] == 'F_0'
        mark = '✓' if (best_is_kinv and worst_is_f0) else '✗'
        print(f"    N={N:>6d}: {' < '.join(f'{k}' for k,_ in ordered)} {mark}")

    return analysis


# ================================================================
# Save & Plot
# ================================================================
def save_and_plot(results, info, analysis, config, path):
    data = {
        'config': config,
        'theory': {nm: {'D': i['D'], 'slope': i['slope']} for nm, i in info.items()},
        'analysis': analysis,
        'raw': {nm: {str(N): v for N, v in d.items()} for nm, d in results.items()},
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n  Saved: {path}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        C = {'F_0':'#E69F00', 'F_inv':'#56B4E9', 'F_K':'#009E73', 'F_Kinv':'#CC79A7'}
        L = {'F_0':r'$\mathcal{F}_0$ (no structure)',
             'F_inv':r'$\mathcal{F}_{\rm inv}$ (inv. only)',
             'F_K':r'$\mathcal{F}_K$ (body-order only)',
             'F_Kinv':r'$\mathcal{F}_{K,{\rm inv}}$ (both)'}

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        # --- Fig 3a: Risk curves ---
        ax = axes[0]
        for nm in ['F_0', 'F_inv', 'F_K', 'F_Kinv']:
            a = analysis[nm]
            ax.errorbar(a['N_values'], a['risk_means'], yerr=a['risk_stds'],
                       fmt='o-', color=C[nm], label=L[nm],
                       markersize=4, capsize=2, linewidth=1.5)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r'Sample size $N$'); ax.set_ylabel(r'$L^2$ risk')
        ax.legend(fontsize=7, loc='upper right')
        ax.set_title(r'(a) Shared $f^* \in \mathcal{F}_{K,\mathrm{inv}}$: risk vs $N$')
        ax.grid(True, alpha=0.3)

        # --- Fig 3b: Risk ratios at multiple N values ---
        ax = axes[1]
        N_vals = analysis['F_Kinv']['N_values']
        # Use last 3-4 N values for bars
        N_show = N_vals[-min(4, len(N_vals)):]
        x = np.arange(len(N_show))
        width = 0.18

        nms_ratio = ['F_K', 'F_inv', 'F_0']
        for j, nm in enumerate(nms_ratio):
            ratios = []
            for N in N_show:
                idx_n = N_vals.index(N)
                r_nm = analysis[nm]['risk_means'][idx_n]
                r_best = analysis['F_Kinv']['risk_means'][idx_n]
                ratios.append(r_nm / r_best if r_best > 0 else 0)
            ax.bar(x + (j-1)*width, ratios, width, label=L[nm], color=C[nm])

        ax.set_xticks(x)
        ax.set_xticklabels([f'N={N}' for N in N_show], fontsize=8)
        ax.set_ylabel(r'Risk ratio vs $\mathcal{F}_{K,\mathrm{inv}}$')
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.legend(fontsize=7)
        ax.set_title('(b) Relative inefficiency without structural priors')
        ax.grid(True, alpha=0.3, axis='y')

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
    print("EXP 3: Four-way risk comparison on shared targets")
    print("="*70)

    # Verify Wigner-D
    beta = np.array([0.5, 1.0, 2.0])
    assert np.allclose(wigner_d_small(1,1,1,beta), np.cos(beta/2)**2)
    print("  Wigner-D ✓")

    if quick:
        cfg = {
            'n': 6, 'K': 2, 's': 2.0, 'sigma': sqrt(0.1),
            'N_values': [500, 1000, 2000, 5000, 10000, 20000],
            'N_test': 2000, 'M_rep': 3,
            'gt_seeds': [42, 43, 44],
            'seed_base': 5000,
            'ell_kinv': 5, 'ell_k1': 1, 'ell_k2': 1, 'ell_inv': 2, 'ell_f0': 1,
        }
        path = 'exp3_quick.json'
    else:
        cfg = {
            'n': 6, 'K': 2, 's': 2.0, 'sigma': sqrt(0.1),
            'N_values': [500, 1000, 2000, 5000, 10000, 20000, 50000],
            'N_test': 5000, 'M_rep': 30,
            'gt_seeds': [42, 43, 44],
            'seed_base': 5000,
            'ell_kinv': 5, 'ell_k1': 1, 'ell_k2': 1, 'ell_inv': 2, 'ell_f0': 1,
        }
        path = 'exp3_data.json'

    t0 = time.time()
    res, info = run(cfg)
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")

    analysis = analyze(res, info, cfg)
    save_and_plot(res, info, analysis, cfg, path)
    print("Done!")
