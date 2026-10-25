"""
Architectural control for the F_{K,inv}-like learner (Appendix P)
=================================================================
A minimal re-implementation of the both-prior learner with a different
architectural formulation, to check that its advantage in the neural bridge
does not depend on one specific implementation:
  * Input: absolute rotation matrices (batch, n, 3, 3)
  * The forward pass computes R_ij = g_i^T g_j internally
  * Pair-swap symmetrization: 0.5 * (phi(R_ij) + phi(R_ji))
  * Sum-pool over pairs (permutation invariant)
Parameters matched to the bridge's F_{K,inv}-like learner (~41K).
Training protocol and data splits are identical to neural_bridge.py (whose
feature, target and model definitions are imported here).

Usage (from results/, after neural_bridge.py has produced nn_bridge_combined.json):
  python ../experiments/neural_bridge_control.py --quick
  python ../experiments/neural_bridge_control.py        # n=6 + n=10, 10 seeds (Appendix P)
  python ../experiments/neural_bridge_control.py --fresh # ignore an existing checkpoint
Outputs: nn_control_n6.json, nn_control_n10.json
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial.transform import Rotation
from scipy.stats import linregress
from math import sqrt
from itertools import combinations
import json, time, sys, os

# Reuse feature/target/model definitions from neural_bridge.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neural_bridge import (
    eval_maier_saupe,
    features_absolute_rotations, features_FKinv_pairs,
    PairInteractionNet, SymmetryRespectingPairNet,
    train_model, rng_state_to_json, rng_state_from_json, CHECKPOINT_FORMAT,
)


# ================================================================
# Experiment runner (train/val/test split, checkpointing)
# ================================================================
def run(config, checkpoint_path=None, resume=True):
    n = config['n']
    sigma = config['sigma']
    N_values = sorted(config['N_values'])
    M_rep = config['M_rep']
    N_max = max(N_values)
    N_val = config.get('N_val', 2000)
    N_test = config['N_test']

    eps2 = config.get('eps2', 1.0)
    eps4 = config.get('eps4', 0.3)

    print(f"\nConfig: n={n}, sigma={sigma:.3f}, N_max={N_max}, N_val={N_val}, N_test={N_test}, reps={M_rep}")
    print(f"N values: {N_values}")
    print(f"Epochs: {config.get('epochs', 500)}, patience: {config.get('patience', 50)}")
    print("Early stopping on the validation loss; final metric is the test MSE at the validation-selected epoch.")

    # Parameter count verification
    p_current = sum(p.numel() for p in PairInteractionNet(pair_in_dim=9, hidden=140).parameters())
    p_toy = sum(p.numel() for p in SymmetryRespectingPairNet(hidden=140).parameters())
    diff_pct = (p_toy - p_current) / p_current * 100
    print(f"\nParameter counts:")
    print(f"  Bridge reference F_Kinv-like : {p_current:,}")
    print(f"  Architectural control        : {p_toy:,}  ({diff_pct:+.1f}%)")

    # Normalize target
    print("\nNormalizing target...")
    ng = [Rotation.random(5000, random_state=999 + i) for i in range(n)]
    vals = eval_maier_saupe(ng, eps2=eps2, eps4=eps4)
    target_std = 0.1
    norm_factor = float(np.std(vals)) / target_std
    print(f"  target std={np.std(vals):.4f}, norm_factor={norm_factor:.3f}")

    results = {'toy_equiv': {N: [] for N in N_values}}
    val_losses = {'toy_equiv': {N: [] for N in N_values}}
    completed = set()
    rng_after = None   # RNG state saved after the last completed (rep, N)
    config_json = json.loads(json.dumps(config))

    if checkpoint_path and resume and os.path.exists(checkpoint_path):
        print(f"\n>>> Loading checkpoint from {checkpoint_path}")
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        if ckpt.get('format') == CHECKPOINT_FORMAT and ckpt.get('experiment_config') == config_json:
            for N_str, v_list in ckpt.get('results', {}).get('toy_equiv', {}).items():
                N_int = int(N_str)
                if N_int in N_values:
                    results['toy_equiv'][N_int] = list(v_list)
            for N_str, v_list in ckpt.get('val_losses', {}).get('toy_equiv', {}).items():
                N_int = int(N_str)
                if N_int in N_values:
                    val_losses['toy_equiv'][N_int] = list(v_list)
            for item in ckpt.get('completed', []):
                completed.add(tuple(item))
            rng_after = ckpt.get('rng_after')
            print(f">>> Resuming: {len(completed)} (rep, N) combinations already done")
        else:
            print(">>> Checkpoint configuration differs from the current one; starting over (the file will be overwritten)")

    def save_checkpoint(rep, N):
        if checkpoint_path is None:
            return
        ckpt = {
            'format': CHECKPOINT_FORMAT,
            'experiment_config': config_json,
            'results': {'toy_equiv': {str(N_): v for N_, v in results['toy_equiv'].items()}},
            'val_losses': {'toy_equiv': {str(N_): v for N_, v in val_losses['toy_equiv'].items()}},
            'completed': [list(x) for x in completed],
            'rng_after': {'rep': rep, 'N': N, 'state': rng_state_to_json()},
        }
        tmp = checkpoint_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(ckpt, f)
        os.replace(tmp, checkpoint_path)   # atomic: an interrupted write never corrupts the previous checkpoint

    t_start_total = time.time()
    for rep in range(M_rep):
        t_rep = time.time()
        rng = np.random.RandomState(config.get('seed_base', 0) + rep)
        torch.manual_seed(config.get('seed_base', 0) + rep)
        np.random.seed(config.get('seed_base', 0) + rep)

        # Generate three independent data splits
        g_tr = [Rotation.random(N_max, random_state=rng.randint(2**31)) for _ in range(n)]
        g_val = [Rotation.random(N_val, random_state=rng.randint(2**31)) for _ in range(n)]
        g_te = [Rotation.random(N_test, random_state=rng.randint(2**31)) for _ in range(n)]

        f_tr = eval_maier_saupe(g_tr, eps2=eps2, eps4=eps4) / norm_factor
        f_val = eval_maier_saupe(g_val, eps2=eps2, eps4=eps4) / norm_factor
        f_te = eval_maier_saupe(g_te, eps2=eps2, eps4=eps4) / norm_factor
        noise = rng.randn(N_max) * sigma
        noise_val = rng.randn(N_val) * sigma

        Y_val = (f_val + noise_val).astype(np.float32)
        Y_te = f_te.astype(np.float32)

        # Absolute rotation features (the control model computes R_ij internally)
        X_tr_abs = features_absolute_rotations(g_tr).astype(np.float32)
        X_val_abs = features_absolute_rotations(g_val).astype(np.float32)
        X_te_abs = features_absolute_rotations(g_te).astype(np.float32)

        for N in N_values:
            if (rep, N) in completed:
                continue
            if rng_after is not None:
                # First (rep, N) after the checkpointed ones: continue the RNG stream
                if rng_after['rep'] == rep:
                    rng_state_from_json(rng_after['state'])
                    print(f"  Restored RNG state saved after rep {rep+1}, N={rng_after['N']}")
                rng_after = None

            idx = np.arange(N)
            Y_tr = (f_tr[:N] + noise[:N]).astype(np.float32)

            model = SymmetryRespectingPairNet(hidden=140)
            test_mse, val_mse = train_model(
                model, X_tr_abs[idx], Y_tr, X_val_abs, Y_val, X_te_abs, Y_te,
                epochs=config.get('epochs', 500),
                batch_size=min(256, N),
                patience=config.get('patience', 50),
            )
            results['toy_equiv'][N].append(test_mse)
            val_losses['toy_equiv'][N].append(val_mse)
            completed.add((rep, N))
            save_checkpoint(rep, N)

            print(f"  Rep {rep+1}/{M_rep}, N={N:>6d}: control test MSE = {test_mse:.4e}  (val MSE = {val_mse:.4e})")

        dt = time.time() - t_rep
        print(f"  Rep {rep+1}/{M_rep} done in {dt/60:.1f} min")

    total_dt = time.time() - t_start_total
    print(f"\nTotal time: {total_dt/60:.1f} min")
    return results, val_losses


# ================================================================
# Analysis and comparison with the F_Kinv-like learner of the bridge
# ================================================================
def analyze_and_compare(toy_results, config, path_out, bridge_path=None):
    """Analyze control results and compare with the F_Kinv-like learner of the bridge."""
    N_values = sorted(config['N_values'])

    means = [float(np.mean(toy_results['toy_equiv'][N])) for N in N_values]
    stds = [float(np.std(toy_results['toy_equiv'][N])) for N in N_values]
    logN = np.array([np.log10(N) for N in N_values])
    logR = np.array([np.log10(max(m, 1e-10)) for m in means])
    sl = linregress(logN, logR)

    print("\n" + "="*70 + "\nARCHITECTURAL CONTROL RESULTS\n" + "="*70)
    print(f"Slope: {sl.slope:.4f} (R^2={sl.rvalue**2:.4f})")
    print(f"{'N':>6s} | {'toy_equiv TEST MSE':>20s}")
    for i, N in enumerate(N_values):
        print(f"{N:>6d} | {means[i]:.4e} +/- {stds[i]:.1e}")

    analysis_toy = {
        'label': 'Control F_{K,inv}-like',
        'N': N_values, 'means': means, 'stds': stds,
        'slope': float(sl.slope), 'R2': float(sl.rvalue**2),
    }

    comparison = None
    if bridge_path and os.path.exists(bridge_path):
        with open(bridge_path) as f:
            bridge = json.load(f)
        n_str = str(config['n'])
        if n_str in bridge.get('results_by_n', {}):
            fkinv = bridge['results_by_n'][n_str]['analysis']['inv_pair']
            print("\nComparison against the F_{K,inv}-like learner of the bridge:")
            print(f"{'N':>6s} | {'control':>17s}  {'bridge reference':>17s}  {'ratio':>8s}")
            comparison = {'N': N_values, 'toy': means, 'current_Fkinv': fkinv['means'], 'ratio': []}
            for i, N in enumerate(N_values):
                if N in fkinv['N']:
                    j = fkinv['N'].index(N)
                    ratio = means[i] / fkinv['means'][j]
                    comparison['ratio'].append(ratio)
                    print(f"{N:>6d} | {means[i]:.4e}      {fkinv['means'][j]:.4e}   {ratio:>7.2f}x")
            print(f"\nControl slope: {sl.slope:.4f}  vs  bridge reference slope: {fkinv['slope']:.4f}")

    out_data = {
        'config': config, 'analysis': analysis_toy, 'comparison': comparison,
        'raw': {'toy_equiv': {str(N): v for N, v in toy_results['toy_equiv'].items()}},
    }
    with open(path_out, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved: {path_out}")
    return analysis_toy


def plot_combined(all_toy_results, all_configs, bridge_path, path_out):
    """Plot control results alongside the F_Kinv-like learner of the bridge."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available, skipping plot")
        return

    with open(bridge_path) as f:
        bridge = json.load(f)

    n_list = list(all_configs.keys())
    fig, axes = plt.subplots(1, len(n_list), figsize=(5.5 * len(n_list), 4.5), sharey=True)
    if len(n_list) == 1:
        axes = [axes]

    for panel_idx, n_val in enumerate(n_list):
        ax = axes[panel_idx]
        config = all_configs[n_val]
        N_values = sorted(config['N_values'])
        toy_results = all_toy_results[n_val]

        toy_means = [float(np.mean(toy_results['toy_equiv'][N])) for N in N_values]
        toy_stds = [float(np.std(toy_results['toy_equiv'][N])) for N in N_values]
        ax.errorbar(N_values, toy_means, yerr=toy_stds, fmt='D-',
                    color='#D55E00', label=r'Architectural control $\mathcal{F}_{K,\rm inv}$-like',
                    markersize=6, capsize=3, linewidth=1.8, zorder=3)

        fkinv = bridge['results_by_n'][str(n_val)]['analysis']['inv_pair']
        ax.errorbar(fkinv['N'], fkinv['means'], yerr=fkinv['stds'], fmt='o-',
                    color='#CC79A7', label=r'Bridge reference $\mathcal{F}_{K,\rm inv}$-like',
                    markersize=5, capsize=2, linewidth=1.5, alpha=0.9)

        for key, color, label in [
            ('no_struct', '#E69F00', r'$\mathcal{F}_0$-like'),
            ('inv_only',  '#56B4E9', r'$\mathcal{F}_{\rm inv}$-like'),
        ]:
            d = bridge['results_by_n'][str(n_val)]['analysis'][key]
            ax.errorbar(d['N'], d['means'], yerr=d['stds'], fmt='o-',
                        color=color, label=label, markersize=4, capsize=1.5,
                        linewidth=1.2, alpha=0.4)

        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r'$N$')
        if panel_idx == 0:
            ax.set_ylabel(r'Test MSE')
            ax.legend(fontsize=8, loc='lower left')
        ax.set_title(f'$n{{=}}{n_val}$')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Architectural control for the F_{K,inv}-like learner', y=1.02)
    plt.tight_layout()
    plt.savefig(path_out, dpi=300, bbox_inches='tight')
    print(f"Figure: {path_out}")


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
    print("ARCHITECTURAL CONTROL (train/val/test split)")
    print("="*70)
    print("Purpose: check whether the both-prior learner's advantage in the bridge")
    print("         is robust to architecture-level reformulation.")

    bridge_path = 'nn_bridge_combined.json'
    if not os.path.exists(bridge_path):
        print(f"\nNOTE: {bridge_path} not found; running the control without the bridge comparison.")
        bridge_path = None

    if quick:
        cfg = {'n': 6, 'sigma': sqrt(0.1),
               'N_values': [500, 2000, 10000],
               'N_test': 5000, 'N_val': 2000, 'M_rep': 1, 'seed_base': 0,
               'epochs': 150, 'patience': 15, 'eps2': 1.0, 'eps4': 0.3}
        t0 = time.time()
        results, val_losses = run(cfg, checkpoint_path='nn_control_quick_checkpoint.json', resume=not args.fresh)
        analyze_and_compare(results, cfg, 'nn_control_quick.json', bridge_path)
        # Save val losses separately
        with open('nn_control_quick_val.json', 'w') as f:
            json.dump({'val_losses': {k: {str(N): v for N, v in d.items()}
                                       for k, d in val_losses.items()}}, f, indent=2)
        print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
        print("Done!")

    else:
        # Default: both n=6 and n=10, matching the bridge setup
        print("\n>>> Running architectural control: n=6 + n=10 (10 seeds each)")
        print(">>> Checkpointed after every (rep, N); an interrupted run can be resumed by rerunning the same command.")

        configs = {
            6:  {'n': 6,  'sigma': sqrt(0.1),
                 'N_values': [500, 1000, 2000, 5000, 10000, 20000],
                 'N_test': 5000, 'N_val': 2000, 'M_rep': 10, 'seed_base': 100,
                 'epochs': 500, 'patience': 50, 'eps2': 1.0, 'eps4': 0.3},
            10: {'n': 10, 'sigma': sqrt(0.1),
                 'N_values': [500, 1000, 2000, 5000, 10000, 20000],
                 'N_test': 5000, 'N_val': 2000, 'M_rep': 10, 'seed_base': 100,
                 'epochs': 500, 'patience': 50, 'eps2': 1.0, 'eps4': 0.3},
        }

        all_toy_results = {}
        all_val_losses = {}
        t_start = time.time()
        for n_val, cfg in configs.items():
            print(f"\n{'#'*70}\n# Architectural control at n={n_val}\n{'#'*70}")
            t_n = time.time()
            checkpoint_file = f'nn_control_n{n_val}_checkpoint.json'
            results, val_losses = run(cfg, checkpoint_path=checkpoint_file, resume=not args.fresh)
            analyze_and_compare(results, cfg, f'nn_control_n{n_val}.json', bridge_path)
            all_toy_results[n_val] = results
            all_val_losses[n_val] = val_losses
            print(f"n={n_val} completed in {(time.time()-t_n)/60:.1f} min")

        # Combined figure (requires the bridge data)
        if bridge_path:
            plot_combined(all_toy_results, configs, bridge_path, 'nn_control_fig.pdf')

        # Validation losses (diagnostic)
        val_out = {str(n_val): {k: {str(N): v for N, v in d.items()}
                                 for k, d in vl.items()}
                   for n_val, vl in all_val_losses.items()}
        with open('nn_control_val_diagnostic.json', 'w') as f:
            json.dump(val_out, f, indent=2)
        print(f"Validation losses saved to nn_control_val_diagnostic.json")

        print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")
        print("Done!")
