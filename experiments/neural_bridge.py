"""
Neural bridge on the Maier-Saupe task (Appendix O)
==================================================
Four parameter-matched MLP learners encode none, one, or both structural priors
and are trained end-to-end on the Maier-Saupe target of Experiment 5:
  F_0-like    : MLP on the concatenated absolute rotations (9n inputs)
  F_inv-like  : MLP on the relative rotations R_{1j}, j=2..n (9(n-1) inputs)
  F_K-like    : shared pair network on (R_i, R_j) (18 inputs), summed over pairs
  F_Kinv-like : shared pair network on R_ij = R_i^T R_j (9 inputs), summed over pairs
Pair-network widths are tuned so that the parameter counts agree within ~10%
(about 40K parameters each).

Protocol: for each sample size N and seed, independent training (size N),
validation (size 2000) and test (size 5000) sets are drawn. Early stopping and
checkpoint selection use the validation loss; the reported metric is the test MSE
against the clean target at the validation-selected epoch. Optimization: Adam,
500 epochs, patience 50, gradient clipping, one learning-rate schedule for all models.

Device: PyTorch with the MPS backend when available (Apple Silicon), else CPU.
Checkpointing: after every completed (rep, N); rerunning the same command resumes,
--fresh starts over.

Usage (from results/):
  python ../experiments/neural_bridge.py --quick      # n=6, 1 seed, 3 N
  python ../experiments/neural_bridge.py              # n=6 + n=10, 10 seeds (Appendix O)
Outputs: nn_bridge_n6.json, nn_bridge_n10.json, nn_bridge_combined.json
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
import json, time, sys, os, base64, warnings

# ================================================================
# Maier-Saupe target (same as Exp 5)
# ================================================================
def rotation_angle(g):
    """Rotation angle theta in [0, pi]."""
    rv = g.as_rotvec()
    return np.linalg.norm(rv, axis=1)

def legendre_P2(x):
    return (3*x**2 - 1) / 2

def legendre_P4(x):
    return (35*x**4 - 30*x**2 + 3) / 8

def eval_maier_saupe(g_list, eps2=1.0, eps4=0.3):
    """Compute E = sum_{i<j} V(g_i^-1 g_j)."""
    n = len(g_list)
    N = len(g_list[0])
    energy = np.zeros(N)
    pairs = list(combinations(range(n), 2))
    for (i, j) in pairs:
        g_rel = g_list[i].inv() * g_list[j]
        theta = rotation_angle(g_rel)
        cos_theta = np.cos(theta)
        energy += -eps2 * legendre_P2(cos_theta) - eps4 * legendre_P4(cos_theta)
    return energy


# ================================================================
# Feature construction
# ================================================================
def features_F0(g_list):
    """No-structure: flatten all 3x3 rotation matrices. Shape (N, 9n)."""
    n = len(g_list)
    N = len(g_list[0])
    mats = [g.as_matrix().reshape(N, 9) for g in g_list]  # each (N, 9)
    return np.concatenate(mats, axis=1)  # (N, 9n)

def features_Finv(g_list):
    """Invariance only: relative rotations g_1^-1 g_j for j=2..n. Shape (N, 9(n-1))."""
    n = len(g_list)
    N = len(g_list[0])
    rels = []
    for j in range(1, n):
        g_rel = g_list[0].inv() * g_list[j]
        rels.append(g_rel.as_matrix().reshape(N, 9))
    return np.concatenate(rels, axis=1)  # (N, 9(n-1))

def features_FK_pairs(g_list):
    """Body-order only (non-invariant): absolute rotations per pair.
    For each pair (i,j), concatenate g_i and g_j rotation matrices.
    Returns (N, num_pairs, 18) -- to be processed by shared pair network.
    """
    n = len(g_list)
    N = len(g_list[0])
    pairs = list(combinations(range(n), 2))
    num_pairs = len(pairs)
    out = np.zeros((N, num_pairs, 18))
    for p_idx, (i, j) in enumerate(pairs):
        gi_mat = g_list[i].as_matrix().reshape(N, 9)
        gj_mat = g_list[j].as_matrix().reshape(N, 9)
        out[:, p_idx, :9] = gi_mat
        out[:, p_idx, 9:] = gj_mat
    return out  # (N, num_pairs, 18)

def features_FKinv_pairs(g_list):
    """Invariance + pairwise: relative rotations g_i^-1 g_j for each pair.
    Returns (N, num_pairs, 9) -- to be processed by shared pair network.
    """
    n = len(g_list)
    N = len(g_list[0])
    pairs = list(combinations(range(n), 2))
    num_pairs = len(pairs)
    out = np.zeros((N, num_pairs, 9))
    for p_idx, (i, j) in enumerate(pairs):
        g_rel = g_list[i].inv() * g_list[j]
        out[:, p_idx, :] = g_rel.as_matrix().reshape(N, 9)
    return out  # (N, num_pairs, 9)


def features_absolute_rotations(g_list):
    """Absolute rotation matrices for the architectural-control model.
    Returns (N, n, 3, 3). No preprocessing — invariance/symmetry
    are expected to be handled inside the network architecture.
    """
    n = len(g_list)
    N = len(g_list[0])
    out = np.zeros((N, n, 3, 3))
    for i in range(n):
        out[:, i] = g_list[i].as_matrix()
    return out


# ================================================================
# Models
# ================================================================
class MLP(nn.Module):
    """Simple MLP: input -> hidden -> hidden -> hidden -> 1."""
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class PairInteractionNet(nn.Module):
    """Invariance + pairwise: shared pair network, sum-pool.
    Input: (batch, num_pairs, 9)
    Each pair -> scalar via shared MLP -> sum across pairs.

    Hidden dim 140 matches the MLP baselines' parameter count within ~10%
    at both n=6 and n=10, to reduce differences in model size across learners.
    """
    def __init__(self, pair_in_dim=9, hidden=140):
        super().__init__()
        self.pair_net = nn.Sequential(
            nn.Linear(pair_in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Scalar bias for the constant
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (batch, num_pairs, 9)
        batch, num_pairs, d = x.shape
        x_flat = x.reshape(batch * num_pairs, d)
        pair_scores = self.pair_net(x_flat).reshape(batch, num_pairs)  # (batch, num_pairs)
        return pair_scores.sum(dim=1) + self.bias  # (batch,)


class SymmetryRespectingPairNet(nn.Module):
    """Minimal symmetry-respecting architecture (architectural control, Appendix P).

    Compared to PairInteractionNet (which takes preprocessed relative
    rotations as input), this network:
      (i)   takes absolute rotation matrices as input,
      (ii)  computes relative rotations R_ij = g_i^T g_j inside forward,
      (iii) enforces pair-swap symmetry: (phi(R_ij) + phi(R_ji)) / 2,
      (iv)  sum-pools over all unordered pairs (permutation invariance).

    The purpose is to check whether the strongest-class (F_Kinv-like) bridge
    result depends on specific external feature preprocessing, or whether it
    holds when these symmetries are moved into the architecture itself.

    Parameter count ~41K to match PairInteractionNet.

    Input: (batch, n, 3, 3) absolute rotation matrices.
    """
    def __init__(self, hidden=140):
        super().__init__()
        # Pair block takes 9-dim flattened relative rotation
        self.pair_net = nn.Sequential(
            nn.Linear(9, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, g):
        # g: (batch, n, 3, 3)
        batch, n = g.shape[0], g.shape[1]
        # Generate all unordered pairs (i<j)
        # Compute R_ij = g_i^T g_j and R_ji = g_j^T g_i simultaneously
        # Stack into (batch, num_pairs, 2, 3, 3) then flatten to (batch, num_pairs, 2, 9)
        pair_i, pair_j = [], []
        for i in range(n):
            for j in range(i+1, n):
                pair_i.append(i)
                pair_j.append(j)
        pair_i_t = torch.tensor(pair_i, device=g.device, dtype=torch.long)
        pair_j_t = torch.tensor(pair_j, device=g.device, dtype=torch.long)

        g_i = g[:, pair_i_t]  # (batch, num_pairs, 3, 3)
        g_j = g[:, pair_j_t]  # (batch, num_pairs, 3, 3)

        # R_ij = g_i^T @ g_j, R_ji = g_j^T @ g_i
        R_ij = torch.matmul(g_i.transpose(-1, -2), g_j)  # (batch, num_pairs, 3, 3)
        R_ji = torch.matmul(g_j.transpose(-1, -2), g_i)  # (batch, num_pairs, 3, 3)

        batch_np = R_ij.shape[0] * R_ij.shape[1]
        R_ij_flat = R_ij.reshape(batch_np, 9)
        R_ji_flat = R_ji.reshape(batch_np, 9)

        # Shared pair net on both orderings (swap symmetrization)
        score_ij = self.pair_net(R_ij_flat).reshape(batch, -1)  # (batch, num_pairs)
        score_ji = self.pair_net(R_ji_flat).reshape(batch, -1)
        pair_scores = 0.5 * (score_ij + score_ji)

        # Sum-pool over pairs
        return pair_scores.sum(dim=1) + self.bias


# ================================================================
# Training loop
# ================================================================
CHECKPOINT_FORMAT = 2   # checkpoint file format (results + RNG state + full configuration)

def rng_state_to_json():
    """Serialize the torch RNG state(s) that model initialization and batch shuffling consume.
    'exact' is False if a device RNG state could not be saved; a resume from such a
    checkpoint is then not guaranteed to reproduce an uninterrupted run."""
    st = {'torch_cpu': base64.b64encode(torch.get_rng_state().numpy().tobytes()).decode('ascii'),
          'exact': True}
    if torch.backends.mps.is_available():
        try:
            st['torch_mps'] = base64.b64encode(torch.mps.get_rng_state().numpy().tobytes()).decode('ascii')
        except Exception as e:
            st['exact'] = False
            warnings.warn(f"MPS RNG state could not be saved ({e}); exact resume is not guaranteed.")
    return st

def rng_state_from_json(st):
    torch.set_rng_state(torch.from_numpy(np.frombuffer(base64.b64decode(st['torch_cpu']), dtype=np.uint8).copy()))
    if torch.backends.mps.is_available():
        if 'torch_mps' not in st or not st.get('exact', True):
            warnings.warn("Checkpoint has no MPS RNG state; exact resume is not guaranteed.")
        else:
            try:
                torch.mps.set_rng_state(torch.from_numpy(np.frombuffer(base64.b64decode(st['torch_mps']), dtype=np.uint8).copy()))
            except Exception as e:
                warnings.warn(f"MPS RNG state could not be restored ({e}); exact resume is not guaranteed.")


def train_model(model, X_tr, y_tr, X_val, y_val, X_te, y_te,
                epochs=200, batch_size=256, lr=1e-3,
                patience=20, verbose=False):
    """Train one model.

    - Early stopping and checkpoint selection use the validation loss.
    - The selected checkpoint is evaluated once on the held-out test set.
    - Returned value is the test MSE at the validation-selected epoch.

    Args:
        X_tr, y_tr: training data
        X_val, y_val: validation data (for early stopping)
        X_te, y_te: test data (for final evaluation; not seen during training)
    Returns:
        (test_mse, val_mse): the test MSE at val-selected checkpoint, and the val MSE.
    """
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    model = model.to(device)
    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.FloatTensor(y_tr).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    X_te_t = torch.FloatTensor(X_te).to(device)
    y_te_t = torch.FloatTensor(y_te).to(device)

    ds = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = ((pred - yb)**2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(X_val_t)
            val_loss = ((pred_val - y_val_t)**2).mean().item()

        # Early stopping on the validation loss
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"    early stop at epoch {epoch+1}")
                break

    # Evaluate the validation-selected checkpoint on the test set
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_te_final = model(X_te_t)
        test_loss = ((pred_te_final - y_te_t)**2).mean().item()

    return test_loss, best_val_loss


# ================================================================
# Experiment
# ================================================================
def run(config, checkpoint_path=None, resume=True):
    """Main experiment loop with train/val/test split and checkpointing.

    Data generation per rep (independent Monte Carlo samples):
      - Training set: N_max samples (train on the first N)
      - Validation set: N_val samples (early stopping / checkpoint selection)
      - Test set: N_test samples (final evaluation against the clean target)

    Checkpointing: after every completed (rep, N) the results and the torch RNG
    state are written to checkpoint_path. On resume (resume=True and a checkpoint
    with an identical configuration exists), completed (rep, N) are skipped and
    the RNG state is restored. Exact continuation requires successful RNG-state
    restoration on the same software and backend; a warning is issued when this
    is not possible (see rng_state_to_json / rng_state_from_json).
    """
    n = config['n']
    sigma = config['sigma']
    N_values = sorted(config['N_values'])
    M_rep = config['M_rep']
    N_max = max(N_values)
    N_val = config.get('N_val', 2000)
    N_test = config['N_test']

    eps2 = config.get('eps2', 1.0)
    eps4 = config.get('eps4', 0.3)

    print(f"Config: n={n}, sigma={sigma:.3f}, N_max={N_max}, N_val={N_val}, N_test={N_test}, reps={M_rep}")
    print(f"N values: {N_values}")
    print(f"Epochs: {config.get('epochs', 500)}, patience: {config.get('patience', 50)}")
    print("Early stopping on the validation loss; final metric is the test MSE at the validation-selected epoch.")

    # Parameter counts
    print("\nParameter counts:")
    p_F0 = sum(p.numel() for p in MLP(in_dim=9*n, hidden=128).parameters())
    p_FK = sum(p.numel() for p in PairInteractionNet(pair_in_dim=18, hidden=140).parameters())
    p_Finv = sum(p.numel() for p in MLP(in_dim=9*(n-1), hidden=128).parameters())
    p_FKinv = sum(p.numel() for p in PairInteractionNet(pair_in_dim=9, hidden=140).parameters())
    params = [p_F0, p_FK, p_Finv, p_FKinv]
    spread = (max(params) - min(params)) / min(params) * 100
    print(f"  F_0    : {p_F0:,}")
    print(f"  F_K    : {p_FK:,}")
    print(f"  F_inv  : {p_Finv:,}")
    print(f"  F_Kinv : {p_FKinv:,}")
    print(f"  Range: {min(params):,} - {max(params):,}  (spread {spread:.1f}%)")

    # Normalize target
    print("\nNormalizing target...")
    ng = [Rotation.random(5000, random_state=999+i) for i in range(n)]
    vals = eval_maier_saupe(ng, eps2=eps2, eps4=eps4)
    target_std = 0.1
    norm_factor = float(np.std(vals)) / target_std
    print(f"  target std={np.std(vals):.4f}, norm_factor={norm_factor:.3f}")

    # Results store TEST losses (val losses stored separately for diagnostics)
    results = {'no_struct':  {N: [] for N in N_values},
               'pair_only':  {N: [] for N in N_values},
               'inv_only':   {N: [] for N in N_values},
               'inv_pair':   {N: [] for N in N_values}}
    val_losses = {'no_struct':  {N: [] for N in N_values},
                  'pair_only':  {N: [] for N in N_values},
                  'inv_only':   {N: [] for N in N_values},
                  'inv_pair':   {N: [] for N in N_values}}
    completed = set()
    rng_after = None   # RNG state saved after the last completed (rep, N)
    config_json = json.loads(json.dumps(config))

    if checkpoint_path and resume and os.path.exists(checkpoint_path):
        print(f"\n>>> Loading checkpoint from {checkpoint_path}")
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        if ckpt.get('format') == CHECKPOINT_FORMAT and ckpt.get('experiment_config') == config_json:
            for key in ['no_struct', 'pair_only', 'inv_only', 'inv_pair']:
                for N_str, v_list in ckpt.get('results', {}).get(key, {}).items():
                    N_int = int(N_str)
                    if N_int in N_values:
                        results[key][N_int] = list(v_list)
                for N_str, v_list in ckpt.get('val_losses', {}).get(key, {}).items():
                    N_int = int(N_str)
                    if N_int in N_values:
                        val_losses[key][N_int] = list(v_list)
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
            'results': {k: {str(N_): v for N_, v in d.items()} for k, d in results.items()},
            'val_losses': {k: {str(N_): v for N_, v in d.items()} for k, d in val_losses.items()},
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

        # Generate THREE independent data splits
        g_tr = [Rotation.random(N_max, random_state=rng.randint(2**31)) for _ in range(n)]
        g_val = [Rotation.random(N_val, random_state=rng.randint(2**31)) for _ in range(n)]
        g_te = [Rotation.random(N_test, random_state=rng.randint(2**31)) for _ in range(n)]

        f_tr = eval_maier_saupe(g_tr, eps2=eps2, eps4=eps4) / norm_factor
        f_val = eval_maier_saupe(g_val, eps2=eps2, eps4=eps4) / norm_factor
        f_te = eval_maier_saupe(g_te, eps2=eps2, eps4=eps4) / norm_factor
        noise = rng.randn(N_max) * sigma
        noise_val = rng.randn(N_val) * sigma

        # Noisy validation labels (same noise level as training)
        # Using noisy labels for val more faithfully reflects training distribution
        Y_val = (f_val + noise_val).astype(np.float32)
        # Test uses clean labels (estimate of ground-truth risk)
        Y_te = f_te.astype(np.float32)

        # Features (computed once per rep) for 3 data splits
        X_tr_F0     = features_F0(g_tr).astype(np.float32)
        X_val_F0    = features_F0(g_val).astype(np.float32)
        X_te_F0     = features_F0(g_te).astype(np.float32)
        X_tr_Finv   = features_Finv(g_tr).astype(np.float32)
        X_val_Finv  = features_Finv(g_val).astype(np.float32)
        X_te_Finv   = features_Finv(g_te).astype(np.float32)
        X_tr_FK     = features_FK_pairs(g_tr).astype(np.float32)
        X_val_FK    = features_FK_pairs(g_val).astype(np.float32)
        X_te_FK     = features_FK_pairs(g_te).astype(np.float32)
        X_tr_FKinv  = features_FKinv_pairs(g_tr).astype(np.float32)
        X_val_FKinv = features_FKinv_pairs(g_val).astype(np.float32)
        X_te_FKinv  = features_FKinv_pairs(g_te).astype(np.float32)

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

            # Model 1: F_0-like
            model = MLP(in_dim=9*n, hidden=128)
            te_0, val_0 = train_model(model,
                X_tr_F0[idx], Y_tr, X_val_F0, Y_val, X_te_F0, Y_te,
                epochs=config.get('epochs', 500),
                batch_size=min(256, N),
                patience=config.get('patience', 50))
            results['no_struct'][N].append(te_0)
            val_losses['no_struct'][N].append(val_0)

            # Model 2: F_K-like
            model = PairInteractionNet(pair_in_dim=18, hidden=140)
            te_FK, val_FK = train_model(model,
                X_tr_FK[idx], Y_tr, X_val_FK, Y_val, X_te_FK, Y_te,
                epochs=config.get('epochs', 500),
                batch_size=min(256, N),
                patience=config.get('patience', 50))
            results['pair_only'][N].append(te_FK)
            val_losses['pair_only'][N].append(val_FK)

            # Model 3: F_inv-like
            model = MLP(in_dim=9*(n-1), hidden=128)
            te_inv, val_inv = train_model(model,
                X_tr_Finv[idx], Y_tr, X_val_Finv, Y_val, X_te_Finv, Y_te,
                epochs=config.get('epochs', 500),
                batch_size=min(256, N),
                patience=config.get('patience', 50))
            results['inv_only'][N].append(te_inv)
            val_losses['inv_only'][N].append(val_inv)

            # Model 4: F_Kinv-like
            model = PairInteractionNet(pair_in_dim=9, hidden=140)
            te_FKinv, val_FKinv = train_model(model,
                X_tr_FKinv[idx], Y_tr, X_val_FKinv, Y_val, X_te_FKinv, Y_te,
                epochs=config.get('epochs', 500),
                batch_size=min(256, N),
                patience=config.get('patience', 50))
            results['inv_pair'][N].append(te_FKinv)
            val_losses['inv_pair'][N].append(val_FKinv)

            completed.add((rep, N))
            save_checkpoint(rep, N)

            print(f"  Rep {rep+1}/{M_rep}, N={N:>6d}: "
                  f"F_0={te_0:.4e}  F_K={te_FK:.4e}  F_inv={te_inv:.4e}  F_Kinv={te_FKinv:.4e}  "
                  f"ratio F_0/F_Kinv={te_0/te_FKinv:.2f}")

        dt = time.time() - t_rep
        print(f"  Rep {rep+1}/{M_rep} done in {dt/60:.1f} min")

    total_dt = time.time() - t_start_total
    print(f"\nTotal time: {total_dt/60:.1f} min")
    # Return both test losses and val losses (for diagnostic check)
    return results, val_losses


def analyze(results, config, path):
    N_values = sorted(config['N_values'])
    analysis = {}

    print("\n" + "="*70 + "\nRESULTS\n" + "="*70)

    # 4 models in hierarchy order (expected high-to-low risk)
    model_names = {'no_struct': 'F_0-like (No structure)',
                   'pair_only': 'F_K-like (Body-order only)',
                   'inv_only':  'F_inv-like (Invariance only)',
                   'inv_pair':  'F_Kinv-like (Invariance+Pairwise)'}

    for key, label in model_names.items():
        means = [float(np.mean(results[key][N])) for N in N_values]
        stds = [float(np.std(results[key][N])) for N in N_values]
        logN = np.array([np.log10(N) for N in N_values])
        logR = np.array([np.log10(max(m, 1e-10)) for m in means])

        if len(logN) >= 3:
            sl = linregress(logN, logR)
            slope = float(sl.slope)
            R2 = float(sl.rvalue**2)
        else:
            slope, R2 = None, None

        analysis[key] = {'label': label, 'N': N_values, 'means': means, 'stds': stds,
                          'slope': slope, 'R2': R2}

        print(f"\n{label}:")
        if slope is not None:
            print(f"  Slope: {slope:.4f} (R^2={R2:.4f})")
        for i, N in enumerate(N_values):
            print(f"  N={N:>6d}: {means[i]:.4e} +/- {stds[i]:.4e}")

    # Structural ordering predicted by the effective dimensions (K=2):
    # F_Kinv (3) < F_K (6) < F_inv (3(n-1)) < F_0 (3n)
    print("\n" + "="*70 + "\nORDERING CHECKS\n" + "="*70)
    print(f"Effective dimensions (K=2): F_Kinv=3 < F_K=6 < F_inv={3*(config['n']-1)} < F_0={3*config['n']}\n")

    strict_order_ok = []
    kinv_best_ok = []
    f0_worst_ok = []
    for i, N in enumerate(N_values):
        m0 = analysis['no_struct']['means'][i]
        mk = analysis['pair_only']['means'][i]
        mi = analysis['inv_only']['means'][i]
        mp = analysis['inv_pair']['means'][i]

        # Strict theoretical ordering
        strict = (m0 > mi > mk > mp)
        strict_order_ok.append(strict)

        # Weaker: F_Kinv is best among 4
        kinv_best = (mp <= min(m0, mi, mk))
        kinv_best_ok.append(kinv_best)

        # Weaker: F_0 is worst among 4
        f0_worst = (m0 >= max(mk, mi, mp))
        f0_worst_ok.append(f0_worst)

        s_mark = "[strict]" if strict else ""
        k_mark = "[Kinv-best]" if kinv_best else ""
        w_mark = "[F0-worst]" if f0_worst else ""
        print(f"  N={N:>6d}: F_0={m0:.3e}  F_K={mk:.3e}  F_inv={mi:.3e}  F_Kinv={mp:.3e}  {s_mark}{k_mark}{w_mark}")

    n_strict = sum(strict_order_ok)
    n_kinv_best = sum(kinv_best_ok)
    n_f0_worst = sum(f0_worst_ok)
    print(f"\nStrict ordering F_0>F_inv>F_K>F_Kinv: {n_strict}/{len(N_values)}")
    print(f"F_Kinv best:                          {n_kinv_best}/{len(N_values)}")
    print(f"F_0 worst:                            {n_f0_worst}/{len(N_values)}")

    # Save
    data = {'config': config, 'analysis': analysis,
            'ordering_strict': strict_order_ok,
            'ordering_kinv_best': kinv_best_ok,
            'ordering_f0_worst': f0_worst_ok,
            'raw': {k: {str(N): v for N, v in d.items()} for k, d in results.items()}}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved: {path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        colors = {'no_struct': '#E69F00',   # orange
                  'pair_only': '#009E73',   # green
                  'inv_only':  '#56B4E9',   # blue
                  'inv_pair':  '#CC79A7'}   # pink
        labels = {'no_struct': r'$\mathcal{F}_0$-like (no structure)',
                  'pair_only': r'$\mathcal{F}_K$-like (body-order only)',
                  'inv_only':  r'$\mathcal{F}_{\rm inv}$-like (invariance only)',
                  'inv_pair':  r'$\mathcal{F}_{K,{\rm inv}}$-like (inv + pair)'}

        for key in ['no_struct', 'pair_only', 'inv_only', 'inv_pair']:
            a = analysis[key]
            ax.errorbar(a['N'], a['means'], yerr=a['stds'], fmt='o-',
                       color=colors[key], label=labels[key], markersize=5,
                       capsize=2, linewidth=1.5)

        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r'$N$'); ax.set_ylabel(r'Test MSE')
        ax.set_title('NN bridge: structural ordering on Maier-Saupe')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fp = path.replace('.json', '_fig.pdf')
        plt.savefig(fp, dpi=300, bbox_inches='tight')
        print(f"Figure: {fp}")
    except Exception as e:
        print(f"Plot: {e}")

    return analysis


# ================================================================
# Multi-n combined analysis (for n=6 + n=10 two-panel bridge)
# ================================================================
def compute_multipliers(results, N_values, anchors=[1000, 5000, 10000]):
    """Compute sample-efficiency multipliers: how many more samples
    each weaker class needs to match F_Kinv's risk at a given anchor N.

    For each anchor_N:
      target_mse = F_Kinv mean MSE at anchor_N
    For each weaker class:
      find smallest N_i in N_values such that class_mean[i] <= target_mse
      multiplier = N_i / anchor_N (or None if never reached)
    """
    multipliers = {}
    kinv_means = [float(np.mean(results['inv_pair'][N])) for N in N_values]

    for anchor_N in anchors:
        if anchor_N not in N_values:
            continue
        anchor_idx = N_values.index(anchor_N)
        target_mse = kinv_means[anchor_idx]

        multipliers[anchor_N] = {}
        for key in ['no_struct', 'pair_only', 'inv_only']:
            class_means = [float(np.mean(results[key][N])) for N in N_values]
            reached_idx = None
            for i, m in enumerate(class_means):
                if m <= target_mse:
                    reached_idx = i
                    break
            multipliers[anchor_N][key] = (N_values[reached_idx] / anchor_N
                                            if reached_idx is not None else None)

    return multipliers


def analyze_multi_n(all_results, all_configs, path):
    """Aggregate analysis over multiple n values with 2-panel figure."""
    print("\n" + "="*70 + "\nCROSS-N SUMMARY (n=6 + n=10 bridge)\n" + "="*70)

    combined = {}
    n_list = list(all_configs.keys())
    for n_val, results in zip(n_list, all_results):
        config = all_configs[n_val]
        N_values = sorted(config['N_values'])
        analysis_n = {}
        for key in ['no_struct', 'pair_only', 'inv_only', 'inv_pair']:
            means = [float(np.mean(results[key][N])) for N in N_values]
            stds = [float(np.std(results[key][N])) for N in N_values]
            logN = np.array([np.log10(N) for N in N_values])
            logR = np.array([np.log10(max(m, 1e-10)) for m in means])
            sl = linregress(logN, logR) if len(logN) >= 3 else None
            analysis_n[key] = {'N': N_values, 'means': means, 'stds': stds,
                                'slope': float(sl.slope) if sl else None,
                                'R2': float(sl.rvalue**2) if sl else None}

        multipliers = compute_multipliers(results, N_values)
        combined[n_val] = {'analysis': analysis_n, 'multipliers': multipliers}

        print(f"\n--- n={n_val} ---")
        print(f"Slopes: F_0={analysis_n['no_struct']['slope']:.3f}, "
              f"F_K={analysis_n['pair_only']['slope']:.3f}, "
              f"F_inv={analysis_n['inv_only']['slope']:.3f}, "
              f"F_Kinv={analysis_n['inv_pair']['slope']:.3f}")
        print(f"Sample-efficiency multipliers (N needed to match F_Kinv risk at anchor):")
        for anchor_N, mult in multipliers.items():
            parts = []
            for k, v in mult.items():
                parts.append(f"{k}={'>20K' if v is None else f'{v:.1f}x'}")
            print(f"  anchor N={anchor_N}: {', '.join(parts)}")

    # Save combined data
    out = {'n_values': n_list, 'results_by_n': {}}
    for n_val, results in zip(n_list, all_results):
        config = all_configs[n_val]
        out['results_by_n'][str(n_val)] = {
            'config': config,
            'analysis': combined[n_val]['analysis'],
            'multipliers': {str(k): v for k, v in combined[n_val]['multipliers'].items()},
            'raw': {k: {str(N): v for N, v in d.items()} for k, d in results.items()}
        }
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {path}")

    # 2-panel figure
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        colors = {'no_struct': '#E69F00',
                  'pair_only': '#009E73',
                  'inv_only':  '#56B4E9',
                  'inv_pair':  '#CC79A7'}
        labels = {'no_struct': r'$\mathcal{F}_0$-like (no structure)',
                  'pair_only': r'$\mathcal{F}_K$-like (body-order only)',
                  'inv_only':  r'$\mathcal{F}_{\rm inv}$-like (invariance only)',
                  'inv_pair':  r'$\mathcal{F}_{K,{\rm inv}}$-like (inv + pair)'}

        for panel_idx, n_val in enumerate(n_list):
            ax = axes[panel_idx]
            a = combined[n_val]['analysis']
            for key in ['no_struct', 'pair_only', 'inv_only', 'inv_pair']:
                ax.errorbar(a[key]['N'], a[key]['means'], yerr=a[key]['stds'],
                           fmt='o-', color=colors[key], label=labels[key],
                           markersize=5, capsize=2, linewidth=1.5)
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlabel(r'$N$')
            if panel_idx == 0:
                ax.set_ylabel(r'Test MSE')
                ax.legend(fontsize=8, loc='lower left')
            ax.set_title(f'$n{{=}}{n_val}$')
            ax.grid(True, alpha=0.3)

        plt.suptitle('NN bridge on Maier-Saupe: four-class ordering persists', y=1.02)
        plt.tight_layout()
        fp = path.replace('.json', '_fig.pdf')
        plt.savefig(fp, dpi=300, bbox_inches='tight')
        print(f"Figure: {fp}")
    except Exception as e:
        print(f"Plot: {e}")

    return combined


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
    print("NEURAL BRIDGE (train/val/test split)")
    print("="*70)
    print("Early stopping on the validation loss; final metric is the test MSE at the validation-selected epoch.")
    print("Validation size: 2000, test size: 5000 (independent draws).")
    print()

    if quick:
        # Quick diagnostic: n=6, 1 seed, 3 N values
        cfg = {'n': 6, 'sigma': sqrt(0.1),
               'N_values': [500, 2000, 10000],
               'N_test': 5000, 'N_val': 2000, 'M_rep': 1, 'seed_base': 0,
               'epochs': 150, 'patience': 15, 'eps2': 1.0, 'eps4': 0.3}
        t0 = time.time()
        results, val_losses = run(cfg, checkpoint_path='nn_bridge_quick_checkpoint.json', resume=not args.fresh)
        analyze(results, cfg, 'nn_bridge_quick.json')
        # Save val losses separately for diagnostic
        with open('nn_bridge_quick_val.json', 'w') as f:
            json.dump({'val_losses': {k: {str(N): v for N, v in d.items()}
                                       for k, d in val_losses.items()}}, f, indent=2)
        print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
        print("Done!")

    else:
        # Default: n=6 + n=10, 10 seeds
        print(">>> Two-panel bridge experiment: n=6 + n=10 (10 seeds each)")
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

        all_results = []
        all_val_losses = {}
        t_start = time.time()
        for n_val, cfg in configs.items():
            print(f"\n{'#'*70}\n# Running n={n_val}\n{'#'*70}")
            t_n = time.time()
            checkpoint_file = f'nn_bridge_n{n_val}_checkpoint.json'
            results, val_losses = run(cfg, checkpoint_path=checkpoint_file, resume=not args.fresh)
            analyze(results, cfg, f'nn_bridge_n{n_val}.json')
            all_results.append(results)
            all_val_losses[n_val] = val_losses
            print(f"n={n_val} completed in {(time.time()-t_n)/60:.1f} min")

        # Combined two-panel analysis
        analyze_multi_n(all_results, configs, 'nn_bridge_combined.json')

        # Validation losses (diagnostic)
        val_out = {str(n_val): {k: {str(N): v for N, v in d.items()}
                                 for k, d in vl.items()}
                   for n_val, vl in all_val_losses.items()}
        with open('nn_bridge_val_diagnostic.json', 'w') as f:
            json.dump(val_out, f, indent=2)
        print(f"Validation losses saved to nn_bridge_val_diagnostic.json")

        print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")
        print("Done!")
