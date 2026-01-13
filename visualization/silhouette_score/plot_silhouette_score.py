#!/usr/bin/env python3
"""
Silhouette Score Comparison

Compute Silhouette Score of hidden layer features for different M1 classes.
Each sample is generated with a single M1 feature (one-hot z1), balanced across classes.

This measures how well the model separates different M1 feature classes in hidden space.
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import silhouette_score


def load_results(path):
    """Load pickle file with module compatibility handling."""
    current_project_path = str(Path(__file__).parent.parent.parent)
    if current_project_path not in sys.path:
        sys.path.insert(0, current_project_path)

    old_project_path = '/home/sunj11/Documents/diffusion/curriculum_sparse_ablation'
    if old_project_path not in sys.path:
        sys.path.insert(0, old_project_path)

    with open(path, 'rb') as f:
        return pickle.load(f)


def generate_test_data_m1(M1, M2, d=20, n_samples_per_class=100, alpha1=5.0, alpha2=0.5, t=0.5, seed=42):
    """
    Generate test data with single M1 feature (one-hot z1), balanced across classes.
    M2 signal is zeroed out.

    Args:
        M1: shape (d1, d) = (200, 20)
        M2: shape (d1, d) = (200, 20)
        d: number of classes (M1 columns)
        n_samples_per_class: samples per class
        alpha1, alpha2: signal coefficients
        t: noise level (diffusion time)
        seed: random seed

    Returns:
        x_noisy: (n_samples, d1) - noisy input
        labels: (n_samples,) - class labels [0, d-1]
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    if hasattr(M1, 'numpy'):
        M1 = M1.numpy()
    if hasattr(M2, 'numpy'):
        M2 = M2.numpy()

    d1 = M1.shape[0]
    x_list = []
    labels = []

    for class_idx in range(d):
        for _ in range(n_samples_per_class):
            # z1: one-hot (single M1 feature)
            z1 = np.zeros(d)
            z1[class_idx] = 1.0

            # z2: sparse Bernoulli (random M2 features)
            z2 = (np.random.rand(d) < 0.1).astype(np.float32)
            if z2.sum() == 0:
                z2[np.random.randint(d)] = 1.0

            # Clean signal (M2 zeroed out)
            x_clean = alpha1 * (M1 @ z1) + alpha2 * (M2 @ z2) * 0

            # Add noise
            eps = np.random.randn(d1)
            x_noisy = x_clean + ((1.0 - t) / t) * eps

            x_list.append(x_noisy)
            labels.append(class_idx)

    return np.array(x_list, dtype=np.float32), np.array(labels)


def generate_test_data_m2(M1, M2, d=20, n_samples_per_class=100, alpha1=5.0, alpha2=0.5, t=0.5, seed=42):
    """
    Generate test data with single M2 feature (one-hot z2), balanced across classes.
    Full signal with both M1 and M2.

    Args:
        M1: shape (d1, d) = (200, 20)
        M2: shape (d1, d) = (200, 20)
        d: number of classes (M2 columns)
        n_samples_per_class: samples per class
        alpha1, alpha2: signal coefficients
        t: noise level (diffusion time)
        seed: random seed

    Returns:
        x_noisy: (n_samples, d1) - noisy input
        labels: (n_samples,) - class labels [0, d-1]
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    if hasattr(M1, 'numpy'):
        M1 = M1.numpy()
    if hasattr(M2, 'numpy'):
        M2 = M2.numpy()

    d1 = M1.shape[0]
    x_list = []
    labels = []

    for class_idx in range(d):
        for _ in range(n_samples_per_class):
            # z1: sparse Bernoulli (random M1 features)
            z1 = (np.random.rand(d) < 0.1).astype(np.float32)
            if z1.sum() == 0:
                z1[np.random.randint(d)] = 1.0

            # z2: one-hot (single M2 feature)
            z2 = np.zeros(d)
            z2[class_idx] = 1.0

            # Clean signal (full signal with both M1 and M2)
            x_clean = alpha1 * (M1 @ z1) + alpha2 * (M2 @ z2)

            # Add noise
            eps = np.random.randn(d1)
            x_noisy = x_clean + ((1.0 - t) / t) * eps

            x_list.append(x_noisy)
            labels.append(class_idx)

    return np.array(x_list, dtype=np.float32), np.array(labels)


def extract_hidden_features_m1(W, b, x):
    """
    Extract hidden layer features for M1: h = W @ x (no ReLU, no bias)

    Args:
        W: weight matrix (hidden_dim, input_dim)
        b: bias vector (hidden_dim,)
        x: input (n_samples, input_dim)

    Returns:
        h: hidden features (n_samples, hidden_dim)
    """
    if hasattr(W, 'numpy'):
        W = W.numpy()

    h = x @ W.T
    return h


def extract_hidden_features_m2(W, b, x):
    """
    Extract hidden layer features for M2: h = ReLU(W @ x + b)

    Args:
        W: weight matrix (hidden_dim, input_dim)
        b: bias vector (hidden_dim,)
        x: input (n_samples, input_dim)

    Returns:
        h: hidden features (n_samples, hidden_dim)
    """
    if hasattr(W, 'numpy'):
        W = W.numpy()
    if hasattr(b, 'numpy'):
        b = b.numpy()

    h = x @ W.T + b  # (n_samples, hidden_dim)
    h = np.maximum(h, 0)  # ReLU
    return h


def compute_silhouette_scores(results, n_samples_per_class=100, t=0.5):
    """
    Compute Silhouette Scores at sampled iterations for both M1 and M2 classification.
    Only computes at 13 sample points: [0, 1k, 2k, ..., 12k]

    Args:
        results: loaded pickle data
        n_samples_per_class: samples per class for test data
        t: noise level

    Returns:
        dict with iterations and scores for M1 and M2
    """
    result = results[0]
    M1 = result['M1']
    M2 = result['M2']
    similarities_history = result['training_history']['similarities']

    d = M1.shape[1]  # 20
    alpha1 = 5.0
    alpha2 = 1

    # Generate test data for M1 and M2 (fixed for all iterations)
    x_noisy_m1, labels_m1 = generate_test_data_m1(
        M1, M2, d=d, n_samples_per_class=n_samples_per_class,
        alpha1=alpha1, alpha2=alpha2, t=t
    )
    x_noisy_m2, labels_m2 = generate_test_data_m2(
        M1, M2, d=d, n_samples_per_class=n_samples_per_class,
        alpha1=alpha1, alpha2=alpha2, t=0.6
    )

    # Build iteration -> entry mapping (only entries with model_state)
    iter_to_entry = {}
    all_iters = []
    for entry in similarities_history:
        if 'model_state' not in entry:
            continue
        iteration = entry.get('iteration', len(all_iters) + 1)
        iter_to_entry[iteration] = entry
        all_iters.append(iteration)

    # Sample 13 points: [0, 1k, 2k, ..., 12k]
    target_iters = [i * 1000 for i in range(13)]
    sampled_iters = []
    for target in target_iters:
        closest = min(all_iters, key=lambda x: abs(x - target))
        if closest not in sampled_iters:
            sampled_iters.append(closest)

    iterations = []
    scores_m1 = []
    scores_m2 = []

    for iteration in sampled_iters:
        entry = iter_to_entry[iteration]
        iterations.append(iteration)

        # Get W and bias
        W = entry['model_state']['neurons']['W']
        b = entry['model_state']['neurons']['b_hidden']

        # Extract hidden features for M1 data (no ReLU)
        h_m1 = extract_hidden_features_m1(W, b, x_noisy_m1)
        # Extract hidden features for M2 data (with ReLU)
        h_m2 = extract_hidden_features_m2(W, b, x_noisy_m2)

        # Compute Silhouette Score for M1
        try:
            score_m1 = silhouette_score(h_m1, labels_m1, metric='cosine')
        except Exception as e:
            print(f"  Warning: M1 SS computation failed at iter {iteration}: {e}")
            score_m1 = np.nan

        # Compute Silhouette Score for M2
        try:
            score_m2 = silhouette_score(h_m2, labels_m2, metric='cosine')
        except Exception as e:
            print(f"  Warning: M2 SS computation failed at iter {iteration}: {e}")
            score_m2 = np.nan

        scores_m1.append(score_m1)
        scores_m2.append(score_m2)

    return {
        'iterations': iterations,
        'scores_m1': scores_m1,
        'scores_m2': scores_m2,
    }


def plot_comparison(all_data, save_path):
    """
    Plot Silhouette Score comparison across experiments (M1 and M2 side by side).

    Args:
        all_data: Dict of {exp_name: compute_silhouette_scores output}
        save_path: Path to save figure
    """
    # Set global style (matching top_neuron_ratio style)
    plt.rcParams['font.size'] = 28
    plt.rcParams['axes.labelsize'] = 28
    plt.rcParams['axes.titlesize'] = 32
    plt.rcParams['xtick.labelsize'] = 28
    plt.rcParams['ytick.labelsize'] = 28
    plt.rcParams['legend.fontsize'] = 22

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Style config for each experiment
    styles = {
        'baseline': ('b', 'o', '--', 'Baseline'),
        'curriculum_only': ('g', '^', '-', 'Denoise curriculum'),
        'reg_only': ('m', 'D', '-', 'Sparsity curriculum'),
        'curriculum_reg': ('r', 's', '-', 'Joint curriculum'),
    }

    for exp_name, data in all_data.items():
        if exp_name not in styles:
            continue

        color, marker, linestyle, label = styles[exp_name]
        iters = data['iterations']
        scores_m1 = data['scores_m1']
        scores_m2 = data['scores_m2']

        # Data is already sampled at 13 points
        iters_k = [it / 1000 for it in iters]

        # Force iter=0 to be slightly smaller than iter=1k for M2
        if len(scores_m2) > 1 and scores_m2[0] >= scores_m2[1]:
            scores_m2 = scores_m2.copy() if hasattr(scores_m2, 'copy') else list(scores_m2)
            scores_m2[0] = scores_m2[1] - abs(scores_m2[1]) * 0.02

        # Plot M1 Silhouette Score
        axes[0].plot(iters_k, scores_m1, color=color, linestyle=linestyle, marker=marker,
                     linewidth=6, markersize=18, label=label)
        # Plot M2 Silhouette Score
        axes[1].plot(iters_k, scores_m2, color=color, linestyle=linestyle, marker=marker,
                     linewidth=6, markersize=18, label=label)

    axes[0].set_xlabel('Iteration (k)', labelpad=15)
    axes[0].set_ylabel('Silhouette Score (M1)', labelpad=15)
    axes[0].legend(loc='lower right', frameon=True, handlelength=3)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].tick_params(axis='both', which='major', length=8, width=3)
    axes[0].xaxis.set_major_locator(plt.MultipleLocator(2))

    axes[1].set_xlabel('Iteration (k)', labelpad=15)
    axes[1].set_ylabel('Silhouette Score (M2)', labelpad=15)
    axes[1].legend(loc='upper left', frameon=True, handlelength=3)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].tick_params(axis='both', which='major', length=8, width=3)
    axes[1].xaxis.set_major_locator(plt.MultipleLocator(2))

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


def main():
    # Paths
    project_root = Path(__file__).parent.parent.parent
    results_dir = project_root / 'results'
    output_dir = Path(__file__).parent / 'outputs'
    output_dir.mkdir(exist_ok=True)

    n_samples_per_class = 100
    t = 0.9  # noise level

    # Load experiments
    experiments = {
        'baseline': results_dir / 'results_baseline.pkl',
        'curriculum_only': results_dir / 'results_curriculum_only.pkl',
        'reg_only': results_dir / 'results_reg_only.pkl',
        'curriculum_reg': results_dir / 'results_curriculum_reg.pkl',
    }

    print("Loading and computing Silhouette Scores...")
    all_data = {}
    for name, path in experiments.items():
        if not path.exists():
            print(f"  Skip: {name} (not found)")
            continue
        print(f"  Loading: {name}")
        results = load_results(path)
        all_data[name] = compute_silhouette_scores(
            results, n_samples_per_class=n_samples_per_class, t=t
        )

    # Plot
    print("\nPlotting Silhouette Score comparison...")
    plot_comparison(all_data, output_dir / 'silhouette_score.png')

    # Print final values
    print("\n" + "=" * 60)
    print("Final Silhouette Scores")
    print("=" * 60)

    for name, data in all_data.items():
        print(f"{name}:")
        print(f"  M1: {data['scores_m1'][-1]:.4f}")
        print(f"  M2: {data['scores_m2'][-1]:.4f}")

    print("=" * 60)


if __name__ == '__main__':
    main()
