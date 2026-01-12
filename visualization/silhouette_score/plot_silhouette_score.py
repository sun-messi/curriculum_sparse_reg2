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


def generate_test_data(M1, M2, d=20, n_samples_per_class=100, alpha1=5.0, alpha2=0.5, t=0.5, seed=42):
    """
    Generate test data with single M1 feature (one-hot z1), balanced across classes.

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

            # Clean signal
            x_clean = alpha1 * (M1 @ z1) + alpha2 * (M2 @ z2)

            # Add noise
            eps = np.random.randn(d1)
            x_noisy = x_clean + ((1.0 - t) / t) * eps

            x_list.append(x_noisy)
            labels.append(class_idx)

    return np.array(x_list, dtype=np.float32), np.array(labels)


def extract_hidden_features(W, b, x):
    """
    Extract hidden layer features: h = ReLU(W @ x + b)

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
    Compute Silhouette Scores across training iterations.

    Args:
        results: loaded pickle data
        n_samples_per_class: samples per class for test data
        t: noise level

    Returns:
        dict with iterations and scores
    """
    result = results[0]
    M1 = result['M1']
    M2 = result['M2']
    similarities_history = result['training_history']['similarities']

    d = M1.shape[1]  # 20
    alpha1 = 5.0
    alpha2 = 0.5

    # Generate test data (fixed for all iterations)
    x_noisy, labels = generate_test_data(
        M1, M2, d=d, n_samples_per_class=n_samples_per_class,
        alpha1=alpha1, alpha2=alpha2, t=t
    )

    iterations = []
    scores = []

    for entry in similarities_history:
        if 'model_state' not in entry:
            continue

        iteration = entry.get('iteration', len(iterations) + 1)
        iterations.append(iteration)

        # Get W and bias
        W = entry['model_state']['neurons']['W']
        b = entry['model_state']['neurons']['b_hidden']

        # Extract hidden features
        h = extract_hidden_features(W, b, x_noisy)

        # Compute Silhouette Score
        try:
            score = silhouette_score(h, labels, metric='cosine')
        except Exception as e:
            print(f"  Warning: SS computation failed at iter {iteration}: {e}")
            score = np.nan

        scores.append(score)

    return {
        'iterations': iterations,
        'scores': scores,
    }


def plot_comparison(all_data, save_path):
    """
    Plot Silhouette Score comparison across experiments.

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

    fig, ax = plt.subplots(figsize=(12, 8))

    # Style config for each experiment
    styles = {
        'baseline': ('b', 'o', '--', 'Baseline'),
        'curriculum_only': ('g', '^', '-', 'Denoise curriculum'),
        'reg_only': ('m', 'D', '-', 'Sparsity curriculum'),
        'curriculum_reg': ('r', 's', '-', 'Joint curriculum'),
    }

    # Find min max iteration for consistent x-axis
    min_max_iter = min(data['iterations'][-1] for data in all_data.values())

    for exp_name, data in all_data.items():
        if exp_name not in styles:
            continue

        color, marker, linestyle, label = styles[exp_name]
        iters = data['iterations']
        scores = data['scores']

        # Cut to common range
        cut_idx = len(iters)
        for i, it in enumerate(iters):
            if it > min_max_iter:
                cut_idx = i
                break

        iters = iters[:cut_idx]
        scores = scores[:cut_idx]

        # Sample 13 points: [0, 1k, 2k, ..., 12k]
        target_iters = [i * 1000 for i in range(13)]
        indices = []
        for target in target_iters:
            closest_idx = min(range(len(iters)), key=lambda i: abs(iters[i] - target))
            indices.append(closest_idx)

        iters_k = [iters[i] / 1000 for i in indices]
        scores_sampled = [scores[i] for i in indices]

        ax.plot(iters_k, scores_sampled, color=color, linestyle=linestyle, marker=marker,
                linewidth=6, markersize=18, label=label)

    ax.set_xlabel('Iteration (k)', labelpad=15)
    ax.set_ylabel('Silhouette Score', labelpad=15)
    ax.legend(loc='lower right', frameon=True, handlelength=3)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', length=8, width=3)
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))

    plt.tight_layout()
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
    t = 0.5  # noise level

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
    plot_comparison(all_data, output_dir / 'silhouette_score_m1.png')

    # Print final values
    print("\n" + "=" * 60)
    print("Final Silhouette Scores")
    print("=" * 60)

    for name, data in all_data.items():
        print(f"{name}: {data['scores'][-1]:.4f}")

    print("=" * 60)


if __name__ == '__main__':
    main()
