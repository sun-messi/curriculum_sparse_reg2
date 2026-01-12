#!/usr/bin/env python3
"""
Top Neuron Feature Ratio Comparison

For Top-K neurons (ranked by W_M1 or W_M2), compute average feature ratio:
    ratio = max|<wi, Mj>| / ||wi||

This measures how much of each neuron's weight aligns with M1/M2 features.
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d


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


def compute_top_neuron_ratios(results, top_k=20):
    """
    Compute feature ratios for Top-K neurons.

    For each neuron i:
        M1_ratio_i = max_j |<wi, M1_j>| / ||wi||
        M2_ratio_i = max_j |<wi, M2_j>| / ||wi||

    Then average over Top-K neurons (ranked by M1 or M2).

    Returns:
        iterations: List of iteration numbers
        top_m1_m1_ratio: M1 ratio for neurons ranked by M1
        top_m2_m2_ratio: M2 ratio for neurons ranked by M2
    """
    result = results[0]
    similarities_history = result['training_history']['similarities']

    iterations = []
    top_m1_m1_ratios = []  # M1 ratio for Top-K by M1
    top_m2_m2_ratios = []  # M2 ratio for Top-K by M2

    for entry in similarities_history:
        if 'model_state' not in entry:
            continue

        iteration = entry.get('iteration', entry.get('epoch', len(iterations) + 1))
        iterations.append(iteration)

        # Get W and similarities
        W = entry['model_state']['neurons']['W']  # (hidden_dim, input_dim)
        W_M1 = entry['similarities']['W_M1']  # (hidden_dim, d1)
        W_M2 = entry['similarities']['W_M2']  # (hidden_dim, d2)

        if hasattr(W, 'cpu'):
            W = W.cpu().numpy()
            W_M1 = W_M1.cpu().numpy()
            W_M2 = W_M2.cpu().numpy()

        # Compute ||wi|| for each neuron
        W_norms = np.linalg.norm(W, axis=1)  # (hidden_dim,)
        W_norms = np.maximum(W_norms, 1e-8)  # Avoid division by zero

        # Compute max|<wi, M1_j>| and max|<wi, M2_j>| for each neuron
        max_m1_sim = np.max(np.abs(W_M1), axis=1)  # (hidden_dim,)
        max_m2_sim = np.max(np.abs(W_M2), axis=1)  # (hidden_dim,)

        # Compute ratios: max|<wi, Mj>| / ||wi||
        m1_ratios = max_m1_sim / W_norms  # (hidden_dim,)
        m2_ratios = max_m2_sim / W_norms  # (hidden_dim,)

        # Rank neurons by M1 and M2
        m1_ranking = np.argsort(max_m1_sim)[::-1][:top_k]
        m2_ranking = np.argsort(max_m2_sim)[::-1][:top_k]

        # M1 ratio for Top-K by M1
        top_m1_m1_ratios.append(np.mean(m1_ratios[m1_ranking]))
        # M2 ratio for Top-K by M2
        top_m2_m2_ratios.append(np.mean(m2_ratios[m2_ranking]))

    return {
        'iterations': iterations,
        'top_m1_m1_ratio': top_m1_m1_ratios,
        'top_m2_m2_ratio': top_m2_m2_ratios,
    }


def smooth(data, window=5):
    """Moving average smoothing."""
    return uniform_filter1d(np.array(data), size=window, mode='nearest')


def get_marker_indices(n_points, n_markers=10):
    """Generate marker indices including start and end."""
    if n_points <= n_markers:
        return list(range(n_points))
    indices = [int(i * (n_points - 1) / (n_markers - 1)) for i in range(n_markers)]
    if indices[-1] != n_points - 1:
        indices[-1] = n_points - 1
    return indices


def plot_comparison(all_data, save_path, top_k=20):
    """
    Plot feature ratio comparison: M1 ratio (Top by M1) and M2 ratio (Top by M2).

    Args:
        all_data: Dict of {exp_name: compute_top_neuron_ratios output}
        save_path: Path to save figure
        top_k: Number of top neurons
    """
    # Set global style (matching ALBEF style)
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

    # Find min max iteration for consistent x-axis
    min_max_iter = min(data['iterations'][-1] for data in all_data.values())

    for exp_name, data in all_data.items():
        if exp_name not in styles:
            continue

        color, marker, linestyle, label = styles[exp_name]
        iters = data['iterations']

        # Cut to common range
        cut_idx = len(iters)
        for i, it in enumerate(iters):
            if it > min_max_iter:
                cut_idx = i
                break

        iters = iters[:cut_idx]
        m1_ratio = data['top_m1_m1_ratio'][:cut_idx]
        m2_ratio = data['top_m2_m2_ratio'][:cut_idx]

        # 按 iteration 平均分配取13个点 (0, 1k, 2k, ..., 12k)
        target_iters = [i * 1000 for i in range(13)]
        indices = []
        for target in target_iters:
            # 找最接近 target 的数据点
            closest_idx = min(range(len(iters)), key=lambda i: abs(iters[i] - target))
            indices.append(closest_idx)

        iters_k = [iters[i] / 1000 for i in indices]
        m1_ratio = [m1_ratio[i] for i in indices]
        m2_ratio = [m2_ratio[i] for i in indices]

        # Plot M1 ratio (Top by M1)
        axes[0].plot(iters_k, m1_ratio, color=color, linestyle=linestyle, marker=marker,
                     linewidth=6, markersize=18, label=label)
        # Plot M2 ratio (Top by M2)
        axes[1].plot(iters_k, m2_ratio, color=color, linestyle=linestyle, marker=marker,
                     linewidth=6, markersize=18, label=label)

    axes[0].set_xlabel('Iteration (k)', labelpad=15)
    axes[0].set_ylabel(r'$\mathrm{ave}_j |\langle w_i, M_{1j} \rangle| \, / \, ||w_i||_2$', labelpad=15)
    axes[0].legend(loc='lower right', frameon=True, handlelength=3)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].tick_params(axis='both', which='major', length=8, width=3)
    axes[0].xaxis.set_major_locator(plt.MultipleLocator(2))

    axes[1].set_xlabel('Iteration (k)', labelpad=15)
    axes[1].set_ylabel(r'$\mathrm{ave}_j |\langle w_i, M_{2j} \rangle| \, / \, ||w_i||_2$', labelpad=15)
    axes[1].legend(loc='upper left', frameon=True, handlelength=3)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].tick_params(axis='both', which='major', length=8, width=3)
    axes[1].xaxis.set_major_locator(plt.MultipleLocator(2))

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


def main():
    # Paths
    project_root = Path(__file__).parent.parent.parent
    results_dir = project_root / 'results'
    output_dir = Path(__file__).parent / 'outputs'
    output_dir.mkdir(exist_ok=True)

    top_k = 20

    # Load experiments
    experiments = {
        'baseline': results_dir / 'results_baseline.pkl',
        'curriculum_only': results_dir / 'results_curriculum_only.pkl',
        'reg_only': results_dir / 'results_reg_only.pkl',
        'curriculum_reg': results_dir / 'results_curriculum_reg.pkl',
    }

    print("Loading and computing feature ratios...")
    all_data = {}
    for name, path in experiments.items():
        if not path.exists():
            print(f"  Skip: {name} (not found)")
            continue
        print(f"  Loading: {name}")
        results = load_results(path)
        all_data[name] = compute_top_neuron_ratios(results, top_k=top_k)

    # Plot
    print(f"\nPlotting Top {top_k} feature ratios...")
    plot_comparison(all_data, output_dir / f'top{top_k}_feature_ratio.png', top_k=top_k)

    # Print final values
    print("\n" + "=" * 60)
    print(f"Final Feature Ratios (Top {top_k} neurons)")
    print("=" * 60)

    for name, data in all_data.items():
        print(f"{name}:")
        print(f"  Top-{top_k} by M1: M1_ratio = {data['top_m1_m1_ratio'][-1]:.4f}")
        print(f"  Top-{top_k} by M2: M2_ratio = {data['top_m2_m2_ratio'][-1]:.4f}")

    print("=" * 60)


if __name__ == '__main__':
    main()
