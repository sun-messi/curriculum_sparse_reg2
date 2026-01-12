#!/usr/bin/env python3
"""
Macro Feature Ratio Comparison

Compare curriculum_only (baseline) vs curriculum_reg by measuring
how much of W aligns with M1 vs M2 features over training.

Metrics:
    M1 Feature Ratio = ||W @ M1||_F^2 / ||W||_F^2
    M2 Feature Ratio = ||W @ M2||_F^2 / ||W||_F^2
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d


def load_results(path):
    """Load pickle file with module compatibility handling."""
    # Add current project path for reg_model
    current_project_path = str(Path(__file__).parent.parent)
    if current_project_path not in sys.path:
        sys.path.insert(0, current_project_path)

    # Add old project path for sparse_model
    old_project_path = '/home/sunj11/Documents/diffusion/curriculum_sparse_ablation'
    if old_project_path not in sys.path:
        sys.path.insert(0, old_project_path)

    with open(path, 'rb') as f:
        return pickle.load(f)


def compute_feature_ratios(results, M1, M2):
    """
    Compute M1/M2 feature ratios from training history.

    Args:
        results: Loaded pickle data
        M1: Feature matrix 1 (input_dim, d1)
        M2: Feature matrix 2 (input_dim, d2)

    Returns:
        iterations: List of iteration numbers
        m1_ratios: List of ||W@M1||^2 / ||W||^2
        m2_ratios: List of ||W@M2||^2 / ||W||^2
    """
    similarities_history = results[0]['training_history']['similarities']

    iterations = []
    m1_ratios = []
    m2_ratios = []

    for entry in similarities_history:
        # Skip entries without model_state
        if 'model_state' not in entry:
            continue

        iteration = entry.get('iteration', entry.get('epoch', len(iterations) + 1))

        # Get W from model_state
        W = entry['model_state']['neurons']['W']  # (hidden_dim, input_dim)

        # Convert to numpy if tensor
        if hasattr(W, 'cpu'):
            W = W.cpu().numpy()
        if hasattr(M1, 'cpu'):
            M1_np = M1.cpu().numpy()
            M2_np = M2.cpu().numpy()
        else:
            M1_np = M1
            M2_np = M2

        # ||W||_F^2
        W_norm_sq = np.sum(W ** 2)

        # ||W @ M1||_F^2
        W_M1 = W @ M1_np
        W_M1_norm_sq = np.sum(W_M1 ** 2)

        # ||W @ M2||_F^2
        W_M2 = W @ M2_np
        W_M2_norm_sq = np.sum(W_M2 ** 2)

        iterations.append(iteration)
        m1_ratios.append(W_M1_norm_sq / W_norm_sq)
        m2_ratios.append(W_M2_norm_sq / W_norm_sq)

    return iterations, m1_ratios, m2_ratios


def plot_feature_ratio_comparison(baseline_data, noise_curr_data, joint_curr_data, reg_only_data, save_path):
    """
    Plot M1/M2 feature ratio comparison for 4 experiments.

    Args:
        baseline_data: (iterations, m1_ratios, m2_ratios) for baseline
        noise_curr_data: (iterations, m1_ratios, m2_ratios) for noise curriculum only
        joint_curr_data: (iterations, m1_ratios, m2_ratios) for joint curriculum
        reg_only_data: (iterations, m1_ratios, m2_ratios) for reg only
        save_path: Path to save the figure
    """
    # Set global style
    plt.rcParams['font.size'] = 32
    plt.rcParams['axes.labelsize'] = 32
    plt.rcParams['axes.titlesize'] = 32
    plt.rcParams['xtick.labelsize'] = 28
    plt.rcParams['ytick.labelsize'] = 28
    plt.rcParams['legend.fontsize'] = 20

    fig, axes = plt.subplots(1, 3, figsize=(36, 10))

    # Unpack data
    base_iters, base_m1, base_m2 = baseline_data
    noise_iters, noise_m1, noise_m2 = noise_curr_data
    joint_iters, joint_m1, joint_m2 = joint_curr_data
    reg_iters, reg_m1, reg_m2 = reg_only_data

    # Convert iterations to k (thousands)
    base_iters_k = [x / 1000 for x in base_iters]
    noise_iters_k = [x / 1000 for x in noise_iters]
    joint_iters_k = [x / 1000 for x in joint_iters]
    reg_iters_k = [x / 1000 for x in reg_iters]

    # 只显示 10 个 marker
    n_markers = 10

    def get_marker_indices(n_points, n_markers):
        """生成包含首尾的 marker 索引列表"""
        if n_points <= n_markers:
            return list(range(n_points))
        indices = [int(i * (n_points - 1) / (n_markers - 1)) for i in range(n_markers)]
        if indices[-1] != n_points - 1:
            indices[-1] = n_points - 1
        return indices

    def smooth(data, window=5):
        """移动平均平滑"""
        return uniform_filter1d(np.array(data), size=window, mode='nearest')

    # 平滑数据
    base_m1 = smooth(base_m1, window=20)
    base_m2 = smooth(base_m2, window=20)
    noise_m1 = smooth(noise_m1, window=5)
    noise_m2 = smooth(noise_m2, window=5)
    joint_m1 = smooth(joint_m1, window=5)
    joint_m2 = smooth(joint_m2, window=5)
    reg_m1 = smooth(reg_m1, window=20)
    reg_m2 = smooth(reg_m2, window=20)

    # 计算噪声比例 (1 - M1 - M2)
    base_noise = 1 - base_m1 - base_m2
    noise_noise = 1 - noise_m1 - noise_m2
    joint_noise = 1 - joint_m1 - joint_m2
    reg_noise = 1 - reg_m1 - reg_m2

    base_markevery = get_marker_indices(len(base_iters), n_markers)
    noise_markevery = get_marker_indices(len(noise_iters), n_markers)
    joint_markevery = get_marker_indices(len(joint_iters), n_markers)
    reg_markevery = get_marker_indices(len(reg_iters), n_markers)

    # 左图: M1
    axes[0].plot(base_iters_k, base_m1, 'b-o', linewidth=6, markersize=18,
                 label='Baseline', markevery=base_markevery)
    axes[0].plot(noise_iters_k, noise_m1, 'g-^', linewidth=6, markersize=18,
                 label='Denoise curriculum', markevery=noise_markevery)
    axes[0].plot(reg_iters_k, reg_m1, 'm-D', linewidth=6, markersize=18,
                 label='Sparsity curriculum', markevery=reg_markevery)
    axes[0].plot(joint_iters_k, joint_m1, 'r-s', linewidth=6, markersize=18,
                 label='Joint curriculum', markevery=joint_markevery)
    axes[0].set_xlabel('Iteration (k)', labelpad=15)
    axes[0].set_ylabel(r'$||W \cdot M_1||^2_F / ||W||^2_F$', labelpad=15)
    axes[0].legend(loc='lower center', handlelength=3, fontsize=28)
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # 右图: M2
    axes[1].plot(base_iters_k, base_m2, 'b-o', linewidth=6, markersize=18,
                 label='Baseline', markevery=base_markevery)
    axes[1].plot(noise_iters_k, noise_m2, 'g-^', linewidth=6, markersize=18,
                 label='Denoise curriculum', markevery=noise_markevery)
    axes[1].plot(reg_iters_k, reg_m2, 'm-D', linewidth=6, markersize=18,
                 label='Sparsity curriculum', markevery=reg_markevery)
    axes[1].plot(joint_iters_k, joint_m2, 'r-s', linewidth=6, markersize=18,
                 label='Joint curriculum', markevery=joint_markevery)
    axes[1].set_xlabel('Iteration (k)', labelpad=15)
    axes[1].set_ylabel(r'$||W \cdot M_2||^2_F / ||W||^2_F$', labelpad=15)
    axes[1].legend(loc='upper left', handlelength=3, fontsize=28)
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # 第三图: Noise (1 - M1 - M2)
    axes[2].plot(base_iters_k, base_noise, 'b-o', linewidth=6, markersize=18,
                 label='Baseline', markevery=base_markevery)
    axes[2].plot(noise_iters_k, noise_noise, 'g-^', linewidth=6, markersize=18,
                 label='Denoise curriculum', markevery=noise_markevery)
    axes[2].plot(reg_iters_k, reg_noise, 'm-D', linewidth=6, markersize=18,
                 label='Sparsity curriculum', markevery=reg_markevery)
    axes[2].plot(joint_iters_k, joint_noise, 'r-s', linewidth=6, markersize=18,
                 label='Joint curriculum', markevery=joint_markevery)
    axes[2].set_xlabel('Iteration (k)', labelpad=15)
    axes[2].set_ylabel(r'$||W \cdot M^\perp||^2_F / ||W||^2_F$', labelpad=15)
    axes[2].legend(loc='upper right', handlelength=3, fontsize=28)
    axes[2].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


def main():
    # Path configuration (relative to visualization/ directory)
    project_root = Path(__file__).parent.parent
    results_dir = project_root / 'results'
    figures_dir = Path(__file__).parent / 'figures'
    figures_dir.mkdir(exist_ok=True)

    # Load data
    print("Loading results...")

    # Four experiments
    baseline_path = results_dir / 'results_baseline.pkl'
    noise_curr_path = results_dir / 'results_curriculum_only.pkl'
    joint_curr_path = results_dir / 'results_curriculum_reg.pkl'
    reg_only_path = results_dir / 'results_reg_only.pkl'

    for path in [baseline_path, noise_curr_path, joint_curr_path, reg_only_path]:
        if not path.exists():
            print(f"Error: {path} not found")
            return

    baseline = load_results(baseline_path)
    noise_curr = load_results(noise_curr_path)
    joint_curr = load_results(joint_curr_path)
    reg_only = load_results(reg_only_path)

    # Get M1, M2 from EACH experiment
    M1_base = baseline[0]['M1']
    M2_base = baseline[0]['M2']
    M1_noise = noise_curr[0]['M1']
    M2_noise = noise_curr[0]['M2']
    M1_joint = joint_curr[0]['M1']
    M2_joint = joint_curr[0]['M2']
    M1_reg = reg_only[0]['M1']
    M2_reg = reg_only[0]['M2']

    print(f"baseline M1 shape: {M1_base.shape}")
    print(f"noise_curriculum M1 shape: {M1_noise.shape}")
    print(f"joint_curriculum M1 shape: {M1_joint.shape}")
    print(f"reg_only M1 shape: {M1_reg.shape}")

    # Compute feature ratios using EACH experiment's own M1/M2
    print("Computing feature ratios...")
    baseline_data = compute_feature_ratios(baseline, M1_base, M2_base)
    noise_curr_data = compute_feature_ratios(noise_curr, M1_noise, M2_noise)
    joint_curr_data = compute_feature_ratios(joint_curr, M1_joint, M2_joint)
    reg_only_data = compute_feature_ratios(reg_only, M1_reg, M2_reg)

    # 找到最小的 max iteration 作为截取点
    max_iters = [baseline_data[0][-1], noise_curr_data[0][-1], joint_curr_data[0][-1], reg_only_data[0][-1]]
    cut_iter = min(max_iters)

    def cut_data(data, cut_iter):
        iters, m1, m2 = data
        cut_idx = len(iters)
        for i, it in enumerate(iters):
            if it > cut_iter:
                cut_idx = i
                break
        return (iters[:cut_idx], m1[:cut_idx], m2[:cut_idx])

    baseline_data = cut_data(baseline_data, cut_iter)
    noise_curr_data = cut_data(noise_curr_data, cut_iter)
    joint_curr_data = cut_data(joint_curr_data, cut_iter)
    reg_only_data = cut_data(reg_only_data, cut_iter)

    print(f"baseline: {len(baseline_data[0])} data points")
    print(f"noise_curriculum: {len(noise_curr_data[0])} data points")
    print(f"joint_curriculum: {len(joint_curr_data[0])} data points")
    print(f"reg_only: {len(reg_only_data[0])} data points")
    print(f"(all cut to iter {cut_iter})")

    # Plot
    print("Plotting...")
    plot_feature_ratio_comparison(
        baseline_data,
        noise_curr_data,
        joint_curr_data,
        reg_only_data,
        figures_dir / 'macro_feature_ratio_comparison.png'
    )

    # Print final values
    print("\n" + "=" * 60)
    print("Final Feature Ratios:")
    print("=" * 60)
    print(f"baseline:            M1={baseline_data[1][-1]:.4f}, M2={baseline_data[2][-1]:.4f}")
    print(f"denoise_curriculum:  M1={noise_curr_data[1][-1]:.4f}, M2={noise_curr_data[2][-1]:.4f}")
    print(f"sparsity_curriculum: M1={reg_only_data[1][-1]:.4f}, M2={reg_only_data[2][-1]:.4f}")
    print(f"joint_curriculum:    M1={joint_curr_data[1][-1]:.4f}, M2={joint_curr_data[2][-1]:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
