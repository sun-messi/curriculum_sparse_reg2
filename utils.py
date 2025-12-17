"""
Utility functions for diffusion experiments
"""
import random
import numpy as np
import torch

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_dictionaries(d1: int, d: int):
    """
    Build orthogonal dictionaries M1 and M2
    
    Args:
        d1: ambient dimension
        d: latent dimension
        
    Returns:
        M1, M2: orthogonal dictionary matrices
    """
    M_total, _ = torch.qr(torch.randn(d1, 2*d))
    M1 = M_total[:, :d]      # (d1 × d)
    M2 = M_total[:, d:]      # (d1 × d)
    return M1, M2

def print_experiment_header(exp_name: str, t_start: float, t_end: float):
    """Print formatted experiment header"""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"Time range: [{t_start:.1f}, {t_end:.2f}]")
    print(f"{'='*60}")

def print_summary_table(results):
    """Print comparison summary table"""
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    print(f"{'Experiment':<12} {'Val MSE':<10} {'W-M1 max':<10} {'W-M2 max':<10}")
    print("-" * 50)
    
    for result in results:
        metrics = result['metrics']
        print(f"{result['name']:<12} "
              f"{result['val_mse']:<10.6f} "
              f"{metrics['W_M1_max']:<10.4f} "
              f"{metrics['W_M2_max']:<10.4f}")
