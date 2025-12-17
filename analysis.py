"""
Analysis functions for neural similarity
"""
import torch
from typing import Dict
from reg_model import RegDenoiser

def analyze_similarities(model, M1: torch.Tensor, M2: torch.Tensor):
    """
    Compute neural similarity metrics between model weights and dictionaries
    
    Args:
        model: trained denoiser model
        M1, M2: dictionary matrices
        
    Returns:
        dict: similarity matrices for different layer-dictionary combinations
    """
    # Handle distributed models
    net = model.module if hasattr(model, 'module') else model
    
    # Move dictionaries to model device
    device = next(net.parameters()).device
    M1, M2 = M1.to(device), M2.to(device)
    
    # Get model weights
    W, V = net.get_weights()
    
    # Compute absolute dot products (similarities)
    similarities = {
        'W_M1': (W @ M1),      # First layer vs M1
        'W_M2': (W @ M2),      # First layer vs M2
        'VT_M1': (V.T @ M1),   # Output layer vs M1
        'VT_M2': (V.T @ M2)    # Output layer vs M2
    }
    
    return similarities

def compute_metrics(similarities: Dict):
    """
    Extract key metrics from similarity matrices
    
    Args:
        similarities: dict of similarity matrices
        
    Returns:
        dict: computed metrics (max, mean, top neurons)
    """
    metrics = {}
    
    # Compute max and mean for each similarity matrix
    for key, sim_matrix in similarities.items():
        metrics[f'{key}_max'] = sim_matrix.max().item()
        metrics[f'{key}_mean'] = sim_matrix.mean().item()
    
    # Find neurons most aligned with each dictionary
    max_w_m1_idx = torch.argmax(similarities['W_M1'].max(dim=1)[0])
    max_w_m2_idx = torch.argmax(similarities['W_M2'].max(dim=1)[0])
    
    metrics['top_neuron_M1'] = max_w_m1_idx.item()
    metrics['top_neuron_M2'] = max_w_m2_idx.item()
    metrics['top_sim_M1'] = similarities['W_M1'][max_w_m1_idx].max().item()
    metrics['top_sim_M2'] = similarities['W_M2'][max_w_m2_idx].max().item()
    
    return metrics

def print_analysis_results(exp_name: str, metrics: Dict):
    """Print formatted analysis results"""
    print(f"\n[{exp_name}] Neural Similarity Analysis:")
    print(f"  W-M1  max: {metrics['W_M1_max']:.4f} | mean: {metrics['W_M1_mean']:.4f}")
    print(f"  W-M2  max: {metrics['W_M2_max']:.4f} | mean: {metrics['W_M2_mean']:.4f}")
    print(f"  VT-M1 max: {metrics['VT_M1_max']:.4f} | mean: {metrics['VT_M1_mean']:.4f}")
    print(f"  VT-M2 max: {metrics['VT_M2_max']:.4f} | mean: {metrics['VT_M2_mean']:.4f}")
    print(f"  Top neurons - M1: {metrics['top_neuron_M1']} (sim: {metrics['top_sim_M1']:.4f})")
    print(f"                M2: {metrics['top_neuron_M2']} (sim: {metrics['top_sim_M2']:.4f})")


def analyze_by_importance(model, M1: torch.Tensor, M2: torch.Tensor, threshold: float = 1e-3) -> Dict:
    """
    Analyze similarities grouped by neuron importance (for RegDenoiser with Group L1)

    Args:
        model: RegDenoiser model
        M1, M2: dictionary matrices
        threshold: importance threshold to distinguish active/inactive neurons

    Returns:
        dict: analysis results for active neurons
    """
    if not isinstance(model, RegDenoiser):
        return {'error': 'Model is not a RegDenoiser'}

    # Handle distributed models
    net = model.module if hasattr(model, 'module') else model

    # Move dictionaries to model device
    device = next(net.parameters()).device
    M1, M2 = M1.to(device), M2.to(device)

    # Get weights and importance
    W, V = net.get_weights()
    importance = net.get_neuron_importance()
    active_mask = importance > threshold

    num_active = active_mask.sum().item()
    num_total = len(importance)

    if num_active == 0:
        return {
            'num_active': 0,
            'num_total': num_total,
            'error': 'No active neurons found'
        }

    # Compute similarities for active neurons
    W_M1 = W @ M1  # (hidden_dim, d)
    W_M2 = W @ M2

    active_W_M1 = W_M1[active_mask]
    active_W_M2 = W_M2[active_mask]

    return {
        'num_active': int(num_active),
        'num_total': num_total,
        'sparsity': 1.0 - num_active / num_total,
        'active_W_M1_max': active_W_M1.abs().max().item(),
        'active_W_M1_mean': active_W_M1.abs().mean().item(),
        'active_W_M2_max': active_W_M2.abs().max().item(),
        'active_W_M2_mean': active_W_M2.abs().mean().item(),
        'importance_mean': importance[active_mask].mean().item(),
        'importance_max': importance[active_mask].max().item(),
    }


def print_importance_analysis(exp_name: str, results: Dict):
    """Print analysis results by importance"""
    print(f"\n[{exp_name}] Similarity Analysis by Importance:")
    print("-" * 60)

    if 'error' in results and results.get('num_active', 0) == 0:
        print(f"  Error: {results['error']}")
        return

    print(f"  Active neurons: {results['num_active']}/{results['num_total']} "
          f"(sparsity: {results['sparsity']:.1%})")
    print(f"  Active W@M1: max={results['active_W_M1_max']:.4f}, mean={results['active_W_M1_mean']:.4f}")
    print(f"  Active W@M2: max={results['active_W_M2_max']:.4f}, mean={results['active_W_M2_mean']:.4f}")
    print(f"  Importance: mean={results['importance_mean']:.4f}, max={results['importance_max']:.4f}")
    print("-" * 60)
