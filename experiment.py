"""
Experiment runner for diffusion experiments with Group L1 regularization
"""
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from model import Denoiser
from reg_model import RegDenoiser
from dataset import DiffusionDataset
from training import (
    train_model, validate_model, print_training_info,
    train_curriculum_model, train_curriculum_model_with_reg
)
from analysis import analyze_similarities, compute_metrics, print_analysis_results
from utils import set_seed, print_experiment_header
from ddp_utils import is_main_process, get_device


def run_experiment(t_start: float, t_end: float, exp_name: str,
                   config, M1: torch.Tensor, M2: torch.Tensor,
                   rank: int = 0, world_size: int = 1):
    """
    Run a single diffusion experiment

    Supports:
    - Standard training (curriculum_enabled=False, reg_enabled=False)
    - Curriculum only (curriculum_enabled=True, reg_enabled=False)
    - Group L1 regularization curriculum (curriculum_enabled=True, reg_enabled=True)
    """
    device = get_device(rank) if world_size > 1 else config.device

    if is_main_process(rank):
        print_experiment_header(exp_name, t_start, t_end)

    set_seed(config.seed + rank)

    # Check enabled modes
    reg_enabled = getattr(config, 'reg_enabled', False)

    # Create model
    if reg_enabled:
        # Group L1 模式
        model = RegDenoiser(config.d1, config.hidden_dim).to(device)
        if is_main_process(rank):
            print(f"\n[{exp_name}] Using RegDenoiser with Group L1 regularization")
            print(f"[{exp_name}] Lambda max: {config.lambda_max}")
    else:
        model = Denoiser(config.d1, config.hidden_dim).to(device)
        if is_main_process(rank):
            print(f"\n[{exp_name}] Using standard Denoiser")

    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
        if is_main_process(rank):
            print(f"[{exp_name}] Using DistributedDataParallel with {world_size} GPUs")

    print_training_info(config, exp_name, rank=rank)

    # Choose training method
    if config.curriculum_enabled and reg_enabled:
        # Curriculum + Group L1 正则化
        if is_main_process(rank):
            print(f"\n[{exp_name}] Using Curriculum Learning + Group L1 Regularization")
        training_history = train_curriculum_model_with_reg(
            model, config, exp_name, M1, M2, t_start, t_end,
            rank=rank, world_size=world_size
        )
    elif config.curriculum_enabled:
        # Curriculum only
        if is_main_process(rank):
            print(f"\n[{exp_name}] Using Curriculum Learning only")
        training_history = train_curriculum_model(
            model, config, exp_name, M1, M2, t_start, t_end,
            rank=rank, world_size=world_size
        )
    elif reg_enabled:
        # Reg only (no curriculum) - lambda decays from lambda_max to 0 over first 30 epochs
        if is_main_process(rank):
            print(f"\n[{exp_name}] Using Group L1 Regularization only (no curriculum)")
            print(f"[{exp_name}] Lambda schedule: {config.lambda_max} -> 0 over 30 epochs")
        dataset = DiffusionDataset(config, M1, M2, t_start, t_end)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        training_history = train_model(
            model, dataloader, config, exp_name, M1, M2,
            rank=rank, world_size=world_size,
            reg_lambda=config.lambda_max,
            use_lambda_schedule=True
        )
    else:
        # Standard training (baseline)
        if is_main_process(rank):
            print(f"\n[{exp_name}] Using Standard Training")
        dataset = DiffusionDataset(config, M1, M2, t_start, t_end)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        training_history = train_model(
            model, dataloader, config, exp_name, M1, M2,
            rank=rank, world_size=world_size
        )

    # Get underlying model
    net = model.module if hasattr(model, 'module') else model

    # Validation
    if is_main_process(rank):
        val_dataset = DiffusionDataset(config, M1, M2, t_start, t_end)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        val_mse = validate_model(model, val_dataloader, config, rank=rank)
        print(f"\n[{exp_name}] Validation MSE: {val_mse:.6f}")

        # Neural similarity analysis
        similarities = analyze_similarities(net, M1, M2)
        metrics = compute_metrics(similarities)
        print_analysis_results(exp_name, metrics)

        # Additional analysis for Group L1
        if isinstance(net, RegDenoiser):
            print(f"\n[{exp_name}] Group L1 Regularization Analysis:")
            net.print_reg_info(threshold=config.reg_threshold)

            # Analyze by importance
            importance = net.get_neuron_importance()
            W, V = net.get_weights()
            active_mask = importance > config.reg_threshold

            if active_mask.sum() > 0:
                # 确保 M1, M2 在正确的设备上
                M1_dev = M1.to(device)
                M2_dev = M2.to(device)
                W_M1 = W @ M1_dev
                W_M2 = W @ M2_dev
                active_W_M1 = W_M1[active_mask]
                active_W_M2 = W_M2[active_mask]
                print(f"    Active neurons alignment:")
                print(f"      W@M1: max={active_W_M1.abs().max().item():.4f}, mean={active_W_M1.abs().mean().item():.4f}")
                print(f"      W@M2: max={active_W_M2.abs().max().item():.4f}, mean={active_W_M2.abs().mean().item():.4f}")
    else:
        val_mse = 0.0
        similarities = {}
        metrics = {}

    # Return results
    return {
        'name': exp_name,
        'val_mse': val_mse,
        'metrics': metrics,
        'model': net,
        'similarities': similarities,
        'training_history': training_history,
        'is_reg': isinstance(net, RegDenoiser),
    }
