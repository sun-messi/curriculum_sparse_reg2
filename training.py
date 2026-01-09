"""
Training and validation functions with Group L1 regularization
使用 Group L1 正则化实现软稀疏
支持 DDP 多 GPU 训练
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataset import DiffusionDataset
from reg_model import RegDenoiser
from ddp_utils import is_main_process, get_device


class FilteredDiffusionDataset(torch.utils.data.Dataset):
    """
    A filtered view of a DiffusionDataset that only returns samples within a specific time range
    """
    def __init__(self, full_dataset, t_start: float, t_end: float, batch_size: int = None):
        self.full_dataset = full_dataset
        self.t_start = t_start
        self.t_end = t_end
        self.batch_size = batch_size

        min_t = min(t_start, t_end)
        max_t = max(t_start, t_end)

        self.valid_indices = []
        for i in range(len(full_dataset)):
            _, _, t, _ = full_dataset[i]
            t_val = t.item()
            if min_t <= t_val <= max_t:
                self.valid_indices.append(i)

        print(f"    FilteredDataset: t ∈ [{t_start:.3f}, {t_end:.3f}] -> {len(self.valid_indices)} samples", end="")

        if len(self.valid_indices) == 0:
            print()
            print(f"    ⚠️  No samples found in exact range, using backup strategy...")

            distances = []
            for i in range(len(full_dataset)):
                _, _, t, _ = full_dataset[i]
                t_val = t.item()
                if t_val < min_t:
                    dist = min_t - t_val
                elif t_val > max_t:
                    dist = t_val - max_t
                else:
                    dist = 0
                distances.append((dist, i))

            distances.sort()

            if batch_size is not None:
                min_samples = batch_size * 10
            else:
                min_samples = 1000

            min_samples = max(min_samples, len(full_dataset) // 10)
            min_samples = min(min_samples, len(full_dataset))

            self.valid_indices = [idx for _, idx in distances[:min_samples]]
            print(f"    → Using {len(self.valid_indices)} closest samples")
        else:
            if batch_size and len(self.valid_indices) < batch_size * 2:
                print(f" ⚠️  Only {len(self.valid_indices) // batch_size} batches")
            else:
                if batch_size:
                    print(f" ({len(self.valid_indices) // batch_size} batches)")
                else:
                    print()

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        return self.full_dataset[actual_idx]


class CombinedDiffusionDataset(torch.utils.data.Dataset):
    """Combine multiple DiffusionDataset instances"""
    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative_sizes = self._get_cumulative_sizes()

    def _get_cumulative_sizes(self):
        cumulative_sizes = []
        cumsum = 0
        for dataset in self.datasets:
            cumsum += len(dataset)
            cumulative_sizes.append(cumsum)
        return cumulative_sizes

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx

        dataset_idx = 0
        for i, cumulative_size in enumerate(self.cumulative_sizes):
            if idx < cumulative_size:
                dataset_idx = i
                break

        if dataset_idx > 0:
            idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx][idx]


def train_stage(model, dataloader, config, exp_name, epochs,
               start_iteration, stage_idx, M1, M2, track_similarities, similarity_freq,
               reg_lambda: float = 0.0,
               rank: int = 0, world_size: int = 1, sampler=None):
    """
    Train model for one curriculum stage with Group L1 regularization

    Args:
        reg_lambda: Group L1 正则化系数 (λ_w = λ_v = reg_lambda)
        sampler: DistributedSampler for DDP
    """
    from analysis import analyze_similarities

    device = get_device(rank) if world_size > 1 else config.device

    lr = getattr(config, 'lr', 1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Warmup scheduler
    warmup_enabled = getattr(config, 'warmup_enabled', False)
    scheduler = None
    if warmup_enabled:
        from torch.optim.lr_scheduler import LinearLR
        warmup_epochs = getattr(config, 'warmup_epochs', 2)
        warmup_start_factor = getattr(config, 'warmup_start_factor', 0.1)
        if world_size > 1:
            warmup_epochs = warmup_epochs * world_size
        warmup_steps = warmup_epochs * len(dataloader)
        scheduler = LinearLR(optimizer, start_factor=warmup_start_factor, total_iters=warmup_steps)
        if is_main_process(rank):
            print(f"[{exp_name}] Warmup enabled: {warmup_epochs} epochs")

    model.train()
    net = model.module if hasattr(model, 'module') else model

    stage_history = {
        'losses': [],
        'mse_losses': [],
        'reg_losses': [],
        'similarities': [],
        'sparsity_history': [],
        'final_iteration': start_iteration,
        'final_loss': 0.0
    }

    iteration_count = start_iteration

    if is_main_process(rank):
        print(f"[{exp_name}] Stage {stage_idx + 1}: λ = {reg_lambda:.6f}")

    for epoch in range(1, epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        total_loss = 0.0
        total_mse = 0.0
        total_reg = 0.0
        num_batches = len(dataloader)

        for batch_idx, (x_t, x_clean, t, eps) in enumerate(dataloader):
            x_t = x_t.to(device)
            x_clean = x_clean.to(device)
            eps = eps.to(device)

            # Forward pass
            pred = model(x_t)
            target = x_clean if config.predict_mode == "x" else eps

            # MSE Loss
            mse_loss = F.mse_loss(pred, target)

            # Group L1 Regularization
            if reg_lambda > 0 and isinstance(net, RegDenoiser):
                reg_loss = net.get_group_l1_penalty(reg_lambda, reg_lambda)
                loss = mse_loss + reg_loss
            else:
                reg_loss = torch.tensor(0.0, device=device)
                loss = mse_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_reg += reg_loss.item()
            iteration_count += 1

            # Track similarities
            if track_similarities and iteration_count % similarity_freq == 0 and is_main_process(rank):
                model.eval()
                with torch.no_grad():
                    similarities = analyze_similarities(net, M1, M2)
                    stage_history['similarities'].append({
                        'iteration': iteration_count,
                        'stage': stage_idx,
                        'epoch': epoch,
                        'batch': batch_idx,
                        'loss': loss.item(),
                        'mse_loss': mse_loss.item(),
                        'reg_loss': reg_loss.item(),
                        'similarities': similarities
                    })
                model.train()

        # Epoch statistics
        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_reg = total_reg / num_batches

        stage_history['losses'].append({
            'iteration': iteration_count,
            'stage': stage_idx,
            'epoch': epoch,
            'avg_loss': avg_loss,
            'avg_mse': avg_mse,
            'avg_reg': avg_reg
        })

        # Record sparsity at epoch end
        if is_main_process(rank) and isinstance(net, RegDenoiser):
            threshold = getattr(config, 'reg_threshold', 1e-3)
            sparsity_stats = net.get_neuron_stats(threshold)
            stage_history['sparsity_history'].append({
                'iteration': iteration_count,
                'stage': stage_idx,
                'epoch': epoch,
                **sparsity_stats
            })

            print(f"[{exp_name}] Stage {stage_idx + 1} - Epoch {epoch:02d}/{epochs} | "
                  f"Loss: {avg_loss:.6f} (MSE: {avg_mse:.6f}, Reg: {avg_reg:.6f}) | "
                  f"Active: {sparsity_stats['active_neurons']}/{sparsity_stats['total_neurons']}")
        elif is_main_process(rank):
            print(f"[{exp_name}] Stage {stage_idx + 1} - Epoch {epoch:02d}/{epochs} | Loss: {avg_loss:.6f}")

        # Record similarities at epoch end
        if track_similarities and is_main_process(rank):
            model.eval()
            with torch.no_grad():
                similarities = analyze_similarities(net, M1, M2)

                W, V = net.get_weights()
                if hasattr(net, 'net'):
                    b_hidden = net.net[0].bias
                    b_output = net.net[2].bias
                else:
                    b_hidden = net.W.bias
                    b_output = net.V.bias

                model_state = {
                    'weights': {name: param.clone().detach().cpu()
                                for name, param in net.state_dict().items()},
                    'neurons': {
                        'W': W.clone().detach().cpu(),
                        'V': V.clone().detach().cpu(),
                        'b_hidden': b_hidden.clone().detach().cpu(),
                        'b_output': b_output.clone().detach().cpu()
                    }
                }

                stage_history['similarities'].append({
                    'iteration': iteration_count,
                    'stage': stage_idx,
                    'epoch': epoch,
                    'batch': 'epoch_end',
                    'loss': avg_loss,
                    'mse_loss': avg_mse,
                    'reg_loss': avg_reg,
                    'similarities': similarities,
                    'model_state': model_state
                })
            model.train()

    stage_history['final_iteration'] = iteration_count
    stage_history['final_loss'] = avg_loss
    return stage_history


def train_curriculum_model_with_reg(model, config, exp_name: str, M1, M2, t_start_full, t_end_full,
                                    rank: int = 0, world_size: int = 1):
    """
    Train model using curriculum learning with Group L1 regularization

    核心特点：
    - 使用 Group L1 正则化驱动稀疏
    - λ 随 stage 递减（从大到小）
    """
    from analysis import analyze_similarities

    device = get_device(rank) if world_size > 1 else config.device
    net = model.module if hasattr(model, 'module') else model

    if is_main_process(rank):
        print(f"[{exp_name}] Curriculum Learning with Group L1 Regularization started...")
        print(f"[{exp_name}] Total stages: {len(config.curriculum_stages)}")
        print(f"[{exp_name}] Epochs per stage: {config.curriculum_epochs_per_stage}")
        print(f"[{exp_name}] Lambda max: {config.lambda_max}")
        print(f"[{exp_name}] Lambda schedule type: {config.lambda_schedule_type}")
        print(f"[{exp_name}] Lambda schedule: {[f'{l:.4f}' for l in config.lambda_schedule]}")

    # Pre-generate full dataset
    if is_main_process(rank):
        print(f"[{exp_name}] Pre-generating full dataset...")
    full_dataset = DiffusionDataset(config, M1, M2, t_start_full, t_end_full)
    if is_main_process(rank):
        print(f"[{exp_name}] Full dataset size: {len(full_dataset):,} samples")

    # Create stage datasets
    if is_main_process(rank):
        print(f"[{exp_name}] Creating {len(config.curriculum_stages)} stage datasets...")
    stage_datasets = []
    for stage_idx, (t_start, t_end, stage_name) in enumerate(config.curriculum_stages):
        if is_main_process(rank):
            print(f"  Stage {stage_idx+1}/{len(config.curriculum_stages)}: {stage_name}")
        stage_dataset = FilteredDiffusionDataset(
            full_dataset, t_start, t_end, batch_size=config.batch_size
        )
        stage_datasets.append(stage_dataset)

    if is_main_process(rank):
        print()

    # Training history
    training_history = {
        'losses': [],
        'similarities': [],
        'stages': [],
        'sparsity_history': [],
        'lambda_history': []
    }

    track_similarities = M1 is not None and M2 is not None
    similarity_freq = getattr(config, 'similarity_freq', 50)

    global_iteration = 0
    accumulated_datasets = []
    max_accumulated_stages = getattr(config, 'max_accumulated_stages', 3)

    # Train through curriculum stages
    for stage_idx, (t_start, t_end, stage_name) in enumerate(config.curriculum_stages):
        if is_main_process(rank):
            print(f"\n{'='*60}")
            print(f"[{exp_name}] CURRICULUM STAGE {stage_idx + 1}/{len(config.curriculum_stages)}")
            print(f"[{exp_name}] Stage: {stage_name} - Time range: [{t_start:.1f}, {t_end:.2f}]")
            print(f"{'='*60}")

        # Get lambda for this stage
        reg_lambda = config.get_lambda_for_stage(stage_idx)
        training_history['lambda_history'].append({
            'stage': stage_idx,
            'lambda': reg_lambda
        })

        # Dataset accumulation
        stage_dataset = stage_datasets[stage_idx]
        accumulated_datasets.append(stage_dataset)
        if len(accumulated_datasets) > max_accumulated_stages:
            accumulated_datasets = accumulated_datasets[-max_accumulated_stages:]

        if len(accumulated_datasets) == 1:
            combined_dataset = accumulated_datasets[0]
        else:
            combined_dataset = CombinedDiffusionDataset(accumulated_datasets)

        start_stage_idx = max(0, stage_idx - max_accumulated_stages + 1)
        included_stages = list(range(start_stage_idx, stage_idx + 1))

        if is_main_process(rank):
            print(f"[{exp_name}] Using data from stages: {[i+1 for i in included_stages]}")
            print(f"[{exp_name}] Combined dataset size: {len(combined_dataset):,} samples")

        # Create DataLoader
        if world_size > 1:
            sampler = DistributedSampler(combined_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            dataloader = DataLoader(combined_dataset, batch_size=config.batch_size,
                                    sampler=sampler, num_workers=0, pin_memory=True)
        else:
            sampler = None
            dataloader = DataLoader(combined_dataset, batch_size=config.batch_size,
                                    shuffle=True, num_workers=4, pin_memory=True)

        if is_main_process(rank):
            print(f"[{exp_name}] DataLoader: {len(dataloader)} batches")

        # Train for this stage (无 regrowth！)
        is_last_stage = (stage_idx == len(config.curriculum_stages) - 1)
        epochs_for_stage = config.curriculum_epochs_per_stage * 2 if is_last_stage else config.curriculum_epochs_per_stage

        if is_last_stage and is_main_process(rank):
            print(f"[{exp_name}] Last stage - training for {epochs_for_stage} epochs (double)")

        stage_history = train_stage(
            model, dataloader, config, exp_name,
            epochs_for_stage, global_iteration, stage_idx, M1, M2,
            track_similarities, similarity_freq,
            reg_lambda=reg_lambda,
            rank=rank, world_size=world_size, sampler=sampler
        )

        global_iteration = stage_history['final_iteration']

        # Record stage info
        training_history['stages'].append({
            'stage_idx': stage_idx,
            'stage_name': stage_name,
            'time_range': (t_start, t_end),
            'dataset_size': len(combined_dataset),
            'included_stages': included_stages,
            'final_loss': stage_history['final_loss'],
            'epochs_trained': epochs_for_stage,
            'lambda': reg_lambda
        })

        training_history['losses'].extend(stage_history['losses'])
        training_history['similarities'].extend(stage_history['similarities'])
        training_history['sparsity_history'].extend(stage_history.get('sparsity_history', []))

        # Print reg info
        if isinstance(net, RegDenoiser) and is_main_process(rank):
            net.print_reg_info()

        if is_main_process(rank):
            print(f"[{exp_name}] Stage {stage_idx + 1} completed. Final loss: {stage_history['final_loss']:.6f}")

    if is_main_process(rank):
        print(f"\n[{exp_name}] All curriculum stages completed!")
        if isinstance(net, RegDenoiser):
            print(f"\n[{exp_name}] Final Regularization Summary:")
            net.print_reg_info()

    return training_history


def train_curriculum_model(model, config, exp_name: str, M1, M2, t_start_full, t_end_full,
                          rank: int = 0, world_size: int = 1):
    """
    Train model using curriculum learning (without sparsity regularization)
    """
    from analysis import analyze_similarities

    device = get_device(rank) if world_size > 1 else config.device

    if is_main_process(rank):
        print(f"[{exp_name}] Curriculum Learning Training started...")
        print(f"[{exp_name}] Total stages: {len(config.curriculum_stages)}")
        print(f"[{exp_name}] Epochs per stage: {config.curriculum_epochs_per_stage}")

    full_dataset = DiffusionDataset(config, M1, M2, t_start_full, t_end_full)
    if is_main_process(rank):
        print(f"[{exp_name}] Full dataset size: {len(full_dataset):,} samples")

    stage_datasets = []
    for stage_idx, (t_start, t_end, stage_name) in enumerate(config.curriculum_stages):
        if is_main_process(rank):
            print(f"  Stage {stage_idx+1}/{len(config.curriculum_stages)}: {stage_name}")
        stage_dataset = FilteredDiffusionDataset(
            full_dataset, t_start, t_end, batch_size=config.batch_size
        )
        stage_datasets.append(stage_dataset)

    training_history = {
        'losses': [],
        'similarities': [],
        'stages': []
    }

    track_similarities = M1 is not None and M2 is not None
    similarity_freq = getattr(config, 'similarity_freq', 50)

    global_iteration = 0
    accumulated_datasets = []
    max_accumulated_stages = getattr(config, 'max_accumulated_stages', 3)

    for stage_idx, (t_start, t_end, stage_name) in enumerate(config.curriculum_stages):
        if is_main_process(rank):
            print(f"\n{'='*60}")
            print(f"[{exp_name}] CURRICULUM STAGE {stage_idx + 1}/{len(config.curriculum_stages)}")
            print(f"{'='*60}")

        stage_dataset = stage_datasets[stage_idx]
        accumulated_datasets.append(stage_dataset)
        if len(accumulated_datasets) > max_accumulated_stages:
            accumulated_datasets = accumulated_datasets[-max_accumulated_stages:]

        if len(accumulated_datasets) == 1:
            combined_dataset = accumulated_datasets[0]
        else:
            combined_dataset = CombinedDiffusionDataset(accumulated_datasets)

        if world_size > 1:
            sampler = DistributedSampler(combined_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            dataloader = DataLoader(combined_dataset, batch_size=config.batch_size,
                                    sampler=sampler, num_workers=0, pin_memory=True)
        else:
            sampler = None
            dataloader = DataLoader(combined_dataset, batch_size=config.batch_size,
                                    shuffle=True, num_workers=4, pin_memory=True)

        is_last_stage = (stage_idx == len(config.curriculum_stages) - 1)
        epochs_for_stage = config.curriculum_epochs_per_stage * 2 if is_last_stage else config.curriculum_epochs_per_stage

        stage_history = train_stage(
            model, dataloader, config, exp_name,
            epochs_for_stage, global_iteration, stage_idx, M1, M2,
            track_similarities, similarity_freq,
            reg_lambda=0.0,  # No regularization
            rank=rank, world_size=world_size, sampler=sampler
        )

        global_iteration = stage_history['final_iteration']

        training_history['stages'].append({
            'stage_idx': stage_idx,
            'stage_name': stage_name,
            'time_range': (t_start, t_end),
            'dataset_size': len(combined_dataset),
            'final_loss': stage_history['final_loss'],
            'epochs_trained': epochs_for_stage
        })

        training_history['losses'].extend(stage_history['losses'])
        training_history['similarities'].extend(stage_history['similarities'])

        if is_main_process(rank):
            print(f"[{exp_name}] Stage {stage_idx + 1} completed. Final loss: {stage_history['final_loss']:.6f}")

    if is_main_process(rank):
        print(f"\n[{exp_name}] All curriculum stages completed!")

    return training_history


def train_model(model, dataloader: DataLoader, config, exp_name: str, M1=None, M2=None,
                rank: int = 0, world_size: int = 1, sampler=None):
    """Train the denoiser model (non-curriculum version)"""
    from analysis import analyze_similarities

    device = get_device(rank) if world_size > 1 else config.device
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    net = model.module if hasattr(model, 'module') else model

    if is_main_process(rank):
        print(f"[{exp_name}] Training started...")

    training_history = {
        'losses': [],
        'similarities': []
    }

    track_similarities = M1 is not None and M2 is not None
    similarity_iter_freq = 30  # 每30个iteration保存一次，与curriculum_reg一致
    next_save_iter = similarity_iter_freq

    iteration_count = 0

    for epoch in range(1, config.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        total_loss = 0.0
        num_batches = len(dataloader)

        for batch_idx, (x_t, x_clean, t, eps) in enumerate(dataloader):
            x_t = x_t.to(device)
            x_clean = x_clean.to(device)
            eps = eps.to(device)

            pred = model(x_t)
            target = x_clean if config.predict_mode == "x" else eps
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            iteration_count += 1

            # Track similarities every similarity_iter_freq iterations
            if track_similarities and iteration_count >= next_save_iter and is_main_process(rank):
                model.eval()
                with torch.no_grad():
                    similarities = analyze_similarities(net, M1, M2)

                    # Get W, V weights
                    if hasattr(net, 'W'):
                        W, V = net.W.weight, net.V.weight
                        b_hidden = net.W.bias
                        b_output = net.V.bias
                    else:
                        W = net.net[0].weight
                        V = net.net[2].weight
                        b_hidden = net.net[0].bias
                        b_output = net.net[2].bias

                    model_state = {
                        'weights': {name: param.clone().detach().cpu()
                                    for name, param in net.state_dict().items()},
                        'neurons': {
                            'W': W.clone().detach().cpu(),
                            'V': V.clone().detach().cpu(),
                            'b_hidden': b_hidden.clone().detach().cpu(),
                            'b_output': b_output.clone().detach().cpu()
                        }
                    }

                    training_history['similarities'].append({
                        'iteration': iteration_count,
                        'epoch': epoch,
                        'batch': batch_idx,
                        'loss': loss.item(),
                        'similarities': similarities,
                        'model_state': model_state
                    })
                model.train()
                next_save_iter += similarity_iter_freq

        avg_loss = total_loss / num_batches
        training_history['losses'].append({
            'iteration': iteration_count,
            'epoch': epoch,
            'avg_loss': avg_loss
        })
        if is_main_process(rank):
            print(f"[{exp_name}] Epoch {epoch:02d}/{config.epochs} | Loss: {avg_loss:.6f} | Iter: {iteration_count}")

    return training_history


def validate_model(model, dataloader: DataLoader, config, rank: int = 0):
    """Validate the trained model"""
    device = get_device(rank) if rank > 0 else config.device
    model.eval()

    with torch.no_grad():
        x_t, x_clean, _, eps = next(iter(dataloader))
        x_t = x_t.to(device)
        x_clean = x_clean.to(device)
        eps = eps.to(device)

        pred = model(x_t)
        target = x_clean if config.predict_mode == "x" else eps
        mse = F.mse_loss(pred, target).item()

    return mse


def print_training_info(config, exp_name: str, rank: int = 0):
    """Print training configuration"""
    if not is_main_process(rank):
        return

    print(f"[{exp_name}] Training Configuration:")
    if config.curriculum_enabled:
        print(f"  Curriculum Learning: ENABLED")
        print(f"  Total stages: {len(config.curriculum_stages)}")
        print(f"  Epochs per stage: {config.curriculum_epochs_per_stage}")
    else:
        print(f"  Curriculum Learning: DISABLED")
        print(f"  Total epochs: {config.epochs}")

    if getattr(config, 'reg_enabled', False):
        print(f"  Group L1 Regularization: ENABLED")
        print(f"  Lambda max: {config.lambda_max}")
        print(f"  Lambda schedule: {config.lambda_schedule_type}")
        print(f"  Reg threshold: {config.reg_threshold}")
    else:
        print(f"  Group L1 Regularization: DISABLED")

    print(f"  Dataset size per stage: {config.n_clean * config.n_steps:,} samples")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Predict mode: {config.predict_mode}")
