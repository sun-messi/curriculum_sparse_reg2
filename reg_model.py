"""
Neural network model for denoising with Group L1 regularization
使用 Group L1 正则化实现软稀疏，无硬掩码
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RegDenoiser(nn.Module):
    """
    Two-layer MLP denoiser with Group L1 regularization

    与硬掩码版本的区别：
    - 无 neuron_mask buffer，不使用硬掩码
    - 通过 Group L1 正则化驱动稀疏
    - Forward pass 直接使用原始权重

    Group L1 正则化形式：
        L_reg = λ_w · Σᵢ ||W[i,:]||_2 + λ_v · Σⱼ ||V[:,j]||_2

    其中：
    - W shape: (hidden_dim, input_dim)，对每行计算 L2 范数
    - V shape: (input_dim, hidden_dim)，对每列计算 L2 范数
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        初始化 RegDenoiser

        Args:
            input_dim: 输入维度 (d1)
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Weight layers
        self.W = nn.Linear(input_dim, hidden_dim)   # First layer: (hidden_dim, input_dim)
        self.V = nn.Linear(hidden_dim, input_dim)   # Second layer: (input_dim, hidden_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small normal distribution"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                nn.init.normal_(m.bias, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - 无掩码，直接计算

        Args:
            x: input tensor (batch_size, input_dim)

        Returns:
            output tensor (batch_size, input_dim)
        """
        h = F.relu(self.W(x))
        return self.V(h)

    # ========== Group L1 正则化方法 ==========

    def get_row_norms(self) -> torch.Tensor:
        """
        计算 W 的每行 L2 范数

        W shape: (hidden_dim, input_dim)
        对每行计算范数，返回 (hidden_dim,)

        Returns:
            tensor of shape (hidden_dim,) - 每个神经元在 W 层的范数
        """
        return torch.norm(self.W.weight, p=2, dim=1)

    def get_col_norms(self) -> torch.Tensor:
        """
        计算 V 的每列 L2 范数

        V shape: (input_dim, hidden_dim)
        对每列计算范数，返回 (hidden_dim,)

        Returns:
            tensor of shape (hidden_dim,) - 每个神经元在 V 层的范数
        """
        return torch.norm(self.V.weight, p=2, dim=0)

    def get_group_l1_penalty(self, lambda_w: float, lambda_v: float) -> torch.Tensor:
        """
        计算 Group L1 正则化惩罚项

        L_reg = λ_w · Σᵢ ||W[i,:]||_2 + λ_v · Σⱼ ||V[:,j]||_2

        Args:
            lambda_w: W 层的正则化系数
            lambda_v: V 层的正则化系数

        Returns:
            标量张量 - Group L1 惩罚值
        """
        row_norms = self.get_row_norms()  # (hidden_dim,)
        col_norms = self.get_col_norms()  # (hidden_dim,)

        penalty = lambda_w * row_norms.sum() + lambda_v * col_norms.sum()
        return penalty

    # ========== 稀疏性分析方法 ==========

    def get_neuron_importance(self) -> torch.Tensor:
        """
        获取每个神经元的综合重要性（W行范数 + V列范数）

        重要性越高，神经元越活跃

        Returns:
            tensor of shape (hidden_dim,) - 每个神经元的重要性分数
        """
        row_norms = self.get_row_norms()
        col_norms = self.get_col_norms()
        return row_norms + col_norms

    def get_active_neurons(self, threshold: float = 1e-3) -> int:
        """
        统计 importance > threshold 的神经元数

        Args:
            threshold: 判断神经元是否活跃的阈值

        Returns:
            活跃神经元数量
        """
        importance = self.get_neuron_importance()
        return (importance > threshold).sum().item()

    def get_current_sparsity(self, threshold: float = 1e-3) -> float:
        """
        获取当前稀疏度（importance <= threshold 的神经元比例）

        Returns:
            稀疏度 (0 = 全部活跃, 1 = 全部静默)
        """
        active = self.get_active_neurons(threshold)
        return 1.0 - (active / self.hidden_dim)

    def get_neuron_stats(self, threshold: float = 1e-3) -> dict:
        """
        获取神经元统计信息

        Args:
            threshold: 判断神经元活跃的阈值

        Returns:
            dict: 详细的神经元统计信息
        """
        importance = self.get_neuron_importance()
        row_norms = self.get_row_norms()
        col_norms = self.get_col_norms()

        return {
            'total_neurons': self.hidden_dim,
            'active_neurons': self.get_active_neurons(threshold),
            'sparsity': self.get_current_sparsity(threshold),
            'importance_mean': importance.mean().item(),
            'importance_std': importance.std().item(),
            'importance_max': importance.max().item(),
            'importance_min': importance.min().item(),
            'row_norms_mean': row_norms.mean().item(),
            'row_norms_std': row_norms.std().item(),
            'col_norms_mean': col_norms.mean().item(),
            'col_norms_std': col_norms.std().item(),
        }

    # ========== 兼容性接口 ==========

    def get_weights(self):
        """
        Get model weights for analysis (与其他模型接口兼容)

        Returns:
            W: first layer weights (hidden_dim, input_dim)
            V: second layer weights (input_dim, hidden_dim)
        """
        return self.W.weight, self.V.weight

    def get_raw_weights(self):
        """Get raw model weights (same as get_weights for this model)"""
        return self.W.weight, self.V.weight

    def get_biases(self):
        """Get bias vectors"""
        return self.W.bias, self.V.bias

    def print_reg_info(self, threshold: float = 1e-3):
        """Print current regularization-induced sparsity information"""
        stats = self.get_neuron_stats(threshold)
        print(f"    Sparsity (threshold={threshold}): {stats['sparsity']:.1%}")
        print(f"    Active neurons: {stats['active_neurons']}/{stats['total_neurons']}")
        print(f"    Importance: mean={stats['importance_mean']:.4f}, "
              f"std={stats['importance_std']:.4f}, "
              f"range=[{stats['importance_min']:.4f}, {stats['importance_max']:.4f}]")
        print(f"    W row norms: mean={stats['row_norms_mean']:.4f}, std={stats['row_norms_std']:.4f}")
        print(f"    V col norms: mean={stats['col_norms_mean']:.4f}, std={stats['col_norms_std']:.4f}")
