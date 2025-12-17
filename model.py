"""
Neural network model for denoising
"""
import torch
import torch.nn as nn

class Denoiser(nn.Module):
    """
    Two-layer MLP denoiser
    Input: noisy signal x_t
    Output: denoised signal (x_clean or epsilon)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim)
        )
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small normal distribution"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                nn.init.normal_(m.bias, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor):
        return self.net(x)
    
    def get_weights(self):
        """
        Get model weights for analysis
        
        Returns:
            W: first layer weights (hidden_dim × input_dim)
            V: second layer weights (input_dim × hidden_dim)
        """
        W = self.net[0].weight  # (hidden_dim × input_dim)
        V = self.net[2].weight  # (input_dim × hidden_dim)
        return W, V
    
    def get_biases(self):
        """获取偏置向量"""
        b_hidden = self.net[0].bias  # (hidden_dim,)
        b_output = self.net[2].bias  # (input_dim,)
        return b_hidden, b_output
