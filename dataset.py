"""
Dataset class for diffusion experiments
"""
import math
import torch
from torch.utils.data import Dataset

class DiffusionDataset(Dataset):
    """
    Multi-step diffusion dataset
    Returns: (x_noisy, x_clean, t, eps)
    """
    
    def __init__(self, config, M1: torch.Tensor, M2: torch.Tensor, 
                 t_start: float, t_end: float):
        self.config = config
        self.n_clean = config.n_clean
        self.n_steps = config.n_steps
        
        # Generate clean signals
        self.x_clean_all = self._generate_clean_signals(M1, M2)
        
        # Generate time values
        self.t_vals = self._generate_time_values(t_start, t_end)
    
    def _generate_clean_signals(self, M1: torch.Tensor, M2: torch.Tensor):
        """
        Generate clean signals x = α1·M1·z1 + α2·M2·z2
        Ensures both z1 AND z2 are not all zeros
        """
        clean_list = []
        rejected_count = 0
        
        for _ in range(self.n_clean):
            # Keep sampling until both have at least one active feature
            while True:
                z1 = torch.bernoulli(torch.full((self.config.d,), 0.1))
                z2 = torch.bernoulli(torch.full((self.config.d,), 0.1))
                
                # Both must be non-zero
                if z1.sum() > 0 and z2.sum() > 0:
                    break
                rejected_count += 1
            
            x = (self.config.alpha1 * (M1 @ z1) + 
                self.config.alpha2 * (M2 @ z2))
            clean_list.append(x)
        
        if rejected_count > 0:
            print(f"  Rejected {rejected_count} samples with z1=0 or z2=0 (kept {self.n_clean} valid samples)")
        
        return torch.stack(clean_list)
    
    def _generate_time_values(self, t_start: float, t_end: float):
        """Generate time schedule"""
        if self.config.schedule == "linear":
            return torch.linspace(t_start, t_end, self.n_steps)
        elif self.config.schedule == "cosine":
            k = torch.arange(self.n_steps)
            cosine_vals = torch.cos(0.5 * math.pi * k / (self.n_steps - 1))
            return t_start + (t_end - t_start) * cosine_vals
        else:
            raise ValueError(f"Unknown schedule: {self.config.schedule}")
    
    def __len__(self):
        return self.n_clean * self.n_steps
    
    def __getitem__(self, idx):
        clean_idx, step_idx = divmod(idx, self.n_steps)
        x_clean = self.x_clean_all[clean_idx]
        t = self.t_vals[step_idx]
        eps = torch.randn(self.config.d1)
        x_noisy =  x_clean + ((1.0 - t) / t)  * eps
        
        return (x_noisy.float(), 
                x_clean.float(), 
                t.unsqueeze(0).float(), 
                eps.float())
