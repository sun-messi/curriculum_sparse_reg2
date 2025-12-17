"""
Configuration file for diffusion experiments with Group L1 regularization
使用 Group L1 正则化实现软稀疏
"""
import torch


class Config:
    # Model parameters
    d = 20
    d1 = 200
    alpha1 = 5
    alpha2 = 0.5

    # Training parameters
    n_clean = 3000
    n_steps = 50
    batch_size = 512
    hidden_dim = 100
    lr = 1e-3  # Base learning rate

    # Warmup parameters
    warmup_enabled = True
    warmup_epochs = 2
    warmup_start_factor = 0.1

    # Experiment settings
    predict_mode = "x"  # "x" or "eps"
    schedule = "linear"  # "linear" or "cosine"
    seed = 42

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # DDP parameters
    world_size = 4
    backend = "nccl"

    # Curriculum Learning parameters
    curriculum_enabled = True
    curriculum_epochs_per_stage = 8
    max_accumulated_stages = 6

    epochs = curriculum_epochs_per_stage * max_accumulated_stages
    epochs = 30

    num_curriculum_stages = 10
    time_ranges = [
        (0.3, 1, "0.3-1"),
    ]

    use_custom_weighting = True

    # ========== Group L1 Regularization 参数 ==========
    reg_enabled = True
    lambda_max = 0.001             # 正则化系数最大值 (λ_w = λ_v 使用相同值) 1e-3
    lambda_schedule_type = "linear"  # "linear", "cosine", "exponential"
    reg_threshold = 1e-1           # 判断神经元活跃的阈值

    def __init__(self):
        # Auto-generate curriculum_stages
        t_start, t_end, _ = self.time_ranges[0]
        step = (t_end - t_start) / self.num_curriculum_stages

        self.curriculum_stages = []
        for i in range(self.num_curriculum_stages):
            stage_start = t_start + i * step
            stage_end = t_start + (i + 1) * step
            if i == self.num_curriculum_stages - 1:
                stage_end = t_end
            stage_name = f"stage_{stage_start:.2f}-{stage_end:.2f}"
            self.curriculum_stages.append((stage_start, stage_end, stage_name))

        # Auto-generate lambda_schedule (aligned with curriculum_stages)
        # λ 从 lambda_max 递减到 0
        if self.reg_enabled:
            self.lambda_schedule = self._generate_lambda_schedule(self.lambda_max)
        else:
            self.lambda_schedule = [0.0] * self.num_curriculum_stages

    def _generate_lambda_schedule(self, lambda_max: float) -> list:
        """
        生成 λ 递减调度表，从 lambda_max 递减到 0

        Args:
            lambda_max: 最大正则化系数

        Returns:
            list: 每个 stage 对应的 λ 值
        """
        import math
        n = self.num_curriculum_stages

        if n <= 1:
            return [0.0]

        if self.lambda_schedule_type == "linear":
            # 线性递减: lambda_max -> 0
            return [lambda_max * (1 - i / (n - 1)) for i in range(n)]

        elif self.lambda_schedule_type == "cosine":
            # 余弦递减: 更平滑的过渡
            return [lambda_max * 0.5 * (1 + math.cos(math.pi * i / (n - 1)))
                    for i in range(n)]

        elif self.lambda_schedule_type == "exponential":
            # 指数递减: 前期快速减小
            decay_rate = 3.0
            return [lambda_max * math.exp(-decay_rate * i / (n - 1))
                    for i in range(n)]

        else:
            # 默认线性
            return [lambda_max * (1 - i / (n - 1)) for i in range(n)]

    def get_lambda_for_stage(self, stage_idx: int) -> float:
        """
        获取指定 stage 的 λ 值 (λ_w = λ_v 使用相同值)

        Args:
            stage_idx: stage 索引

        Returns:
            float: 该 stage 的 λ 值
        """
        if stage_idx >= len(self.lambda_schedule):
            return 0.0
        return self.lambda_schedule[stage_idx]
