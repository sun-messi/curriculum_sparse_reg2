# Curriculum Learning + Group L1 Regularization

Diffusion model denoising experiments combining curriculum learning with Group L1 regularization for soft sparsity.

## Core Ideas

- **Curriculum Learning**: Train in stages from large to small time steps (high noise to low noise)
- **Group L1 Regularization**: Apply L1 regularization on hidden layer neurons to drive sparsity

### Regularization Formula

```
L_total = MSE(pred, target) + λ_w · Σᵢ ||W[i,:]||_2 + λ_v · Σⱼ ||V[:,j]||_2
```

- `W`: First layer weights (hidden_dim, input_dim), L2 norm computed per row
- `V`: Second layer weights (input_dim, hidden_dim), L2 norm computed per column
- `λ`: Decreases from `lambda_max` to 0 across curriculum stages

## Ablation Experiments

| Experiment | Curriculum | Reg | Description |
|------------|------------|-----|-------------|
| 1 - baseline | OFF | OFF | Baseline |
| 2 - curriculum_only | ON | OFF | Curriculum learning only |
| 3 - reg_only | OFF | ON | Regularization only |
| 4 - curriculum_reg | ON | ON | Curriculum + Regularization |

## Usage

```bash
# Run specific experiments
python run_ablation.py 2 4                    # curriculum_only + curriculum_reg
python run_ablation.py curriculum both        # Same as above (using names)

# Specify GPUs
python run_ablation.py 2 4 --gpus 0,1,2,3     # Use GPUs 0,1,2,3

# Parallel mode (recommended): each GPU runs one experiment
python run_ablation.py 2 4 --gpus 1,4 --parallel

# Run all experiments
python run_ablation.py --all

# List available experiments
python run_ablation.py --list
```

## Configuration

Edit `config.py`:

```python
# Curriculum Learning
curriculum_enabled = True
curriculum_epochs_per_stage = 8
num_curriculum_stages = 10

# Group L1 Regularization
reg_enabled = True
lambda_max = 0.01              # Maximum λ value
lambda_schedule_type = "linear"  # "linear", "cosine", "exponential"
reg_threshold = 1e-3           # Threshold for active neuron detection
```

## File Structure

```
├── config.py          # Configuration parameters
├── model.py           # Standard Denoiser
├── reg_model.py       # RegDenoiser (with Group L1)
├── training.py        # Training functions
├── experiment.py      # Experiment runner
├── run_ablation.py    # Ablation study entry point
├── analysis.py        # Similarity analysis
├── dataset.py         # Dataset
├── ddp_utils.py       # DDP utilities
├── utils.py           # Common utilities
├── compare_curriculum_reg_ablation.ipynb  # Result visualization
└── results/           # Experiment results (.pkl)
```

## Result Analysis

View results using Jupyter notebook:

```bash
jupyter notebook compare_curriculum_reg_ablation.ipynb
```

Visualizations include:
- W-M1, W-M2 similarity comparisons
- Top N neuron training curves
- Sparsity over training
- λ schedule effects

---

# 中文版 (Chinese Version)

## 核心思想

- **课程学习 (Curriculum Learning)**: 按时间步从大到小分阶段训练（从噪声大到噪声小）
- **Group L1 正则化**: 对隐藏层神经元施加 L1 正则化，驱动稀疏

### 正则化公式

```
L_total = MSE(pred, target) + λ_w · Σᵢ ||W[i,:]||_2 + λ_v · Σⱼ ||V[:,j]||_2
```

- `W`: 第一层权重 (hidden_dim, input_dim)，按行计算 L2 范数
- `V`: 第二层权重 (input_dim, hidden_dim)，按列计算 L2 范数
- `λ`: 随 curriculum stage 从 `lambda_max` 递减到 0

## 消融实验

| 实验 | Curriculum | Reg | 说明 |
|------|------------|-----|------|
| 1 - baseline | OFF | OFF | 基线 |
| 2 - curriculum_only | ON | OFF | 仅课程学习 |
| 3 - reg_only | OFF | ON | 仅正则化 |
| 4 - curriculum_reg | ON | ON | 课程学习 + 正则化 |

## 使用方法

```bash
# 运行指定实验
python run_ablation.py 2 4                    # curriculum_only + curriculum_reg

# 并行模式（推荐）：每个 GPU 运行一个实验
python run_ablation.py 2 4 --gpus 1,4 --parallel

# 运行全部实验
python run_ablation.py --all
```

## 配置参数

编辑 `config.py`:

```python
# 课程学习
curriculum_enabled = True
curriculum_epochs_per_stage = 8
num_curriculum_stages = 10

# Group L1 正则化
reg_enabled = True
lambda_max = 0.01              # λ 最大值
lambda_schedule_type = "linear"  # "linear", "cosine", "exponential"
reg_threshold = 1e-3           # 判断神经元活跃的阈值
```
