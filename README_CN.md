# 课程学习 + Group L1 正则化

Diffusion 模型去噪实验，结合课程学习和 Group L1 正则化实现软稀疏。

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
python run_ablation.py curriculum both        # 同上（用名字）

# 指定 GPU
python run_ablation.py 2 4 --gpus 0,1,2,3     # 使用 GPU 0,1,2,3

# 并行模式（推荐）：每个 GPU 运行一个实验
python run_ablation.py 2 4 --gpus 1,4 --parallel

# 运行全部实验
python run_ablation.py --all

# 查看可用实验
python run_ablation.py --list
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

## 文件结构

```
├── config.py          # 配置参数
├── model.py           # 标准 Denoiser
├── reg_model.py       # RegDenoiser（带 Group L1）
├── training.py        # 训练函数
├── experiment.py      # 实验运行器
├── run_ablation.py    # 消融实验入口
├── analysis.py        # 相似度分析
├── dataset.py         # 数据集
├── ddp_utils.py       # DDP 工具
├── utils.py           # 通用工具
├── compare_curriculum_reg_ablation.ipynb  # 结果可视化
└── results/           # 实验结果 (.pkl)
```

## 结果分析

使用 Jupyter notebook 查看结果：

```bash
jupyter notebook compare_curriculum_reg_ablation.ipynb
```

可视化内容：
- W-M1, W-M2 相似度对比
- Top N 神经元训练曲线
- 稀疏度随训练变化
- λ schedule 效果

---

[English Version](README.md)
