#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ablation Study: Curriculum Learning x Sparsity Curriculum
消融实验：课程学习 x 稀疏课程

支持 DistributedDataParallel (DDP) 多GPU训练

可用实验:
  1 or baseline           - Curriculum OFF, Reg OFF (基线)
  2 or curriculum         - Curriculum ON,  Reg OFF (仅课程学习)
  3 or reg                - Curriculum OFF, Reg ON  (仅正则化)
  4 or both               - Curriculum ON,  Reg ON  (课程学习+正则化)

使用方法:
    # 运行指定实验 (用编号或名字)
    python run_ablation.py 2 4                    # curriculum_only + curriculum_reg
    python run_ablation.py curriculum both        # 同上

    # 指定 GPU
    python run_ablation.py 2 4 --gpus 0,1,2,3,4,5     # 使用 GPU 0,1,2,3
    python run_ablation.py 2 4 --gpus 1,3,4,5     # 使用 GPU 1,3,4,5
    python run_ablation.py 2 4 --gpus 1,4 --parallel # 每个 GPU 运行一个实验  推荐

    # 运行全部实验
    python run_ablation.py --all
    python run_ablation.py --all --gpus 0,1       # 全部实验，使用 GPU 0,1

    # 查看可用实验
    python run_ablation.py --list
    
pkill -f anaconda3/envs/lavis/bin/python
"""

import os
import sys
import argparse
import pickle
import torch
import torch.multiprocessing as mp

from config import Config
from utils import build_dictionaries, print_summary_table
from experiment import run_experiment
from ddp_utils import setup_ddp, is_main_process, barrier, cleanup_ddp


# ============================================================================
# 实验配置定义
# ============================================================================
ALL_EXPERIMENTS = {
    # 编号方式
    '1': (False, False, "results_baseline"),           # 基线：无课程学习，无正则化
    '2': (True,  False, "results_curriculum_only"),    # 仅课程学习
    '3': (False, True,  "results_reg_only"),           # 仅正则化
    '4': (True,  True,  "results_curriculum_reg"),     # 课程学习 + 正则化
    # 名称方式 (别名)
    'baseline': (False, False, "results_baseline"),
    'curriculum': (True,  False, "results_curriculum_only"),
    'reg': (False, True,  "results_reg_only"),
    'both': (True,  True,  "results_curriculum_reg"),
}


# ============================================================================
# 命令行参数解析
# ============================================================================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='消融实验: Curriculum x Sparsity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_ablation.py 2 4                    # 运行实验 2 和 4
  python run_ablation.py curriculum both        # 同上 (用名字)
  python run_ablation.py 2 4 --gpus 0,1,2,3     # 指定使用 GPU 0,1,2,3
  python run_ablation.py --all                  # 运行全部 4 个实验
  python run_ablation.py --all --parallel       # 每个 GPU 运行一个实验（推荐！）
  python run_ablation.py --list                 # 查看可用实验
        """
    )
    parser.add_argument(
        'experiments',
        nargs='*',
        help='要运行的实验 (1-4 或名字: baseline, curriculum, reg, both)'
    )
    parser.add_argument(
        '--gpus', '-g',
        type=str,
        default=None,
        help='指定使用的 GPU，用逗号分隔 (例如: 0,1,2,3 或 1,3,4,5)'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='运行全部 4 个实验'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='列出可用实验并退出'
    )
    parser.add_argument(
        '--parallel', '-p',
        action='store_true',
        help='并行模式：每个 GPU 运行一个实验（推荐！避免 DDP 梯度同步问题）'
    )
    return parser.parse_args()


def get_selected_combinations(args):
    """根据命令行参数获取要运行的实验组合"""
    # 显示可用实验列表
    if args.list:
        print("\n可用实验:")
        print("  1 or 'baseline'    - Curriculum OFF, Reg OFF (基线)")
        print("  2 or 'curriculum'  - Curriculum ON,  Reg OFF (仅课程学习)")
        print("  3 or 'reg'         - Curriculum OFF, Reg ON  (仅正则化)")
        print("  4 or 'both'        - Curriculum ON,  Reg ON  (课程学习+正则化)")
        print("\n示例:")
        print("  python run_ablation.py 2 4")
        print("  python run_ablation.py curriculum both")
        print("  python run_ablation.py 2 4 --gpus 0,1,2,3")
        print("  python run_ablation.py --all")
        sys.exit(0)

    # 运行全部实验
    if args.all:
        return [
            (False, False, "results_baseline"),
            (True,  False, "results_curriculum_only"),
            (False, True,  "results_reg_only"),
            (True,  True,  "results_curriculum_reg"),
        ]

    # 未指定实验
    if not args.experiments:
        print("错误: 未指定实验")
        print("使用 --list 查看可用实验，或 --all 运行全部")
        print("示例: python run_ablation.py 2 4")
        sys.exit(1)

    # 解析指定的实验
    combinations = []
    seen_names = set()

    for exp in args.experiments:
        exp_lower = exp.lower()
        if exp_lower not in ALL_EXPERIMENTS:
            print(f"错误: 未知实验 '{exp}'")
            print("使用 --list 查看可用实验")
            sys.exit(1)

        curriculum, reg, name = ALL_EXPERIMENTS[exp_lower]
        if name not in seen_names:
            combinations.append((curriculum, reg, name))
            seen_names.add(name)

    return combinations


def setup_gpus(gpu_str):
    """
    设置要使用的 GPU

    Args:
        gpu_str: GPU 编号字符串，如 "0,1,2,3" 或 "1,3,4,5"

    Returns:
        int: 可用 GPU 数量
    """
    if gpu_str is None:
        # 未指定，使用默认配置
        return torch.cuda.device_count()

    # 设置 CUDA_VISIBLE_DEVICES 环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str

    # 解析 GPU 数量
    gpu_list = [g.strip() for g in gpu_str.split(',') if g.strip()]
    num_gpus = len(gpu_list)

    print(f"已设置 CUDA_VISIBLE_DEVICES={gpu_str}")
    print(f"使用 {num_gpus} 个 GPU: {gpu_list}")

    return num_gpus


# ============================================================================
# DDP 多 GPU 训练
# ============================================================================
def run_ablation_ddp(rank: int, world_size: int, combinations: list):
    """
    在单个 GPU 上运行消融实验 (DDP 模式下每个进程调用)

    Args:
        rank: 当前进程的 rank (0 到 world_size-1)
        world_size: 总进程数/GPU数
        combinations: 要运行的实验组合列表
    """

    # 初始化 DDP
    setup_ddp(rank, world_size)

    # 输出目录
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    if is_main_process(rank):
        os.makedirs(output_dir, exist_ok=True)

    # 同步所有进程
    barrier()

    # 构建共享的正交字典 M1, M2
    config = Config()

    if is_main_process(rank):
        print("=" * 70)
        print(f"消融实验: {len(combinations)} 个实验")
        print(f"使用 {world_size} 个 GPU (DistributedDataParallel)")
        print("=" * 70)
        print(f"\n构建共享正交字典...")

    M1, M2 = build_dictionaries(config.d1, config.d)

    if is_main_process(rank):
        print(f"  M1 shape: {M1.shape}")
        print(f"  M2 shape: {M2.shape}")
        print(f"\n输出目录: {output_dir}")
        print("=" * 70)

    # 运行每个实验组合
    for i, (curriculum_enabled, reg_enabled, name) in enumerate(combinations, 1):
        if is_main_process(rank):
            print(f"\n{'='*70}")
            print(f"[{i}/{len(combinations)}] 运行: {name}")
            print(f"       Curriculum: {'ON' if curriculum_enabled else 'OFF'}")
            print(f"       Reg:        {'ON' if reg_enabled else 'OFF'}")
            print(f"{'='*70}\n")

        # 为每个组合创建新的配置
        config = Config()
        config.curriculum_enabled = curriculum_enabled
        config.reg_enabled = reg_enabled

        # 运行所有时间范围
        results = []
        for t_start, t_end, exp_name in config.time_ranges:
            # 在 exp_name 中加入实验名称，方便区分
            full_exp_name = f"{name.replace('results_', '')}_{exp_name}"
            result = run_experiment(
                t_start, t_end, full_exp_name, config, M1, M2,
                rank=rank, world_size=world_size
            )
            results.append(result)

        # 保存结果 (仅主进程)
        if is_main_process(rank):
            for result in results:
                # 移到 CPU 以避免 pickle GPU 张量的问题
                result['M1'] = M1.cpu() if hasattr(M1, 'cpu') else M1
                result['M2'] = M2.cpu() if hasattr(M2, 'cpu') else M2

            output_path = os.path.join(output_dir, f"{name}.pkl")
            with open(output_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"\n>>> 已保存: {output_path}")

        # 同步后再进行下一个实验
        barrier()

    # 最终总结
    if is_main_process(rank):
        print(f"\n{'='*70}")
        print("消融实验完成")
        print(f"{'='*70}")
        print(f"\n结果保存到: {output_dir}/")
        for _, _, name in combinations:
            print(f"  - {name}.pkl")
        print(f"{'='*70}\n")

    # DDP 清理：显式销毁进程组以避免 NCCL 超时警告
    cleanup_ddp()


# ============================================================================
# 单 GPU 训练
# ============================================================================
def run_ablation_single_gpu(combinations):
    """单 GPU 模式运行消融实验"""

    # 输出目录
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)

    # 构建共享的正交字典
    config = Config()
    print("=" * 70)
    print(f"消融实验: {len(combinations)} 个实验")
    print("使用单 GPU")
    print("=" * 70)
    print(f"\n构建共享正交字典...")
    M1, M2 = build_dictionaries(config.d1, config.d)
    print(f"  M1 shape: {M1.shape}")
    print(f"  M2 shape: {M2.shape}")
    print(f"\n输出目录: {output_dir}")
    print("=" * 70)

    # 运行每个实验组合
    for i, (curriculum_enabled, reg_enabled, name) in enumerate(combinations, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(combinations)}] 运行: {name}")
        print(f"       Curriculum: {'ON' if curriculum_enabled else 'OFF'}")
        print(f"       Reg:        {'ON' if reg_enabled else 'OFF'}")
        print(f"{'='*70}\n")

        # 为每个组合创建新的配置
        config = Config()
        config.curriculum_enabled = curriculum_enabled
        config.reg_enabled = reg_enabled

        # 运行所有时间范围
        results = []
        for t_start, t_end, exp_name in config.time_ranges:
            # 在 exp_name 中加入实验名称，方便区分
            full_exp_name = f"{name.replace('results_', '')}_{exp_name}"
            result = run_experiment(t_start, t_end, full_exp_name, config, M1, M2)
            results.append(result)

        # 保存结果 (移到 CPU 以避免 pickle GPU 张量的问题)
        for result in results:
            result['M1'] = M1.cpu() if hasattr(M1, 'cpu') else M1
            result['M2'] = M2.cpu() if hasattr(M2, 'cpu') else M2

        output_path = os.path.join(output_dir, f"{name}.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"\n>>> 已保存: {output_path}")

    # 最终总结
    print(f"\n{'='*70}")
    print("消融实验完成")
    print(f"{'='*70}")
    print(f"\n结果保存到: {output_dir}/")
    for _, _, name in combinations:
        print(f"  - {name}.pkl")
    print(f"{'='*70}\n")


# ============================================================================
# 并行模式：每个 GPU 运行不同实验（推荐）
# ============================================================================
def run_single_experiment_on_gpu(gpu_id: int, experiment: tuple, output_dir: str, M1, M2):
    """
    在指定 GPU 上运行单个实验（独立进程）

    Args:
        gpu_id: GPU 编号
        experiment: (curriculum_enabled, reg_enabled, name)
        output_dir: 输出目录
        M1, M2: 正交字典
    """
    import torch
    torch.cuda.set_device(gpu_id)

    curriculum_enabled, reg_enabled, name = experiment

    print(f"\n[GPU {gpu_id}] 开始运行: {name}")
    print(f"[GPU {gpu_id}] Curriculum: {'ON' if curriculum_enabled else 'OFF'}, Reg: {'ON' if reg_enabled else 'OFF'}")

    # 创建配置
    config = Config()
    config.curriculum_enabled = curriculum_enabled
    config.reg_enabled = reg_enabled
    config.device = torch.device(f"cuda:{gpu_id}")

    # 注意：M1, M2 保持在 CPU 上，让 run_experiment 内部处理设备
    # 这样 DiffusionDataset 可以在 CPU 上生成数据

    # 运行实验
    results = []
    for t_start, t_end, exp_name in config.time_ranges:
        # 在 exp_name 中加入实验名称，方便区分
        full_exp_name = f"{name.replace('results_', '')}_{exp_name}"
        result = run_experiment(t_start, t_end, full_exp_name, config, M1, M2)
        results.append(result)

    # 保存结果
    for result in results:
        result['M1'] = M1.cpu()
        result['M2'] = M2.cpu()

    output_path = os.path.join(output_dir, f"{name}.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n[GPU {gpu_id}] 完成: {name} -> {output_path}")


def run_parallel_experiments(combinations: list, gpu_list: list):
    """
    并行运行多个实验，每个 GPU 运行一个实验

    Args:
        combinations: 实验组合列表
        gpu_list: 可用 GPU 列表
    """
    import torch.multiprocessing as mp

    # 设置 spawn 方法以支持 CUDA
    mp.set_start_method('spawn', force=True)

    # 输出目录
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)

    # 构建共享的正交字典（在 CPU 上）
    config = Config()
    print("=" * 70)
    print(f"并行实验模式: {len(combinations)} 个实验, {len(gpu_list)} 个 GPU")
    print("每个 GPU 独立运行一个实验（无 DDP 梯度同步）")
    print("=" * 70)
    print(f"\n构建共享正交字典...")
    M1, M2 = build_dictionaries(config.d1, config.d)
    print(f"  M1 shape: {M1.shape}")
    print(f"  M2 shape: {M2.shape}")
    print(f"\n输出目录: {output_dir}")
    print("=" * 70)

    # 分配实验到 GPU
    processes = []
    for i, experiment in enumerate(combinations):
        gpu_id = gpu_list[i % len(gpu_list)]
        p = mp.Process(
            target=run_single_experiment_on_gpu,
            args=(gpu_id, experiment, output_dir, M1.clone(), M2.clone())
        )
        p.start()
        processes.append((p, experiment[2], gpu_id))
        print(f"启动: {experiment[2]} -> GPU {gpu_id}")

    # 等待所有进程完成
    for p, name, gpu_id in processes:
        p.join()
        print(f"[GPU {gpu_id}] {name} 已完成")

    # 最终总结
    print(f"\n{'='*70}")
    print("所有实验完成！")
    print(f"{'='*70}")
    print(f"\n结果保存到: {output_dir}/")
    for _, _, name in combinations:
        print(f"  - {name}.pkl")
    print(f"{'='*70}\n")


# ============================================================================
# 主入口
# ============================================================================
def main():
    """主入口函数"""
    # 解析命令行参数
    args = parse_args()

    # 设置 GPU (必须在其他 CUDA 操作之前)
    if args.gpus:
        num_gpus = setup_gpus(args.gpus)
    else:
        num_gpus = None  # 使用 config 中的设置

    # 获取要运行的实验组合
    combinations = get_selected_combinations(args)

    # 显示选中的实验
    print(f"\n选中的实验:")
    for curriculum, reg, name in combinations:
        print(f"  - {name} (Curriculum: {'ON' if curriculum else 'OFF'}, Reg: {'ON' if reg else 'OFF'})")
    print()

    # 确定 world_size 和 GPU 列表
    config = Config()
    if num_gpus is not None:
        world_size = num_gpus
        # 解析 GPU 列表 (CUDA_VISIBLE_DEVICES 已设置，所以 GPU 0,1,2... 是映射后的)
        gpu_list = list(range(num_gpus))
    else:
        world_size = config.world_size
        gpu_list = list(range(min(world_size, torch.cuda.device_count())))

    # 选择运行模式
    if args.parallel:
        # 并行模式：每个 GPU 运行一个实验（推荐！）
        if len(gpu_list) == 0:
            print("错误: 没有可用的 GPU")
            sys.exit(1)
        print(f"\n使用并行模式：每个 GPU 运行一个实验")
        print(f"可用 GPU: {len(gpu_list)}, 实验数: {len(combinations)}")
        run_parallel_experiments(combinations, gpu_list)
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # 通过 torchrun 启动
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        run_ablation_ddp(rank, world_size, combinations)
    elif world_size > 1 and torch.cuda.device_count() >= world_size:
        # 多 GPU 模式 (使用 mp.spawn)
        print(f"启动 DDP 训练，使用 {world_size} 个 GPU...")
        mp.spawn(
            run_ablation_ddp,
            args=(world_size, combinations),  # 传递 combinations 给子进程
            nprocs=world_size,
            join=True
        )
    else:
        # 单 GPU 模式
        if world_size > 1:
            available = torch.cuda.device_count()
            print(f"警告: world_size={world_size} 但只有 {available} 个 GPU 可用")
            print("回退到单 GPU 模式")
        run_ablation_single_gpu(combinations)


if __name__ == "__main__":
    main()

