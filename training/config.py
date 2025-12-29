"""
训练超参数配置
"""

from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass
class TrainingConfig:
    """训练配置"""
    
    # 数据相关
    data_root: str = './data'
    img_size: int = 64
    num_classes: int = 62
    
    # 模型相关
    model_type: str = 'base'  # 'base' 或 'large'
    dropout_rate: float = 0.5
    
    # 训练相关
    batch_size: int = 128
    epochs: int = 50
    num_workers: int = 4
    
    # 优化器相关
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)
    
    # 学习率调度
    scheduler_type: str = 'cosine'  # 'cosine', 'step', 'plateau'
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # 正则化
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mixup_prob: float = 0.5
    
    # 保存和日志
    save_dir: str = './checkpoints'
    log_interval: int = 100
    save_interval: int = 5
    
    # 设备
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision: bool = True  # 使用AMP混合精度训练
    
    # 早停
    early_stopping_patience: int = 10
    
    # 随机种子
    seed: int = 42


@dataclass
class InferenceConfig:
    """推理配置"""
    
    model_path: str = '../backend/weights/best_model.pt'
    model_type: str = 'base'
    img_size: int = 64
    num_classes: int = 62
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Grad-CAM
    target_layer: str = 'stage3'
    
    # 预处理
    normalize_mean: float = 0.5
    normalize_std: float = 0.5


# 预设配置
CONFIGS = {
    'default': TrainingConfig(),
    
    'fast_debug': TrainingConfig(
        batch_size=32,
        epochs=5,
        num_workers=0,
        log_interval=10,
        mixed_precision=False
    ),
    
    'high_accuracy': TrainingConfig(
        model_type='large',
        batch_size=64,
        epochs=100,
        learning_rate=5e-4,
        weight_decay=5e-5,
        label_smoothing=0.1,
        mixup_alpha=0.4,
        warmup_epochs=10
    ),
    
    'rtx5080_optimized': TrainingConfig(
        model_type='base',
        batch_size=256,  # RTX 5080 16GB 可以用更大batch
        epochs=50,
        num_workers=8,
        learning_rate=2e-3,  # 更大batch需要更大lr
        mixed_precision=True,
        warmup_epochs=3
    )
}


def get_config(name: str = 'default') -> TrainingConfig:
    """获取预设配置"""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]

