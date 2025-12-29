"""
EMNIST数据集加载与数据增强

EMNIST ByClass: 62类 (0-9, A-Z, a-z)
- 训练集: 697,932 样本
- 测试集: 116,323 样本
- 图像尺寸: 28x28 -> 放大到 64x64
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import EMNIST
import numpy as np
from typing import Tuple, Optional, Callable
import random


class ElasticTransform:
    """弹性变形增强 - 模拟手写的自然变化"""

    def __init__(self, alpha: float = 30, sigma: float = 4):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > 0.5:
            return img

        # 简化版弹性变形
        _, h, w = img.shape

        # 生成随机位移场
        dx = torch.randn(h, w) * self.alpha
        dy = torch.randn(h, w) * self.alpha

        # 高斯平滑
        kernel_size = int(self.sigma * 4) | 1  # 确保奇数
        if kernel_size > 1:
            from torchvision.transforms.functional import gaussian_blur
            dx = gaussian_blur(dx.unsqueeze(0).unsqueeze(0),
                               kernel_size).squeeze()
            dy = gaussian_blur(dy.unsqueeze(0).unsqueeze(0),
                               kernel_size).squeeze()

        # 归一化到较小范围
        dx = dx / (dx.abs().max() + 1e-8) * 3
        dy = dy / (dy.abs().max() + 1e-8) * 3

        # 创建采样网格
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w),
            indexing='ij'
        )

        grid = torch.stack([
            x + dx * 2 / w,
            y + dy * 2 / h
        ], dim=-1).unsqueeze(0)

        # 应用变形
        img_batch = img.unsqueeze(0)
        warped = torch.nn.functional.grid_sample(
            img_batch, grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )

        return warped.squeeze(0)


class RandomMorphology:
    """随机形态学操作 - 模拟不同笔画粗细"""

    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img

        # 简单的膨胀/腐蚀效果通过最大/最小池化模拟
        kernel_size = random.choice([2, 3])
        padding = kernel_size // 2

        img_4d = img.unsqueeze(0)

        if random.random() > 0.5:
            # 膨胀 (加粗)
            result = torch.nn.functional.max_pool2d(
                img_4d, kernel_size, stride=1, padding=padding
            )
        else:
            # 腐蚀 (变细)
            result = -torch.nn.functional.max_pool2d(
                -img_4d, kernel_size, stride=1, padding=padding
            )

        # 调整尺寸匹配
        if result.shape[-2:] != img.shape[-2:]:
            result = torch.nn.functional.interpolate(
                result, size=img.shape[-2:], mode='bilinear', align_corners=False
            )

        return result.squeeze(0)


def get_train_transforms(img_size: int = 64) -> transforms.Compose:
    """训练集数据增强变换"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # 随机旋转
        transforms.RandomRotation(degrees=15, fill=0),
        # 随机仿射变换 (平移、缩放、剪切)
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=10,
            fill=0
        ),
        # 随机透视变换
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3, fill=0),
        # 弹性变形
        ElasticTransform(alpha=20, sigma=3),
        # 形态学变换
        RandomMorphology(p=0.2),
        # 随机擦除
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        # 高斯噪声
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        # 归一化到 [-1, 1]
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def get_val_transforms(img_size: int = 64) -> transforms.Compose:
    """验证/测试集变换 (无增强)"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


class EMNISTDataset(Dataset):
    """
    EMNIST数据集包装器

    EMNIST ByClass 的标签映射:
    0-9: 数字 '0'-'9'
    10-35: 大写字母 'A'-'Z'
    36-61: 小写字母 'a'-'z'
    """

    # EMNIST ByClass 标签到字符的映射
    LABEL_MAP = (
        list('0123456789') +
        list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') +
        list('abcdefghijklmnopqrstuvwxyz')
    )

    def __init__(
        self,
        root: str = './data',
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = True
    ):
        """
        Args:
            root: 数据存储路径
            train: True为训练集，False为测试集
            transform: 数据变换
            download: 是否自动下载
        """
        self.transform = transform

        # 加载EMNIST ByClass数据集
        self.dataset = EMNIST(
            root=root,
            split='byclass',
            train=train,
            download=download,
            transform=None  # 我们自己处理transform
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.dataset[idx]

        # EMNIST图像需要转置 (原始是转置的)
        img = img.transpose(method=0)  # PIL的transpose

        if self.transform:
            img = self.transform(img)

        return img, label

    @classmethod
    def idx_to_char(cls, idx: int) -> str:
        """标签索引转字符"""
        return cls.LABEL_MAP[idx]

    @classmethod
    def char_to_idx(cls, char: str) -> int:
        """字符转标签索引"""
        return cls.LABEL_MAP.index(char)


def create_dataloaders(
    root: str = './data',
    batch_size: int = 128,
    img_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器

    Args:
        root: 数据存储路径
        batch_size: 批次大小
        img_size: 图像尺寸
        num_workers: 数据加载线程数
        pin_memory: 是否使用固定内存

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = EMNISTDataset(
        root=root,
        train=True,
        transform=get_train_transforms(img_size),
        download=True
    )

    val_dataset = EMNISTDataset(
        root=root,
        train=False,
        transform=get_val_transforms(img_size),
        download=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # 验证时可以用更大batch
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, val_loader


def get_class_weights(dataset: EMNISTDataset) -> torch.Tensor:
    """
    计算类别权重，用于处理类别不平衡
    """
    labels = []
    for _, label in dataset:
        labels.append(label)

    labels = torch.tensor(labels)
    class_counts = torch.bincount(labels, minlength=62).float()

    # 使用逆频率作为权重
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * len(weights)

    return weights


if __name__ == '__main__':
    # 测试数据加载
    print("正在下载/加载EMNIST数据集...")

    train_loader, val_loader = create_dataloaders(
        root='./data',
        batch_size=64,
        num_workers=0
    )

    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")

    # 获取一个批次测试
    images, labels = next(iter(train_loader))
    print(f"\n批次图像形状: {images.shape}")
    print(f"批次标签形状: {labels.shape}")
    print(f"图像值范围: [{images.min():.2f}, {images.max():.2f}]")

    # 显示一些标签
    print("\n前10个标签:")
    for i in range(10):
        print(
            f"  {labels[i].item()} -> '{EMNISTDataset.idx_to_char(labels[i].item())}'")
