"""
深度CNN模型架构 - 用于手写字符识别
支持 A-Z, a-z, 0-9 共62类字符

模型特点:
- 残差连接 (Skip Connections)
- 批归一化 (Batch Normalization)  
- 渐进式通道扩展
- 全局平均池化
- 约3M参数量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class ConvBlock(nn.Module):
    """基础卷积块: Conv -> BN -> ReLU"""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """残差块: 两个卷积层 + 跳跃连接"""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        stride: int = 1
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 当通道数或尺寸改变时，需要投影shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        
        return out


class DeepCharCNN(nn.Module):
    """
    深度字符识别CNN
    
    架构:
    - 输入: 1x64x64 灰度图像
    - 初始卷积: 1 -> 64 通道
    - Stage 1: 64 -> 128 通道, 2个残差块
    - Stage 2: 128 -> 256 通道, 3个残差块
    - Stage 3: 256 -> 512 通道, 3个残差块
    - 全局平均池化
    - 全连接: 512 -> 256 -> 62类
    
    参数量: ~3.2M
    """
    
    # 类别标签映射
    CLASSES = (
        list('0123456789') +  # 0-9
        list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') +  # A-Z
        list('abcdefghijklmnopqrstuvwxyz')  # a-z
    )
    NUM_CLASSES = 62
    
    def __init__(self, num_classes: int = 62, dropout_rate: float = 0.5):
        super().__init__()
        
        self.num_classes = num_classes
        
        # 保存中间层激活的钩子
        self.activations: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}
        
        # 初始卷积块: 1 -> 64
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )  # 64x64 -> 16x16
        
        # Stage 1: 64 -> 128, 2个残差块
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 128, stride=1),
            ResidualBlock(128, 128, stride=1)
        )  # 16x16 -> 16x16
        
        # Stage 2: 128 -> 256, 3个残差块
        self.stage2 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1),
            ResidualBlock(256, 256, stride=1)
        )  # 16x16 -> 8x8
        
        # Stage 3: 256 -> 512, 3个残差块
        self.stage3 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=1),
            ResidualBlock(512, 512, stride=1)
        )  # 8x8 -> 4x4
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """使用Kaiming初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
    def forward_with_activations(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播并返回中间层激活
        用于3D可视化
        """
        activations = {}
        
        # Stem
        x = self.stem(x)
        activations['stem'] = x.detach().clone()
        
        # Stage 1
        x = self.stage1(x)
        activations['stage1'] = x.detach().clone()
        
        # Stage 2
        x = self.stage2(x)
        activations['stage2'] = x.detach().clone()
        
        # Stage 3
        x = self.stage3(x)
        activations['stage3'] = x.detach().clone()
        
        # 分类
        x = self.global_pool(x)
        activations['global_pool'] = x.detach().clone()
        
        x = torch.flatten(x, 1)
        
        # 全连接层中间结果
        x = self.classifier[0](x)  # Dropout
        x = self.classifier[1](x)  # Linear 512->256
        x = self.classifier[2](x)  # ReLU
        activations['fc1'] = x.detach().clone()
        
        x = self.classifier[3](x)  # Dropout
        x = self.classifier[4](x)  # Linear 256->62
        activations['output'] = x.detach().clone()
        
        return x, activations
    
    def get_layer_for_gradcam(self) -> nn.Module:
        """返回用于Grad-CAM的目标层"""
        return self.stage3[-1]  # 最后一个残差块
    
    @classmethod
    def idx_to_class(cls, idx: int) -> str:
        """索引转类别标签"""
        return cls.CLASSES[idx]
    
    @classmethod
    def class_to_idx(cls, label: str) -> int:
        """类别标签转索引"""
        return cls.CLASSES.index(label)
    
    def count_parameters(self) -> int:
        """统计可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DeepCharCNNLarge(DeepCharCNN):
    """
    更大的模型变体 (~8M参数)
    适用于有充足GPU内存的情况
    """
    
    def __init__(self, num_classes: int = 62, dropout_rate: float = 0.5):
        # 先调用nn.Module的初始化，跳过父类的初始化逻辑
        nn.Module.__init__(self)
        
        self.num_classes = num_classes
        self.activations: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}
        
        # 初始卷积块: 1 -> 96
        self.stem = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Stage 1: 96 -> 192
        self.stage1 = nn.Sequential(
            ResidualBlock(96, 192, stride=1),
            ResidualBlock(192, 192, stride=1),
            ResidualBlock(192, 192, stride=1)
        )
        
        # Stage 2: 192 -> 384
        self.stage2 = nn.Sequential(
            ResidualBlock(192, 384, stride=2),
            ResidualBlock(384, 384, stride=1),
            ResidualBlock(384, 384, stride=1),
            ResidualBlock(384, 384, stride=1)
        )
        
        # Stage 3: 384 -> 768
        self.stage3 = nn.Sequential(
            ResidualBlock(384, 768, stride=2),
            ResidualBlock(768, 768, stride=1),
            ResidualBlock(768, 768, stride=1),
            ResidualBlock(768, 768, stride=1)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(768, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()


def create_model(
    model_type: str = 'base',
    num_classes: int = 62,
    pretrained_path: Optional[str] = None
) -> DeepCharCNN:
    """
    工厂函数创建模型
    
    Args:
        model_type: 'base' 或 'large'
        num_classes: 类别数
        pretrained_path: 预训练权重路径
    
    Returns:
        模型实例
    """
    if model_type == 'base':
        model = DeepCharCNN(num_classes=num_classes)
    elif model_type == 'large':
        model = DeepCharCNNLarge(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict)
    
    return model


if __name__ == '__main__':
    # 测试模型
    model = DeepCharCNN()
    print(f"模型参数量: {model.count_parameters():,}")
    
    # 测试前向传播
    x = torch.randn(1, 1, 64, 64)
    output = model(x)
    print(f"输出形状: {output.shape}")
    
    # 测试带激活的前向传播
    output, activations = model.forward_with_activations(x)
    print("\n各层激活形状:")
    for name, act in activations.items():
        print(f"  {name}: {act.shape}")
    
    # 测试大模型
    model_large = DeepCharCNNLarge()
    print(f"\n大模型参数量: {model_large.count_parameters():,}")

