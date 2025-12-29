"""
Grad-CAM 实现

Gradient-weighted Class Activation Mapping
用于可视化CNN的注意力区域
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import cv2
from PIL import Image
import io
import base64


class GradCAM:
    """
    Grad-CAM 可视化
    
    原理:
    1. 前向传播获取目标层的特征图
    2. 对目标类别进行反向传播获取梯度
    3. 使用梯度对特征图通道进行加权平均
    4. ReLU激活并归一化得到热力图
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        target_layer: nn.Module,
        device: torch.device = None
    ):
        """
        Args:
            model: CNN模型
            target_layer: 用于生成热力图的目标层
            device: 计算设备
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device or next(model.parameters()).device
        
        # 存储特征图和梯度
        self.feature_maps: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        
        # 注册钩子
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向和反向钩子"""
        
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        normalize: bool = True
    ) -> Tuple[np.ndarray, int, float]:
        """
        生成Grad-CAM热力图
        
        Args:
            input_tensor: 输入图像张量 [1, C, H, W]
            target_class: 目标类别索引，None则使用预测类别
            normalize: 是否归一化到[0,1]
        
        Returns:
            (热力图数组, 预测类别, 置信度)
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        
        # 前向传播
        output = self.model(input_tensor)
        
        # 获取预测类别和置信度
        probs = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
        
        # 如果未指定目标类别，使用预测类别
        if target_class is None:
            target_class = pred_class
        
        # 反向传播
        self.model.zero_grad()
        
        # 创建one-hot向量
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 计算Grad-CAM
        # 全局平均池化梯度作为权重
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # 加权求和特征图
        cam = (weights * self.feature_maps).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        
        # ReLU激活 (只保留正贡献)
        cam = F.relu(cam)
        
        # 上采样到输入尺寸
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        # 转换为numpy
        cam = cam.squeeze().cpu().numpy()
        
        # 归一化
        if normalize:
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)
        
        return cam, pred_class, confidence
    
    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, int, float]:
        """
        生成彩色热力图和叠加图
        
        Args:
            input_tensor: 输入图像张量
            target_class: 目标类别
            colormap: OpenCV颜色映射
            alpha: 叠加透明度
        
        Returns:
            (热力图, 叠加图, 预测类别, 置信度)
        """
        # 生成CAM
        cam, pred_class, confidence = self.generate(input_tensor, target_class)
        
        # 转换为彩色热力图
        cam_uint8 = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(cam_uint8, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 获取原始图像
        img = input_tensor.squeeze().detach().cpu().numpy()
        
        # 反归一化 (假设使用 mean=0.5, std=0.5)
        img = img * 0.5 + 0.5
        img = np.clip(img, 0, 1)
        
        # 转换为RGB (灰度图复制到3通道)
        if img.ndim == 2:
            img_rgb = np.stack([img] * 3, axis=-1)
        else:
            img_rgb = img.transpose(1, 2, 0)
        
        img_rgb = np.uint8(255 * img_rgb)
        
        # 叠加
        overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap, alpha, 0)
        
        # 逆变换：将图像旋转回用户视角
        # 因为预处理时做了 TRANSPOSE + FLIP_LEFT_RIGHT，需要逆操作
        # 逆时针旋转90度可以恢复原始方向
        img_rgb = cv2.rotate(img_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        heatmap = cv2.rotate(heatmap, cv2.ROTATE_90_COUNTERCLOCKWISE)
        overlay = cv2.rotate(overlay, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        return heatmap, overlay, pred_class, confidence


class LayerActivationExtractor:
    """
    中间层激活提取器
    用于3D可视化
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.activations: Dict[str, torch.Tensor] = {}
        self.input_tensor: Optional[torch.Tensor] = None
        
        # 注册钩子
        self._register_hooks()
    
    def _register_hooks(self):
        """为所有主要层注册钩子"""
        
        def get_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # 注册stem - 同时捕获输入
        if hasattr(self.model, 'stem'):
            def stem_hook(module, input, output):
                self.activations['stem'] = output.detach()
                # 保存输入用于input层可视化
                if len(input) > 0:
                    self.input_tensor = input[0].detach()
            self.model.stem.register_forward_hook(stem_hook)
        
        # 注册各个stage
        for name in ['stage1', 'stage2', 'stage3']:
            if hasattr(self.model, name):
                getattr(self.model, name).register_forward_hook(get_hook(name))
        
        # 注册classifier中的第一个Linear层后的ReLU输出 (fc1)
        if hasattr(self.model, 'classifier'):
            # classifier[1] 是第一个Linear, classifier[2] 是ReLU
            if len(self.model.classifier) > 2:
                self.model.classifier[2].register_forward_hook(get_hook('fc1'))
    
    def extract(
        self, 
        input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Dict]]:
        """
        提取所有层的激活
        
        Args:
            input_tensor: 输入图像张量
        
        Returns:
            (模型输出, 激活字典)
        """
        self.model.eval()
        self.activations.clear()
        self.input_tensor = None
        
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # 处理激活数据用于传输
        processed = {}
        
        # 添加输入层数据
        if self.input_tensor is not None:
            inp = self.input_tensor.squeeze(0)  # [1, H, W] 或 [H, W]
            if inp.dim() == 3:
                inp = inp.squeeze(0)  # [H, W]
            # 下采样到8x8
            inp_resized = F.adaptive_avg_pool2d(
                inp.unsqueeze(0).unsqueeze(0), (8, 8)
            ).squeeze()
            processed['input'] = {
                'shape': list(inp.shape),
                'values': inp_resized.cpu().numpy().flatten().tolist(),
                'mean': float(inp.mean()),
                'std': float(inp.std()),
                'max': float(inp.max())
            }
        
        for name, activation in self.activations.items():
            act = activation.squeeze(0)  # 移除batch维度
            
            # 对于卷积层，获取通道平均激活
            if act.dim() == 3:  # [C, H, W]
                # 计算每个通道的平均激活值
                channel_activations = act.mean(dim=(1, 2)).cpu().numpy()
                
                # 对激活图进行下采样以减少传输数据量
                # 只保留前N个最活跃的通道
                n_top = min(32, act.shape[0])
                top_indices = channel_activations.argsort()[-n_top:][::-1].copy()  # copy()修复负步长问题
                
                # 获取这些通道的下采样激活图
                h, w = act.shape[1], act.shape[2]
                target_size = 8  # 下采样到8x8
                
                if h > target_size:
                    sampled_maps = F.adaptive_avg_pool2d(
                        act[top_indices].unsqueeze(0),
                        (target_size, target_size)
                    ).squeeze(0)
                else:
                    sampled_maps = act[top_indices]
                
                processed[name] = {
                    'shape': list(act.shape),
                    'channel_activations': channel_activations.tolist(),
                    'top_channels': top_indices.tolist(),
                    'sampled_maps': sampled_maps.cpu().numpy().tolist(),
                    'mean': float(act.mean()),
                    'std': float(act.std()),
                    'max': float(act.max())
                }
            else:  # 全连接层或池化后
                processed[name] = {
                    'shape': list(act.shape),
                    'values': act.cpu().numpy().flatten().tolist()[:256],  # 限制数据量
                    'mean': float(act.mean()),
                    'std': float(act.std()),
                    'max': float(act.max())
                }
        
        # 添加输出层数据 (softmax后的概率)
        probs = F.softmax(output, dim=1).squeeze(0)
        processed['output'] = {
            'shape': [62],
            'values': probs.cpu().numpy().tolist(),
            'mean': float(probs.mean()),
            'std': float(probs.std()),
            'max': float(probs.max())
        }
        
        return output, processed


def numpy_to_base64(arr: np.ndarray, format: str = 'PNG') -> str:
    """将numpy数组转换为base64字符串"""
    if arr.dtype != np.uint8:
        arr = np.uint8(np.clip(arr, 0, 255))
    
    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    
    return f"data:image/{format.lower()};base64," + base64.b64encode(buffer.read()).decode()


def create_visualization_data(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: nn.Module,
    target_class: Optional[int] = None,
    device: torch.device = None
) -> Dict:
    """
    创建完整的可视化数据
    
    Returns:
        包含预测结果、Grad-CAM、激活数据的字典
    """
    device = device or next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Grad-CAM
    gradcam = GradCAM(model, target_layer, device)
    heatmap, overlay, pred_class, confidence = gradcam.generate_heatmap(
        input_tensor, target_class
    )
    
    # 激活提取
    extractor = LayerActivationExtractor(model, device)
    output, activations = extractor.extract(input_tensor)
    
    # Top-5预测
    probs = F.softmax(output, dim=1).squeeze()
    top5_probs, top5_indices = probs.topk(5)
    
    # 获取类别标签
    from backend.app.model import DeepCharCNN
    top5 = [
        [DeepCharCNN.idx_to_class(idx.item()), prob.item()]
        for idx, prob in zip(top5_indices, top5_probs)
    ]
    
    return {
        'prediction': DeepCharCNN.idx_to_class(pred_class),
        'confidence': confidence,
        'top5': top5,
        'gradcam_heatmap': numpy_to_base64(heatmap),
        'gradcam_overlay': numpy_to_base64(overlay),
        'activations': activations
    }


if __name__ == '__main__':
    # 测试代码
    from backend.app.model import DeepCharCNN
    
    # 创建模型
    model = DeepCharCNN()
    model.eval()
    
    # 创建测试输入
    x = torch.randn(1, 1, 64, 64)
    
    # 测试Grad-CAM
    target_layer = model.stage3[-1]
    gradcam = GradCAM(model, target_layer)
    cam, pred_class, confidence = gradcam.generate(x)
    print(f"CAM shape: {cam.shape}")
    print(f"预测类别: {pred_class}, 置信度: {confidence:.4f}")
    
    # 测试激活提取
    extractor = LayerActivationExtractor(model)
    output, activations = extractor.extract(x)
    print("\n激活层:")
    for name, data in activations.items():
        print(f"  {name}: shape={data['shape']}")
    
    # 测试完整可视化数据
    vis_data = create_visualization_data(model, x, target_layer)
    print(f"\n可视化数据键: {vis_data.keys()}")

