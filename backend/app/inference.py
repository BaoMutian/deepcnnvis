"""
推理服务模块

提供图像预处理和模型推理功能
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import base64
from typing import Dict, List, Tuple, Optional, Union
import cv2

from .model import DeepCharCNN, create_model
from .gradcam import GradCAM, LayerActivationExtractor, numpy_to_base64


class InferenceEngine:
    """推理引擎"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = 'base',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model_path: 模型权重路径
            model_type: 模型类型 ('base' 或 'large')
            device: 计算设备
        """
        self.device = torch.device(device)
        self.img_size = 64
        
        # 加载模型
        print(f"加载模型到 {self.device}...")
        self.model = create_model(
            model_type=model_type,
            num_classes=62,
            pretrained_path=model_path
        ).to(self.device)
        self.model.eval()
        
        # 初始化Grad-CAM
        self.target_layer = self.model.stage3[-1]
        self.gradcam = GradCAM(self.model, self.target_layer, self.device)
        
        # 初始化激活提取器
        self.activation_extractor = LayerActivationExtractor(self.model, self.device)
        
        print(f"模型加载完成，参数量: {self.model.count_parameters():,}")
    
    def preprocess_image(
        self, 
        image: Union[str, bytes, np.ndarray, Image.Image]
    ) -> torch.Tensor:
        """
        预处理图像
        
        Args:
            image: 输入图像，可以是:
                - base64字符串
                - bytes
                - numpy数组
                - PIL Image
        
        Returns:
            预处理后的张量 [1, 1, 64, 64]
        """
        # 解码图像
        if isinstance(image, str):
            # 处理base64
            if image.startswith('data:image'):
                image = image.split(',')[1]
            image_bytes = base64.b64decode(image)
            img = Image.open(io.BytesIO(image_bytes))
        elif isinstance(image, bytes):
            img = Image.open(io.BytesIO(image))
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # 转换为灰度图
        img = img.convert('L')
        
        # 调整大小
        img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        
        # 转换为numpy
        img_array = np.array(img, dtype=np.float32)
        
        # 归一化到 [-1, 1]
        img_array = img_array / 255.0
        img_array = (img_array - 0.5) / 0.5
        
        # 转换为张量
        tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def preprocess_canvas(
        self,
        image_data: str,
        invert: bool = True
    ) -> torch.Tensor:
        """
        预处理Canvas手写图像
        
        Canvas通常是白底黑字或透明背景，需要特殊处理
        
        Args:
            image_data: base64图像数据
            invert: 是否反转颜色（Canvas通常需要）
        
        Returns:
            预处理后的张量
        """
        # 解码
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes))
        
        # 转换为RGBA以处理透明度
        if img.mode == 'RGBA':
            # 创建白色背景
            background = Image.new('RGBA', img.size, (255, 255, 255, 255))
            background.paste(img, mask=img.split()[3])  # 使用alpha通道作为mask
            img = background.convert('L')
        else:
            img = img.convert('L')
        
        # 找到内容边界并裁剪
        img_array = np.array(img)
        
        if invert:
            # 反转：白变黑，黑变白 (手写通常是深色笔画)
            img_array = 255 - img_array
        
        # 找到非零区域
        coords = np.column_stack(np.where(img_array > 20))
        
        if len(coords) > 0:
            # 获取边界框
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # 添加padding
            padding = 20
            y_min = max(0, y_min - padding)
            x_min = max(0, x_min - padding)
            y_max = min(img_array.shape[0], y_max + padding)
            x_max = min(img_array.shape[1], x_max + padding)
            
            # 裁剪
            cropped = img_array[y_min:y_max, x_min:x_max]
            
            # 确保是正方形
            h, w = cropped.shape
            if h > w:
                pad_left = (h - w) // 2
                pad_right = h - w - pad_left
                cropped = np.pad(cropped, ((0, 0), (pad_left, pad_right)), 
                               mode='constant', constant_values=0)
            elif w > h:
                pad_top = (w - h) // 2
                pad_bottom = w - h - pad_top
                cropped = np.pad(cropped, ((pad_top, pad_bottom), (0, 0)),
                               mode='constant', constant_values=0)
            
            img_array = cropped
        
        # 调整大小
        img = Image.fromarray(img_array)
        img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        
        # 关键：EMNIST训练数据是转置的，需要对用户输入做相同变换
        # 转置图像以匹配训练数据格式
        img = img.transpose(Image.Transpose.TRANSPOSE)
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        
        # 转换为float并归一化
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = (img_array - 0.5) / 0.5
        
        # 转换为张量
        tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        
        return tensor.to(self.device)
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[str, bytes, np.ndarray, Image.Image],
        from_canvas: bool = True
    ) -> Dict:
        """
        执行推理
        
        Args:
            image: 输入图像
            from_canvas: 是否来自Canvas（需要特殊预处理）
        
        Returns:
            预测结果字典
        """
        # 预处理
        if from_canvas and isinstance(image, str):
            tensor = self.preprocess_canvas(image)
        else:
            tensor = self.preprocess_image(image)
        
        # 推理
        output = self.model(tensor)
        
        # Softmax获取概率
        probs = F.softmax(output, dim=1).squeeze()
        
        # Top-5预测
        top5_probs, top5_indices = probs.topk(5)
        
        prediction = DeepCharCNN.idx_to_class(top5_indices[0].item())
        confidence = top5_probs[0].item()
        
        top5 = [
            [DeepCharCNN.idx_to_class(idx.item()), prob.item()]
            for idx, prob in zip(top5_indices, top5_probs)
        ]
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'top5': top5
        }
    
    def predict_with_visualization(
        self,
        image: Union[str, bytes, np.ndarray, Image.Image],
        from_canvas: bool = True,
        include_activations: bool = True
    ) -> Dict:
        """
        执行推理并生成可视化数据
        
        Args:
            image: 输入图像
            from_canvas: 是否来自Canvas
            include_activations: 是否包含激活数据
        
        Returns:
            包含预测结果和可视化数据的字典
        """
        # 预处理
        if from_canvas and isinstance(image, str):
            tensor = self.preprocess_canvas(image)
        else:
            tensor = self.preprocess_image(image)
        
        # 需要开启梯度用于Grad-CAM
        tensor.requires_grad_(True)
        
        # 生成Grad-CAM
        heatmap, overlay, pred_class, confidence = self.gradcam.generate_heatmap(
            tensor, target_class=None
        )
        
        # 获取Top-5预测
        with torch.no_grad():
            output = self.model(tensor)
            probs = F.softmax(output, dim=1).squeeze()
            top5_probs, top5_indices = probs.topk(5)
        
        top5 = [
            [DeepCharCNN.idx_to_class(idx.item()), prob.item()]
            for idx, prob in zip(top5_indices, top5_probs)
        ]
        
        # 反向转置 Grad-CAM 图像以匹配用户输入方向
        # 原始转换: TRANSPOSE -> FLIP_LEFT_RIGHT
        # 反向转换: FLIP_LEFT_RIGHT -> TRANSPOSE
        if from_canvas:
            heatmap = np.fliplr(heatmap.transpose(1, 0, 2))
            overlay = np.fliplr(overlay.transpose(1, 0, 2))
        
        result = {
            'prediction': DeepCharCNN.idx_to_class(pred_class),
            'confidence': confidence,
            'top5': top5,
            'gradcam_heatmap': numpy_to_base64(heatmap),
            'gradcam_overlay': numpy_to_base64(overlay)
        }
        
        # 提取激活（如果需要）
        if include_activations:
            tensor_no_grad = tensor.detach()
            _, activations = self.activation_extractor.extract(tensor_no_grad)
            result['activations'] = activations
        
        return result
    
    def get_preprocessed_image(
        self,
        image: Union[str, bytes, np.ndarray, Image.Image],
        from_canvas: bool = True
    ) -> str:
        """
        获取预处理后的图像（base64格式）
        用于前端显示预处理结果
        """
        if from_canvas and isinstance(image, str):
            tensor = self.preprocess_canvas(image)
        else:
            tensor = self.preprocess_image(image)
        
        # 反归一化
        img = tensor.squeeze().cpu().numpy()
        img = img * 0.5 + 0.5
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        # 反向转置以匹配用户输入方向
        # 原始转换: TRANSPOSE -> FLIP_LEFT_RIGHT
        # 反向转换: FLIP_LEFT_RIGHT -> TRANSPOSE
        if from_canvas:
            img = np.fliplr(img.T)
        
        return numpy_to_base64(img)


# 全局推理引擎实例（延迟初始化）
_engine: Optional[InferenceEngine] = None


def get_engine() -> InferenceEngine:
    """获取全局推理引擎实例"""
    global _engine
    if _engine is None:
        import os
        from pathlib import Path
        
        # 查找模型权重
        weights_dir = Path(__file__).parent.parent / 'weights'
        model_path = weights_dir / 'best_model.pt'
        
        if model_path.exists():
            _engine = InferenceEngine(model_path=str(model_path))
        else:
            print("警告: 未找到训练好的模型，使用随机初始化")
            _engine = InferenceEngine(model_path=None)
    
    return _engine


def init_engine(model_path: Optional[str] = None, **kwargs):
    """初始化全局推理引擎"""
    global _engine
    _engine = InferenceEngine(model_path=model_path, **kwargs)
    return _engine

