# DeepCNN Visualization

实时手写字符识别与深度神经网络可视化交互平台。

## 功能特性

- **手写识别**: 支持 A-Z、a-z、0-9 共62类字符实时识别
- **3D可视化**: Three.js 构建的神经网络激活状态3D展示
- **Grad-CAM热力图**: 可视化AI注意力分布
- **Apple风格UI**: 简约科技感设计

## 技术栈

### 后端
- Python 3.10+
- PyTorch 2.x (CUDA 12.x)
- FastAPI
- torchvision

### 前端
- 原生 HTML/CSS/JavaScript
- Three.js (3D可视化)
- Canvas API (手写输入)

## 快速开始

### 环境准备

```bash
conda activate icml26
cd backend
pip install -r requirements.txt
```

### 训练模型

```bash
cd training
python train.py
```

### 启动服务

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 访问应用

打开浏览器访问 `http://localhost:8000`

## 项目结构

```
deepcnnvis/
├── backend/           # 后端服务
│   ├── app/          # FastAPI 应用
│   └── weights/      # 模型权重
├── training/         # 训练脚本
└── frontend/         # 前端页面
```

## 模型架构

采用深度残差CNN架构:
- 8个残差块
- ~3M 参数
- 输入: 64x64 灰度图
- 输出: 62类字符概率

## License

MIT

