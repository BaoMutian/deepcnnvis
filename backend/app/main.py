"""
FastAPI 后端服务

提供实时推理API和静态文件服务
"""

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from .inference import get_engine, init_engine


# 创建FastAPI应用
app = FastAPI(
    title="DeepCNN Visualization API",
    description="手写字符识别与深度神经网络可视化服务",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求模型
class PredictRequest(BaseModel):
    """预测请求"""
    image: str  # base64编码的图像
    from_canvas: bool = True  # 是否来自Canvas
    include_activations: bool = True  # 是否包含激活数据


class PredictResponse(BaseModel):
    """预测响应"""
    prediction: str
    confidence: float
    top5: list
    gradcam_heatmap: Optional[str] = None
    gradcam_overlay: Optional[str] = None
    activations: Optional[dict] = None
    preprocessed_image: Optional[str] = None


# API路由
@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "message": "服务运行正常"}


@app.get("/api/model/info")
async def model_info():
    """获取模型信息"""
    engine = get_engine()
    return {
        "model_type": "DeepCharCNN",
        "num_classes": 62,
        "parameters": engine.model.count_parameters(),
        "input_size": [1, 1, 64, 64],
        "device": str(engine.device),
        "classes": engine.model.CLASSES
    }


@app.post("/api/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    执行推理
    
    接收base64编码的图像，返回预测结果和可视化数据
    """
    try:
        engine = get_engine()
        
        # 执行推理
        result = engine.predict_with_visualization(
            image=request.image,
            from_canvas=request.from_canvas,
            include_activations=request.include_activations
        )
        
        # 获取预处理后的图像
        result['preprocessed_image'] = engine.get_preprocessed_image(
            image=request.image,
            from_canvas=request.from_canvas
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/simple")
async def predict_simple(request: PredictRequest):
    """
    简单推理（无可视化数据）
    用于快速预测
    """
    try:
        engine = get_engine()
        result = engine.predict(
            image=request.image,
            from_canvas=request.from_canvas
        )
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/upload")
async def predict_upload(file: UploadFile = File(...)):
    """
    通过文件上传进行推理
    """
    try:
        contents = await file.read()
        engine = get_engine()
        
        result = engine.predict_with_visualization(
            image=contents,
            from_canvas=False,
            include_activations=True
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 静态文件服务
frontend_path = Path(__file__).parent.parent.parent / "frontend"

if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
    
    @app.get("/")
    async def serve_index():
        """提供前端页面"""
        index_path = frontend_path / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return {"message": "前端页面未找到"}
else:
    @app.get("/")
    async def root():
        return {"message": "DeepCNN Visualization API", "docs": "/docs"}


# 启动事件
@app.on_event("startup")
async def startup_event():
    """服务启动时初始化模型"""
    print("正在初始化推理引擎...")
    try:
        engine = get_engine()
        print("推理引擎初始化完成")
    except Exception as e:
        print(f"警告: 推理引擎初始化失败 - {e}")


def main():
    """主函数"""
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )


if __name__ == "__main__":
    main()

