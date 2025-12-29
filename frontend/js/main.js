/**
 * 主应用程序入口
 * 协调Canvas、3D可视化、Grad-CAM和API调用
 */

class DeepCNNApp {
    constructor() {
        // API配置
        this.apiBase = window.location.origin;
        this.apiEndpoint = '/api/predict';
        
        // 组件
        this.canvas = null;
        this.network3d = null;
        this.predictionDisplay = null;
        
        // 状态
        this.isConnected = false;
        this.isProcessing = false;
        this.lastPrediction = null;
        
        // 元素引用
        this.elements = {
            statusDot: document.getElementById('status-dot'),
            statusText: document.getElementById('status-text'),
            loadingOverlay: document.getElementById('loading-overlay'),
            btnClear: document.getElementById('btn-clear'),
            btnUndo: document.getElementById('btn-undo'),
            imgOriginal: document.getElementById('img-original'),
            imgHeatmap: document.getElementById('img-heatmap'),
            imgOverlay: document.getElementById('img-overlay')
        };
        
        // 初始化
        this.init();
    }
    
    async init() {
        try {
            // 检查服务连接
            await this.checkConnection();
            
            // 初始化组件
            this.initCanvas();
            this.init3DVisualization();
            this.initPredictionDisplay();
            this.initLayerControls();
            
            // 绑定事件
            this.bindEvents();
            
            console.log('DeepCNN App initialized successfully');
        } catch (error) {
            console.error('Initialization error:', error);
            this.setStatus(false, '初始化失败');
        }
    }
    
    async checkConnection() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            if (response.ok) {
                const data = await response.json();
                this.isConnected = true;
                this.setStatus(true, '服务已连接');
                return true;
            }
        } catch (error) {
            console.warn('Health check failed:', error);
        }
        
        this.isConnected = false;
        this.setStatus(false, '服务未连接');
        return false;
    }
    
    setStatus(connected, text) {
        this.elements.statusDot.classList.toggle('connected', connected);
        this.elements.statusText.textContent = text;
    }
    
    initCanvas() {
        this.canvas = new HandwritingCanvas('drawing-canvas', {
            lineWidth: 16,
            lineColor: '#1D1D1F',
            backgroundColor: '#FFFFFF',
            debounceDelay: 200,
            onDrawStart: () => this.onDrawStart(),
            onDrawEnd: (imageData) => this.onDrawEnd(imageData)
        });
        
        // 添加drawing状态
        const container = document.getElementById('canvas-container');
        this.canvas.canvas.addEventListener('mousedown', () => {
            container.classList.add('drawing');
        });
        document.addEventListener('mouseup', () => {
            container.classList.remove('drawing');
        });
    }
    
    init3DVisualization() {
        try {
            this.network3d = new Network3DVisualizer('network-3d', {
                backgroundColor: 0xf5f5f7,
                cameraDistance: 18
            });
        } catch (error) {
            console.error('3D visualization init error:', error);
        }
    }
    
    initPredictionDisplay() {
        this.predictionDisplay = new PredictionDisplay('prediction-display');
    }
    
    initLayerControls() {
        const layerBtns = document.querySelectorAll('.layer-btn');
        
        layerBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                // 更新active状态
                layerBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                // 移动相机
                const layerIndex = parseInt(btn.dataset.layer);
                if (this.network3d) {
                    if (layerIndex === -1) {
                        this.network3d.resetCamera();
                    } else {
                        this.network3d.focusOnLayer(layerIndex);
                    }
                }
            });
        });
    }
    
    bindEvents() {
        // 清除按钮
        this.elements.btnClear.addEventListener('click', () => {
            this.clearAll();
        });
        
        // 撤销按钮
        this.elements.btnUndo.addEventListener('click', () => {
            if (this.canvas) {
                this.canvas.undo();
            }
        });
        
        // 键盘快捷键
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.clearAll();
            } else if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
                e.preventDefault();
                if (this.canvas) {
                    this.canvas.undo();
                }
            }
        });
    }
    
    onDrawStart() {
        // 绘制开始时可以做一些准备工作
    }
    
    async onDrawEnd(imageData) {
        if (this.isProcessing || !imageData) return;
        
        // 检查画布是否为空
        if (this.canvas.isEmpty()) {
            this.resetVisualization();
            return;
        }
        
        try {
            this.isProcessing = true;
            
            // 显示处理状态
            this.setStatus(true, '识别中...');
            
            // 调用API
            const result = await this.predict(imageData);
            
            if (result) {
                // 更新可视化
                this.updateVisualization(result);
                this.setStatus(true, '识别完成');
            }
        } catch (error) {
            console.error('Prediction error:', error);
            this.setStatus(false, '识别失败');
        } finally {
            this.isProcessing = false;
        }
    }
    
    async predict(imageData) {
        try {
            const response = await fetch(`${this.apiBase}${this.apiEndpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: imageData,
                    from_canvas: true,
                    include_activations: true
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API error:', error);
            
            // 如果API不可用，尝试重新连接
            this.checkConnection();
            
            throw error;
        }
    }
    
    updateVisualization(result) {
        this.lastPrediction = result;
        
        // 更新预测显示
        if (this.predictionDisplay) {
            this.predictionDisplay.update(result);
        }
        
        // 更新3D网络
        if (this.network3d && result.activations) {
            this.network3d.updateActivations(result.activations);
            this.network3d.highlightPrediction(
                result.prediction,
                result.top5
            );
        }
        
        // 更新Grad-CAM图像
        this.updateGradCAMImages(result);
    }
    
    updateGradCAMImages(result) {
        // 更新预处理后的原始图像
        if (result.preprocessed_image) {
            this.updateImage(this.elements.imgOriginal, result.preprocessed_image);
        }
        
        // 更新热力图
        if (result.gradcam_heatmap) {
            this.updateImage(this.elements.imgHeatmap, result.gradcam_heatmap);
        }
        
        // 更新叠加图
        if (result.gradcam_overlay) {
            this.updateImage(this.elements.imgOverlay, result.gradcam_overlay);
        }
    }
    
    updateImage(imgElement, src) {
        if (!imgElement) return;
        
        const placeholder = imgElement.parentElement.querySelector('.panel-placeholder');
        
        // 淡出占位符
        if (placeholder) {
            placeholder.style.opacity = '0';
        }
        
        // 加载新图像
        imgElement.style.opacity = '0';
        imgElement.onload = () => {
            imgElement.style.transition = 'opacity 0.3s ease';
            imgElement.style.opacity = '1';
        };
        imgElement.src = src;
    }
    
    resetVisualization() {
        this.lastPrediction = null;
        
        // 重置预测显示
        if (this.predictionDisplay) {
            this.predictionDisplay.reset();
        }
        
        // 重置3D网络
        if (this.network3d) {
            this.network3d.reset();
        }
        
        // 重置Grad-CAM图像
        this.resetGradCAMImages();
    }
    
    resetGradCAMImages() {
        [this.elements.imgOriginal, this.elements.imgHeatmap, this.elements.imgOverlay].forEach(img => {
            if (img) {
                img.style.opacity = '0';
                img.src = '';
                
                const placeholder = img.parentElement.querySelector('.panel-placeholder');
                if (placeholder) {
                    placeholder.style.opacity = '1';
                }
            }
        });
    }
    
    clearAll() {
        // 清除画布
        if (this.canvas) {
            this.canvas.clear();
        }
        
        // 重置可视化
        this.resetVisualization();
        
        // 重置状态
        this.setStatus(this.isConnected, this.isConnected ? '服务已连接' : '服务未连接');
    }
    
    showLoading(show = true) {
        if (this.elements.loadingOverlay) {
            this.elements.loadingOverlay.classList.toggle('visible', show);
        }
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    window.app = new DeepCNNApp();
});

// 处理页面可见性变化
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && window.app) {
        window.app.checkConnection();
    }
});

