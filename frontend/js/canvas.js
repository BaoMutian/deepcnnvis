/**
 * 手写画布模块
 * 支持鼠标和触摸输入，实时捕捉手写轨迹
 */

class HandwritingCanvas {
    constructor(canvasId, options = {}) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            throw new Error(`Canvas element not found: ${canvasId}`);
        }
        
        this.ctx = this.canvas.getContext('2d');
        
        // 配置选项
        this.options = {
            lineWidth: options.lineWidth || 12,
            lineColor: options.lineColor || '#1D1D1F',
            lineCap: options.lineCap || 'round',
            lineJoin: options.lineJoin || 'round',
            backgroundColor: options.backgroundColor || '#FFFFFF',
            smoothing: options.smoothing !== false,
            debounceDelay: options.debounceDelay || 300,
            onDrawEnd: options.onDrawEnd || null,
            onDrawStart: options.onDrawStart || null
        };
        
        // 状态
        this.isDrawing = false;
        this.lastPoint = null;
        this.points = [];
        this.strokes = [];
        this.debounceTimer = null;
        
        // 初始化
        this.init();
    }
    
    init() {
        // 设置画布尺寸
        this.resizeCanvas();
        
        // 初始化画布样式
        this.clear();
        
        // 绑定事件
        this.bindEvents();
        
        // 监听窗口大小变化
        window.addEventListener('resize', () => this.resizeCanvas());
    }
    
    resizeCanvas() {
        const rect = this.canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        
        // 设置实际分辨率
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        
        // 缩放上下文以匹配CSS尺寸
        this.ctx.scale(dpr, dpr);
        
        // 重新应用样式
        this.applyStyles();
        
        // 重绘已有笔画
        this.redraw();
    }
    
    applyStyles() {
        this.ctx.lineWidth = this.options.lineWidth;
        this.ctx.strokeStyle = this.options.lineColor;
        this.ctx.lineCap = this.options.lineCap;
        this.ctx.lineJoin = this.options.lineJoin;
        this.ctx.imageSmoothingEnabled = true;
        this.ctx.imageSmoothingQuality = 'high';
    }
    
    bindEvents() {
        // 鼠标事件
        this.canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
        this.canvas.addEventListener('mousemove', (e) => this.draw(e));
        this.canvas.addEventListener('mouseup', () => this.stopDrawing());
        this.canvas.addEventListener('mouseleave', () => this.stopDrawing());
        
        // 触摸事件
        this.canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.startDrawing(e.touches[0]);
        }, { passive: false });
        
        this.canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            this.draw(e.touches[0]);
        }, { passive: false });
        
        this.canvas.addEventListener('touchend', () => this.stopDrawing());
        this.canvas.addEventListener('touchcancel', () => this.stopDrawing());
    }
    
    getPoint(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top,
            time: Date.now()
        };
    }
    
    startDrawing(e) {
        this.isDrawing = true;
        this.points = [];
        this.lastPoint = this.getPoint(e);
        this.points.push(this.lastPoint);
        
        // 清除防抖定时器
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
            this.debounceTimer = null;
        }
        
        // 触发开始绘制回调
        if (this.options.onDrawStart) {
            this.options.onDrawStart();
        }
        
        // 开始新路径
        this.ctx.beginPath();
        this.ctx.moveTo(this.lastPoint.x, this.lastPoint.y);
    }
    
    draw(e) {
        if (!this.isDrawing) return;
        
        const currentPoint = this.getPoint(e);
        this.points.push(currentPoint);
        
        if (this.options.smoothing && this.points.length >= 3) {
            // 使用二次贝塞尔曲线平滑
            const p1 = this.points[this.points.length - 3];
            const p2 = this.points[this.points.length - 2];
            const p3 = currentPoint;
            
            const midPoint1 = {
                x: (p1.x + p2.x) / 2,
                y: (p1.y + p2.y) / 2
            };
            const midPoint2 = {
                x: (p2.x + p3.x) / 2,
                y: (p2.y + p3.y) / 2
            };
            
            this.ctx.beginPath();
            this.ctx.moveTo(midPoint1.x, midPoint1.y);
            this.ctx.quadraticCurveTo(p2.x, p2.y, midPoint2.x, midPoint2.y);
            this.ctx.stroke();
        } else {
            // 直接画线
            this.ctx.lineTo(currentPoint.x, currentPoint.y);
            this.ctx.stroke();
            this.ctx.beginPath();
            this.ctx.moveTo(currentPoint.x, currentPoint.y);
        }
        
        this.lastPoint = currentPoint;
    }
    
    stopDrawing() {
        if (!this.isDrawing) return;
        
        this.isDrawing = false;
        
        // 保存当前笔画
        if (this.points.length > 1) {
            this.strokes.push([...this.points]);
        }
        
        this.points = [];
        this.lastPoint = null;
        
        // 防抖触发绘制结束回调
        if (this.options.onDrawEnd) {
            if (this.debounceTimer) {
                clearTimeout(this.debounceTimer);
            }
            this.debounceTimer = setTimeout(() => {
                this.options.onDrawEnd(this.getImageData());
            }, this.options.debounceDelay);
        }
    }
    
    redraw() {
        // 清空画布
        this.ctx.fillStyle = this.options.backgroundColor;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // 重新应用样式
        this.applyStyles();
        
        // 重绘所有笔画
        for (const stroke of this.strokes) {
            if (stroke.length < 2) continue;
            
            this.ctx.beginPath();
            this.ctx.moveTo(stroke[0].x, stroke[0].y);
            
            for (let i = 1; i < stroke.length; i++) {
                if (this.options.smoothing && i < stroke.length - 1) {
                    const p1 = stroke[i - 1];
                    const p2 = stroke[i];
                    const p3 = stroke[i + 1];
                    
                    const midPoint = {
                        x: (p2.x + p3.x) / 2,
                        y: (p2.y + p3.y) / 2
                    };
                    
                    this.ctx.quadraticCurveTo(p2.x, p2.y, midPoint.x, midPoint.y);
                } else {
                    this.ctx.lineTo(stroke[i].x, stroke[i].y);
                }
            }
            
            this.ctx.stroke();
        }
    }
    
    clear() {
        this.strokes = [];
        this.points = [];
        this.ctx.fillStyle = this.options.backgroundColor;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.applyStyles();
        
        // 清除防抖定时器
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
            this.debounceTimer = null;
        }
    }
    
    undo() {
        if (this.strokes.length > 0) {
            this.strokes.pop();
            this.redraw();
            
            // 触发更新
            if (this.options.onDrawEnd && this.strokes.length > 0) {
                if (this.debounceTimer) {
                    clearTimeout(this.debounceTimer);
                }
                this.debounceTimer = setTimeout(() => {
                    this.options.onDrawEnd(this.getImageData());
                }, this.options.debounceDelay);
            }
        }
    }
    
    isEmpty() {
        return this.strokes.length === 0;
    }
    
    getImageData(format = 'image/png', quality = 1.0) {
        return this.canvas.toDataURL(format, quality);
    }
    
    getImageBlob(format = 'image/png', quality = 1.0) {
        return new Promise((resolve) => {
            this.canvas.toBlob(resolve, format, quality);
        });
    }
    
    // 设置线条粗细
    setLineWidth(width) {
        this.options.lineWidth = width;
        this.ctx.lineWidth = width;
    }
    
    // 设置线条颜色
    setLineColor(color) {
        this.options.lineColor = color;
        this.ctx.strokeStyle = color;
    }
    
    // 导出用于模型的图像数据
    exportForModel(size = 64) {
        // 创建临时画布
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = size;
        tempCanvas.height = size;
        const tempCtx = tempCanvas.getContext('2d');
        
        // 白色背景
        tempCtx.fillStyle = '#FFFFFF';
        tempCtx.fillRect(0, 0, size, size);
        
        // 找到内容边界
        const imageData = this.ctx.getImageData(
            0, 0,
            this.canvas.width / (window.devicePixelRatio || 1),
            this.canvas.height / (window.devicePixelRatio || 1)
        );
        
        let minX = Infinity, minY = Infinity;
        let maxX = -Infinity, maxY = -Infinity;
        
        const data = imageData.data;
        const width = imageData.width;
        const height = imageData.height;
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = (y * width + x) * 4;
                // 检查是否有内容（非白色像素）
                if (data[idx] < 250 || data[idx + 1] < 250 || data[idx + 2] < 250) {
                    minX = Math.min(minX, x);
                    minY = Math.min(minY, y);
                    maxX = Math.max(maxX, x);
                    maxY = Math.max(maxY, y);
                }
            }
        }
        
        // 如果没有内容，返回空白
        if (minX === Infinity) {
            return tempCanvas.toDataURL('image/png');
        }
        
        // 添加padding
        const padding = 10;
        minX = Math.max(0, minX - padding);
        minY = Math.max(0, minY - padding);
        maxX = Math.min(width - 1, maxX + padding);
        maxY = Math.min(height - 1, maxY + padding);
        
        const contentWidth = maxX - minX;
        const contentHeight = maxY - minY;
        
        // 保持纵横比，居中绘制
        const scale = Math.min(
            (size - 8) / contentWidth,
            (size - 8) / contentHeight
        );
        
        const scaledWidth = contentWidth * scale;
        const scaledHeight = contentHeight * scale;
        const offsetX = (size - scaledWidth) / 2;
        const offsetY = (size - scaledHeight) / 2;
        
        // 绘制到临时画布
        tempCtx.drawImage(
            this.canvas,
            minX * (window.devicePixelRatio || 1),
            minY * (window.devicePixelRatio || 1),
            contentWidth * (window.devicePixelRatio || 1),
            contentHeight * (window.devicePixelRatio || 1),
            offsetX, offsetY,
            scaledWidth, scaledHeight
        );
        
        return tempCanvas.toDataURL('image/png');
    }
}

// 导出
window.HandwritingCanvas = HandwritingCanvas;

