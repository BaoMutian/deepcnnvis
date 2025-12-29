/**
 * Grad-CAM 热力图显示模块
 * 在原图上叠加显示AI注意力分布
 */

class GradCAMDisplay {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container not found: ${containerId}`);
        }
        
        // 配置
        this.options = {
            showOriginal: options.showOriginal !== false,
            showHeatmap: options.showHeatmap !== false,
            showOverlay: options.showOverlay !== false,
            overlayOpacity: options.overlayOpacity || 0.5,
            animationDuration: options.animationDuration || 300
        };
        
        // 创建显示元素
        this.createElements();
    }
    
    createElements() {
        this.container.innerHTML = '';
        this.container.classList.add('gradcam-container');
        
        // 创建图像显示区域
        const imagesWrapper = document.createElement('div');
        imagesWrapper.className = 'gradcam-images';
        
        // 原始输入图像
        if (this.options.showOriginal) {
            this.originalPanel = this.createImagePanel('原始输入', 'original');
            imagesWrapper.appendChild(this.originalPanel.container);
        }
        
        // 热力图
        if (this.options.showHeatmap) {
            this.heatmapPanel = this.createImagePanel('热力图', 'heatmap');
            imagesWrapper.appendChild(this.heatmapPanel.container);
        }
        
        // 叠加图
        if (this.options.showOverlay) {
            this.overlayPanel = this.createImagePanel('注意力叠加', 'overlay');
            imagesWrapper.appendChild(this.overlayPanel.container);
        }
        
        this.container.appendChild(imagesWrapper);
        
        // 创建说明区域
        this.infoPanel = document.createElement('div');
        this.infoPanel.className = 'gradcam-info';
        this.infoPanel.innerHTML = `
            <div class="info-item">
                <span class="info-label">Grad-CAM</span>
                <span class="info-desc">显示模型关注的图像区域</span>
            </div>
        `;
        this.container.appendChild(this.infoPanel);
    }
    
    createImagePanel(title, type) {
        const container = document.createElement('div');
        container.className = `gradcam-panel gradcam-${type}`;
        
        const label = document.createElement('div');
        label.className = 'panel-label';
        label.textContent = title;
        
        const imageWrapper = document.createElement('div');
        imageWrapper.className = 'panel-image-wrapper';
        
        const image = document.createElement('img');
        image.className = 'panel-image';
        image.alt = title;
        
        // 占位符
        const placeholder = document.createElement('div');
        placeholder.className = 'panel-placeholder';
        placeholder.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <rect x="3" y="3" width="18" height="18" rx="2"/>
                <circle cx="8.5" cy="8.5" r="1.5"/>
                <path d="M21 15l-5-5L5 21"/>
            </svg>
            <span>等待输入</span>
        `;
        
        imageWrapper.appendChild(image);
        imageWrapper.appendChild(placeholder);
        
        container.appendChild(label);
        container.appendChild(imageWrapper);
        
        return { container, image, placeholder };
    }
    
    update(data) {
        if (!data) {
            this.reset();
            return;
        }
        
        // 更新原始图像
        if (this.originalPanel && data.preprocessed_image) {
            this.updateImage(this.originalPanel, data.preprocessed_image);
        }
        
        // 更新热力图
        if (this.heatmapPanel && data.gradcam_heatmap) {
            this.updateImage(this.heatmapPanel, data.gradcam_heatmap);
        }
        
        // 更新叠加图
        if (this.overlayPanel && data.gradcam_overlay) {
            this.updateImage(this.overlayPanel, data.gradcam_overlay);
        }
        
        // 更新信息
        if (data.prediction && data.confidence) {
            this.updateInfo(data.prediction, data.confidence);
        }
    }
    
    updateImage(panel, src) {
        const { image, placeholder } = panel;
        
        // 淡出占位符
        placeholder.style.opacity = '0';
        
        // 加载新图像
        const tempImg = new Image();
        tempImg.onload = () => {
            // 淡入新图像
            image.style.opacity = '0';
            image.src = src;
            
            requestAnimationFrame(() => {
                image.style.transition = `opacity ${this.options.animationDuration}ms ease`;
                image.style.opacity = '1';
            });
        };
        tempImg.src = src;
    }
    
    updateInfo(prediction, confidence) {
        const confidencePercent = (confidence * 100).toFixed(1);
        
        this.infoPanel.innerHTML = `
            <div class="info-item">
                <span class="info-label">模型关注区域</span>
                <span class="info-desc">红色区域表示高注意力</span>
            </div>
            <div class="info-item">
                <span class="info-label">预测置信度</span>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                    <span class="confidence-text">${confidencePercent}%</span>
                </div>
            </div>
        `;
    }
    
    reset() {
        // 重置所有面板
        [this.originalPanel, this.heatmapPanel, this.overlayPanel].forEach(panel => {
            if (panel) {
                panel.image.style.opacity = '0';
                panel.image.src = '';
                panel.placeholder.style.opacity = '1';
            }
        });
        
        // 重置信息
        this.infoPanel.innerHTML = `
            <div class="info-item">
                <span class="info-label">Grad-CAM</span>
                <span class="info-desc">显示模型关注的图像区域</span>
            </div>
        `;
    }
    
    setOverlayOpacity(opacity) {
        this.options.overlayOpacity = opacity;
        if (this.overlayPanel) {
            this.overlayPanel.image.style.opacity = opacity;
        }
    }
}

/**
 * 预测结果显示组件
 */
class PredictionDisplay {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container not found: ${containerId}`);
        }
        
        this.createElements();
    }
    
    createElements() {
        this.container.innerHTML = '';
        this.container.classList.add('prediction-container');
        
        // 主预测结果
        this.mainPrediction = document.createElement('div');
        this.mainPrediction.className = 'main-prediction';
        this.mainPrediction.innerHTML = `
            <div class="prediction-char">?</div>
            <div class="prediction-label">等待输入</div>
        `;
        
        // Top-5 预测
        this.top5Container = document.createElement('div');
        this.top5Container.className = 'top5-predictions';
        this.top5Container.innerHTML = '<div class="top5-title">可能的字符</div>';
        
        this.container.appendChild(this.mainPrediction);
        this.container.appendChild(this.top5Container);
    }
    
    update(data) {
        if (!data || !data.prediction) {
            this.reset();
            return;
        }
        
        const { prediction, confidence, top5 } = data;
        
        // 更新主预测
        const charEl = this.mainPrediction.querySelector('.prediction-char');
        const labelEl = this.mainPrediction.querySelector('.prediction-label');
        
        // 添加动画
        charEl.style.transform = 'scale(0.8)';
        charEl.style.opacity = '0';
        
        setTimeout(() => {
            charEl.textContent = prediction;
            charEl.style.transform = 'scale(1)';
            charEl.style.opacity = '1';
        }, 150);
        
        labelEl.textContent = `置信度: ${(confidence * 100).toFixed(1)}%`;
        
        // 更新Top-5
        this.updateTop5(top5);
    }
    
    updateTop5(top5) {
        if (!top5 || top5.length === 0) return;
        
        let html = '<div class="top5-title">可能的字符</div>';
        
        top5.forEach(([char, prob], index) => {
            const percent = (prob * 100).toFixed(1);
            const isTop = index === 0;
            
            html += `
                <div class="top5-item ${isTop ? 'top5-best' : ''}">
                    <span class="top5-rank">${index + 1}</span>
                    <span class="top5-char">${char}</span>
                    <div class="top5-bar">
                        <div class="top5-fill" style="width: ${percent}%"></div>
                    </div>
                    <span class="top5-percent">${percent}%</span>
                </div>
            `;
        });
        
        this.top5Container.innerHTML = html;
        
        // 动画
        const items = this.top5Container.querySelectorAll('.top5-item');
        items.forEach((item, index) => {
            item.style.opacity = '0';
            item.style.transform = 'translateX(-10px)';
            
            setTimeout(() => {
                item.style.transition = 'all 0.3s ease';
                item.style.opacity = '1';
                item.style.transform = 'translateX(0)';
            }, index * 50);
        });
    }
    
    reset() {
        const charEl = this.mainPrediction.querySelector('.prediction-char');
        const labelEl = this.mainPrediction.querySelector('.prediction-label');
        
        charEl.textContent = '?';
        labelEl.textContent = '等待输入';
        
        this.top5Container.innerHTML = '<div class="top5-title">可能的字符</div>';
    }
}

// 导出
window.GradCAMDisplay = GradCAMDisplay;
window.PredictionDisplay = PredictionDisplay;

