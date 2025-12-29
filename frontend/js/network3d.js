/**
 * Three.js 3D神经网络可视化模块
 * 展示CNN各层的激活状态
 */

class Network3DVisualizer {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container not found: ${containerId}`);
        }
        
        // 配置
        this.options = {
            backgroundColor: options.backgroundColor || 0xf5f5f7,
            cameraDistance: options.cameraDistance || 15,
            layerSpacing: options.layerSpacing || 3,
            neuronSize: options.neuronSize || 0.15,
            colorScheme: options.colorScheme || {
                cold: new THREE.Color(0x3b82f6),  // 蓝色 - 低激活
                warm: new THREE.Color(0xef4444),  // 红色 - 高激活
                neutral: new THREE.Color(0xe5e7eb) // 灰色 - 中性
            },
            animationSpeed: options.animationSpeed || 0.01
        };
        
        // Three.js 组件
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.layers = [];
        this.connections = [];
        this.animationId = null;
        
        // 网络结构定义
        this.networkStructure = [
            { name: 'input', label: '输入层', size: [1, 64, 64], displaySize: [8, 8], type: 'input' },
            { name: 'stem', label: 'Stem', size: [64, 16, 16], displaySize: [8, 8], type: 'conv' },
            { name: 'stage1', label: 'Stage 1', size: [128, 16, 16], displaySize: [8, 8], type: 'conv' },
            { name: 'stage2', label: 'Stage 2', size: [256, 8, 8], displaySize: [8, 8], type: 'conv' },
            { name: 'stage3', label: 'Stage 3', size: [512, 4, 4], displaySize: [8, 8], type: 'conv' },
            { name: 'fc1', label: 'FC-256', size: [256], displaySize: [16, 16], type: 'fc' },
            { name: 'output', label: '输出层', size: [62], displaySize: [8, 8], type: 'output' }
        ];
        
        this.init();
    }
    
    init() {
        this.setupScene();
        this.setupCamera();
        this.setupRenderer();
        this.setupLights();
        this.setupControls();
        this.createNetwork();
        this.animate();
        
        // 监听窗口大小变化
        window.addEventListener('resize', () => this.onResize());
    }
    
    setupScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(this.options.backgroundColor);
        
        // 添加雾效果增加深度感
        this.scene.fog = new THREE.Fog(this.options.backgroundColor, 20, 50);
    }
    
    setupCamera() {
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
        this.camera.position.set(0, 5, this.options.cameraDistance);
        this.camera.lookAt(0, 0, 0);
    }
    
    setupRenderer() {
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);
    }
    
    setupLights() {
        // 环境光
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);
        
        // 主方向光
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 20, 10);
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);
        
        // 补光
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
        fillLight.position.set(-10, 10, -10);
        this.scene.add(fillLight);
    }
    
    setupControls() {
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.minDistance = 5;
        this.controls.maxDistance = 50;
        this.controls.maxPolarAngle = Math.PI * 0.9;
    }
    
    createNetwork() {
        const totalLayers = this.networkStructure.length;
        const totalWidth = (totalLayers - 1) * this.options.layerSpacing;
        const startX = -totalWidth / 2;
        
        this.networkStructure.forEach((layerDef, index) => {
            const x = startX + index * this.options.layerSpacing;
            const layer = this.createLayer(layerDef, x, index);
            this.layers.push(layer);
            this.scene.add(layer.group);
        });
        
        // 创建层间连接（可选）
        this.createConnections();
        
        // 添加层标签
        this.createLabels();
    }
    
    createLayer(layerDef, x, index) {
        const group = new THREE.Group();
        group.position.x = x;
        
        const neurons = [];
        const [displayW, displayH] = layerDef.displaySize;
        
        // 创建神经元网格
        const geometry = new THREE.SphereGeometry(this.options.neuronSize, 16, 16);
        
        for (let row = 0; row < displayH; row++) {
            for (let col = 0; col < displayW; col++) {
                const material = new THREE.MeshPhongMaterial({
                    color: this.options.colorScheme.neutral,
                    shininess: 100,
                    transparent: true,
                    opacity: 0.9
                });
                
                const neuron = new THREE.Mesh(geometry, material);
                
                // 网格布局
                const y = (row - displayH / 2 + 0.5) * this.options.neuronSize * 2.5;
                const z = (col - displayW / 2 + 0.5) * this.options.neuronSize * 2.5;
                
                neuron.position.set(0, y, z);
                neuron.castShadow = true;
                neuron.receiveShadow = true;
                
                group.add(neuron);
                neurons.push(neuron);
            }
        }
        
        // 添加层背板
        const backplaneGeometry = new THREE.PlaneGeometry(
            displayW * this.options.neuronSize * 3,
            displayH * this.options.neuronSize * 3
        );
        const backplaneMaterial = new THREE.MeshPhongMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.1,
            side: THREE.DoubleSide
        });
        const backplane = new THREE.Mesh(backplaneGeometry, backplaneMaterial);
        backplane.rotation.y = Math.PI / 2;
        backplane.position.x = -0.1;
        group.add(backplane);
        
        return {
            group,
            neurons,
            def: layerDef,
            index
        };
    }
    
    createConnections() {
        // 创建层间的连接线（简化显示）
        const material = new THREE.LineBasicMaterial({
            color: 0xcccccc,
            transparent: true,
            opacity: 0.2
        });
        
        for (let i = 0; i < this.layers.length - 1; i++) {
            const fromLayer = this.layers[i];
            const toLayer = this.layers[i + 1];
            
            // 只连接部分神经元（避免过于密集）
            const step = Math.max(1, Math.floor(fromLayer.neurons.length / 8));
            
            for (let j = 0; j < fromLayer.neurons.length; j += step) {
                const fromNeuron = fromLayer.neurons[j];
                const fromPos = new THREE.Vector3();
                fromNeuron.getWorldPosition(fromPos);
                
                // 连接到下一层的随机神经元
                const toIdx = Math.floor(Math.random() * toLayer.neurons.length);
                const toNeuron = toLayer.neurons[toIdx];
                const toPos = new THREE.Vector3();
                toNeuron.getWorldPosition(toPos);
                
                const points = [fromPos, toPos];
                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                const line = new THREE.Line(geometry, material);
                
                this.connections.push(line);
                this.scene.add(line);
            }
        }
    }
    
    createLabels() {
        // 使用CSS2DRenderer或简单的sprite创建标签
        // 这里使用canvas纹理创建sprite标签
        this.layers.forEach((layer, index) => {
            const sprite = this.createTextSprite(layer.def.label, {
                fontSize: 24,
                fontFace: 'Inter, SF Pro Display, system-ui',
                textColor: '#1D1D1F'
            });
            
            sprite.position.set(
                layer.group.position.x,
                -2.5,
                0
            );
            sprite.scale.set(2, 1, 1);
            
            this.scene.add(sprite);
        });
    }
    
    createTextSprite(text, options = {}) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        
        canvas.width = 256;
        canvas.height = 128;
        
        context.font = `${options.fontSize || 24}px ${options.fontFace || 'Arial'}`;
        context.fillStyle = options.textColor || '#000000';
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        context.fillText(text, canvas.width / 2, canvas.height / 2);
        
        const texture = new THREE.CanvasTexture(canvas);
        texture.needsUpdate = true;
        
        const material = new THREE.SpriteMaterial({
            map: texture,
            transparent: true
        });
        
        return new THREE.Sprite(material);
    }
    
    updateActivations(activations) {
        if (!activations) return;
        
        this.layers.forEach((layer) => {
            const layerData = activations[layer.def.name];
            if (!layerData) return;
            
            let values;
            
            if (layerData.channel_activations) {
                // 卷积层 - 使用通道激活
                values = layerData.channel_activations;
            } else if (layerData.values) {
                // 全连接层
                values = layerData.values;
            } else {
                return;
            }
            
            // 归一化值
            const maxVal = Math.max(...values.map(Math.abs));
            const normalizedValues = values.map(v => v / (maxVal + 1e-8));
            
            // 更新神经元颜色
            layer.neurons.forEach((neuron, idx) => {
                const valueIdx = idx % normalizedValues.length;
                const activation = normalizedValues[valueIdx];
                
                // 使用激活值插值颜色
                const color = this.getActivationColor(activation);
                
                // 平滑过渡
                gsap.to(neuron.material.color, {
                    r: color.r,
                    g: color.g,
                    b: color.b,
                    duration: 0.3,
                    ease: 'power2.out'
                });
                
                // 根据激活调整大小
                const scale = 0.8 + Math.abs(activation) * 0.4;
                gsap.to(neuron.scale, {
                    x: scale,
                    y: scale,
                    z: scale,
                    duration: 0.3,
                    ease: 'power2.out'
                });
            });
        });
    }
    
    getActivationColor(activation) {
        const { cold, warm, neutral } = this.options.colorScheme;
        
        if (activation > 0) {
            return neutral.clone().lerp(warm, activation);
        } else {
            return neutral.clone().lerp(cold, -activation);
        }
    }
    
    // 类别标签到索引的映射 (0-9, A-Z, a-z)
    labelToIndex(label) {
        if (!label || label.length !== 1) return -1;
        const char = label.charAt(0);
        const code = char.charCodeAt(0);
        
        // 0-9 -> 索引 0-9
        if (code >= 48 && code <= 57) {
            return code - 48;
        }
        // A-Z -> 索引 10-35
        if (code >= 65 && code <= 90) {
            return code - 65 + 10;
        }
        // a-z -> 索引 36-61
        if (code >= 97 && code <= 122) {
            return code - 97 + 36;
        }
        return -1;
    }
    
    highlightPrediction(classIndex, top5) {
        // 高亮输出层的预测结果
        const outputLayer = this.layers[this.layers.length - 1];
        if (!outputLayer) return;
        
        // 重置所有神经元为低亮度
        outputLayer.neurons.forEach((neuron, idx) => {
            gsap.to(neuron.material, {
                opacity: 0.3,
                duration: 0.3
            });
            gsap.to(neuron.scale, {
                x: 0.8,
                y: 0.8,
                z: 0.8,
                duration: 0.3
            });
            // 重置颜色为中性
            gsap.to(neuron.material.color, {
                r: this.options.colorScheme.neutral.r,
                g: this.options.colorScheme.neutral.g,
                b: this.options.colorScheme.neutral.b,
                duration: 0.3
            });
        });
        
        // 高亮top5 - 使用正确的类别索引
        if (top5) {
            top5.forEach(([label, prob], rank) => {
                const idx = this.labelToIndex(label);  // 根据标签获取正确的索引
                if (idx >= 0 && idx < outputLayer.neurons.length) {
                    const neuron = outputLayer.neurons[idx];
                    
                    gsap.to(neuron.material, {
                        opacity: 1,
                        duration: 0.3
                    });
                    
                    const scale = 1 + prob * 0.5;
                    gsap.to(neuron.scale, {
                        x: scale,
                        y: scale,
                        z: scale,
                        duration: 0.3
                    });
                    
                    // rank 0 是最高概率，颜色最暖（红色）
                    const color = new THREE.Color().setHSL(0.3 - rank * 0.06, 0.8, 0.5);
                    gsap.to(neuron.material.color, {
                        r: color.r,
                        g: color.g,
                        b: color.b,
                        duration: 0.3
                    });
                }
            });
        }
    }
    
    reset() {
        // 重置所有神经元到中性状态
        this.layers.forEach(layer => {
            layer.neurons.forEach(neuron => {
                gsap.to(neuron.material.color, {
                    r: this.options.colorScheme.neutral.r,
                    g: this.options.colorScheme.neutral.g,
                    b: this.options.colorScheme.neutral.b,
                    duration: 0.5
                });
                gsap.to(neuron.scale, {
                    x: 1,
                    y: 1,
                    z: 1,
                    duration: 0.5
                });
                gsap.to(neuron.material, {
                    opacity: 0.9,
                    duration: 0.5
                });
            });
        });
    }
    
    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());
        
        // 更新控制器
        this.controls.update();
        
        // 渲染场景
        this.renderer.render(this.scene, this.camera);
    }
    
    onResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        
        this.renderer.setSize(width, height);
    }
    
    dispose() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        // 清理Three.js资源
        this.scene.traverse((object) => {
            if (object.geometry) {
                object.geometry.dispose();
            }
            if (object.material) {
                if (Array.isArray(object.material)) {
                    object.material.forEach(m => m.dispose());
                } else {
                    object.material.dispose();
                }
            }
        });
        
        this.renderer.dispose();
        this.container.removeChild(this.renderer.domElement);
    }
    
    // 相机动画
    focusOnLayer(layerIndex) {
        if (layerIndex < 0 || layerIndex >= this.layers.length) return;
        
        const layer = this.layers[layerIndex];
        const targetPosition = new THREE.Vector3(
            layer.group.position.x + 5,
            2,
            5
        );
        
        gsap.to(this.camera.position, {
            x: targetPosition.x,
            y: targetPosition.y,
            z: targetPosition.z,
            duration: 1,
            ease: 'power2.inOut'
        });
        
        gsap.to(this.controls.target, {
            x: layer.group.position.x,
            y: 0,
            z: 0,
            duration: 1,
            ease: 'power2.inOut'
        });
    }
    
    resetCamera() {
        gsap.to(this.camera.position, {
            x: 0,
            y: 5,
            z: this.options.cameraDistance,
            duration: 1,
            ease: 'power2.inOut'
        });
        
        gsap.to(this.controls.target, {
            x: 0,
            y: 0,
            z: 0,
            duration: 1,
            ease: 'power2.inOut'
        });
    }
}

// 导出
window.Network3DVisualizer = Network3DVisualizer;

