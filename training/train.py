"""
深度CNN训练脚本

使用方法:
    python train.py --config default
    python train.py --config rtx5080_optimized
    python train.py --epochs 100 --batch_size 128
"""

import os
import sys
import argparse
import time
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.model import DeepCharCNN, DeepCharCNNLarge, create_model
from training.dataset import create_dataloaders, EMNISTDataset
from training.config import TrainingConfig, get_config


def set_seed(seed: int):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=0.2):
    """Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class Trainer:
    """训练器类"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 设置随机种子
        set_seed(config.seed)
        
        # 创建模型
        print(f"创建模型: {config.model_type}")
        self.model = create_model(
            model_type=config.model_type,
            num_classes=config.num_classes
        ).to(self.device)
        
        print(f"模型参数量: {self.model.count_parameters():,}")
        
        # 创建数据加载器
        print("加载数据集...")
        self.train_loader, self.val_loader = create_dataloaders(
            root=config.data_root,
            batch_size=config.batch_size,
            img_size=config.img_size,
            num_workers=config.num_workers
        )
        print(f"训练集: {len(self.train_loader.dataset):,} 样本")
        print(f"验证集: {len(self.val_loader.dataset):,} 样本")
        
        # 损失函数 (带标签平滑)
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing
        )
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas
        )
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 混合精度训练
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # 日志
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        log_dir = self.save_dir / 'logs' / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(log_dir)
        
        # 训练状态
        self.best_acc = 0
        self.epochs_no_improve = 0
        self.global_step = 0
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        config = self.config
        total_steps = len(self.train_loader) * config.epochs
        warmup_steps = len(self.train_loader) * config.warmup_epochs
        
        if config.scheduler_type == 'cosine':
            # Cosine Annealing with Warmup
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return config.min_lr / config.learning_rate + \
                    (1 - config.min_lr / config.learning_rate) * \
                    0.5 * (1 + np.cos(np.pi * progress))
            
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        elif config.scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=config.epochs // 3, 
                gamma=0.1
            )
        
        elif config.scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=config.min_lr
            )
        
        else:
            raise ValueError(f"Unknown scheduler: {config.scheduler_type}")
    
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Mixup数据增强
            use_mixup = (
                self.config.mixup_alpha > 0 and 
                random.random() < self.config.mixup_prob
            )
            
            if use_mixup:
                images, labels_a, labels_b, lam = mixup_data(
                    images, labels, self.config.mixup_alpha
                )
            
            self.optimizer.zero_grad()
            
            # 混合精度前向传播
            if self.scaler:
                with autocast():
                    outputs = self.model(images)
                    if use_mixup:
                        loss = mixup_criterion(
                            self.criterion, outputs, labels_a, labels_b, lam
                        )
                    else:
                        loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                if use_mixup:
                    loss = mixup_criterion(
                        self.criterion, outputs, labels_a, labels_b, lam
                    )
                else:
                    loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
            
            # 更新学习率
            if self.config.scheduler_type == 'cosine':
                self.scheduler.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            if not use_mixup:
                correct += predicted.eq(labels).sum().item()
            else:
                correct += (
                    lam * predicted.eq(labels_a).sum().item() +
                    (1 - lam) * predicted.eq(labels_b).sum().item()
                )
            
            self.global_step += 1
            
            # 日志
            if batch_idx % self.config.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', current_lr, self.global_step)
                
                elapsed = time.time() - start_time
                samples_per_sec = (batch_idx + 1) * self.config.batch_size / elapsed
                
                print(
                    f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Acc: {100. * correct / total:.2f}% "
                    f"LR: {current_lr:.2e} "
                    f"Speed: {samples_per_sec:.0f} samples/s"
                )
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self, epoch: int):
        """验证"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            if self.scaler:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/accuracy', accuracy, epoch)
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, accuracy: float, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'accuracy': accuracy,
            'config': self.config
        }
        
        # 保存最新
        torch.save(checkpoint, self.save_dir / 'latest_checkpoint.pt')
        
        # 保存最佳
        if is_best:
            torch.save(self.model.state_dict(), self.save_dir / 'best_model.pt')
            # 同时复制到backend/weights
            backend_weights = Path(__file__).parent.parent / 'backend' / 'weights'
            backend_weights.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), backend_weights / 'best_model.pt')
            print(f"  >> 保存最佳模型 (Acc: {accuracy:.2f}%)")
        
        # 定期保存
        if epoch % self.config.save_interval == 0:
            torch.save(checkpoint, self.save_dir / f'checkpoint_epoch_{epoch}.pt')
    
    def train(self):
        """完整训练流程"""
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)
        
        for epoch in range(1, self.config.epochs + 1):
            print(f"\n--- Epoch {epoch}/{self.config.epochs} ---")
            
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate(epoch)
            
            print(
                f"Epoch {epoch} 完成: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            # 更新学习率 (对于非cosine调度器)
            if self.config.scheduler_type == 'step':
                self.scheduler.step()
            elif self.config.scheduler_type == 'plateau':
                self.scheduler.step(val_acc)
            
            # 保存检查点
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
            
            self.save_checkpoint(epoch, val_acc, is_best)
            
            # 早停
            if self.epochs_no_improve >= self.config.early_stopping_patience:
                print(f"\n早停: {self.epochs_no_improve} epochs 无改进")
                break
        
        print("\n" + "=" * 60)
        print(f"训练完成! 最佳验证准确率: {self.best_acc:.2f}%")
        print("=" * 60)
        
        self.writer.close()
        
        return self.best_acc


def main():
    parser = argparse.ArgumentParser(description='训练深度CNN字符识别模型')
    
    # 配置选择
    parser.add_argument('--config', type=str, default='rtx5080_optimized',
                        choices=['default', 'fast_debug', 'high_accuracy', 'rtx5080_optimized'],
                        help='预设配置名称')
    
    # 覆盖配置的参数
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--model_type', type=str, default=None)
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    
    args = parser.parse_args()
    
    # 加载配置
    config = get_config(args.config)
    
    # 覆盖指定参数
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.model_type is not None:
        config.model_type = args.model_type
    if args.data_root is not None:
        config.data_root = args.data_root
    if args.save_dir is not None:
        config.save_dir = args.save_dir
    
    # 打印配置
    print("\n训练配置:")
    print("-" * 40)
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
    print("-" * 40)
    
    # 检查GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\nGPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("\n警告: 未检测到GPU，将使用CPU训练")
    
    # 开始训练
    trainer = Trainer(config)
    best_acc = trainer.train()
    
    return best_acc


if __name__ == '__main__':
    main()

