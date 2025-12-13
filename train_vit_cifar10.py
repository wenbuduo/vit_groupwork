"""
在 CIFAR-10 数据集上训练 Vision Transformer
"""
import sys
sys.path.append("/kaggle/input/vit-groupwork/vit_groupwork")

from vit_from_scratch import vit_tiny_patch16_224

import os
import logging
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# Logger 设置
def setup_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(save_dir, f"training_{time_str}.log")

    logger = logging.getLogger("ViT-Training")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_path)
    console_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_path

# EMA 
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()
        self.backup = {}

# 学习率预热
def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    return optim.lr_scheduler.LambdaLR(optimizer, f)

# 数据加载
def get_dataloaders(batch_size=64, validation_split=0.1):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    # 加载训练集并划分验证集
    full_train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    
    # 划分训练集和验证集
    indices = list(range(len(full_train_dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=validation_split, random_state=42, stratify=full_train_dataset.targets
    )
    
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(
        datasets.CIFAR10(root="./data", train=True, download=False, transform=transform_val),
        val_indices
    )
    
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader

# 训练 / 验证
def train_epoch(model, loader, criterion, optimizer, device, ema=None, grad_clip=1.0):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    
    progress_bar = tqdm(loader, desc="Training", leave=False)
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 梯度裁剪
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # 更新EMA
        if ema is not None:
            ema.update()

        # 计算指标
        loss_sum += loss.item()
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
        
        # 更新进度条
        avg_loss = loss_sum / (batch_idx + 1)
        acc = 100. * correct / total
        progress_bar.set_postfix({'loss': f'{avg_loss:.3f}', 'acc': f'{acc:.2f}%'})

    return loss_sum / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_sum += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算每个类别的准确率
    class_correct = np.zeros(10)
    class_total = np.zeros(10)
    for pred, label in zip(all_preds, all_labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
    
    class_acc = class_correct / (class_total + 1e-6)
    
    return loss_sum / len(loader), 100. * correct / total, class_acc

# 曲线绘制
def plot_curves(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    axes[0, 0].plot(epochs, history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[0, 1].plot(epochs, history['train_acc'], label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 学习率曲线
    if 'lr' in history:
        axes[1, 0].plot(epochs, history['lr'], label='Learning Rate', color='red', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 训练/验证差距
    if len(history['train_acc']) > 0 and len(history['val_acc']) > 0:
        gap = np.array(history['train_acc']) - np.array(history['val_acc'])
        axes[1, 1].plot(epochs, gap, label='Train-Val Gap', color='purple', linewidth=2)
        axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Gap (%)')
        axes[1, 1].set_title('Train-Validation Accuracy Gap')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 单独保存损失和准确率曲线
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    
    axes2[0].plot(epochs, history['train_loss'], label='Train', linewidth=2)
    axes2[0].plot(epochs, history['val_loss'], label='Validation', linewidth=2)
    axes2[0].set_xlabel('Epoch')
    axes2[0].set_ylabel('Loss')
    axes2[0].set_title('Loss Curve')
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)
    
    axes2[1].plot(epochs, history['train_acc'], label='Train', linewidth=2)
    axes2[1].plot(epochs, history['val_acc'], label='Validation', linewidth=2)
    axes2[1].set_xlabel('Epoch')
    axes2[1].set_ylabel('Accuracy (%)')
    axes2[1].set_title('Accuracy Curve')
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_acc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

# 保存配置
def save_config(config, save_dir):
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

# 主函数
def main():
    # 超参数配置
    config = {
        'batch_size': 64,
        'epochs': 100,
        'lr': 3e-4,
        'weight_decay': 0.05,
        'label_smoothing': 0.1,
        'warmup_epochs': 5,
        'patience': 15,
        'grad_clip': 1.0,
        'ema_decay': 0.999,
        'validation_split': 0.1,
    }
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_DIR = "/kaggle/working/outputs"
    
    # 创建保存目录
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(SAVE_DIR, f"run_{time_str}")
    os.makedirs(run_dir, exist_ok=True)
    
    logger, log_path = setup_logger(run_dir)
    
    logger.info("启动 ViT CIFAR-10 训练")
    logger.info(f"日志文件: {log_path}")
    logger.info(f"运行目录: {run_dir}")
    logger.info(f"设备: {DEVICE}")
    logger.info("超参数配置:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 50)
    
    # 保存配置
    save_config(config, run_dir)
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=config['batch_size'], 
        validation_split=config['validation_split']
    )
    
    logger.info(f"训练集大小: {len(train_loader.dataset)}")
    logger.info(f"验证集大小: {len(val_loader.dataset)}")
    logger.info(f"测试集大小: {len(test_loader.dataset)}")
    
    # 初始化模型
    model = vit_tiny_patch16_224(num_classes=10).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数量: {total_params:,}")
    logger.info(f"可训练参数量: {trainable_params:,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器
    warmup_scheduler = warmup_lr_scheduler(
        optimizer, 
        warmup_iters=config['warmup_epochs'] * len(train_loader), 
        warmup_factor=0.001
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['epochs'] * len(train_loader)
    )
    
    # EMA
    ema = ModelEMA(model, decay=config['ema_decay'])
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': [],
        'best_val_acc': 0.0,
        'best_epoch': 0
    }
    
    best_acc = 0.0
    patience_cnt = 0
    
    logger.info("开始训练")
    
    for epoch in range(config['epochs']):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{config['epochs']}")
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, 
            ema=ema, grad_clip=config['grad_clip']
        )
        
        # 更新学习率调度器
        warmup_scheduler.step()
        cosine_scheduler.step()
        
        # 使用EMA模型进行验证
        ema.apply_shadow()
        val_loss, val_acc, class_acc = validate(model, val_loader, criterion, DEVICE)
        ema.restore()
        
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # 输出类别准确率
        logger.info("类别准确率:")
        for i, acc in enumerate(class_acc):
            logger.info(f"  类别 {i}: {acc*100:.2f}%")
        
        logger.info(
            f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
            f"Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}% | "
            f"LR: {current_lr:.2e}"
        )
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            history['best_val_acc'] = best_acc
            history['best_epoch'] = epoch + 1
            patience_cnt = 0
            
            # 保存最佳模型（使用EMA权重）
            ema.apply_shadow()
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'history': history,
            }, os.path.join(run_dir, "vit_cifar10_best.pth"))
            ema.restore()
            
            logger.info(f"✓ 最佳模型已保存 (Acc: {best_acc:.2f}%)")
            
            # 同时保存检查点
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': cosine_scheduler.state_dict(),
                'ema_shadow': ema.shadow,
                'val_acc': val_acc,
                'history': history,
            }, os.path.join(run_dir, f"checkpoint_epoch_{epoch+1}.pth"))
        else:
            patience_cnt += 1
            if patience_cnt >= config['patience']:
                logger.info(f"早停触发 (Patience={config['patience']})")
                break
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': cosine_scheduler.state_dict(),
                'ema_shadow': ema.shadow,
                'val_acc': val_acc,
                'history': history,
            }, os.path.join(run_dir, f"checkpoint_epoch_{epoch+1}.pth"))
            logger.info(f"检查点已保存 (Epoch {epoch+1})")
    
    # 最终测试
    logger.info("\n" + "="*60)
    logger.info("最终测试")
    
    # 使用最佳模型进行测试
    checkpoint = torch.load(os.path.join(run_dir, "vit_cifar10_best.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_class_acc = validate(model, test_loader, criterion, DEVICE)
    logger.info(f"测试结果: Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
    logger.info("测试集类别准确率:")
    for i, acc in enumerate(test_class_acc):
        logger.info(f"  类别 {i}: {acc*100:.2f}%")
    
    # 绘制曲线
    plot_curves(history, run_dir)
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_acc': test_acc,
        'history': history,
        'config': config,
    }, os.path.join(run_dir, "vit_cifar10_final.pth"))
    
    logger.info("\n" + "="*60)
    logger.info(f"训练完成!")
    logger.info(f"最佳验证准确率: {history['best_val_acc']:.2f}% (Epoch {history['best_epoch']})")
    logger.info(f"最终测试准确率: {test_acc:.2f}%")
    logger.info(f"所有文件保存在: {run_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
