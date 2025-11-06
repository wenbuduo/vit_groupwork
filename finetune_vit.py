"""
使用预训练ViT模型进行迁移学习
在自定义数据集上微调
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm


def get_dataloaders(data_dir='./data/custom_dataset', batch_size=32):
    """
    准备自定义数据集的数据加载器
    数据目录结构:
    data_dir/
        train/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img1.jpg
        val/
            class1/
            class2/
    """
    # 使用ViT的图像处理器
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    
    # 数据增强
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=processor.image_mean,
            std=processor.image_std
        )
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=processor.image_mean,
            std=processor.image_std
        )
    ])
    
    # 如果使用CIFAR-10作为示例
    train_dataset = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train
    )
    
    val_dataset = datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform_val
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader


def finetune_vit(num_classes=10, num_epochs=10):
    """微调预训练的ViT模型"""
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载预训练模型
    print("加载预训练ViT模型...")
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=num_classes,
        ignore_mismatched_sizes=True  # 允许分类头大小不匹配
    )
    model = model.to(device)
    
    # 冻结部分层（可选）
    # 只训练分类头，冻结backbone
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
    
    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params:,} / {total_params:,}")
    
    # 准备数据
    train_loader, val_loader = get_dataloaders(batch_size=32)
    
    # 优化器和损失函数
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5,
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 训练循环
    best_acc = 0
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = outputs.loss if hasattr(outputs, 'loss') else nn.CrossEntropyLoss()(outputs.logits, labels)
            
            if not hasattr(outputs, 'loss'):
                loss = nn.CrossEntropyLoss()(outputs.logits, labels)
            else:
                loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': train_loss / (pbar.n + 1),
                'acc': 100. * train_correct / train_total
            })
        
        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                if not hasattr(outputs, 'loss'):
                    loss = nn.CrossEntropyLoss()(outputs.logits, labels)
                else:
                    loss = outputs.loss
                
                val_loss += loss.item()
                _, predicted = outputs.logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # 统计
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained('./vit_finetuned')
            print(f"保存最佳模型! 验证准确率: {val_acc:.2f}%")
        
        scheduler.step()
    
    print(f"\n微调完成! 最佳验证准确率: {best_acc:.2f}%")
    return model


if __name__ == "__main__":
    # 开始微调
    model = finetune_vit(num_classes=10, num_epochs=5)
    
    print("\n模型已保存到 './vit_finetuned' 目录")
    print("使用以下代码加载模型:")
    print("from transformers import ViTForImageClassification")
    print("model = ViTForImageClassification.from_pretrained('./vit_finetuned')")
