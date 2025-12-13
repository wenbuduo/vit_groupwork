"""
使用预训练 ViT 模型进行迁移学习
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm
import os
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_curves(history, save_path='./logs/training_curves.png'):
    """绘制模型训练曲线"""
    if not history['train_loss']: return
    epochs = range(1, len(history['train_loss']) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()
    
    # 绘制曲线(训练损失、训练准确率、验证准确率)
    line1 = ax.plot(epochs, history['train_loss'], 'b-', linewidth=2, marker='o', label='Train Loss')
    line2 = ax2.plot(epochs, history['train_acc'], 'g-', linewidth=2, marker='s', label='Train Acc')
    line3 = ax2.plot(epochs, history['val_acc'], 'r-', linewidth=2, marker='^', label='Val Acc')
    
    # 坐标轴设置
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', color='b', fontsize=12)
    ax.tick_params(axis='y', labelcolor='b')
    ax2.set_ylabel('Accuracy (%)', color='g', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='g')
    
    # 合并图例
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left', fontsize=10)
    
    # 标题和网格
    ax.set_title('Training Progress', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # # 在最佳 epoch 处标注
    # best_epoch = history['val_acc'].index(max(history['val_acc'])) + 1
    # best_val_acc = max(history['val_acc'])
    # ax2.annotate(f'Best: {best_val_acc:.2f}%\n(Epoch {best_epoch})', xy=(best_epoch, best_val_acc), 
    #             xytext=(best_epoch+0.2, best_val_acc-1), arrowprops=dict(arrowstyle='->', color='purple', lw=1.5),
    #             fontsize=10, color='purple', fontweight='bold')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
def setup_logger(log_dir='./logs', is_main=True):
    """设置日志，将 DDP 非主进程设置为静默，避免非主进程重复输出日志内容"""
    if not is_main:
        logging.basicConfig(level=logging.CRITICAL, force=True)
        return None
    
    # 创建日志目录和文件，文件名附带时间戳
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # 设置日志格式
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()], force=True)
    return log_file

class TransformSubset(Dataset):  
    def __init__(self, dataset, indices, transform):
        self.dataset, self.indices, self.transform = dataset, indices, transform
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        return self.transform(img) if self.transform else img, label

def get_dataloaders(args, rank, gpu_num, is_ddp):
    """构建 DataLoader"""
    processor = ViTImageProcessor.from_pretrained(args.model_dir)
    
    # 图像增强策略
    tf_train = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        # 随机采用 2 种增强策略, 强度为 9
        transforms.RandAugment(2, 9),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        transforms.RandomErasing(0.25)   # 随机擦除, 概率 25%
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])
    
    torch.manual_seed(42)
    
    # 数据集加载
    train_val_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True)
    
    # 数据集划分和变换
    total_size = len(train_val_dataset)
    train_size = int(0.9 * total_size)   # 90% 训练集, 10% 验证集
    indices = torch.randperm(total_size).tolist()
    train_ds = TransformSubset(train_val_dataset, indices[:train_size], tf_train)
    val_ds = TransformSubset(train_val_dataset, indices[train_size:], tf_eval)
    test_ds = TransformSubset(test_dataset, list(range(len(test_dataset))), tf_eval)
    
    # 构建 DataLoader
    loaders = {}
    train_sampler = None
    for name, ds, shuffle in [('train', train_ds, True), ('val', val_ds, False), ('test', test_ds, False)]:
        sampler = DistributedSampler(ds, num_replicas=gpu_num, rank=rank, shuffle=shuffle) if is_ddp else None
        loaders[name] = DataLoader(ds, batch_size=args.batch_size, shuffle=(shuffle and not is_ddp),
                                   sampler=sampler, num_workers=8, pin_memory=True, persistent_workers=True)
        if name == 'train': train_sampler = sampler  # 训练集采样器
            
    return loaders, train_sampler   # 返回所有 DataLoader 及训练集采样器

def run_epoch(model, loader, device, is_main, epoch, num_epochs, mode='Train', optimizer=None):
    """训练、验证和测试循环"""
    is_train = (optimizer is not None)
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0, 0, 0  # 累计损失、正确数和样本数
    
    if epoch == -1: desc = f'[Final Test]'
    else: desc = f'Epoch {epoch+1}/{num_epochs} [{mode}]' 
    # 仅主进程显示进度条
    iterator = tqdm(loader, desc=desc) if is_main else loader
    
    with torch.set_grad_enabled(is_train):
        for imgs, labels in iterator:
            imgs, labels = imgs.to(device), labels.to(device)
            batch_size = labels.size(0) 
            if is_train: optimizer.zero_grad()
            outputs = model(imgs, labels=labels) #前向传播
            if is_train:
                outputs.loss.backward() # 反向传播
                optimizer.step()   # 参数更新
            
            total_loss += outputs.loss.item() * batch_size
            correct += outputs.logits.argmax(1).eq(labels).sum().item()
            total += batch_size
            
            # 更新进度条
            if is_main and is_train and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({'loss': total_loss/total, 'acc': 100.*correct/total})
                
    return total_loss, correct, total  # 返回累计损失、正确数和样本数

def sync_metrics(loss, correct, total, device, is_ddp):
    """同步 DDP 指标"""
    
    if not is_ddp: return loss, correct, total
    metric = torch.tensor([loss, correct, total], dtype=torch.float32, device=device)
    dist.all_reduce(metric, op=dist.ReduceOp.SUM)
    return metric.tolist()

# rank: 当前进程的编号， gpu_num: 总GPU数, gpu_list: 使用的GPU列表, args: 命令行参数
def train(rank, gpu_num, gpu_list, args):
    is_ddp = gpu_num > 1   # 是否启用多卡训练
    if is_ddp:
        # 分布式的相关设置
        os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'] = 'localhost', '12355'
        dist.init_process_group('nccl', rank=rank, world_size=gpu_num)
        if rank != 0: 
            import transformers; transformers.logging.set_verbosity_error()
            import warnings; warnings.filterwarnings('ignore')
    # 根据 rank 选择当前进程使用的 GPU
    gpu = gpu_list[rank]
    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')
    is_main = (rank == 0)   # DDP 主进程的标志
    
    try:
        # 记录日志信息
        log_file = setup_logger(is_main=is_main)
        logging.info(f"启动训练: {'DDP' if is_ddp else '单卡'} | GPU: {gpu_list if is_ddp else [gpu]}")
        if is_main and log_file:
            logging.info(f"日志文件: {log_file}"); logging.info("="*50); logging.info("超参数配置:")
            logging.info(f"  - Batch Size: {args.batch_size}")
            logging.info(f"  - Epochs: {args.num_epochs}")
            logging.info(f"  - Learning Rate: {1e-5 if args.full_finetune=='y' else 5e-5}")
            logging.info(f"  - Patience: {args.patience}")
            logging.info("="*50)
        # 加载预训练模型, model_dir 可以是本地路径或 HuggingFace 模型名
        model = ViTForImageClassification.from_pretrained(
            args.model_dir, num_labels=args.num_classes, 
            ignore_mismatched_sizes=True,       # 保留 backbone, 重新初始化分类头
            attention_probs_dropout_prob=0.05,   # 将注意力 Dropout
            hidden_dropout_prob=0.05             # 将隐藏层 Dropout
        ).to(device)
        
        # 选择微调模式：冻结 Backbone 或 全参微调
        if args.full_finetune == 'n':
            for n, p in model.named_parameters(): 
                if 'classifier' not in n: p.requires_grad = False
            logging.info("模式: 冻结 Backbone")  # 仅微调分类头
        else: logging.info("模式: 全参微调")     # 微调所有参数
        
        # 使用 DDP 包装模型
        if is_ddp: model = DDP(model, device_ids=[gpu])
        
        # 计算可训练参数量
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"可训练参数: {trainable:,} / {total_params:,}")
        
        # 准备数据、优化器和学习率调度器
        loaders, train_sampler = get_dataloaders(args, rank, gpu_num, is_ddp)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=1e-5 if args.full_finetune=='y' else 5e-5,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
        
        # 初始化训练的状态
        history = {'train_loss': [], 'train_acc': [], 'val_acc': []} if is_main else None
        best_acc, patience = 0.0, 0 
        early_stop = torch.tensor([0], dtype=torch.int32, device=device)
        
        # 在训练前测试模型在测试集上的准确率
        logging.info("="*20 + " 训练前测试 " + "="*20)
        initial_loss, initial_corr, initial_tot = run_epoch(model, loaders['test'], device, is_main, -1, args.num_epochs, 'Initial Test')
        initial_loss, initial_corr, initial_tot = sync_metrics(initial_loss, initial_corr, initial_tot, device, is_ddp)
        initial_acc = 100. * initial_corr / initial_tot   # 计算准确率
        avg_initial_loss = initial_loss / initial_tot     # 计算平均损失
        logging.info(f"训练前测试集 Loss: {avg_initial_loss:.4f} | Acc: {initial_acc:.2f}%")
        logging.info("="*50 + "\n")
        logging.info("="*20 + " 开始训练 " + "="*20)
        
        # 主循环
        for epoch in range(args.num_epochs):
            if is_ddp: train_sampler.set_epoch(epoch) # 设置 epoch 以打乱数据
            
            # 训练过程和验证过程
            t_loss, t_corr, t_tot = run_epoch(model, loaders['train'], device, is_main, epoch, args.num_epochs, 'Train', optimizer)
            v_loss, v_corr, v_tot = run_epoch(model, loaders['val'], device, is_main, epoch, args.num_epochs, 'Val')
            
            # 指标同步与计算
            t_loss, t_corr, t_tot = sync_metrics(t_loss, t_corr, t_tot, device, is_ddp)
            v_loss, v_corr, v_tot = sync_metrics(v_loss, v_corr, v_tot, device, is_ddp)
            t_acc, v_acc = 100. * t_corr / t_tot, 100. * v_corr / v_tot
            avg_t_loss = t_loss / t_tot     # 训练集平均损失
            avg_v_loss = v_loss / v_tot     # 验证集平均损失
            logging.info(f"Epoch {epoch+1}/{args.num_epochs} | T-Loss: {avg_t_loss:.4f} T-Acc: {t_acc:.2f}% | V-Loss: {avg_v_loss:.4f} V-Acc: {v_acc:.2f}%")

            # 记录与保存训练数据和模型
            if is_main:
                history['train_loss'].append(avg_t_loss)
                history['train_acc'].append(t_acc)
                history['val_acc'].append(v_acc)
                plot_curves(history)   # 绘制训练曲线
                # 保存最佳模型
                if v_acc > best_acc:
                    best_acc, patience = v_acc, 0
                    try:
                        save_dir = './vit_finetuned'
                        os.makedirs(save_dir, exist_ok=True)
                        (model.module if is_ddp else model).save_pretrained(save_dir)
                        logging.info(f"✓ 最佳模型已保存 (Acc: {v_acc:.2f}%)")
                    except Exception as e:
                        logging.error(f"✗ 保存模型失败: {e}")
                else:
                    patience += 1      # 模型在验证集上性能未提升时耐心值加1
                    if patience >= args.patience:
                        logging.info(f"早停触发 (Patience: {patience})")
                        early_stop[0] = 1
            
            # 同步 early_stop 状态
            if is_ddp: dist.broadcast(early_stop, src=0)
            if early_stop.item(): break
            # 调整学习率
            scheduler.step()

        # 最终测试，加载最佳模型
        logging.info("="*20 + " 最终测试 " + "="*20)
        best_model_path = './vit_finetuned'
        config_file = os.path.join(best_model_path, 'config.json')
        if os.path.exists(config_file):
            if is_main: logging.info("加载最佳模型进行测试...")
            if is_ddp: dist.barrier() # 同步等待主进程完成模型保存
            test_model = ViTForImageClassification.from_pretrained(best_model_path).to(device)
            test_model.eval()
            if is_ddp: test_model = DDP(test_model, device_ids=[gpu])
        else:
            logging.warning("未找到保存的最佳模型，使用当前模型")
            test_model = model
            test_model.eval()
        
        # 在测试集上评估模型
        test_loss, test_corr, test_total = run_epoch(test_model, loaders['test'], device, is_main, -1, args.num_epochs, 'Test')
        test_loss, test_corr, test_total = sync_metrics(test_loss, test_corr, test_total, device, is_ddp)
        test_acc = 100. * test_corr / test_total
        avg_test_loss = test_loss / test_total
        logging.info(f"测试集 Loss: {avg_test_loss:.4f} | Acc: {test_acc:.2f}%")
        logging.info(f"最佳验证准确率: {best_acc:.2f}%")
        logging.info("="*50)
    finally:
        if is_ddp: dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ViT Fine-tuning')
    parser.add_argument('--num_classes', type=int, default=10, help='分类类别数')
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--model_dir', type=str, default='/disk3/model/vit-base-patch16-224', help='预训练模型目录或名称')
    # parser.add_argument('--model_dir', type=str, default='google/vit-base-patch16-224', help='预训练模型目录或名称')
    parser.add_argument('--gpu_ids', type=str, default='0', help='使用的GPU ID列表, 逗号分隔')
    parser.add_argument('--patience', type=int, default=5, help='早停耐心值')
    parser.add_argument('--full_finetune', type=str, default='n', choices=['y', 'n'], help='是否全参数微调 (y/n)')
    args = parser.parse_args()
    
    gpus = [int(x) for x in args.gpu_ids.split(',')]
    if len(gpus) > 1:
        print(f"检测到多GPU: {gpus}, 启动DDP训练...")
        mp.spawn(train, args=(len(gpus), gpus, args), nprocs=len(gpus), join=True)
    else:
        train(0, 1, gpus, args)
    
    print("\n最佳模型已保存到 './vit_finetuned' 目录")