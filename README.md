# Vision Transformer (ViT) 复现指南

## 📋 目录
1. [环境配置](#环境配置)
2. [快速开始](#快速开始)
3. [从零实现](#从零实现)
4. [训练自己的模型](#训练自己的模型)
5. [常见问题](#常见问题)

---

## 🔧 环境配置

### 方法1: 使用conda（推荐）
```bash
# 创建虚拟环境
conda create -n vit python=3.9
conda activate vit

# 安装PyTorch（根据你的CUDA版本选择）
# CPU版本
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# GPU版本 (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install transformers pillow requests tqdm
```

### 方法2: 使用pip
```bash
# 创建虚拟环境
python -m venv vit_env
source vit_env/bin/activate  # Linux/Mac
# vit_env\Scripts\activate  # Windows

# 安装依赖
pip install torch torchvision transformers pillow requests tqdm
```

---

## 🚀 快速开始（5分钟跑通）

### 方案A: 使用预训练模型推理（最简单）

```bash
python vit_quickstart.py
```

**这个脚本会：**
- 自动下载预训练的ViT模型
- 下载测试图片
- 进行图像分类
- 输出Top-5预测结果

**示例输出：**
```
正在加载ViT模型...
正在进行推理...
预测类别ID: 281
预测类别: tabby cat

Top-5 预测结果:
1. tabby cat: 0.4123
2. Egyptian cat: 0.3456
3. tiger cat: 0.1234
...
```

---

## 🔨 从零实现ViT架构

### 方案B: 理解并测试ViT架构

```bash
python vit_from_scratch.py
```

**这个脚本包含：**
- ✅ Patch Embedding层
- ✅ Multi-Head Self-Attention
- ✅ Transformer Encoder Block
- ✅ 完整的ViT模型
- ✅ 三种模型大小：Tiny/Small/Base

**代码结构：**
```
VisionTransformer
├── PatchEmbedding      # 图像切分为patches
├── TransformerBlock    # Transformer编码器
│   ├── MultiHeadAttention
│   └── MLP
└── Classification Head # 分类层
```

**模型参数对比：**
| 模型 | 参数量 | 适用场景 |
|------|--------|----------|
| ViT-Tiny | ~5M | 快速实验 |
| ViT-Small | ~22M | 中等数据集 |
| ViT-Base | ~86M | 大规模训练 |

---

## 🏋️ 训练自己的模型

### 方案C: 在CIFAR-10上从头训练

```bash
# 训练ViT-Tiny模型
python train_vit_cifar10.py
```

**训练配置：**
- 数据集: CIFAR-10 (60,000张图片, 10类)
- 模型: ViT-Tiny
- 训练轮数: 50 epochs
- 批次大小: 64
- 学习率: 0.001 (cosine decay)

**预期结果：**
- 训练时间: ~2小时 (GPU) / ~10小时 (CPU)
- 验证准确率: 70-75% (ViT-Tiny)

**提示：** 如果GPU内存不足，可以：
1. 减小batch_size
2. 使用更小的图像尺寸
3. 减少模型深度

### 方案D: 使用预训练模型微调（推荐）

```bash
# 微调预训练ViT模型
python finetune_vit.py
```

**优势：**
- ✅ 收敛更快
- ✅ 准确率更高
- ✅ 需要更少数据
- ✅ 训练时间更短

**微调策略：**
1. **冻结backbone，只训练分类头**（最快）
   - 训练时间: ~30分钟
   - 适合小数据集
   
2. **解冻所有层**（最好效果）
   - 训练时间: ~2小时
   - 需要更多数据

---

## 📊 性能对比

### 不同方案的对比

| 方案 | 准确率 | 训练时间 | 难度 | 推荐指数 |
|------|--------|----------|------|----------|
| 预训练推理 | - | 0分钟 | ⭐ | ⭐⭐⭐⭐⭐ |
| 从零训练(Tiny) | 70-75% | 2小时 | ⭐⭐⭐ | ⭐⭐⭐ |
| 从零训练(Base) | 80-85% | 8小时 | ⭐⭐⭐⭐ | ⭐⭐ |
| 预训练微调 | 85-90% | 30分钟 | ⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🎯 使用自己的数据集

### 准备数据集
```
your_dataset/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class2/
│       ├── img1.jpg
│       └── img2.jpg
└── val/
    ├── class1/
    └── class2/
```

### 修改代码
```python
# 在 finetune_vit.py 中修改：
train_dataset = datasets.ImageFolder(
    root='your_dataset/train',
    transform=transform_train
)

val_dataset = datasets.ImageFolder(
    root='your_dataset/val',
    transform=transform_val
)
```

---

## ❓ 常见问题

### Q1: GPU内存不足怎么办？
**A:** 尝试以下方法：
```python
# 1. 减小批次大小
BATCH_SIZE = 32  # 改为 16 或 8

# 2. 使用梯度累积
accumulation_steps = 4
loss = loss / accumulation_steps
loss.backward()
if (step + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()

# 3. 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### Q2: 训练速度慢怎么办？
**A:** 优化建议：
- 使用多GPU训练：`torch.nn.DataParallel`
- 增加num_workers：`DataLoader(..., num_workers=4)`
- 使用更小的模型：ViT-Tiny代替ViT-Base
- 使用预训练模型微调

### Q3: 准确率不高怎么办？
**A:** 改进方法：
1. 使用预训练模型
2. 增加数据增强
3. 调整学习率
4. 训练更多epochs
5. 使用更大的模型

### Q4: 如何可视化注意力图？
**A:** 添加以下代码：
```python
# 获取注意力权重
attention_weights = model.vit.encoder.layer[-1].attention.self.attention_probs
# 可视化（需要安装matplotlib）
import matplotlib.pyplot as plt
plt.imshow(attention_weights[0, 0].detach().cpu())
plt.show()
```

---

## 📚 进阶阅读

### 原始论文
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

### 相关资源
- HuggingFace ViT文档: https://huggingface.co/docs/transformers/model_doc/vit
- PyTorch Image Models: https://github.com/rwightman/pytorch-image-models
- 论文解读视频: [推荐搜索相关视频]

---

## 🎉 总结

### 推荐学习路径：

1. **第一步：快速体验**
   - 运行 `vit_quickstart.py`
   - 感受ViT的推理效果

2. **第二步：理解原理**
   - 阅读 `vit_from_scratch.py`
   - 理解各个组件的作用

3. **第三步：实战训练**
   - 运行 `finetune_vit.py`
   - 在自己的数据上微调

4. **第四步：深入研究**
   - 阅读原始论文
   - 尝试改进模型架构

### 下一步可以做什么？

- 🔍 研究其他ViT变体：DeiT, Swin Transformer
- 🎨 尝试Vision-Language模型：CLIP, BLIP
- 🚀 优化模型：剪枝、量化、蒸馏
- 📱 部署模型：ONNX导出、移动端部署

---

## 💡 提示

- 建议从**预训练模型微调**开始，这是最高效的方式
- 从头训练需要大量数据和计算资源
- ViT在大规模数据集上表现最好
- 对于小数据集，CNN可能表现更好

祝你复现顺利！🎊
