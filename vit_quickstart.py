"""
Vision Transformer (ViT) 快速入门示例
使用HuggingFace预训练模型进行图像分类
"""

import os
# 必须在最开始就设置镜像源，在import transformers之前
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

# 下载测试图片
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# 加载预训练的ViT模型和处理器
print("正在加载ViT模型...")
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# 预处理图像
inputs = processor(images=image, return_tensors="pt")

# 模型推理
print("正在进行推理...")
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 获取预测结果
predicted_class_idx = logits.argmax(-1).item()
print(f"预测类别ID: {predicted_class_idx}")
print(f"预测类别: {model.config.id2label[predicted_class_idx]}")

# 显示top-5预测
probs = torch.nn.functional.softmax(logits, dim=-1)[0]
top5_prob, top5_idx = torch.topk(probs, 5)

print("\nTop-5 预测结果:")
for i in range(5):
    print(f"{i+1}. {model.config.id2label[top5_idx[i].item()]}: {top5_prob[i].item():.4f}")