# U2NET 分割分类双任务模型

本项目基于U2NET模型开发了一个同时支持分割和分类的双任务模型。该模型保留了U2NET的分割能力，同时增加了一个复杂的分类头，可以将图像分类为指定的类别。

## 特点

- **双任务学习**：同时执行分割和分类任务
- **预训练模型利用**：利用预训练的U2NET权重进行微调
- **复杂分类头**：使用多尺度特征、注意力机制和残差连接
- **分阶段训练**：先训练分类头，再微调整个网络

## 项目结构

- `u2net_classify.py`：双任务模型定义和训练代码
- `u2net_classify_test.py`：模型测试和推理代码
- `daizhuang_saved_models_classify/`：保存训练模型的目录

## 依赖项

- PyTorch >= 1.7.0
- Torchvision
- NumPy
- OpenCV
- scikit-image

## 使用方法

### 1. 训练模型

使用以下命令训练模型：

```bash
python u2net_classify.py
```

默认参数：
- 预训练权重路径：`daizhuang_saved_models_512/u2net/u2net_best_acc_0.9140_epoch_50.pth`
- 分类类别数：3
- 批次大小：2
- 学习率：0.001 (微调时使用0.0001)
- 训练epochs：100 (前10个epochs冻结U2NET主干网络)

可以在`ConfigWithClassifier`类中修改这些参数。

### 2. 测试模型

单张图像测试：

```bash
# 修改u2net_classify_test.py中的相关路径后运行
python u2net_classify_test.py
```

或者在Python代码中使用：

```python
from u2net_classify_test import test_single_image

test_single_image(
    model_path="your_model_path.pth",
    image_path="path_to_test_image.jpg",
    output_path="result.png"
)
```

批量测试：

```python
from u2net_classify_test import test_batch

test_batch(
    model_path="your_model_path.pth",
    image_dir="test_images_folder",
    output_dir="results_folder",
    ext='.jpg'
)
```

## 模型架构

### 总体架构

```
U2NetWithClassifier
├── U2NET (分割主干网络)
│   └── 7个不同尺度的预测输出 (d0-d6)
└── ClassifierHead (分类头)
    ├── 图像处理分支
    ├── 分割特征处理
    ├── 注意力模块
    ├── 特征融合
    ├── 残差块
    └── 分类器
```

### 分类头

分类头使用以下组件：

1. **多尺度特征融合**：结合U2NET的多个尺度预测输出
2. **通道和空间注意力**：增强关键特征
3. **残差连接**：提高训练稳定性
4. **多层分类器**：提高分类能力

## 训练策略

训练分为两个阶段：

1. **第一阶段（默认10个epochs）**：
   - 冻结U2NET主干网络
   - 只训练分类头
   - 使用较大的学习率 (0.001)

2. **第二阶段（剩余epochs）**：
   - 解冻整个网络
   - 使用较小的学习率 (0.0001)
   - 整体微调模型

## 损失函数

总损失由两部分组成：

- **分割损失**：使用U2NET原有的多尺度BCE损失
- **分类损失**：使用交叉熵损失

总损失 = 分割损失权重 * 分割损失 + 分类损失权重 * 分类损失

## 数据格式要求

训练数据要求：
- 分割标签中的值应该能指示类别（1、2或3）
- 每个样本应至少包含一个前景像素

测试数据要求：
- 输入图像格式支持jpg、png等常见格式
- 尺寸不限，会自动调整为模型所需尺寸

## 常见问题

**Q1: 如何调整分类类别数？**  
A1: 修改`ConfigWithClassifier`类中的`n_classes`参数。

**Q2: 如何改变分割和分类的相对重要性？**  
A2: 调整`seg_loss_weight`和`cls_loss_weight`参数。

**Q3: 如何使用其他预训练模型？**  
A3: 修改`pretrained_model_path`参数指向您的预训练模型路径。

## 示例结果

处理后的结果包含：
- 原始图像
- 分割掩码叠加（绿色区域）
- 分类结果和置信度（显示在图像上方）

## 参考文献

1. U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection
2. 注意力机制在视觉任务中的应用
3. 多任务学习：理论与实践 