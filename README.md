# U²-Net 晶界分割项目

## 项目概述

本项目基于U²-Net架构，专门针对材料科学中的晶界分割任务进行了优化。项目包含原始的U²-Net模型和专门为晶界分割优化的U²-Net-Grain模型。

## 模型架构

### 1. 原始U²-Net模型
- **U2NET**: 完整版本，参数量约173.6MB
- **U2NETP**: 轻量版本，参数量约4.7MB

### 2. 晶界分割优化模型 (U²-Net-Grain)
- **U2NET_GRAIN**: 针对晶界分割优化的完整版本
- **U2NETP_GRAIN**: 针对晶界分割优化的轻量版本

#### 优化特性：
1. **减少下采样层数**: 保留更多细节信息，适合细粒度的晶界检测
2. **密集跳跃连接**: 增强特征融合能力
3. **注意力机制**: 
   - 通道注意力 (Channel Attention)
   - 空间注意力 (Spatial Attention)
   - CBAM (Convolutional Block Attention Module)
4. **边缘增强模块**: 使用Sobel算子增强边缘特征
5. **转置卷积上采样**: 替代双线性插值，提高细节恢复能力

## 项目结构

```
U-2-Net/
├── model/
│   ├── __init__.py          # 模型导入
│   ├── u2net.py            # 原始U²-Net模型
│   └── u2net_grain.py      # 晶界分割优化模型
├── data_loader.py          # 数据加载和预处理
├── u2net_train.py          # 训练脚本
├── u2net_infer_single_img.py # 推理脚本
├── select_model.py         # 模型对比分析脚本
├── test_u2net_grain.py     # 模型测试脚本
├── saved_models/           # 保存的模型权重
│   ├── u2net/             # U2NET模型权重
│   └── u2net_grain/       # U2NET_GRAIN模型权重
├── test_results/           # 推理结果
├── model_comparison_results/ # 模型对比结果
└── README.md              # 项目说明文档
```

## 安装依赖

```bash
pip install torch torchvision
pip install numpy pillow scikit-image
pip install opencv-python
```

## 使用方法

### 1. 训练模型

#### 准备数据
创建 `train.txt` 文件，每行包含图像路径和标签路径，用空格分隔：
```
path/to/image1.jpg path/to/label1.png
path/to/image2.jpg path/to/label2.png
...
```

#### 开始训练
```bash
python u2net_train.py
```

#### 训练配置
在 `u2net_train.py` 中可以修改以下参数：
- `model_name`: 选择模型类型 ('u2net', 'u2netp', 'u2net_grain', 'u2netp_grain')
- `epoch_num`: 训练轮数
- `batch_size_train`: 训练批次大小
- `pretrained_model_path`: 预训练模型路径（可选）

### 2. 模型推理

#### 批量推理
```python
from u2net_infer_single_img import inference_folder

inference_folder(
    image_dir='path/to/images/',
    model_path='saved_models/u2net_grain/model.pth',
    output_dir='results/',
    model_type='u2net_grain'
)
```

#### 单张图片推理
```python
from u2net_infer_single_img import inference_single_image

inference_single_image(
    image_path='path/to/image.jpg',
    model_path='saved_models/u2net_grain/model.pth',
    output_path='result.png',
    model_type='u2net_grain'
)
```

### 3. 模型对比分析

使用 `select_model.py` 脚本可以批量对比同一文件夹中不同模型的推理效果：

#### 基本用法
```python
from select_model import compare_models_in_folder

# 对比u2net_grain文件夹中的所有模型
compare_models_in_folder(
    model_folder_path="saved_models/u2net_grain",
    test_image_path="path/to/test/image.jpg",
    output_dir="model_comparison_results"
)

# 对比u2net文件夹中的所有模型
compare_models_in_folder(
    model_folder_path="saved_models/u2net",
    test_image_path="path/to/test/image.jpg", 
    output_dir="u2net_comparison_results"
)
```

#### 直接运行脚本
```bash
python select_model.py
```

#### 功能特点
- **自动模型类型识别**: 根据文件夹名称自动判断模型类型（u2net、u2netp、u2net_grain、u2netp_grain）
- **批量推理**: 自动加载文件夹中的所有.pth模型文件进行推理
- **智能命名**: 输出文件以模型名称命名（如 `u2net_grain_best_acc_0.9556_epoch_336.png`）
- **错误处理**: 单个模型失败不影响其他模型的处理
- **内存管理**: 每次推理后自动清理内存，避免内存溢出
- **进度跟踪**: 显示处理进度和最终统计信息

#### 使用场景
1. **模型版本对比**: 比较同一架构不同训练轮次的模型效果
2. **超参数调优**: 评估不同超参数设置的模型性能
3. **模型选择**: 在多个候选模型中选择最优模型
4. **质量评估**: 快速评估所有保存模型的推理质量

#### 输出示例
```
Using device: mps
Model type determined: u2net_grain
Found 30 model files:
  - u2net_grain_best_acc_0.9189_epoch_1.pth
  - u2net_grain_best_acc_0.9192_epoch_2.pth
  - ...

Processing model: u2net_grain_best_acc_0.9189_epoch_1
✓ Result saved: model_comparison_results/u2net_grain_best_acc_0.9189_epoch_1.png

Processing model: u2net_grain_best_acc_0.9192_epoch_2
✓ Result saved: model_comparison_results/u2net_grain_best_acc_0.9192_epoch_2.png

=== Processing Summary ===
Total models: 30
Successful inferences: 30
Failed inferences: 0
Results saved in: model_comparison_results
```

### 4. 模型测试

运行测试脚本验证模型功能：
```bash
python test_u2net_grain.py
```

## 模型性能对比

| 模型 | 参数量 | 输出数量 | 特殊优化 |
|------|--------|----------|----------|
| U2NET | ~44M | 7个侧输出 | 标准U²-Net |
| U2NETP | ~1.1M | 7个侧输出 | 轻量化版本 |
| U2NET_GRAIN | ~35M | 5个侧输出 | 晶界分割优化 |
| U2NETP_GRAIN | ~0.8M | 5个侧输出 | 轻量化+晶界优化 |

## 数据增强

项目支持多种数据增强技术：
- **颜色抖动** (ColorJitter): 亮度、对比度、饱和度、色调调整
- **随机最大滤波** (RandomMaxFilter): 模拟图像噪声
- **随机划痕** (RandomScratch): 模拟图像缺陷
- **尺寸调整** (RescaleT): 统一输入尺寸

## 损失函数

使用多尺度二元交叉熵损失 (Multi-scale BCE Loss)：
- 对所有侧输出计算BCE损失
- 加权融合多个尺度的损失
- 支持不同模型的输出数量

## 设备支持

自动检测并使用最佳可用设备：
1. CUDA GPU (如果可用)
2. Apple Silicon MPS (如果可用)
3. CPU (备选)

## 模型保存策略

训练过程中会自动保存：
- 每2000次迭代保存一次
- 每5个epoch保存一次
- 准确率提升时保存最佳模型

保存格式：
```
{model_name}_epoch_{epoch}_loss_{loss:.4f}_acc_{accuracy:.4f}.pth
{model_name}_best_acc_{accuracy:.4f}_epoch_{epoch}.pth
```

## 注意事项

1. **内存管理**: 训练和推理过程中会自动清理临时变量以节省内存
2. **模型兼容性**: 推理脚本自动适配不同模型的输出格式
3. **图像格式**: 支持常见图像格式 (.jpg, .jpeg, .png, .bmp)
4. **输入尺寸**: 默认将输入图像调整为512x512，输出时恢复原始尺寸

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少batch_size
   - 使用轻量版模型 (U2NETP_GRAIN)

2. **训练损失不收敛**
   - 检查数据标签质量
   - 调整学习率
   - 使用预训练模型

3. **推理结果不理想**
   - 确保使用正确的模型类型
   - 检查模型权重路径
   - 验证输入图像质量

## 更新日志

### v1.0 (当前版本)
- 实现原始U²-Net模型
- 添加晶界分割优化版本
- 支持多种数据增强
- 完善的训练和推理流程
- 自动设备检测和内存管理

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。在提交代码前，请确保：
1. 代码符合项目风格
2. 添加必要的注释
3. 测试新功能的正确性

## 许可证

本项目遵循MIT许可证。详见LICENSE文件。

## 联系方式

如有问题或建议，请通过GitHub Issues联系我们。