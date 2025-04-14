# U2Net 多模型评估工具

这个工具用于评估和比较多个U2Net模型在单张图像上的分割性能。它可以加载多个模型，计算各类别的分割指标（IoU、Dice系数、准确率等），并生成可视化结果和比较图表。

## 功能特点

- 支持同时评估多个模型
- 计算详细的分割评估指标：IoU、Dice系数、准确率、精确率、召回率
- 对每个类别单独计算评估指标
- 生成直观的可视化结果，便于比较不同模型的性能
- 生成各种比较图表（柱状图、雷达图）展示模型间差异
- 导出详细的评估指标表格（CSV格式）

## 安装依赖

确保已安装以下Python库：

```bash
pip install torch numpy matplotlib scikit-image pillow tqdm pandas seaborn
```

## 使用方法

### 基本用法

```bash
python u2net_model_evaluator.py --input_image <图像路径> --models_dir <模型目录> --gt_mask <真实标签路径> --num_classes <类别数>
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--input_image` | 输入图像路径（必需） | - |
| `--models_dir` | 模型所在目录（必需） | - |
| `--gt_mask` | 真实标签图像路径（必需） | - |
| `--output_dir` | 评估结果输出目录 | model_evaluation_results |
| `--num_classes` | 分割类别数量 | 8 |
| `--input_size` | 输入图像大小 | 512 |
| `--threshold` | 分割二值化阈值 | 0.5 |
| `--flag` | 颜色空间标志 (0:RGB, 1:Lab, 2:RGB+Lab) | 0 |
| `--class_names` | 类别名称列表 | ['BG', 'A', 'B', 'C', ...] |
| `--device` | 设备选择 (auto, cpu, cuda) | auto |

### 示例

评估JZ模型目录下的所有模型，使用8个类别：

```bash
python u2net_model_evaluator.py \
  --input_image /path/to/test_image.jpg \
  --models_dir JZ_saved_models_multichannel_512 \
  --gt_mask /path/to/test_mask.png \
  --num_classes 8 \
  --input_size 512
```

为所有类别指定名称：

```bash
python u2net_model_evaluator.py \
  --input_image /path/to/test_image.jpg \
  --models_dir JZ_saved_models_multichannel_512 \
  --gt_mask /path/to/test_mask.png \
  --num_classes 8 \
  --class_names BG ClassA ClassB ClassC ClassD ClassE ClassF ClassG
```

## 输出结果

执行完成后，程序将在指定的输出目录（默认为`model_evaluation_results`）生成以下文件：

1. **model_comparison.png**：主要可视化结果，包含原图、GT掩码和每个模型的预测结果，同时显示详细评估指标。
2. **IoU_comparison.png**、**Dice_comparison.png**、**Accuracy_comparison.png**等：各个评估指标的比较柱状图。
3. **mean_metrics_radar.png**：使用雷达图展示各模型的平均性能指标。
4. **metrics_comparison.csv**：包含所有详细评估指标的CSV表格文件。

## 评估指标说明

- **IoU (Intersection over Union)**：交并比，计算预测区域与真实区域的重叠程度，值越大越好。
- **Dice系数**：评估预测区域与真实区域的重叠程度，计算公式为`(2 * Intersection) / (Area_Pred + Area_GT)`，值越大越好。
- **准确率 (Accuracy)**：像素级别的分类准确率，值越大越好。
- **精确率 (Precision)**：正确预测的前景像素数量占总预测前景像素的比例，值越大越好。
- **召回率 (Recall)**：正确预测的前景像素数量占真实前景像素的比例，值越大越好。

## 注意事项

1. 确保真实标签掩码格式正确：单通道掩码中，不同类别应使用不同的像素值（0, 1, 2...）表示；或者使用多通道掩码，每个通道对应一个类别。
2. 对于大型模型或高分辨率图像，请确保有足够的GPU内存或考虑使用较小的`input_size`。
3. 如果评估大量模型，可能需要较长时间，请耐心等待。

## 高级用法

### 支持不同数据格式的真实标签

该工具尝试自动处理不同格式的真实标签掩码：
- 单通道整数索引掩码（每个像素值表示一个类别）
- 多通道二值掩码（每个通道表示一个类别）
- RGB颜色编码掩码（如果提供，程序会尝试将其转换为多通道格式）

### 批量评估

如果需要对多张图像进行批量评估，可以创建一个脚本循环调用此工具，或者修改代码以支持批量处理。 