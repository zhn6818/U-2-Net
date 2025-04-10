#!/bin/bash

# 设置CUDA设备可见性 - 只使用1,2,3,4,5这五张卡
export CUDA_VISIBLE_DEVICES=1,2,3,4,5

# 获取可用GPU数量 - 这将只计算我们设置为可见的GPU
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "使用 $NUM_GPUS 个GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# 确保至少有1个GPU
if [ $NUM_GPUS -lt 1 ]; then
    echo "错误: 没有设置可用的GPU!"
    exit 1
fi

# 默认配置
BATCH_SIZE=10
EPOCHS=100000
LEARNING_RATE=0.001
MODEL="u2net"
NUM_CLASSES=7
DATA_DIR="/data1/zhn/JZ/train_512/"
USE_SYNC_BN=""
BOUNDARY_LOSS=""
BOUNDARY_WEIGHT=0.5

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --num-classes)
            NUM_CLASSES="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --sync-bn)
            USE_SYNC_BN="--sync-bn"
            shift
            ;;
        --no-boundary-loss)
            BOUNDARY_LOSS="--no-boundary-loss"
            shift
            ;;
        --boundary-weight)
            BOUNDARY_WEIGHT="$2"
            shift 2
            ;;
        --pretrained)
            PRETRAINED="--pretrained $2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            shift
            ;;
    esac
done

# 打印训练配置
echo "训练配置:"
echo "----------------------------------------------------------------"
echo "模型: $MODEL"
echo "批次大小: $BATCH_SIZE"
echo "学习率: $LEARNING_RATE"
echo "训练轮次: $EPOCHS"
echo "分割通道数: $NUM_CLASSES"
echo "数据目录: $DATA_DIR"
echo "是否使用SyncBN: ${USE_SYNC_BN:-否}"
echo "是否禁用边界损失: ${BOUNDARY_LOSS:-否}"
echo "边界损失权重: $BOUNDARY_WEIGHT"
echo "预训练模型: ${PRETRAINED:-无}"
echo "GPU数量: $NUM_GPUS"
echo "使用的GPU设备: $CUDA_VISIBLE_DEVICES"
echo "----------------------------------------------------------------"

# 如果有多个GPU，使用分布式训练
if [ $NUM_GPUS -gt 1 ]; then
    echo "使用 $NUM_GPUS 个GPU进行分布式训练..."
    python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
        u2net_train_multichannel_multigpu.py \
        --distributed \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LEARNING_RATE \
        --model $MODEL \
        --num-classes $NUM_CLASSES \
        --data-dir "$DATA_DIR" \
        --boundary-weight $BOUNDARY_WEIGHT \
        $USE_SYNC_BN $BOUNDARY_LOSS $PRETRAINED
else
    # 单GPU训练
    echo "使用单个GPU进行训练..."
    python u2net_train_multichannel_multigpu.py \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LEARNING_RATE \
        --model $MODEL \
        --num-classes $NUM_CLASSES \
        --data-dir "$DATA_DIR" \
        --boundary-weight $BOUNDARY_WEIGHT \
        $BOUNDARY_LOSS $PRETRAINED
fi 