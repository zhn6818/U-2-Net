#!/bin/bash

# 设置环境变量（如果需要）
# export PYTHONPATH=$PYTHONPATH:/path/to/project

# 输出当前时间和开始消息
echo "开始训练 U2-Net 多通道模型 - $(date)"

# 使用nohup命令在后台运行训练脚本，将输出重定向到train.log文件
# 2>&1表示将标准错误也重定向到标准输出，&表示在后台运行
nohup python u2net_train_multichannel.py > train.log 2>&1 &

# 获取后台进程的PID
TRAIN_PID=$!

# 将训练进程的PID保存到文件中，方便后续停止训练
echo $TRAIN_PID > .train_pid

echo "训练已启动，进程ID: $TRAIN_PID"
echo "使用 'tail -f train.log' 命令查看训练日志"
echo "使用 'bash stop_training.sh' 命令停止训练"
