#!/bin/bash

# 设置训练日志文件
LOG_FILE="train_$(date +%Y%m%d_%H%M%S).log"
echo "启动U-2-Net训练，日志将保存到：$LOG_FILE"

# 启动训练进程并在后台运行
nohup python u2net_train_dai.py > $LOG_FILE 2>&1 &

# 获取进程ID并保存
PID=$!
echo $PID > .train_pid
echo "训练进程已启动，PID: $PID"
echo "查看训练日志：tail -f $LOG_FILE" 