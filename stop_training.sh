#!/bin/bash

# 检查PID文件是否存在
if [ ! -f .train_pid ]; then
    echo "未找到训练进程PID文件，可能没有正在运行的训练进程"
    exit 1
fi

# 读取PID
PID=$(cat .train_pid)

# 检查进程是否存在
if ps -p $PID > /dev/null; then
    echo "正在停止U-2-Net训练进程 (PID: $PID)..."
    
    # 先尝试优雅地终止进程
    kill -15 $PID
    
    # 等待5秒看进程是否结束
    sleep 5
    
    # 如果进程仍在运行，强制终止
    if ps -p $PID > /dev/null; then
        echo "进程未响应，强制终止..."
        kill -9 $PID
    fi
    
    echo "训练进程已停止"
else
    echo "PID为 $PID 的进程不存在，可能已经停止"
fi

# 删除PID文件
rm -f .train_pid
echo "已清理PID文件" 