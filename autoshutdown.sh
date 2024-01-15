#!/bin/bash

# 从命令行参数获取自动关机前的等待时间，默认为20分钟
idle_time=${1:-1200}
print_time=${2:-30}

# 初始化计数器
count=0
print_count=0
total_gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)

# 循环检查GPU占用率
while :
do
    # 查询当前的GPU显存占用和利用率
    gpu_memory_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    gpu_utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    
    # 计算显存占用率
    gpu_memory_utilization=$((gpu_memory_usage * 100 / total_gpu_memory))

    # 每30秒打印一次显存占用和利用率
    if [ $print_count -ge $print_time ]; then
        echo "当前GPU显存占用: ${gpu_memory_usage}MiB, 显存占用率: ${gpu_memory_utilization}%, 利用率: ${gpu_utilization}%"
        print_count=0
    fi

    # 如果GPU利用率小于5%
    if [ $gpu_memory_utilization -lt 5 ]; then
        # 增加计数器
        count=$((count + 1))

        # 如果连续1200s都没有任何GPU占用
        if [ $count -ge $idle_time ]; then
            # 关机
            echo "GPU空闲，准备关机..."
            shutdown -h now
        fi
    else
        # 如果有GPU占用，则重置计数器
        count=0
    fi

    # 等待1秒，增加打印计数器
    sleep 1
    print_count=$((print_count + 1))
done
