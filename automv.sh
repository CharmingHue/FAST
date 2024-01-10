#!/bin/bash

# 从命令行参数获取自动移动前的等待时间，默认为3600秒
wait_time=${1:-3600}

# 初始化计数器
count_mv=0

while :
do
    # 累加计数器
    count_mv=$((count_mv + 1))

    # 检查是否达到等待时间
    if [ $count_mv -ge $wait_time ]; then
        # 执行文件移动操作
        mv ~/apieceofshit/FAST/checkpoints/fast_base_fsnet_ctw_512_finetune_ic17mlt/*.tar ~/autodl-tmp/fast_base_fsnet_ctw_512_finetune_ic17mlt_init/
        echo "成功移动权重文件"
        # 重置计数器
        count_mv=0
    fi

    # 每秒检查一次
    sleep 1
done
