#!/bin/bash
config=${1}
wait_time=${2:-3600}

outpath="$HOME/autodl-fs/$config"
if [ ! -d $outpath ]; then
    mkdir -p $outpath
    echo "成功创建文件夹"
else
    echo "文件夹已存在"
fi
# 初始化计数器
count_mv=0

while :
do
    # 累加计数器
    count_mv=$((count_mv + 1))

    # 检查是否达到等待时间
    if [ $count_mv -ge $wait_time ]; then
        # 执行文件移动操作
        mv ~/FAST/checkpoints/$config/*.tar $outpath
        echo "成功移动权重文件"
        # 重置计数器
        count_mv=0
    fi

    # 每秒检查一次
    sleep 1
done
