#!/bin/bash
# 下载 CBraMod 预训练权重脚本
# 使用进程守护(nohup)运行，避免终端关闭导致下载中断

unset LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com

DOWNLOAD_DIR="/local4/home/lizuotong/EEG-FM-bench/external/models/CBraMod-main/pretrained_weights"
REPO_ID="weighting666/CBraMod"
PYTHON="/local4/home/lizuotong/.conda/envs/cbramod/bin/python"

echo "========================================"
echo "开始下载 CBraMod 预训练权重"
echo "时间: $(date)"
echo "仓库: ${REPO_ID}"
echo "目标目录: ${DOWNLOAD_DIR}"
echo "========================================"

${PYTHON} -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${REPO_ID}',
    local_dir='${DOWNLOAD_DIR}',
    local_dir_use_symlinks=False
)
"

if [ $? -eq 0 ]; then
    echo "========================================"
    echo "下载完成!"
    echo "时间: $(date)"
    echo "========================================"
    ls -lh "${DOWNLOAD_DIR}"
else
    echo "========================================"
    echo "下载失败! 退出码: $?"
    echo "时间: $(date)"
    echo "========================================"
    exit 1
fi
