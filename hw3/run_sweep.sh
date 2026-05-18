#!/bin/bash

# 基准配置文件路径
BASE_YAML="experiments/sac/halfcheetah.yaml"
# 温度参数 beta 列表 (已经去掉了 0.1)
BETAS=(0.01 0.05 0.5 1.0)
# 临时配置文件的保存目录
TEMP_DIR="experiments/sac"

# 检查基准配置是否存在
if [ ! -f "$BASE_YAML" ]; then
    echo "Error: Base config file $BASE_YAML not found!"
    exit 1
fi

# 限制并发任务数为 2
MAX_JOBS=2
job_count=0
pids=()

for beta in "${BETAS[@]}"; do
    echo "========================================"
    echo "Starting run with temperature (beta) = $beta in background..."
    echo "========================================"

    # 1. 临时文件保存到 experiments/sac/ 目录下，每个 beta 独立一个文件
    TEMP_YAML="${TEMP_DIR}/tmp_halfcheetah_sweep_beta_${beta}.yaml"
    cp "$BASE_YAML" "$TEMP_YAML"

    # 2. 使用 sed 替换临时配置文件中的 temperature 值
    sed -i "s/^temperature: .*/temperature: $beta/" "$TEMP_YAML"

    # 3. 使用 sed 替换 exp_name，修改 WandB 上的记录名称
    sed -i "s/^exp_name: .*/exp_name: sac_beta_$beta/" "$TEMP_YAML"

    # 4. 在后台运行训练脚本并传入临时配置文件
    uv run src/scripts/run_sac.py -cfg "$TEMP_YAML" &
    
    # 获取刚刚放到后台的进程 PID 并记录
    pids+=($!)
    ((job_count++))
    
    # 5. 当达到并发上限（2个）时，停下来等待这两个任务跑完
    if [[ $job_count -ge $MAX_JOBS ]]; then
        echo "-> Reached $MAX_JOBS parallel jobs. Waiting for this group to finish before continuing..."
        for pid in "${pids[@]}"; do
            wait $pid
        done
        echo "-> Group finished. Proceeding to next group (if any)..."
        
        # 清空计数器和 PID 列表，准备迎接下一组
        job_count=0
        pids=()
    fi
done

# 6. 如果最后一组不足 MAX_JOBS（例如数组长度不是 2 的倍数），需要再等待剩下的任务完成
if [[ ${#pids[@]} -gt 0 ]]; then
    echo "-> Waiting for remaining background jobs to finish..."
    for pid in "${pids[@]}"; do
        wait $pid
    done
fi

# 7. 清理临时文件
echo "Cleaning up temporary config files..."
rm -f "${TEMP_DIR}"/tmp_halfcheetah_sweep_beta_*.yaml

echo "All sweep runs completed successfully!"
