#!/bin/bash
# ============================================================
# MorphoBench Evaluation Agent - 批量评测脚本
# ============================================================
# 用法: bash eval_agent/run_eval.sh
# ============================================================

# 加载环境变量
ENV_FILE="./.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
    echo "✅ 成功加载环境变量从 $ENV_FILE"
else
    echo "⚠️ 未找到 $ENV_FILE，使用默认配置"
fi

# 检查必要的环境变量
if [ -z "$API_KEY" ]; then
    echo "❌ 环境变量 API_KEY 未设置"
    exit 1
fi

# 目录配置
PRED_DIR="./output/infer_result"
OUTPUT_DIR="./output/eval_agent_result"
TRACE_DIR="./output/eval_agent_traces"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$TRACE_DIR"

# 并发配置
NUM_WORKERS=100  # 降低并发，因为推理质量评测需要多次 API 调用
MAX_TOKENS=4096

echo "============================================================"
echo "MorphoBench Evaluation Agent - Batch Evaluation"
echo "============================================================"
echo "Predictions dir: $PRED_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Traces dir: $TRACE_DIR"
echo "============================================================"

# 遍历所有预测文件
for file in "$PRED_DIR"/*.json; do
    fname=$(basename "$file")
    echo ""
    echo ">>> Processing: $fname"
    
    # 根据文件名判断数据集和难度
    if [[ "$fname" == *"Lite"* ]]; then
        dataset="./data/Morpho_R_Lite"
        difficulty="easy"
    elif [[ "$fname" == *"Complex"* ]]; then
        dataset="./data/Morpho_R_Complex"
        difficulty="hard"
    elif [[ "$fname" == *"Perturbed"* ]]; then
        dataset="./data/Morpho_P_Perturbed"
        difficulty="perturbed"
    elif [[ "$fname" == *"_R_v0"* ]]; then
        dataset="./data/Morpho_R_v0"
        difficulty="v0"
    else
        dataset="./data/Morpho_P_v0"
        difficulty="v0"
    fi
    
    model_name=$(echo "$fname" | sed 's/.json//')
    
    echo "  Dataset: $dataset"
    echo "  Difficulty: $difficulty"
    echo "  Model: $model_name"
    
    # 运行评测
    # trace_dir 现在会按 reasoning_quality/dataset/model 和 hint_follow/dataset/model 组织
    python -m eval_agent.runner \
        --dataset "$dataset" \
        --predictions "$file" \
        --difficulty "$difficulty" \
        --model_name "$model_name" \
        --output_dir "$OUTPUT_DIR" \
        --trace_dir "$TRACE_DIR" \
        --num_workers "$NUM_WORKERS" \
        --max_tokens "$MAX_TOKENS" \
        --api_key "$API_KEY" \
        --base_url "$API_BASE" &
    
    # 限制并行任务数（每次最多 2 个模型同时评测）
    while [ $(jobs -rp | wc -l) -ge 2 ]; do
        sleep 5
    done
done

# 等待所有任务完成
wait

echo ""
echo "============================================================"
echo "✅ All evaluations completed!"
echo "============================================================"
echo "Results saved to: $OUTPUT_DIR"
echo "Traces saved to: $TRACE_DIR"


