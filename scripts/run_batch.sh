#!/bin/bash
# 批量跑 MorphoBench 多模型多数据集评测
# 用法: bash run_all_models.sh
ENV_FILE="./.env"

if [ -f "$ENV_FILE" ]; then
    # set -a 会让 source 时自动 export 变量
    set -a
    source "$ENV_FILE"
    set +a
    echo "✅ 成功加载环境变量从 $ENV_FILE"
else
    echo "⚠️ 未找到 $ENV_FILE "
fi

# 简单检查一下关键变量是否存在
if [ -z "$API_KEY" ]; then
    echo "❌ 环境变量 API_KEY 未设置，请在 .env 中写 API_KEY=xxx"
    exit 1
fi

if [ -z "$API_BASE" ]; then
    echo "⚠️ 环境变量 API_BASE 未设置，将使用脚本里默认的 base_url（如果有的话）"
fi


SCRIPT_PATH="./scripts/run_model_predictions.py"

DATASETS=(
"./data/Morpho_P_Perturbed"
"./data/Morpho_P_v0"
"./data/Morpho_R_Complex"
"./data/Morpho_R_Lite"
"./data/Morpho_R_v0"
)


MODELS=(
"o3"
"o4-mini"
"gpt-5.1-2025-11-13"
"gemini-2.5-pro-thinking"
"gemini-2.5-flash-thinking"
"grok-4-0709"
"gemini-3-pro-preview-thinking"
# "claude-sonnet-4-20250514-thinking"
)

OUTPUT_BASE="./output/infer_result"
MAX_JOBS=1 


run_with_limit() {
    while [ $(jobs -rp | wc -l) -ge $MAX_JOBS ]; do
        sleep 1
    done
    "$@" &
}


for dataset in "${DATASETS[@]}"; do
    dataset_name=$(basename "$dataset")  

    if [[ "$dataset" == *"Lite"* ]]; then
        DIFF="easy"
    elif [[ "$dataset" == *"Complex"* ]]; then
        DIFF="hard"
    elif [[ "$dataset" == *"Perturbed"* ]]; then
        DIFF="perturbed"
    else
        DIFF="v0"
    fi

    for model in "${MODELS[@]}"; do
        model_safe=$(echo "$model" | tr ' ' '_' | tr '/' '_')

        OUTPUT_FILE="${OUTPUT_BASE}/${dataset_name}_${model_safe}.json"

        echo ">>> Running dataset=${dataset_name} (${DIFF}), model=${model}"
        run_with_limit python "$SCRIPT_PATH" \
            --dataset "$dataset" \
            --model "$model" \
            --output "$OUTPUT_FILE" \
            --num_workers 1000 \
            --max_completion_tokens 32760 \
            --base_url "$API_BASE" \
            --api_key "$API_KEY" 
    done
done

wait
echo "✅ 全部任务完成！"




