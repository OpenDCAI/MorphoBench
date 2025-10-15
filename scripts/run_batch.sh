#!/bin/bash
# 批量跑 MorphoBench 多模型多数据集评测
# 用法: bash run_all_models.sh

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
"gpt-5-2025-08-07"
"gemini-2.5-pro-thinking-8192"
"gemini-2.5-flash-thinking-24576"
"grok-4-0709"
"claude-sonnet-4-20250514-thinking"
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
            --base_url "API_BASE_URL" \
            --api_key "YOUR_API_KEY"
    done
done

wait
echo "✅ 全部任务完成！"




