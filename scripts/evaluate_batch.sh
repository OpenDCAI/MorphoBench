#!/bin/bash
# 批量评测 MorphoBench 模型结果
# 用法: bash batch_evaluate.sh

PRED_DIR="./output/infer_result"
OUTPUT_DIR="./output/eval_result"

for file in "$PRED_DIR"/*.json; do
  fname=$(basename "$file")
  echo "Processing $fname"

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

  model_name=$(echo "$fname" | sed 's/.json//' )

  python ./scripts/evaluate_judge.py \
    --dataset "$dataset" \
    --predictions "$file" \
    --difficulty "$difficulty" \
    --model_name "$model_name" \
    --output_dir "$OUTPUT_DIR" \
    --num_workers 1000 \
    --judge "o3-mini-2025-01-31" &
done

wait
echo "✅ All evaluations completed."