#!/bin/bash
source ~/miniconda3/bin/activate llava-hr
cd ~/work/LLaVA

gpu_list="${CUDA_VISIBLE_DEVICES:-4}"
IFS=',' read -ra GPULIST <<<"$gpu_list"
CHUNKS=${#GPULIST[@]}

BASE_MODEL=${1:-"liuhaotian/llava-v1.5-13b"}
CKPT=${2:-"mamachang/llava-chart-13b_lora"}

# 评估文件名称
FILE_NAME="model_vqa_chartqa"

OUTPUT_DIR="./playground/eval/chartCaption/$(basename $BASE_MODEL)/$(basename $CKPT)/${FILE_NAME}"

# 创建输出目录，如果不存在的话
mkdir -p $OUTPUT_DIR

# 清空 OUTPUT_DIR 目录下的文件
rm -rf $OUTPUT_DIR/*

# 定义数据集
declare -A QUESTION_FILES=(
    ["vistext"]="./playground/data/chartCaption/test.jsonl"
)

# 评估函数
run_evaluation() {
    local split=$1
    local question_file=${QUESTION_FILES[$split]}
    echo "Evaluating $split split..."
    for IDX in $(seq 0 $((CHUNKS - 1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.chartqa.demo \
            --model-base $BASE_MODEL \
            --model-path $CKPT \
            --question-file $question_file \
            --image-folder ./images/chart \
            --answers-file $OUTPUT_DIR/${CHUNKS}_${IDX}_${split}.jsonl \
            --temperature 0 \
            --conv-mode vicuna_v1 & # --model-base $BASE_MODEL \
    done
    wait

    output_file=$OUTPUT_DIR/${split}.jsonl
    # Clear out the output file if it exists.
    >"$output_file"

    for IDX in $(seq 0 $((CHUNKS - 1))); do
        cat $OUTPUT_DIR/${CHUNKS}_${IDX}_${split}.jsonl >>"$output_file"
        rm -f $OUTPUT_DIR/${CHUNKS}_${IDX}_${split}.jsonl
    done

    # metric_file=$OUTPUT_DIR/${split}_metric.txt
    # echo -e "===========================\nMetrics for $split test set:" >>"$metric_file"
    # python eval_vqa_relaxed_acc.py \
    #     --result-file $output_file \
    #     >>"$metric_file"
}

for split in "${!QUESTION_FILES[@]}"; do
    run_evaluation $split
done
