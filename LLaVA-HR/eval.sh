#!/bin/bash
source ~/miniconda3/bin/activate llava-hr

gpu_list="${CUDA_VISIBLE_DEVICES:-1,3,4,6}"
IFS=',' read -ra GPULIST <<<"$gpu_list"
CHUNKS=${#GPULIST[@]}

# 设置模型路径
CKPT=${2:-"/home/lhy/models/llava-hr-ChartInstruction"}

# 图片目录
IMAGE_DIR="/home/lhy/work/front-tech/playground/data/chartqa/png"

# 输出目录
OUTPUT_DIR="./playground/eval/chartqa/$(basename $CKPT)/detailed_output"
mkdir -p $OUTPUT_DIR

# 定义数据集
declare -A QUESTION_FILES=(
    # ["aug"]="/home/lhy/work/front-tech/playground/data/chartqa/test_aug.jsonl"
    ["human"]="/home/lhy/work/front-tech/playground/data/chartqa/test_human.jsonl"
)

# 评估函数
run_evaluation() {
    local split=$1
    local question_file=${QUESTION_FILES[$split]}
    echo "Evaluating $split split..."
    echo "CHUNKS: $CHUNKS"

    # 运行评估
    for IDX in $(seq 0 $((CHUNKS - 1))); do
        echo "Evaluating $split split on GPU ${GPULIST[$IDX]}..."
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file $question_file \
            --image-folder $IMAGE_DIR \
            --answers-file $OUTPUT_DIR/${CHUNKS}_${IDX}_${split}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0 \
            --conv-mode vicuna_v1 &
    done
    wait

    output_file=$OUTPUT_DIR/${split}.jsonl
    # Clear out the output file if it exists.
    >"$output_file"

    for IDX in $(seq 0 $((CHUNKS - 1))); do
        cat $OUTPUT_DIR/${CHUNKS}_${IDX}_${split}.jsonl >>"$output_file"
        rm -f $OUTPUT_DIR/${CHUNKS}_${IDX}_${split}.jsonl
    done

    metric_file=$OUTPUT_DIR/${split}_metric.txt
    python /home/lhy/work/front-tech/eval_vqa_relaxed_acc.py \
        --result-file $output_file \
        >>"$metric_file"
}

for split in "${!QUESTION_FILES[@]}"; do
    run_evaluation $split
done
