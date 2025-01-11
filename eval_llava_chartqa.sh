#!/bin/bash
source ~/miniconda3/bin/activate llava-hr
cd ~/work/LLaVA-HR

gpu_list="${CUDA_VISIBLE_DEVICES:-3,4}"
IFS=',' read -ra GPULIST <<<"$gpu_list"
CHUNKS=${#GPULIST[@]}

# 设置模型路径
BASE_MODEL=${1:-"/home/lhy/models/llava-hr-ChartInstruction"}
# BASE_MODEL=${1:-"liuhaotian/llava-v1.5-13b"}
CKPT=${2:-"mamachang/llava-chart-13b_lora"}
# CKPT=${2:-"~/checkpoints/llava-v1.5-13b-lora-youho99-ChartLlama"}
# CKPT=${2:-"/home/lhy/work/front-tech/checkpoints/llava-v1.5-13b-chartllama-lora-blue"}

# 图片目录
IMAGE_DIR="/home/lhy/datasets/chartqa/test/png"

# 输出目录
OUTPUT_DIR="./playground/eval/chartqa/$(basename $BASE_MODEL)/$(date +%Y%m%d%H%M%S)"
mkdir -p $OUTPUT_DIR

# 定义数据集
declare -A QUESTION_FILES=(
    ["aug"]="/home/lhy/datasets/chartqa/test/aug.jsonl"
    ["human"]="/home/lhy/datasets/chartqa/test/human.jsonl"
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
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.chartqa.model_vqa_chartqa \
            --model-path $BASE_MODEL \
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
        --qtype True
}

for split in "${!QUESTION_FILES[@]}"; do
    run_evaluation $split
done
