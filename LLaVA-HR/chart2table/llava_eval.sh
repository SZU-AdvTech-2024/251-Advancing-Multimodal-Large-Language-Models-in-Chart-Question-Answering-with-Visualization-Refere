#!/bin/bash
source ~/miniconda3/bin/activate llava-hr

gpu_list="${CUDA_VISIBLE_DEVICES:-3}"
IFS=',' read -ra GPULIST <<<"$gpu_list"
CHUNKS=${#GPULIST[@]}

# 设置模型路径
MODEL_PATH=${1:-"/home/lhy/models/llava-hr-ChartInstruction"}

# 图片目录
IMAGE_DIR="images/ai4sci"

# 推理文件
INFERENCE_FILE=${2:-"model_vqa_chart2table_COT2"}
# 输出目录
OUTPUT_DIR="./playground/eval/chart2table/$(basename $MODEL_PATH)/$INFERENCE_FILE/$(date +%Y%m%d%H%M%S)"
mkdir -p $OUTPUT_DIR

# 定义数据集
declare -A QUESTION_FILES=(
    ["test"]="./playground/data/ai4sci/test.jsonl"
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
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.chart2table.$INFERENCE_FILE \
            --model-path $MODEL_PATH \
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

    python playground/eval/chart2table/tool.py --file $output_file
}

for split in "${!QUESTION_FILES[@]}"; do
    run_evaluation $split
done
