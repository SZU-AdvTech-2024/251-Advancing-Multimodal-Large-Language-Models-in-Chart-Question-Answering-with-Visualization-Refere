export CUDA_VISIBLE_DEVICES=1

python -m llava.serve.cli \
    --model-path /home/lhy/models/llava-hr-ChartInstruction \
    --image-file "./images/ai4sci/ch4.png"
