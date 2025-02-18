import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import re

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math


#################################### Extract one word/phrase answer
def extract_answer(s):
    match = re.search(r"the answer is:\s*(.+)", s, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        if answer.endswith("."):
            answer = answer[:-1]
        return answer
    else:
        return None


####################################

################################################################## few show examples
template = """
Question: What was the coverage ratio for bad loans in the Italian banking system as of 2017?answer the question step by step..
Answer: The Answer is: 61. This is the highest value on the graph, which shows the coverage ratio for bad loans in the Italian banking system from 2010 to 2017. The coverage ratio is the ratio of loan loss reserves to non-performing loans, and it indicates the ability of the banking system to cover potential losses from bad loans. In this case, the coverage ratio is 61, which suggests that the Italian banking system has a significant amount of loan loss reserves relative to the amount of non-performing loans.

Question: What was the total assets of the Bank of Communications in 2019?answer the question step by step.
Answer: The Answer is: 9905.6. To determine the total assets of the Bank of Communications in 2019, we need to look at the blue bar on the right side of the graph that corresponds to the year 2019. The total assets for the Bank of Communications in 2019 were 9,950.6 billion yuan.

Question: What was the average cost of a Big Mac in 2021?answer the question step by step.
Answer: The Answer is: 3.56. To calculate the average cost of a Big Mac in 2021, we need to add the costs for each year and divide by the number of years.\n\nIn the image, the costs for a Big Mac in 2019, 2020, and 2021 are 3.18, 3.84, and 3.56, respectively.To find the average, we add the costs:\n3.18 + 3.84 + 3.56 = 10.58 Now, we divide by the number of years: 10.58 / 3 = 3.56 So, the average cost of a Big Mac in 2021 was $3.56.

Question: How many Indian rupees were incurred by Reliance Jio at the end of fiscal year 2020?answer the question step by step..
Answer: The Answer is: 165.82. To determine the amount of Indian rupees incurred by Reliance Jio at the end of fiscal year 2020, we need to look at the last data point on the graph, which is for FY 2020.
"""
################################################################## few show examples


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        # qs = line["question"]
        # qs += "Answer the question using a single word or phrase"

        qs = "I need you to answer the question based on a chart image."
        qs += "answer the question step by step. Here are some answer format examples you should follow:\
        Question: What was the coverage ratio for bad loans in the Italian banking system as of 2017?answer the question step by step.\
        Answer: The Answer is: 61. This is the highest value on the graph, which shows the coverage ratio for bad loans in the Italian banking system from 2010 to 2017. The coverage ratio is the ratio of loan loss reserves to non-performing loans, and it indicates the ability of the banking system to cover potential losses from bad loans. In this case, the coverage ratio is 61, which suggests that the Italian banking system has a significant amount of loan loss reserves relative to the amount of non-performing loans.\
        Question: What was the total assets of the Bank of Communications in 2019?answer the question step by step.\
        Answer: The Answer is: 9905.6. To determine the total assets of the Bank of Communications in 2019, we need to look at the blue bar on the right side of the graph that corresponds to the year 2019. The total assets for the Bank of Communications in 2019 were 9,950.6 billion yuan.\
        Question: What was the average cost of a Big Mac in 2021?answer the question step by step.\
        Answer: The Answer is: 3.56. To calculate the average cost of a Big Mac in 2021, we need to add the costs for each year and divide by the number of years.In the image, the costs for a Big Mac in 2019, 2020, and 2021 are 3.18, 3.84, and 3.56, respectively.To find the average, we add the costs:\n3.18 + 3.84 + 3.56 = 10.58 Now, we divide by the number of years: 10.58 / 3 = 3.56 So, the average cost of a Big Mac in 2021 was $3.56.\
        Question: How many Indian rupees were incurred by Reliance Jio at the end of fiscal year 2020?answer the question step by step.\
        Answer: The Answer is: 165.82. To determine the amount of Indian rupees incurred by Reliance Jio at the end of fiscal year 2020, we need to look at the last data point on the graph, which is for FY 2020."
        qs += "Question:"
        qs += line["question"]
        qs += "answer the question step by step."
        qs += "Answer:"

        label = line["answer"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True,
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        outputs = outputs.split("\n")[-1]

        ################ extract one word/phrase answer
        # outputs = extract_answer(outputs)
        ################

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "image": image_file,
                    "answer": outputs,
                    "annotation": label,
                    "question": cur_prompt,
                }
            )
            + "\n"
        )
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="~/models/llava-hr-ChartInstruction")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="./playground/data/chartqa/png")
    parser.add_argument(
        "--question-file",
        type=str,
        default="./playground/data/chartqa/test_aug.jsonl",
    )
    parser.add_argument(
        "--answers-file",
        type=str,
        default="./playground/eval/chartqa/outputs/test_aug_answers.jsonl",
    )

    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
