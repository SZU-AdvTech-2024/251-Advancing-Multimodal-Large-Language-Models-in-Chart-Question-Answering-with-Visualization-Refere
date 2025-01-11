import argparse
import itertools
import json
import os
import random
import time
from functools import partial
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vqa import VQA
from vqa_eval import VQAEval

# 设置图片根目录
IMAGE_ROOT_DIR = "playground/data/chartqa/png"  # 替换为您的图片根目录

ds_collections = {
    "chartqa_test_human": {
        "train": "playground/data/chartqa/train_human.jsonl",
        "test": "playground/data/chartqa/test_human.jsonl",
        "metric": "relaxed_accuracy",
        "max_new_tokens": 100,
    },
    "chartqa_test_aug": {
        "train": "playground/data/chartqa/train_aug.jsonl",
        "test": "playground/data/chartqa/test_aug.jsonl",
        "metric": "relaxed_accuracy",
        "max_new_tokens": 100,
    },
}


# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_correctness(target: str, prediction: str, max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith("%"):
                # Convert percentages to floats.
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem["annotation"], str):
            elem["annotation"] = [elem["annotation"]]
        score = max([relaxed_correctness(elem["answer"].strip(), ann) for ann in elem["annotation"]])
        scores.append(score)
    return sum(scores) / len(scores)


def evaluate_exact_match_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem["annotation"], str):
            elem["annotation"] = [elem["annotation"]]
        score = max(
            [(1.0 if (elem["answer"].strip().lower() == ann.strip().lower()) else 0.0) for ann in elem["annotation"]]
        )
        scores.append(score)
    return sum(scores) / len(scores)


def collate_fn(batches, tokenizer):

    questions = [_["question"] for _ in batches]
    question_ids = [_["question_id"] for _ in batches]
    annotations = [_["annotation"] for _ in batches]

    input_ids = tokenizer(questions, return_tensors="pt", padding="longest")

    return question_ids, input_ids.input_ids, input_ids.attention_mask, annotations


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, train, test, prompt, few_shot):
        self.test = open(test).readlines()
        self.prompt = prompt

        self.few_shot = few_shot
        if few_shot > 0:
            self.train = open(train).readlines()

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = json.loads(self.test[idx].strip())
        image, question, question_id, annotation = (
            os.path.join(IMAGE_ROOT_DIR, data["image"]),
            data["question"],
            data["question_id"],
            data.get("answer", None),
        )

        few_shot_prompt = ""
        if self.few_shot > 0:
            few_shot_samples = random.sample(self.train, self.few_shot)
            for sample in few_shot_samples:
                sample = json.loads(sample.strip())
                few_shot_prompt += (
                    self.prompt.format(os.path.join(IMAGE_ROOT_DIR, sample["image"]), sample["question"])
                    + f" {sample['answer']}"
                )

        return {
            "question": few_shot_prompt + self.prompt.format(image, question),
            "question_id": question_id,
            "annotation": annotation,
        }


class InferenceSampler:

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._local_indices = range(size)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen-VL-Chat")
    parser.add_argument("--dataset", type=str, default="chartqa_test_aug")
    parser.add_argument("--image-folder", type=str, default="playground/data/chartqa/png")
    parser.add_argument("--output-dir", type=str, default="playground/eval/chartqa/Qwen-VL-Chat")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--few-shot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="cuda", trust_remote_code=True).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eod_id

    prompt = "<img>{}</img>{} Please answer with a single word or phrase."

    random.seed(args.seed)
    dataset = VQADataset(
        train=ds_collections[args.dataset]["train"],
        test=ds_collections[args.dataset]["test"],
        prompt=prompt,
        few_shot=args.few_shot,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    outputs = []
    for _, (question_ids, input_ids, attention_mask, annotations) in tqdm(enumerate(dataloader)):
        pred = model.generate(
            input_ids=input_ids.cuda(),
            attention_mask=attention_mask.cuda(),
            do_sample=False,
            num_beams=1,
            max_new_tokens=ds_collections[args.dataset]["max_new_tokens"],
            min_new_tokens=1,
            length_penalty=1,
            num_return_sequences=1,
            output_hidden_states=True,
            use_cache=True,
            pad_token_id=tokenizer.eod_id,
            eos_token_id=tokenizer.eod_id,
        )
        answers = [tokenizer.decode(_[input_ids.size(1) :].cpu(), skip_special_tokens=True).strip() for _ in pred]

        for question_id, answer, annotation in zip(question_ids, answers, annotations):
            outputs.append(
                {
                    "answer": answer,
                    "annotation": annotation,
                }
            )

    print(f"Evaluating {args.dataset} ...")
    time_prefix = time.strftime("%y%m%d%H%M%S", time.localtime())
    results_file = f"{args.dataset}_{time_prefix}_fs{args.few_shot}_s{args.seed}.jsonl"
    # 保存结果
    with open(os.path.join(args.output_dir, results_file), "w") as f:
        for output in outputs:
            json.dump(output, f, ensure_ascii=False)
            f.write("\n")

    accuracy = evaluate_relaxed_accuracy(outputs)
    print({"relaxed_accuracy": accuracy})
    with open(os.path.join(args.output_dir, f"{args.dataset}_relaxed_accuracy.txt"), "w") as f:
        f.write(f"relaxed_accuracy: {accuracy}")
