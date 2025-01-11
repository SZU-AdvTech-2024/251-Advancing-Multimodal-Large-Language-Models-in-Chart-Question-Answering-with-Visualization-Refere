from datetime import datetime
import json
import os
import argparse
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image
import pandas as pd
from io import StringIO
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/lhy/models/deplot")
    parser.add_argument("--image-folder", type=str, default="./images/ai4sci")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)

    args = parser.parse_args()

    model_path = args.model_path
    image_folder = args.image_folder
    question_file = args.question_file
    answers_file = args.answers_file

    processor = Pix2StructProcessor.from_pretrained(model_path)
    model = Pix2StructForConditionalGeneration.from_pretrained(model_path).to(device="cuda")

    questions = []
    with open(question_file, "r") as f:
        for line in f:
            questions.append(json.loads(line))

    for i in tqdm(range(len(questions))):
        question = questions[i]
        qs_id = question["question_id"]
        image_name = question["image"]
        image = Image.open(os.path.join(image_folder, image_name))

        inputs = processor(
            images=image,
            text="Generate underlying data table of the figure below using markdown format",
            return_tensors="pt",
        ).to(device="cuda")

        predictions = model.generate(**inputs, max_new_tokens=512)
        output = processor.decode(predictions[0], skip_special_tokens=True)

        with open(answers_file, "a") as f:
            f.write(json.dumps({"image": image_name, "answer": output}) + "\n")


if __name__ == "__main__":
    main()
