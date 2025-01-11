import json
import argparse
import os


def transform_josnl_txt(jsonl_file, txt_file):
    with open(jsonl_file, "r") as f:
        for line in f:
            data = json.loads(line)
            answer = data["answer"]
            if "<0x0A>" in answer:
                answer = answer.replace("<0x0A>", "|\n|")
            with open(txt_file, "a") as f:
                f.write(answer + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default="/home/lhy/work/InternVL/results/ai4science_241211180426.json",
    )
    args = parser.parse_args()
    file_name = os.path.basename(args.file).split(".")[0]
    output_dir = os.path.dirname(args.file)

    transform_josnl_txt(args.file, os.path.join(output_dir, f"{file_name}.txt"))
