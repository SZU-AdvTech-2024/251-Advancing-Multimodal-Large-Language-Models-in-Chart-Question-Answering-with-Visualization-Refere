import argparse
import json


def main(args):
    with open(args.result_file, "r") as f:
        data = [json.loads(line) for line in f]
        for item in data:
            if item["question_type"] == "bar":
                print(item["question_type"], item["question"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", type=str, required=True)
    args = parser.parse_args()
    main(args)
