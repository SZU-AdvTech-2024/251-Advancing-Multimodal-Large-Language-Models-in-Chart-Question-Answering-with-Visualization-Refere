import os
import json


def transform_chartqa_data(input_file):
    file_name = os.path.basename(input_file).split(".")[0]
    output_file = os.path.join(cur_dir, f"{file_name}-transformed.json")
    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            new_data = {
                "image": data["imgname"],
                "question": data["query"],
                "annotation": data["label"],
                "qtype": data["qtype"],
                "reason": data["reason"],
            }
            with open(output_file, "a") as out:
                out.write(json.dumps(new_data) + "\n")


def calc_qtype_num(input_file):
    file_name = os.path.basename(input_file).split(".")[0]
    output_file = os.path.join(cur_dir, f"{file_name}-qtype-num.json")
    qtype_num = {}
    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            qtype_num[data["qtype"]] = qtype_num.get(data["qtype"], 0) + 1
    with open(output_file, "w") as out:
        json.dump(qtype_num, out)
    return qtype_num


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    jsonl_files = [f for f in os.listdir(cur_dir) if f.endswith(".jsonl")]
    for file in jsonl_files:
        # transform_chartqa_data(os.path.join(cur_dir, file))
        qtype_num = calc_qtype_num(os.path.join(cur_dir, file))
        print(qtype_num)
