import json
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
files = os.listdir(cur_dir)
print(files)
for file in files:
    if file.endswith(".jsonl"):
        path = os.path.join(cur_dir, file)
        output_path = os.path.join(cur_dir, f"simplified_{file}.jsonl")
        simplified_data = []
        with open(path, "r") as f:
            for line in f:
                data = json.loads(line)
                simplified_data.append({"answer": data["answer"], "annotation": data["annotation"]})
        with open(output_path, "a") as f:
            for item in simplified_data:
                f.write(json.dumps(item) + "\n")
