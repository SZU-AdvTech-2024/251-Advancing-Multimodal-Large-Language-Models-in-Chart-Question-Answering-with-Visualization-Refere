import json


file = "/home/lhy/work/InternVL/results/chartqa_test_human_compositional_241221122421.json"


def process_answer(type, data):
    if "sum up" in type:
        return str(sum(data))
    elif "average" in type:
        return str(sum(data) / len(data))
    elif "difference" in type:
        return str(abs(data[0] - data[1]))
    elif "percentage" in type:
        return str(data[0] / data[1])
    else:
        return "other"


def count_qs_type(data_list):
    type_count = {}
    for item in data_list:
        type_count[item["type"]] = type_count.get(item["type"], 0) + 1
    return type_count


def read_json(file):
    with open(file, "r") as f:
        data_list = json.load(f)
        ret = []
        for item in data_list:
            answer = item["answer"]
            annotation = item["annotation"]
            try:
                i1 = answer.index("type")
                i2 = answer.index("data")
                type = answer[i1 + 5 : i2 - 2].strip()
                i3 = answer.index("[")
                i4 = answer.index("]")
                data = answer[i3 + 1 : i4].strip()
                data = [float(item.strip()) for item in data.split(",")]
                item = {
                    "type": type,
                    "answer": process_answer(type, data),
                    "annotation": annotation,
                }
                ret.append(item)
            except Exception as e:
                print(e)
                continue
        return ret


def read_jsonl(file):
    with open(file, "r") as f:
        data_list = []
        for line in f:
            line = json.loads(line)
            answer = line["answer"]
            annotation = line["annotation"]
            try:
                i1 = answer.index("type")
                i2 = answer.index("data")
                type = answer[i1 + 5 : i2 - 2].strip()
                i3 = answer.index("[")
                i4 = answer.index("]")
                data = answer[i3 + 1 : i4].strip()
                data = [float(item.strip()) for item in data.split(",")]
                item = {
                    "type": type,
                    "data": data,
                    "final_answer": process_answer(type, data),
                    "annotation": annotation,
                }
                data_list.append(item)
            except Exception as e:
                # print(e)
                continue
        return data_list


data_list = read_json(file)
print(data_list)
types = count_qs_type(data_list)
print(types)
total = 0
for type in types:
    total += types[type]
print(total)
with open("intern_data_list.json", "w") as f:
    json.dump(data_list, f)

total = len(data_list)
right = 0
for data in data_list:
    if data["annotation"] == data["answer"]:
        right += 1
print(right / total)
