import os
import json


def convert_json_files_to_jsonl(dir_path="./"):
    # 获取当前目录下的所有文件
    files = os.listdir(dir_path)

    # 筛选出以 .json 结尾的文件
    json_files = [f for f in files if f.endswith(".json")]

    if not json_files:
        print("当前目录下没有找到 JSON 文件。")
        return

    for json_file in json_files:
        try:
            # 生成对应的 .jsonl 文件名
            jsonl_file = f"{os.path.splitext(json_file)[0]}.jsonl"

            # 读取 JSON 文件
            with open(os.path.join(dir_path, json_file), "r", encoding="utf-8") as f:
                data = json.load(f)

            # 如果 JSON 是对象，将其包装成列表处理
            if isinstance(data, dict):
                data = [data]

            # 添加 question_id 并重命名字段
            for idx, item in enumerate(data):
                item["answer"] = item.pop("answer")
                item["annotation"] = item.pop("annotation")

            # 写入 JSONL 文件
            with open(jsonl_file, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            print(f"成功将 {json_file} 转换为 {jsonl_file}")
        except Exception as e:
            print(f"转换 {json_file} 失败：{e}")


# 调用函数
if __name__ == "__main__":
    convert_json_files_to_jsonl("/home/lhy/work/LLaVA/playground/eval/chartqa/Qwen-VL-Chat/model_vqa_chartqa")
