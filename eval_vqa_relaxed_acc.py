import os
from typing import Optional
import json
import argparse


def relaxed_correctness(prediction: str, target: str, max_relative_change: float = 0.05) -> bool:
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


def relaxed_accuracy(eval_file_path, qtype=False):
    with open(eval_file_path, "r") as eval_file:
        if eval_file_path.endswith("json"):
            entries = json.load(eval_file)
        elif eval_file_path.endswith("jsonl"):
            entries = [json.loads(q) for q in open(eval_file_path, "r")]
        else:
            raise OSError
        if qtype:
            qtype_key = set(entry["type"] for entry in entries)
            data = []
            for qtype in qtype_key:
                tmp = []
                for entry in entries:
                    if entry["type"] == qtype:
                        tmp.append(entry)
                data.append(tmp)
            metrics = []
            for entries in data:
                qtype = entries[0]["type"]
                acc = evaluate_relaxed_accuracy(entries)
                metrics.append(f"{qtype}: {acc * 100:.2f}\n")
            return metrics

        return evaluate_relaxed_accuracy(entries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", type=str, required=True)
    parser.add_argument("--qtype", type=bool, default=False)
    args = parser.parse_args()
    result_file = args.result_file
    result_file_name = os.path.basename(result_file).split(".")[0]
    output_dir = os.path.dirname(result_file)

    if args.qtype:
        output_file_name = f"{result_file_name}_metric_qtype.txt"
    else:
        output_file_name = f"{result_file_name}_metric.txt"
    with open(os.path.join(output_dir, output_file_name), "w") as output_file:
        if args.qtype:
            metrics = relaxed_accuracy(result_file, args.qtype)
            for metric in metrics:
                output_file.write(metric)
        else:
            metrics = relaxed_accuracy(result_file)
            output_file.write(f"relaxed_accuracy: {metrics}")
