import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_sft.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "dataset_analysis"
    / "sft_token_length_summary.json"
)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def get_token_lengths(records, tokenizer):
    lengths = []

    for record in records:
        text = tokenizer.apply_chat_template(
            record["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

        tokens = tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]

        lengths.append(len(tokens))

    return lengths


def summarize(lengths):
    values = pd.Series(lengths)

    return {
        "model": MODEL_NAME,
        "examples": len(values),
        "min_tokens": int(values.min()),
        "median_tokens": int(values.median()),
        "p90_tokens": int(values.quantile(0.90)),
        "p95_tokens": int(values.quantile(0.95)),
        "p99_tokens": int(values.quantile(0.99)),
        "max_tokens": int(values.max()),
        "recommended_max_seq_length": 3072,
    }


def main():
    records = load_jsonl(DATA_PATH)

    print(f"\nLoading tokenizer: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    lengths = get_token_lengths(
        records,
        tokenizer,
    )

    stats = summarize(lengths)

    print("\nTOKEN LENGTH SUMMARY")
    print("-" * 50)

    print(f"Examples      : {stats['examples']}")
    print(f"Min tokens    : {stats['min_tokens']}")
    print(f"Median tokens : {stats['median_tokens']}")
    print(f"P90 tokens    : {stats['p90_tokens']}")
    print(f"P95 tokens    : {stats['p95_tokens']}")
    print(f"P99 tokens    : {stats['p99_tokens']}")
    print(f"Max tokens    : {stats['max_tokens']}")

    print(
        "\nRecommended max sequence length:",
        stats["recommended_max_seq_length"],
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            stats,
            f,
            indent=2,
        )

    print("\nSaved token analysis to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()