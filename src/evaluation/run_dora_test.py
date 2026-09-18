import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_eval.jsonl"
)

ADAPTER_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "qwen_dora_r8"
    / "final_adapter"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "finetuned"
    / "qwen_dora_r8_test_predictions.jsonl"
)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Frozen after validation analysis
MAX_NEW_TOKENS = 256


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU is required for DoRA inference."
        )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {INPUT_PATH}"
        )

    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(
            f"DoRA adapter not found: {ADAPTER_PATH}"
        )

    print("\nDORA TEST INFERENCE")
    print("-" * 55)

    print(
        f"GPU     : {torch.cuda.get_device_name(0)}"
    )

    print(
        f"Model   : {MODEL_NAME}"
    )

    print(
        f"Adapter : {ADAPTER_PATH}"
    )

    print(
        f"Test set: {INPUT_PATH}"
    )

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    # --------------------------------------------------
    # Base model
    # --------------------------------------------------

    print("\nLoading base model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    # --------------------------------------------------
    # Load DoRA adapter
    # --------------------------------------------------

    print("Loading DoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )

    # --------------------------------------------------
    # Merge adapter
    # --------------------------------------------------

    print(
        "Merging DoRA adapter into base model..."
    )

    model = model.merge_and_unload(
        safe_merge=True
    )

    model.config.use_cache = True
    model.eval()

    print("DoRA merge complete.")

    # --------------------------------------------------
    # Test data
    # --------------------------------------------------

    records = load_jsonl(
        INPUT_PATH
    )

    print(
        f"Test examples: {len(records)}"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    latencies = []
    generated_token_counts = []

    # --------------------------------------------------
    # Test inference
    # --------------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as output_file:

        for record in tqdm(
            records,
            desc="Running DoRA test",
        ):
            # Same input format used during
            # validation and zero-shot inference.
            input_messages = (
                record["messages"][:2]
            )

            prompt = tokenizer.apply_chat_template(
                input_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
            ).to(model.device)

            torch.cuda.synchronize()

            start_time = time.time()

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=(
                        tokenizer.eos_token_id
                    ),
                )

            torch.cuda.synchronize()

            latency = (
                time.time() - start_time
            )

            generated_tokens = outputs[
                0,
                inputs["input_ids"].shape[1]:
            ]

            prediction = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()

            gold = (
                record["messages"][2]["content"]
            )

            result = {
                "contract": record["contract"],
                "category": record["category"],
                "is_positive": record[
                    "is_positive"
                ],
                "gold": gold,
                "prediction": prediction,
                "latency_seconds": round(
                    latency,
                    4,
                ),
                "generated_tokens": int(
                    len(generated_tokens)
                ),
            }

            output_file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            latencies.append(
                latency
            )

            generated_token_counts.append(
                len(generated_tokens)
            )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    average_latency = (
        sum(latencies) / len(latencies)
        if latencies
        else 0
    )

    average_generated_tokens = (
        sum(generated_token_counts)
        / len(generated_token_counts)
        if generated_token_counts
        else 0
    )

    max_generated_tokens = (
        max(generated_token_counts)
        if generated_token_counts
        else 0
    )

    hit_token_limit = sum(
        count == MAX_NEW_TOKENS
        for count in generated_token_counts
    )

    print("\nDORA TEST INFERENCE COMPLETE")
    print("-" * 55)

    print(
        f"Examples                 : "
        f"{len(records)}"
    )

    print(
        f"Average latency          : "
        f"{average_latency:.2f} seconds"
    )

    print(
        f"Average generated tokens : "
        f"{average_generated_tokens:.1f}"
    )

    print(
        f"Maximum generated tokens : "
        f"{max_generated_tokens}"
    )

    print(
        f"Hit {MAX_NEW_TOKENS}-token limit      : "
        f"{hit_token_limit}"
    )

    print(
        f"Predictions              : "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()