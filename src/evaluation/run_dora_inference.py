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
    / "validation_sft.jsonl"
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
    / "qwen_dora_r8_validation_predictions.jsonl"
)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

MAX_NEW_TOKENS = 150


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

    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(
            f"DoRA adapter not found: {ADAPTER_PATH}"
        )

    print("\nDORA VALIDATION INFERENCE")
    print("-" * 50)

    print(
        f"GPU     : {torch.cuda.get_device_name(0)}"
    )

    print(
        f"Model   : {MODEL_NAME}"
    )

    print(
        f"Adapter : {ADAPTER_PATH}"
    )

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    # Load trained DoRA adapter
    # --------------------------------------------------

    print("Loading DoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )

    model.eval()

    # --------------------------------------------------
    # Validation data
    # --------------------------------------------------

    records = load_jsonl(
        INPUT_PATH
    )

    print(
        f"Validation examples: {len(records)}"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    latencies = []

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as output_file:

        for record in tqdm(
            records,
            desc="Running DoRA validation",
        ):
            # Use exactly the same input as the
            # zero-shot baseline:
            # system + user only.
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
            }

            output_file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            latencies.append(latency)

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    average_latency = (
        sum(latencies) / len(latencies)
        if latencies
        else 0
    )

    print("\nDORA INFERENCE COMPLETE")
    print("-" * 50)

    print(
        f"Examples        : {len(records)}"
    )

    print(
        f"Average latency : "
        f"{average_latency:.2f} seconds"
    )

    print(
        f"Predictions     : {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()