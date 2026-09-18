import argparse
import gc
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_ADAPTER = "artifacts/qwen_dora_r8/final_adapter"
DEFAULT_OUTPUT = "results/general_capability/general_capability_results.json"


TESTS = [
    {
        "id": "math_1",
        "category": "reasoning",
        "prompt": "A factory produces 120 units per hour. How many units are produced in 7 hours? Answer with only the number.",
        "type": "exact",
        "expected": "840",
    },
    {
        "id": "math_2",
        "category": "reasoning",
        "prompt": "Calculate 1000 - 375. Answer with only the number.",
        "type": "exact",
        "expected": "625",
    },
    {
        "id": "reasoning_1",
        "category": "reasoning",
        "prompt": "If today is Monday, what day will it be three days later? Answer with only the day.",
        "type": "exact",
        "expected": "Thursday",
    },
    {
        "id": "factual_1",
        "category": "general_qa",
        "prompt": "What is the capital of Japan? Answer with only the city name.",
        "type": "exact",
        "expected": "Tokyo",
    },
    {
        "id": "factual_2",
        "category": "general_qa",
        "prompt": "Which planet is known as the Red Planet? Answer with only the planet name.",
        "type": "exact",
        "expected": "Mars",
    },
    {
        "id": "instruction_1",
        "category": "instruction_following",
        "prompt": "Respond with exactly the word BLUE and nothing else.",
        "type": "exact",
        "expected": "BLUE",
    },
    {
        "id": "instruction_2",
        "category": "instruction_following",
        "prompt": "Convert the word 'enterprise' to uppercase. Return only the converted word.",
        "type": "exact",
        "expected": "ENTERPRISE",
    },
    {
        "id": "sorting_1",
        "category": "reasoning",
        "prompt": "Sort these numbers in ascending order: 9, 2, 7, 1. Return only: 1, 2, 7, 9",
        "type": "exact",
        "expected": "1, 2, 7, 9",
    },
    {
        "id": "json_1",
        "category": "structured_output",
        "prompt": """
Return the following information as valid JSON only.

Name: Alice
Department: Finance
Location: London

Use exactly these keys:
name
department
location
""",
        "type": "json",
        "expected": {
            "name": "Alice",
            "department": "Finance",
            "location": "London",
        },
    },
    {
        "id": "json_2",
        "category": "structured_output",
        "prompt": """
Return valid JSON only.

Product: Sensor
Quantity: 12
Available: true

Use keys:
product
quantity
available
""",
        "type": "json",
        "expected": {
            "product": "Sensor",
            "quantity": 12,
            "available": True,
        },
    },
    {
        "id": "classification_1",
        "category": "classification",
        "prompt": """
Classify the sentiment of this sentence as POSITIVE or NEGATIVE.

"The system worked correctly and completed the task successfully."

Return only the label.
""",
        "type": "exact",
        "expected": "POSITIVE",
    },
    {
        "id": "classification_2",
        "category": "classification",
        "prompt": """
Classify the sentiment of this sentence as POSITIVE or NEGATIVE.

"The application repeatedly failed and produced incorrect results."

Return only the label.
""",
        "type": "exact",
        "expected": "NEGATIVE",
    },
    {
        "id": "extraction_1",
        "category": "information_extraction",
        "prompt": """
Extract only the person's name from this sentence:

"Sarah Johnson joined Acme Corporation in 2024."

Return only the name.
""",
        "type": "exact",
        "expected": "Sarah Johnson",
    },
    {
        "id": "summary_1",
        "category": "summarization",
        "prompt": """
Summarize the following in one short sentence:

Machine learning systems should be monitored after deployment because
data distributions may change over time, causing model performance to degrade.
""",
        "type": "keywords",
        "expected": ["monitor", "data", "performance"],
    },
    {
        "id": "summary_2",
        "category": "summarization",
        "prompt": """
Summarize the following in one short sentence:

Predictive maintenance uses equipment data to identify early signs of
failure so maintenance can be performed before an unexpected breakdown.
""",
        "type": "keywords",
        "expected": ["maintenance", "failure"],
    },
]


def normalize(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def extract_json(text):
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None

    return None


def evaluate_output(output, test):
    if test["type"] == "exact":
        return normalize(output) == normalize(str(test["expected"]))

    if test["type"] == "json":
        parsed = extract_json(output)

        if parsed is None:
            return False

        return parsed == test["expected"]

    if test["type"] == "keywords":
        output_lower = output.lower()

        return all(
            keyword.lower() in output_lower
            for keyword in test["expected"]
        )

    return False


def load_base_model(model_name):
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()

        if major >= 8:
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )

    model.eval()

    return model


def generate(model, tokenizer, prompt):
    messages = [
        {
            "role": "system",
            "content": (
                "Follow the user's instruction exactly. "
                "Keep the answer concise."
            ),
        },
        {
            "role": "user",
            "content": prompt.strip(),
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    device = next(model.parameters()).device

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    generated_tokens = generated[0][inputs["input_ids"].shape[1]:]

    output = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return output.strip()


def evaluate_model(model, tokenizer):
    results = []

    for i, test in enumerate(TESTS, start=1):
        print(f"[{i}/{len(TESTS)}] {test['id']}")

        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=test["prompt"],
        )

        passed = evaluate_output(output, test)

        results.append(
            {
                "id": test["id"],
                "category": test["category"],
                "prompt": test["prompt"].strip(),
                "expected": test["expected"],
                "output": output,
                "passed": passed,
            }
        )

        print("Output :", output)
        print("Passed :", passed)
        print()

    return results


def release_model(model):
    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-model",
        default=BASE_MODEL,
    )

    parser.add_argument(
        "--adapter",
        default=DEFAULT_ADAPTER,
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nGENERAL CAPABILITY SANITY CHECK")
    print("=" * 60)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: CUDA not available. CPU inference will be slow.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # ---------------------------------------------------------
    # Base model
    # ---------------------------------------------------------

    print("\nLoading BASE model...\n")

    base_model = load_base_model(args.base_model)

    base_results = evaluate_model(
        base_model,
        tokenizer,
    )

    release_model(base_model)

    # ---------------------------------------------------------
    # Fine-tuned model
    # ---------------------------------------------------------

    print("\nLoading BASE + DoRA adapter...\n")

    finetuned_model = load_base_model(args.base_model)

    finetuned_model = PeftModel.from_pretrained(
        finetuned_model,
        args.adapter,
    )

    finetuned_model = finetuned_model.merge_and_unload(
        safe_merge=True,
    )

    finetuned_model.eval()

    finetuned_results = evaluate_model(
        finetuned_model,
        tokenizer,
    )

    # ---------------------------------------------------------
    # Compare
    # ---------------------------------------------------------

    base_pass = sum(x["passed"] for x in base_results)
    finetuned_pass = sum(x["passed"] for x in finetuned_results)

    regressions = []

    for base, tuned in zip(base_results, finetuned_results):
        if base["passed"] and not tuned["passed"]:
            regressions.append(base["id"])

    summary = {
        "total_prompts": len(TESTS),
        "base_passed": base_pass,
        "base_pass_rate": base_pass / len(TESTS),
        "finetuned_passed": finetuned_pass,
        "finetuned_pass_rate": finetuned_pass / len(TESTS),
        "regression_count": len(regressions),
        "regressions": regressions,
    }

    output_data = {
        "base_model": args.base_model,
        "adapter": args.adapter,
        "summary": summary,
        "base_results": base_results,
        "finetuned_results": finetuned_results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            output_data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"Prompts              : {len(TESTS)}")
    print(f"Base passed          : {base_pass}/{len(TESTS)}")
    print(f"Fine-tuned passed    : {finetuned_pass}/{len(TESTS)}")
    print(f"Regressions          : {len(regressions)}")

    if regressions:
        print("Regression cases     :", ", ".join(regressions))
    else:
        print("Regression cases     : None")

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()