import json
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "dora_qwen.yaml"
)


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_example(
    example,
    tokenizer,
    max_seq_length,
):
    """
    Build a causal-LM training example.

    Loss is calculated only on the assistant response.
    System + user prompt tokens are masked with -100.
    """

    messages = example["messages"]

    prompt_text = tokenizer.apply_chat_template(
        messages[:2],
        tokenize=False,
        add_generation_prompt=True,
    )

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=False,
    )

    prompt_tokens = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=False,
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    labels = input_ids.copy()

    prompt_length = min(
        len(prompt_tokens["input_ids"]),
        len(labels),
    )

    # Do not train on the system/user prompt.
    labels[:prompt_length] = (
        [-100] * prompt_length
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class DataCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = (
            tokenizer.pad_token_id
        )

    def __call__(self, examples):
        max_length = max(
            len(example["input_ids"])
            for example in examples
        )

        batch_input_ids = []
        batch_attention_masks = []
        batch_labels = []

        for example in examples:
            length = len(
                example["input_ids"]
            )

            padding = max_length - length

            input_ids = (
                example["input_ids"]
                + [self.pad_token_id] * padding
            )

            attention_mask = (
                example["attention_mask"]
                + [0] * padding
            )

            labels = (
                example["labels"]
                + [-100] * padding
            )

            batch_input_ids.append(
                input_ids
            )

            batch_attention_masks.append(
                attention_mask
            )

            batch_labels.append(
                labels
            )

        return {
            "input_ids": torch.tensor(
                batch_input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                batch_attention_masks,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                batch_labels,
                dtype=torch.long,
            ),
        }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA GPU is required for DoRA training."
        )

    config = load_config()

    model_name = config["model_name"]
    max_seq_length = config[
        "max_seq_length"
    ]

    print("\nDORA FINE-TUNING")
    print("-" * 50)

    print(
        "GPU          :",
        torch.cuda.get_device_name(0),
    )

    print(
        "Model        :",
        model_name,
    )

    print(
        "Max sequence :",
        max_seq_length,
    )

    print(
        "LoRA rank    :",
        config["lora"]["rank"],
    )

    print(
        "LoRA alpha   :",
        config["lora"]["alpha"],
    )

    # -------------------------
    # Tokenizer
    # -------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = "right"

    # -------------------------
    # Base model
    # -------------------------

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    model.config.use_cache = False

    # -------------------------
    # DoRA configuration
    # -------------------------

    peft_config = LoraConfig(
        r=config["lora"]["rank"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config[
            "lora"
        ]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",

        # This turns LoRA into DoRA
        use_dora=True,
    )

    model = get_peft_model(
        model,
        peft_config,
    )

    print("\nTRAINABLE PARAMETERS")
    print("-" * 50)

    model.print_trainable_parameters()

    # -------------------------
    # Dataset
    # -------------------------

    train_path = (
        PROJECT_ROOT
        / config["train_file"]
    )

    validation_path = (
        PROJECT_ROOT
        / config["validation_file"]
    )

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "validation": str(
                validation_path
            ),
        },
    )

    original_columns = (
        dataset["train"].column_names
    )

    tokenized_dataset = dataset.map(
        lambda example: prepare_example(
            example,
            tokenizer,
            max_seq_length,
        ),
        remove_columns=original_columns,
        desc="Tokenizing SFT dataset",
    )

    print("\nDATASET")
    print("-" * 50)

    print(
        "Train examples      :",
        len(
            tokenized_dataset[
                "train"
            ]
        ),
    )

    print(
        "Validation examples :",
        len(
            tokenized_dataset[
                "validation"
            ]
        ),
    )

    # -------------------------
    # Training arguments
    # -------------------------

    training_config = config[
        "training"
    ]

    output_dir = (
        PROJECT_ROOT
        / config["output_dir"]
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),

        num_train_epochs=training_config[
            "epochs"
        ],

        per_device_train_batch_size=(
            training_config[
                "batch_size"
            ]
        ),

        per_device_eval_batch_size=1,

        gradient_accumulation_steps=(
            training_config[
                "gradient_accumulation_steps"
            ]
        ),

        learning_rate=training_config[
            "learning_rate"
        ],

        weight_decay=training_config[
            "weight_decay"
        ],

        warmup_ratio=training_config[
            "warmup_ratio"
        ],

        bf16=True,

        gradient_checkpointing=True,

        logging_steps=training_config[
            "logging_steps"
        ],

        eval_strategy="steps",

        eval_steps=training_config[
            "eval_steps"
        ],

        save_strategy="steps",

        save_steps=training_config[
            "save_steps"
        ],

        save_total_limit=2,

        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",

        greater_is_better=False,

        report_to="none",

        seed=training_config[
            "seed"
        ],

        remove_unused_columns=False,
    )

    # -------------------------
    # Trainer
    # -------------------------

    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=(
            tokenized_dataset["train"]
        ),

        eval_dataset=(
            tokenized_dataset[
                "validation"
            ]
        ),

        data_collator=DataCollator(
            tokenizer
        ),
    )

    print("\nSTARTING TRAINING")
    print("-" * 50)

    train_result = trainer.train()

    # -------------------------
    # Save adapter
    # -------------------------

    final_adapter_dir = (
        output_dir / "final_adapter"
    )

    trainer.save_model(
        str(final_adapter_dir)
    )

    tokenizer.save_pretrained(
        str(final_adapter_dir)
    )

    # -------------------------
    # Final validation loss
    # -------------------------

    eval_metrics = trainer.evaluate()

    metrics = {
        "train_loss": (
            train_result.training_loss
        ),
        "eval_loss": (
            eval_metrics.get(
                "eval_loss"
            )
        ),
        "rank": config["lora"][
            "rank"
        ],
        "alpha": config["lora"][
            "alpha"
        ],
        "epochs": training_config[
            "epochs"
        ],
        "learning_rate": (
            training_config[
                "learning_rate"
            ]
        ),
    }

    metrics_path = (
        output_dir
        / "training_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    print("\nTRAINING COMPLETE")
    print("-" * 50)

    print(
        "Adapter saved to:",
        final_adapter_dir,
    )

    print(
        "Metrics saved to:",
        metrics_path,
    )


if __name__ == "__main__":
    main()