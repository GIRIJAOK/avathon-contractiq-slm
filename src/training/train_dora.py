import json
from pathlib import Path

import peft
import torch
import transformers
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
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


def prepare_example(example, tokenizer, max_seq_length):
    """
    Convert one chat example into input_ids, attention_mask
    and labels.

    Loss is calculated only on the assistant response.
    System and user tokens are masked with -100.
    """

    messages = example["messages"]

    # System + user + start of assistant response
    prompt_ids = tokenizer.apply_chat_template(
        messages[:2],
        tokenize=True,
        add_generation_prompt=True,
    )

    # Complete conversation including gold assistant answer
    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )

    # Our token analysis showed all training examples are
    # below 3072, but keep truncation as a safety measure.
    input_ids = full_ids[:max_seq_length]

    prompt_length = min(
        len(prompt_ids),
        len(input_ids),
    )

    if prompt_length >= len(input_ids):
        raise ValueError(
            "Assistant response was removed by truncation. "
            "Increase max_seq_length."
        )

    labels = input_ids.copy()

    # Ignore system + user prompt when calculating loss
    labels[:prompt_length] = [-100] * prompt_length

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class DataCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, examples):
        max_length = max(
            len(example["input_ids"])
            for example in examples
        )

        batch_input_ids = []
        batch_attention_masks = []
        batch_labels = []

        for example in examples:
            padding_length = (
                max_length
                - len(example["input_ids"])
            )

            batch_input_ids.append(
                example["input_ids"]
                + [self.pad_token_id] * padding_length
            )

            batch_attention_masks.append(
                example["attention_mask"]
                + [0] * padding_length
            )

            batch_labels.append(
                example["labels"]
                + [-100] * padding_length
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
            "CUDA GPU is required for DoRA fine-tuning."
        )

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "This configuration expects a BF16-capable GPU."
        )

    config = load_config()

    training_config = config["training"]
    lora_config = config["lora"]

    set_seed(training_config["seed"])

    model_name = config["model_name"]
    max_seq_length = config["max_seq_length"]

    print("\nDORA FINE-TUNING")
    print("-" * 50)

    print(
        f"GPU          : {torch.cuda.get_device_name(0)}"
    )
    print(
        f"Model        : {model_name}"
    )
    print(
        f"Transformers : {transformers.__version__}"
    )
    print(
        f"PEFT         : {peft.__version__}"
    )
    print(
        f"Max sequence : {max_seq_length}"
    )
    print(
        f"DoRA rank    : {lora_config['rank']}"
    )
    print(
        f"DoRA alpha   : {lora_config['alpha']}"
    )

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # --------------------------------------------------
    # Base model
    # --------------------------------------------------

    print("\nLoading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
    )

    # Required when gradient checkpointing is enabled
    model.config.use_cache = False

    # --------------------------------------------------
    # DoRA adapter
    # --------------------------------------------------

    dora_config = LoraConfig(
        r=lora_config["rank"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        target_modules=lora_config[
            "target_modules"
        ],
        bias="none",
        task_type="CAUSAL_LM",

        # Enables DoRA
        use_dora=True,
    )

    model = get_peft_model(
        model,
        dora_config,
    )

    # Important for PEFT + gradient checkpointing
    model.enable_input_require_grads()

    print("\nTRAINABLE PARAMETERS")
    print("-" * 50)

    model.print_trainable_parameters()

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

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

    train_columns = (
        dataset["train"].column_names
    )

    tokenized_dataset = dataset.map(
        lambda example: prepare_example(
            example,
            tokenizer,
            max_seq_length,
        ),
        remove_columns=train_columns,
        desc="Tokenizing SFT dataset",
    )

    print("\nDATASET")
    print("-" * 50)

    print(
        "Train examples      :",
        len(tokenized_dataset["train"]),
    )

    print(
        "Validation examples :",
        len(
            tokenized_dataset[
                "validation"
            ]
        ),
    )

    # Quick sanity check
    first_example = (
        tokenized_dataset["train"][0]
    )

    trainable_label_tokens = sum(
        label != -100
        for label in first_example["labels"]
    )

    print(
        "Assistant tokens in first example:",
        trainable_label_tokens,
    )

    if trainable_label_tokens == 0:
        raise ValueError(
            "No assistant tokens are available "
            "for loss calculation."
        )

    # --------------------------------------------------
    # Training configuration
    # --------------------------------------------------

    output_dir = (
        PROJECT_ROOT
        / config["output_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
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

        per_device_eval_batch_size=(
            training_config[
                "eval_batch_size"
            ]
        ),

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

        lr_scheduler_type=training_config[
            "lr_scheduler_type"
        ],

        # Transformers v5 uses warmup_steps.
        # 0.05 means 5% of total training steps.
        warmup_steps=training_config[
            "warmup_steps"
        ],

        bf16=True,

        # A100 supports TF32 matrix multiplication
        tf32=True,

        gradient_checkpointing=True,

        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },

        logging_strategy="steps",

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

        save_total_limit=training_config[
            "save_total_limit"
        ],

        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",

        greater_is_better=False,

        report_to="none",

        seed=training_config[
            "seed"
        ],

        remove_unused_columns=False,
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    print("\nSTARTING TRAINING")
    print("-" * 50)

    train_result = trainer.train()

    # --------------------------------------------------
    # Final evaluation
    # --------------------------------------------------

    print("\nFINAL VALIDATION LOSS")
    print("-" * 50)

    eval_metrics = trainer.evaluate()

    # --------------------------------------------------
    # Save best DoRA adapter
    # --------------------------------------------------

    final_adapter_dir = (
        output_dir
        / "final_adapter"
    )

    trainer.model.save_pretrained(
        final_adapter_dir
    )

    tokenizer.save_pretrained(
        final_adapter_dir
    )

    # --------------------------------------------------
    # Save training metrics
    # --------------------------------------------------

    metrics = {
        "model": model_name,
        "peft_method": "DoRA",
        "rank": lora_config["rank"],
        "alpha": lora_config["alpha"],
        "dropout": lora_config["dropout"],
        "max_seq_length": max_seq_length,
        "epochs": training_config["epochs"],
        "learning_rate": training_config[
            "learning_rate"
        ],
        "effective_batch_size": (
            training_config["batch_size"]
            * training_config[
                "gradient_accumulation_steps"
            ]
        ),
        "train_loss": float(
            train_result.training_loss
        ),
        "eval_loss": float(
            eval_metrics["eval_loss"]
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
        "Training loss :",
        round(
            train_result.training_loss,
            4,
        ),
    )

    print(
        "Validation loss:",
        round(
            eval_metrics["eval_loss"],
            4,
        ),
    )

    print(
        "Adapter saved :",
        final_adapter_dir,
    )

    print(
        "Metrics saved :",
        metrics_path,
    )


if __name__ == "__main__":
    main()