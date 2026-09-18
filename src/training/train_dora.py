import json
from pathlib import Path

import peft
import torch
import transformers
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

# --------------------------------------------------
# Experiment configuration
# --------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_sft.jsonl"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_sft.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "qwen_dora_r8"
)

MAX_SEQ_LENGTH = 3072

# DoRA
DORA_RANK = 8
DORA_ALPHA = 16
DORA_DROPOUT = 0.05

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Training
LEARNING_RATE = 1e-4
EPOCHS = 2

BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8

WARMUP_STEPS = 0.05
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"

LOGGING_STEPS = 10
EVAL_STEPS = 100
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 2

SEED = 42


def prepare_example(example, tokenizer):
    """
    Convert one chat example into a causal language-model
    training example.

    Loss is calculated only on the assistant response.
    System and user tokens are masked with -100.
    """

    messages = example["messages"]

    # System + user + assistant-generation marker
    prompt_text = tokenizer.apply_chat_template(
        messages[:2],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Full training conversation including gold assistant answer
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Chat templates already include their required special tokens,
    # so do not add another set here.
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]

    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
    )["input_ids"]

    # Do not silently truncate training targets.
    if len(full_ids) > MAX_SEQ_LENGTH:
        raise ValueError(
            f"Training example has {len(full_ids)} tokens, "
            f"which exceeds MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}."
        )

    if len(prompt_ids) >= len(full_ids):
        raise ValueError(
            "No assistant response tokens found in training example."
        )

    # The prompt should be the prefix of the complete conversation.
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Prompt tokens are not aligned with the full conversation."
        )

    input_ids = full_ids

    attention_mask = [1] * len(input_ids)

    labels = input_ids.copy()

    # Ignore system + user tokens during loss calculation.
    labels[:len(prompt_ids)] = (
        [-100] * len(prompt_ids)
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class DataCollator:
    """
    Dynamically pad each batch.

    Label padding uses -100 so padded tokens are ignored
    by the causal language-model loss.
    """

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
    # --------------------------------------------------
    # Hardware checks
    # --------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for DoRA fine-tuning."
        )

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "This experiment requires a BF16-capable GPU."
        )

    set_seed(SEED)

    print("\nDORA FINE-TUNING")
    print("-" * 55)

    print(
        f"GPU          : {torch.cuda.get_device_name(0)}"
    )
    print(
        f"Model        : {MODEL_NAME}"
    )
    print(
        f"Transformers : {transformers.__version__}"
    )
    print(
        f"PEFT         : {peft.__version__}"
    )
    print(
        f"Max sequence : {MAX_SEQ_LENGTH}"
    )
    print(
        f"DoRA rank    : {DORA_RANK}"
    )
    print(
        f"DoRA alpha   : {DORA_ALPHA}"
    )
    print(
        f"Learning rate: {LEARNING_RATE}"
    )
    print(
        f"Epochs       : {EPOCHS}"
    )

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # --------------------------------------------------
    # Base model
    # --------------------------------------------------

    print("\nLoading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
    )

    model.config.use_cache = False

    # --------------------------------------------------
    # DoRA
    # --------------------------------------------------

    dora_config = LoraConfig(
        r=DORA_RANK,
        lora_alpha=DORA_ALPHA,
        lora_dropout=DORA_DROPOUT,

        target_modules=TARGET_MODULES,

        bias="none",
        task_type="CAUSAL_LM",

        # This converts LoRA adaptation to DoRA.
        use_dora=True,
    )

    model = get_peft_model(
        model,
        dora_config,
    )

    # Required for PEFT with gradient checkpointing.
    model.enable_input_require_grads()

    print("\nTRAINABLE PARAMETERS")
    print("-" * 55)

    model.print_trainable_parameters()

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_FILE),
            "validation": str(
                VALIDATION_FILE
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
        ),
        remove_columns=train_columns,
        desc="Tokenizing SFT dataset",
    )

    train_lengths = [
        len(example["input_ids"])
        for example
        in tokenized_dataset["train"]
    ]

    validation_lengths = [
        len(example["input_ids"])
        for example
        in tokenized_dataset["validation"]
    ]

    print("\nDATASET")
    print("-" * 55)

    print(
        "Train examples            :",
        len(tokenized_dataset["train"]),
    )

    print(
        "Validation examples       :",
        len(
            tokenized_dataset[
                "validation"
            ]
        ),
    )

    print(
        "Maximum train tokens      :",
        max(train_lengths),
    )

    print(
        "Maximum validation tokens :",
        max(validation_lengths),
    )

    # Verify that assistant tokens exist
    first_example = (
        tokenized_dataset["train"][0]
    )

    assistant_tokens = sum(
        label != -100
        for label
        in first_example["labels"]
    )

    print(
        "Assistant tokens example  :",
        assistant_tokens,
    )

    if assistant_tokens == 0:
        raise ValueError(
            "No assistant tokens are available "
            "for training loss."
        )

    # --------------------------------------------------
    # TrainingArguments
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),

        num_train_epochs=EPOCHS,

        per_device_train_batch_size=(
            BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            EVAL_BATCH_SIZE
        ),

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        learning_rate=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

        lr_scheduler_type=LR_SCHEDULER,

        # Transformers 5.x accepts a float below 1
        # as a fraction of total training steps.
        warmup_steps=WARMUP_STEPS,

        bf16=True,

        # Useful on NVIDIA A100
        tf32=True,

        gradient_checkpointing=True,

        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },

        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,

        eval_strategy="steps",
        eval_steps=EVAL_STEPS,

        save_strategy="steps",
        save_steps=SAVE_STEPS,

        save_total_limit=SAVE_TOTAL_LIMIT,

        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",
        greater_is_better=False,

        report_to="none",

        seed=SEED,

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
    # Training
    # --------------------------------------------------

    print("\nSTARTING TRAINING")
    print("-" * 55)

    train_result = trainer.train()

    # --------------------------------------------------
    # Final validation loss
    # --------------------------------------------------

    print("\nFINAL VALIDATION")
    print("-" * 55)

    eval_metrics = trainer.evaluate()

    # --------------------------------------------------
    # Save DoRA adapter
    # --------------------------------------------------

    final_adapter_dir = (
        OUTPUT_DIR
        / "final_adapter"
    )

    trainer.model.save_pretrained(
        final_adapter_dir
    )

    tokenizer.save_pretrained(
        final_adapter_dir
    )

    # --------------------------------------------------
    # Save experiment metrics
    # --------------------------------------------------

    metrics = {
        "model": MODEL_NAME,
        "peft_method": "DoRA",

        "rank": DORA_RANK,
        "alpha": DORA_ALPHA,
        "dropout": DORA_DROPOUT,

        "target_modules": TARGET_MODULES,

        "max_seq_length": MAX_SEQ_LENGTH,

        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,

        "batch_size": BATCH_SIZE,

        "gradient_accumulation_steps": (
            GRADIENT_ACCUMULATION_STEPS
        ),

        "effective_batch_size": (
            BATCH_SIZE
            * GRADIENT_ACCUMULATION_STEPS
        ),

        "warmup_steps": WARMUP_STEPS,

        "train_loss": float(
            train_result.training_loss
        ),

        "eval_loss": float(
            eval_metrics["eval_loss"]
        ),

        "train_runtime_seconds": (
            train_result.metrics.get(
                "train_runtime"
            )
        ),

        "gpu": torch.cuda.get_device_name(0),

        "transformers_version": (
            transformers.__version__
        ),

        "peft_version": (
            peft.__version__
        ),
    }

    metrics_path = (
        OUTPUT_DIR
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
    print("-" * 55)

    print(
        "Training loss   :",
        round(
            train_result.training_loss,
            4,
        ),
    )

    print(
        "Validation loss :",
        round(
            eval_metrics["eval_loss"],
            4,
        ),
    )

    print(
        "Adapter saved   :",
        final_adapter_dir,
    )

    print(
        "Metrics saved   :",
        metrics_path,
    )


if __name__ == "__main__":
    main()