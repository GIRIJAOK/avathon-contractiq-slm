# ContractIQ-SLM

Parameter-efficient domain adaptation of a small language model for grounded contract clause verification and evidence extraction.

## Assignment Setup

- **Track:** B — Small Language Model Fine-Tuning
- **Scenario:** S2 — Enterprise Contract Intelligence
- **Domain:** Commercial contract clause verification and grounded evidence extraction
- **Base model:** Qwen/Qwen2.5-7B-Instruct
- **Adaptation method:** DoRA

## Overview

ContractIQ-SLM explores whether a compact instruction-tuned language model can be adapted to better understand commercial contract clauses without full-model fine-tuning.

For each example, the model receives:

- a target contract clause
- short guidance describing the clause
- a passage from a commercial contract

The model decides whether the clause is present and returns supporting evidence as structured JSON.

```json
{
  "present": true,
  "evidence": "exact supporting text from the contract"
}
```

If the clause is not present:

```json
{
  "present": false,
  "evidence": null
}
```

The project focuses on classification quality, evidence grounding, structured-output reliability, and parameter-efficient adaptation.

---

## Dataset

The project uses **CUAD v1 (Contract Understanding Atticus Dataset)**.

CUAD contains:

- 510 commercial contracts
- 41 legal clause categories
- question-answer style annotations
- evidence spans linked to contract text

For this experiment, eight clause categories were selected:

1. Cap On Liability
2. Audit Rights
3. Termination For Convenience
4. Exclusivity
5. Renewal Term
6. Change Of Control
7. Uncapped Liability
8. Notice Period To Terminate Renewal

Some categories are intentionally close in meaning, for example:

- Cap On Liability vs Uncapped Liability
- Renewal Term vs Notice Period To Terminate Renewal

This makes the task more meaningful than simple keyword matching.

---

## Data Split

Contracts were split at the **contract level** to avoid leakage between training, validation, and test sets.

| Split | Contracts | Examples |
|---|---:|---:|
| Train | 357 | 2,856 |
| Validation | 76 | 608 |
| Test | 77 | 616 |

The final test set was kept untouched until model selection and validation analysis were complete.

---

## Training Data Construction

Each training example contains:

- system instruction
- target clause
- clause guidance
- contract passage
- structured JSON answer

For positive examples, the passage contains the annotated evidence span.

For negative examples, passages were selected from the same contract where possible, so the model sees realistic legal text rather than artificially easy negatives.

The model is trained only on the assistant response. System and user prompt tokens are masked from the training loss.

---

## Base Model

The base model is:

**Qwen/Qwen2.5-7B-Instruct**

It was selected because it provides a practical balance between:

- instruction following
- structured generation
- model size
- local deployment potential
- PEFT support

The maximum training sequence length was set to **3,072 tokens** based on token-length analysis of the generated training examples.

---

## Why Fine-Tuning?

A zero-shot baseline was evaluated first using the same Qwen model.

The baseline showed high precision but low recall. When it identified a clause, it was usually correct, but it missed many clauses that were actually present.

The goal of fine-tuning was therefore to improve:

- recognition of domain-specific legal language
- recall of positive clauses
- evidence extraction
- JSON output consistency
- grounding to the supplied contract passage

---

## Parameter-Efficient Fine-Tuning

The final model was adapted using **DoRA — Weight-Decomposed Low-Rank Adaptation**.

| Parameter | Value |
|---|---|
| Base model | Qwen2.5-7B-Instruct |
| PEFT method | DoRA |
| Rank | 8 |
| Alpha | 16 |
| Dropout | 0.05 |
| Precision | BF16 |
| Quantization | None |
| Max sequence length | 3072 |
| Learning rate | 1e-4 |
| Epochs | 2 |
| Batch size | 1 |
| Gradient accumulation | 8 |
| Effective batch size | 8 |
| Trainable parameters | 21.6M |
| Trainable percentage | ~0.28% |

DoRA was applied to:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Training was performed on an **NVIDIA A100-SXM4-80GB** GPU.

Final training loss:

```text
0.01985
```

Final validation loss:

```text
0.009995
```

---

## Why DoRA?

Several PEFT approaches were considered.

**LoRA** is simple and widely used.

**QLoRA** is attractive when GPU memory is limited because it keeps the base model in 4-bit precision.

For this experiment, an A100 80GB GPU was available, so BF16 DoRA was used without quantization. This kept the experiment focused on domain adaptation without adding quantization as another variable.

DoRA was chosen as the main PEFT method because it provides a more expressive low-rank adaptation while still training only a small fraction of the full model parameters.

---

## Zero-Shot Baseline vs DoRA

The same 608-example validation set was used for both models.

| Metric | Zero-Shot Qwen | DoRA |
|---|---:|---:|
| Accuracy | 0.8289 | **0.9622** |
| Precision | **0.9429** | 0.9387 |
| Recall | 0.3976 | **0.9217** |
| F1 | 0.5593 | **0.9301** |
| Macro F1 | 0.7266 | **0.9521** |
| Evidence Token F1 | 0.2885 | **0.7751** |
| Grounded Evidence Rate | 0.9286 | **0.9939** |
| Unsupported Evidence Rate | 0.0714 | **0.0061** |
| Valid JSON Rate | 0.9770 | **1.0000** |

The largest improvement was in recall.

The zero-shot model identified only about 40% of the positive clauses, while the adapted model identified more than 92% on the validation set, while maintaining high precision.

---

## Final Test Results

After model selection was completed using the validation set, the configuration was frozen and evaluated once on the untouched test set.

| Metric | Test Result |
|---|---:|
| Accuracy | **0.9448** |
| Precision | **0.9451** |
| Recall | **0.8776** |
| F1 | **0.9101** |
| Macro F1 | **0.9351** |
| Evidence Token F1 | **0.7054** |
| Grounded Evidence Rate | **0.9834** |
| Unsupported Evidence Rate | **0.0166** |
| Valid JSON Rate | **0.9951** |
| Valid Schema Rate | **0.9935** |
| Average Inference Latency | **1.039 s/example** |

The final test set contains **616 examples**.

---

## Performance by Clause Type

| Clause | Precision | Recall | F1 |
|---|---:|---:|---:|
| Audit Rights | 1.0000 | 0.9286 | 0.9630 |
| Cap On Liability | 0.9697 | 0.9412 | 0.9552 |
| Change Of Control | 0.9474 | 0.9474 | 0.9474 |
| Exclusivity | 0.9677 | 0.9091 | 0.9375 |
| Renewal Term | 1.0000 | 0.8400 | 0.9130 |
| Uncapped Liability | 0.8333 | 0.9375 | 0.8824 |
| Termination For Convenience | 0.8696 | 0.8000 | 0.8333 |
| Notice Period To Terminate Renewal | 0.9091 | 0.6250 | 0.7407 |

The weakest test category was **Notice Period To Terminate Renewal**, mainly because of missed positive examples.

---

## Error Analysis

On the final test set:

```text
False negatives : 24
False positives : 10
```

False negatives were concentrated mainly in:

```text
Notice Period To Terminate Renewal    6
Termination For Convenience           5
Renewal Term                          4
Exclusivity                           3
Audit Rights                          2
Cap On Liability                      2
Uncapped Liability                    1
Change Of Control                     1
```

Additional observations:

- 3 invalid JSON outputs
- 4 schema-invalid outputs
- 3 unsupported-evidence cases
- 3 outputs reached the 256-token generation limit

The test configuration was not changed after observing these cases.

---

## Evidence Evaluation

Exact string matching is too strict for this task because a model may return a longer valid supporting span than the CUAD annotation.

Evidence quality is therefore measured using:

- token-level evidence F1
- evidence grounding against the supplied passage
- unsupported evidence rate

This provides a more realistic measure of grounded extraction.

---

## Inference

For inference, the trained DoRA adapter is loaded on top of Qwen2.5-7B and merged into the base model before generation.

```text
model = model.merge_and_unload(
    safe_merge=True
)
```

This removes PEFT runtime overhead.

Generation is deterministic:

```text
do_sample = False
max_new_tokens = 256
```

The merged model achieved approximately **1.04 seconds per test example** on the A100 GPU.

---

## Environment

The repository is intended to run with:

- **Python:** 3.12
- **Training environment:** Google Colab
- **Training GPU:** NVIDIA A100-SXM4-80GB
- **Training precision:** BF16

The main Python dependencies are pinned in `requirements.txt`.

---

## Project Structure

```text
avathon-contractiq-slm/
│
├── data/
│   └── processed/
│
├── results/
│   ├── baselines/
│   ├── dataset_analysis/
│   ├── error_analysis/
│   └── finetuned/
│
├── src/
│   ├── benchmark/
│   │   └── run_qwen_baseline.py
│   │
│   ├── data/
│   │   ├── download_dataset.py
│   │   ├── analyze_dataset.py
│   │   ├── analyze_context_lengths.py
│   │   ├── analyze_sft_tokens.py
│   │   ├── build_splits.py
│   │   └── build_sft_dataset.py
│   │
│   ├── evaluation/
│   │   ├── evaluate_predictions.py
│   │   ├── run_dora_inference.py
│   │   ├── run_dora_test.py
│   │   └── analyze_test_errors.py
│   │
│   └── training/
│       └── train_dora.py
│
├── write-up/
│   └── technical_writeup.md
│
├── .gitignore
├── README.md
└── requirements.txt
```

Generated datasets and large model artifacts are intentionally excluded from Git.

---

## Reproducing the Results

### 1. Create an environment

Use Python 3.12.

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Download CUAD

```bash
python -m src.data.download_dataset
```

### 3. Build contract-level splits

```bash
python -m src.data.build_splits
```

### 4. Build SFT examples

```bash
python -m src.data.build_sft_dataset
```

This generates the train, validation, and test JSONL files under `data/processed/`.

### 5. Run the zero-shot baseline

```bash
python -m src.benchmark.run_qwen_baseline
```

### 6. Evaluate the zero-shot baseline

```bash
python -m src.evaluation.evaluate_predictions   --input results/baselines/qwen_zero_shot_predictions.jsonl   --output-dir results/baselines   --name qwen_zero_shot
```

### 7. Fine-tune Qwen with DoRA

```bash
python -m src.training.train_dora
```

### 8. Run DoRA validation inference

```bash
python -m src.evaluation.run_dora_inference
```

### 9. Evaluate validation predictions

```bash
python -m src.evaluation.evaluate_predictions   --input results/finetuned/qwen_dora_r8_validation_predictions.jsonl   --output-dir results/finetuned   --name qwen_dora_r8
```

### 10. Run final test inference

```bash
python -m src.evaluation.run_dora_test
```

### 11. Evaluate the test set

```bash
python -m src.evaluation.evaluate_predictions   --input results/finetuned/qwen_dora_r8_test_predictions.jsonl   --output-dir results/finetuned   --name qwen_dora_r8_test   --validation data/processed/test_eval.jsonl   --split test
```

### 12. Run final error analysis

```bash
python -m src.evaluation.analyze_test_errors
```

---

## Model Artifacts

Large checkpoints and trained adapter weights are intentionally excluded from Git.

The repository contains:

- source code
- deterministic data-generation logic
- split metadata
- baseline predictions
- fine-tuned predictions
- evaluation metrics
- error analysis
- technical write-up

The trained DoRA adapter can be loaded with the original Qwen2.5-7B-Instruct model for reproduction or deployment.

---

## Limitations

This experiment focuses on eight CUAD clause categories rather than all 41 categories.

The model is evaluated on contract passages containing relevant context rather than retrieving evidence from an entire long contract.

Some legal concepts remain difficult when wording is indirect or overlaps with another clause category.

Evidence generation may also produce a longer valid supporting span than the reference annotation.

A production system could extend this work as:

```text
contract ingestion
→ passage retrieval
→ domain-adapted model
→ structured output validation
→ evidence grounding check
→ downstream application
```

Additional evaluation across more contract types, clause categories, and external datasets would be needed before production deployment.

---

## Key Takeaway

The zero-shot Qwen model was already precise when it predicted that a clause was present, but it missed many positive clauses.

DoRA domain adaptation substantially improved recall, evidence extraction, and grounding while preserving high precision.

On the untouched test set, the adapted 7B model achieved:

```text
F1                  0.9101
Macro F1            0.9351
Grounded evidence   0.9834
```

The experiment shows that a relatively small parameter-efficient update can meaningfully improve domain-specific contract understanding without full-model fine-tuning.
