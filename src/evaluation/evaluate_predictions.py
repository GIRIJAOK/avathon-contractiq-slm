import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "baselines"
    / "qwen_zero_shot_predictions.jsonl"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_sft.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "baselines"
)


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def parse_prediction(text):
    """
    Parse model output as JSON.

    Handles:
    - plain JSON
    - ```json ... ```
    - JSON embedded in extra text
    """

    text = text.strip()

    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        pass

    cleaned = (
        text.replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:
        return json.loads(cleaned), True
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1:
        try:
            return (
                json.loads(
                    cleaned[start:end + 1]
                ),
                True,
            )
        except json.JSONDecodeError:
            pass

    return None, False


def valid_schema(result):
    if not isinstance(result, dict):
        return False

    if "present" not in result:
        return False

    if "evidence" not in result:
        return False

    if not isinstance(
        result["present"],
        bool,
    ):
        return False

    if result["present"]:
        return isinstance(
            result["evidence"],
            str,
        )

    return result["evidence"] is None


def normalize(text):
    return " ".join(
        str(text).split()
    )


def tokenize(text):
    return re.findall(
        r"\w+",
        str(text).lower(),
    )


def calculate_evidence_f1(
    predicted,
    gold,
):
    predicted_tokens = tokenize(predicted)
    gold_tokens = tokenize(gold)

    if not predicted_tokens:
        return 0.0

    if not gold_tokens:
        return 0.0

    predicted_counts = Counter(
        predicted_tokens
    )

    gold_counts = Counter(
        gold_tokens
    )

    overlap = sum(
        (
            predicted_counts
            & gold_counts
        ).values()
    )

    if overlap == 0:
        return 0.0

    precision = (
        overlap
        / len(predicted_tokens)
    )

    recall = (
        overlap
        / len(gold_tokens)
    )

    return (
        2
        * precision
        * recall
        / (precision + recall)
    )


def main():
    predictions = load_jsonl(
        PREDICTIONS_PATH
    )

    validation = load_jsonl(
        VALIDATION_PATH
    )

    if len(predictions) != len(validation):
        raise ValueError(
            "Prediction count does not match "
            "validation count."
        )

    rows = []

    gold_labels = []
    predicted_labels = []

    valid_json_count = 0
    valid_schema_count = 0

    evidence_predictions = 0
    grounded_evidence_count = 0

    evidence_scores = []

    for prediction_record, validation_record in zip(
        predictions,
        validation,
    ):
        # Check that prediction order matches
        # the original validation dataset.
        if (
            prediction_record["contract"]
            != validation_record["contract"]
            or
            prediction_record["category"]
            != validation_record["category"]
        ):
            raise ValueError(
                "Prediction order does not match "
                "validation data."
            )

        gold_result = json.loads(
            prediction_record["gold"]
        )

        gold_label = bool(
            gold_result["present"]
        )

        parsed, json_valid = parse_prediction(
            prediction_record["prediction"]
        )

        if json_valid:
            valid_json_count += 1

        schema_valid = (
            json_valid
            and valid_schema(parsed)
        )

        if schema_valid:
            valid_schema_count += 1
            predicted_label = bool(
                parsed["present"]
            )

        else:
            predicted_label = None

        # Invalid JSON/schema should count as
        # an incorrect classification.
        if predicted_label is None:
            strict_prediction = (
                not gold_label
            )
        else:
            strict_prediction = (
                predicted_label
            )

        gold_labels.append(
            gold_label
        )

        predicted_labels.append(
            strict_prediction
        )

        grounded = None
        current_evidence_f1 = None

        # -------------------------
        # Evidence grounding
        # -------------------------
        if (
            schema_valid
            and parsed["present"]
            and parsed["evidence"]
        ):
            evidence_predictions += 1

            predicted_evidence = (
                parsed["evidence"]
            )

            passage = validation_record[
                "messages"
            ][1]["content"]

            grounded = (
                normalize(
                    predicted_evidence
                )
                in normalize(
                    passage
                )
            )

            if grounded:
                grounded_evidence_count += 1

        # -------------------------
        # Evidence F1
        # -------------------------
        # Calculate on gold-positive examples.
        # False negatives receive F1 = 0.
        if gold_label:
            gold_evidence = (
                gold_result["evidence"]
            )

            if (
                schema_valid
                and parsed["present"]
                and parsed["evidence"]
            ):
                current_evidence_f1 = (
                    calculate_evidence_f1(
                        parsed["evidence"],
                        gold_evidence,
                    )
                )

            else:
                current_evidence_f1 = 0.0

            evidence_scores.append(
                current_evidence_f1
            )

        rows.append(
            {
                "contract": (
                    prediction_record[
                        "contract"
                    ]
                ),
                "category": (
                    prediction_record[
                        "category"
                    ]
                ),
                "gold_present": gold_label,
                "predicted_present": (
                    predicted_label
                ),
                "strict_prediction": (
                    strict_prediction
                ),
                "json_valid": json_valid,
                "schema_valid": schema_valid,
                "grounded_evidence": grounded,
                "evidence_f1": (
                    current_evidence_f1
                ),
                "latency_seconds": (
                    prediction_record[
                        "latency_seconds"
                    ]
                ),
            }
        )

    results_df = pd.DataFrame(
        rows
    )

    # -------------------------
    # Overall classification
    # -------------------------

    accuracy = accuracy_score(
        gold_labels,
        predicted_labels,
    )

    precision = precision_score(
        gold_labels,
        predicted_labels,
        zero_division=0,
    )

    recall = recall_score(
        gold_labels,
        predicted_labels,
        zero_division=0,
    )

    f1 = f1_score(
        gold_labels,
        predicted_labels,
        zero_division=0,
    )

    macro_f1 = f1_score(
        gold_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    # -------------------------
    # Output quality
    # -------------------------

    total_examples = len(
        results_df
    )

    valid_json_rate = (
        valid_json_count
        / total_examples
    )

    valid_schema_rate = (
        valid_schema_count
        / total_examples
    )

    # -------------------------
    # Evidence metrics
    # -------------------------

    grounded_evidence_rate = (
        grounded_evidence_count
        / evidence_predictions
        if evidence_predictions
        else 0.0
    )

    unsupported_evidence_rate = (
        1 - grounded_evidence_rate
        if evidence_predictions
        else 0.0
    )

    mean_evidence_f1 = (
        sum(evidence_scores)
        / len(evidence_scores)
        if evidence_scores
        else 0.0
    )

    average_latency = (
        results_df[
            "latency_seconds"
        ].mean()
    )

    summary = {
        "model": (
            "Qwen/Qwen2.5-7B-Instruct"
        ),
        "evaluation_split": (
            "validation"
        ),
        "total_examples": (
            total_examples
        ),
        "accuracy": round(
            accuracy,
            4,
        ),
        "precision": round(
            precision,
            4,
        ),
        "recall": round(
            recall,
            4,
        ),
        "f1": round(
            f1,
            4,
        ),
        "macro_f1": round(
            macro_f1,
            4,
        ),
        "valid_json_rate": round(
            valid_json_rate,
            4,
        ),
        "valid_schema_rate": round(
            valid_schema_rate,
            4,
        ),
        "evidence_token_f1": round(
            mean_evidence_f1,
            4,
        ),
        "grounded_evidence_rate": round(
            grounded_evidence_rate,
            4,
        ),
        "unsupported_evidence_rate": round(
            unsupported_evidence_rate,
            4,
        ),
        "average_latency_seconds": round(
            average_latency,
            4,
        ),
    }

    # -------------------------
    # Per-category metrics
    # -------------------------

    category_results = []

    for category, group in (
        results_df.groupby("category")
    ):
        # Explicit bool conversion fixes:
        # "mix of binary and unknown targets"
        category_gold = (
            group["gold_present"]
            .astype(bool)
            .to_numpy()
        )

        category_predicted = (
            group["strict_prediction"]
            .astype(bool)
            .to_numpy()
        )

        category_results.append(
            {
                "category": category,
                "examples": len(group),
                "positive_examples": int(
                    category_gold.sum()
                ),
                "precision": round(
                    precision_score(
                        category_gold,
                        category_predicted,
                        zero_division=0,
                    ),
                    4,
                ),
                "recall": round(
                    recall_score(
                        category_gold,
                        category_predicted,
                        zero_division=0,
                    ),
                    4,
                ),
                "f1": round(
                    f1_score(
                        category_gold,
                        category_predicted,
                        zero_division=0,
                    ),
                    4,
                ),
            }
        )

    category_df = pd.DataFrame(
        category_results
    ).sort_values(
        "f1",
        ascending=True,
    )

    # -------------------------
    # Save results
    # -------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_DIR
        / "qwen_zero_shot_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    category_df.to_csv(
        OUTPUT_DIR
        / "qwen_zero_shot_category_metrics.csv",
        index=False,
    )

    results_df.to_csv(
        OUTPUT_DIR
        / "qwen_zero_shot_evaluation_details.csv",
        index=False,
    )

    # -------------------------
    # Print results
    # -------------------------

    print(
        "\nQWEN ZERO-SHOT BASELINE"
    )
    print("-" * 55)

    for key, value in summary.items():
        print(
            f"{key:32}: {value}"
        )

    print(
        "\nPER-CATEGORY PERFORMANCE"
    )
    print("-" * 80)

    print(
        category_df.to_string(
            index=False
        )
    )

    print(
        "\nEvaluation results saved to:"
    )
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()