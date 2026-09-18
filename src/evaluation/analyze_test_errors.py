import json
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DETAILS_PATH = (
    PROJECT_ROOT
    / "results"
    / "finetuned"
    / "qwen_dora_r8_test_evaluation_details.csv"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "finetuned"
    / "qwen_dora_r8_test_predictions.jsonl"
)


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def main():
    details = pd.read_csv(DETAILS_PATH)
    predictions = load_jsonl(PREDICTIONS_PATH)

    # --------------------------------------------------
    # False negatives / false positives
    # --------------------------------------------------

    false_negatives = details[
        (details["gold_present"] == True)
        & (details["strict_prediction"] == False)
    ]

    false_positives = details[
        (details["gold_present"] == False)
        & (details["strict_prediction"] == True)
    ]

    print("\nFINAL TEST ERROR ANALYSIS")
    print("-" * 60)

    print(
        f"Total examples     : {len(details)}"
    )

    print(
        f"False negatives    : {len(false_negatives)}"
    )

    print(
        f"False positives    : {len(false_positives)}"
    )

    print("\nFALSE NEGATIVES BY CATEGORY")
    print("-" * 60)

    print(
        false_negatives[
            "category"
        ].value_counts()
    )

    print("\nFALSE POSITIVES BY CATEGORY")
    print("-" * 60)

    print(
        false_positives[
            "category"
        ].value_counts()
    )

    # --------------------------------------------------
    # JSON / schema errors
    # --------------------------------------------------

    invalid_json = details[
        details["json_valid"] == False
    ]

    invalid_schema = details[
        details["schema_valid"] == False
    ]

    print("\nOUTPUT FORMAT")
    print("-" * 60)

    print(
        f"Invalid JSON       : {len(invalid_json)}"
    )

    print(
        f"Invalid schema     : {len(invalid_schema)}"
    )

    # --------------------------------------------------
    # Unsupported evidence
    # --------------------------------------------------

    unsupported = details[
        details["grounded_evidence"] == False
    ]

    print("\nGROUNDING")
    print("-" * 60)

    print(
        f"Unsupported evidence: {len(unsupported)}"
    )

    if len(unsupported) > 0:
        print(
            unsupported[
                "category"
            ].value_counts()
        )

    # --------------------------------------------------
    # Evidence quality
    # --------------------------------------------------

    evidence_rows = details[
        details["evidence_f1"].notna()
    ].copy()

    lowest_evidence = evidence_rows.sort_values(
        "evidence_f1"
    ).head(10)

    print("\nLOWEST EVIDENCE-F1 EXAMPLES")
    print("-" * 60)

    print(
        lowest_evidence[
            [
                "category",
                "gold_present",
                "predicted_present",
                "evidence_f1",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------
    # Generation limit
    # --------------------------------------------------

    hit_limit = [
        row
        for row in predictions
        if row.get("generated_tokens") == 256
    ]

    print("\nGENERATION LIMIT")
    print("-" * 60)

    print(
        f"Hit 256-token limit: {len(hit_limit)}"
    )

    for row in hit_limit:
        print(
            f"- {row['category']}"
        )


if __name__ == "__main__":
    main()