
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "CUAD_v1.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "dataset_analysis"


def load_cuad():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    data = load_cuad()

    rows = []

    for contract in data["data"]:
        contract_name = contract["title"]

        for paragraph in contract["paragraphs"]:
            for qa in paragraph["qas"]:

                question = qa["question"]
                answers = qa.get("answers", [])

                # CUAD question IDs contain the clause/category name
                category = qa["id"].split("__")[-1]

                rows.append(
                    {
                        "contract": contract_name,
                        "category": category,
                        "question": question,
                        "is_positive": len(answers) > 0,
                        "answer_count": len(answers),
                    }
                )

    df = pd.DataFrame(rows)

    print("\nCUAD DATASET SUMMARY")
    print("-" * 40)

    print(f"Contracts        : {df['contract'].nunique()}")
    print(f"Questions        : {len(df)}")
    print(f"Categories       : {df['category'].nunique()}")
    print(f"Positive samples : {df['is_positive'].sum()}")
    print(f"Negative samples : {(~df['is_positive']).sum()}")

    category_summary = (
        df.groupby("category")
        .agg(
            total_samples=("category", "size"),
            positive_samples=("is_positive", "sum"),
            contracts=("contract", "nunique"),
        )
        .reset_index()
    )

    category_summary["negative_samples"] = (
        category_summary["total_samples"]
        - category_summary["positive_samples"]
    )

    category_summary["positive_ratio"] = (
        category_summary["positive_samples"]
        / category_summary["total_samples"]
    )

    category_summary = category_summary.sort_values(
        "positive_samples",
        ascending=False,
    )

    print("\nCATEGORY DISTRIBUTION")
    print("-" * 80)

    print(
        category_summary[
            [
                "category",
                "positive_samples",
                "negative_samples",
                "contracts",
                "positive_ratio",
            ]
        ].to_string(index=False)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "category_summary.csv"

    category_summary.to_csv(
        output_file,
        index=False,
    )

    print(f"\nSaved category summary to:")
    print(output_file)


if __name__ == "__main__":
    main()