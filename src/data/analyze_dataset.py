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
                answers = qa.get("answers", [])

                # CUAD question IDs contain the clause/category name
                category = qa["id"].split("__")[-1]

                rows.append(
                    {
                        "contract": contract_name,
                        "category": category,
                        "question": qa["question"],
                        "is_positive": len(answers) > 0,
                        "answer_count": len(answers),
                    }
                )

    df = pd.DataFrame(rows)

    # Dataset summary
    dataset_summary = {
        "contracts": int(df["contract"].nunique()),
        "questions": int(len(df)),
        "categories": int(df["category"].nunique()),
        "positive_samples": int(df["is_positive"].sum()),
        "negative_samples": int((~df["is_positive"]).sum()),
    }

    print("\nCUAD DATASET SUMMARY")
    print("-" * 40)

    print(f"Contracts        : {dataset_summary['contracts']}")
    print(f"Questions        : {dataset_summary['questions']}")
    print(f"Categories       : {dataset_summary['categories']}")
    print(f"Positive samples : {dataset_summary['positive_samples']}")
    print(f"Negative samples : {dataset_summary['negative_samples']}")

    # Category-level analysis
    category_summary = (
        df.groupby("category")
        .agg(
            total_samples=("category", "size"),
            positive_samples=("is_positive", "sum"),
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

    # Number of contracts where the clause is actually present
    positive_contracts = (
        df[df["is_positive"]]
        .groupby("category")["contract"]
        .nunique()
    )

    category_summary["positive_contracts"] = (
        category_summary["category"]
        .map(positive_contracts)
        .fillna(0)
        .astype(int)
    )

    category_summary = category_summary.sort_values(
        "positive_samples",
        ascending=False,
    )

    print("\nCATEGORY DISTRIBUTION")
    print("-" * 90)

    print(
        category_summary[
            [
                "category",
                "positive_samples",
                "negative_samples",
                "positive_contracts",
                "positive_ratio",
            ]
        ].to_string(index=False)
    )

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    category_summary.to_csv(
        OUTPUT_DIR / "category_summary.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "dataset_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(dataset_summary, f, indent=2)

    print("\nSaved EDA results to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()