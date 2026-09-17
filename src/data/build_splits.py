import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "CUAD_v1.json"
CONFIG_PATH = PROJECT_ROOT / "configs" / "selected_categories.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_SEED = 42


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_selected_categories():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config["selected_categories"]


def extract_records(data, selected_categories):
    rows = []

    for contract in data["data"]:
        contract_name = contract["title"]

        for paragraph in contract["paragraphs"]:
            context = paragraph["context"]

            for qa in paragraph["qas"]:
                category = qa["id"].split("__")[-1]

                if category not in selected_categories:
                    continue

                answers = qa.get("answers", [])
                answer_texts = [answer["text"] for answer in answers]

                rows.append(
                    {
                        "contract": contract_name,
                        "category": category,
                        "question": qa["question"],
                        "context": context,
                        "is_positive": len(answers) > 0,
                        "answers": json.dumps(
                            answer_texts,
                            ensure_ascii=False,
                        ),
                    }
                )

    return pd.DataFrame(rows)


def main():
    data = load_json(DATASET_PATH)
    selected_categories = load_selected_categories()

    df = extract_records(
        data,
        selected_categories,
    )

    contracts = sorted(
        df["contract"].unique()
    )

    # 70% train, 30% temporary split
    train_contracts, temp_contracts = train_test_split(
        contracts,
        test_size=0.30,
        random_state=RANDOM_SEED,
    )

    # Temporary split is divided equally:
    # 15% validation and 15% test
    val_contracts, test_contracts = train_test_split(
        temp_contracts,
        test_size=0.50,
        random_state=RANDOM_SEED,
    )

    train_contracts = set(train_contracts)
    val_contracts = set(val_contracts)
    test_contracts = set(test_contracts)

    # Make sure there is no contract leakage
    assert train_contracts.isdisjoint(val_contracts)
    assert train_contracts.isdisjoint(test_contracts)
    assert val_contracts.isdisjoint(test_contracts)

    df["split"] = df["contract"].apply(
        lambda contract: (
            "train"
            if contract in train_contracts
            else "validation"
            if contract in val_contracts
            else "test"
        )
    )

    print("\nCONTRACT SPLIT")
    print("-" * 40)

    print(f"Train contracts      : {len(train_contracts)}")
    print(f"Validation contracts : {len(val_contracts)}")
    print(f"Test contracts       : {len(test_contracts)}")

    print("\nSAMPLE DISTRIBUTION")
    print("-" * 40)

    print(
        df.groupby("split").size()
    )

    print("\nPOSITIVE SAMPLES BY CATEGORY AND SPLIT")
    print("-" * 80)

    positive_distribution = (
        df[df["is_positive"]]
        .groupby(["category", "split"])
        .size()
        .unstack(fill_value=0)
    )

    print(
        positive_distribution
    )

    # Save outputs
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_info = {
        "seed": RANDOM_SEED,
        "train_contracts": sorted(train_contracts),
        "validation_contracts": sorted(val_contracts),
        "test_contracts": sorted(test_contracts),
    }

    with open(
        OUTPUT_DIR / "contract_splits.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            split_info,
            f,
            indent=2,
        )

    df.to_csv(
        OUTPUT_DIR / "selected_clause_samples.csv",
        index=False,
    )

    positive_distribution.to_csv(
        OUTPUT_DIR / "split_category_distribution.csv"
    )

    print("\nSaved split files to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()