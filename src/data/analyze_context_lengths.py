import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "selected_clause_samples.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "dataset_analysis"


def main():
    df = pd.read_csv(DATA_PATH)

    # The same contract context appears for several clause categories.
    # Keep only one copy of each contract for length analysis.
    contracts = (
        df[["contract", "context", "split"]]
        .drop_duplicates(subset=["contract"])
        .copy()
    )

    contracts["character_count"] = (
        contracts["context"]
        .fillna("")
        .str.len()
    )

    contracts["word_count"] = (
        contracts["context"]
        .fillna("")
        .str.split()
        .str.len()
    )

    print("\nCONTEXT LENGTH ANALYSIS")
    print("-" * 50)

    print(f"Contracts analyzed : {len(contracts)}")

    print("\nWORD COUNT")
    print("-" * 50)

    print(
        contracts["word_count"]
        .describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
    )

    print("\nCONTEXT LENGTH BY SPLIT")
    print("-" * 50)

    split_summary = (
        contracts.groupby("split")["word_count"]
        .agg(
            contracts="count",
            min_words="min",
            median_words="median",
            mean_words="mean",
            max_words="max",
        )
        .reset_index()
    )

    print(split_summary.to_string(index=False))

    summary = {
        "contracts": int(len(contracts)),
        "min_words": int(contracts["word_count"].min()),
        "median_words": float(
            contracts["word_count"].median()
        ),
        "mean_words": float(
            contracts["word_count"].mean()
        ),
        "p90_words": float(
            contracts["word_count"].quantile(0.90)
        ),
        "p95_words": float(
            contracts["word_count"].quantile(0.95)
        ),
        "p99_words": float(
            contracts["word_count"].quantile(0.99)
        ),
        "max_words": int(
            contracts["word_count"].max()
        ),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    contracts[
        [
            "contract",
            "split",
            "word_count",
            "character_count",
        ]
    ].to_csv(
        OUTPUT_DIR / "context_lengths.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "context_length_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\nSaved context-length analysis to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()