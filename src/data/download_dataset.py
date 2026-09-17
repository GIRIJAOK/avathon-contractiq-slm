"""Download the official CUAD v1 dataset.

The dataset is downloaded from the official Atticus Project
Hugging Face repository rather than through the legacy CUAD-QA
dataset loading script.

CUAD:
Contract Understanding Atticus Dataset
License: CC BY 4.0
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "theatticusproject/cuad"
REPO_TYPE = "dataset"
HF_FILENAME = "CUAD_v1/CUAD_v1.json"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
LOCAL_DATASET_PATH = RAW_DATA_DIR / "CUAD_v1.json"
METADATA_PATH = RAW_DATA_DIR / "dataset_metadata.json"


def sha256sum(path: Path) -> str:
    """Return SHA256 checksum for reproducibility."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def inspect_dataset(path: Path) -> dict:
    """Return basic statistics from CUAD's SQuAD-style JSON."""
    with path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    contracts = dataset["data"]

    paragraph_count = 0
    question_count = 0
    positive_questions = 0
    negative_questions = 0

    for contract in contracts:
        for paragraph in contract.get("paragraphs", []):
            paragraph_count += 1

            for qa in paragraph.get("qas", []):
                question_count += 1

                if qa.get("answers"):
                    positive_questions += 1
                else:
                    negative_questions += 1

    return {
        "contracts": len(contracts),
        "paragraphs": paragraph_count,
        "questions": question_count,
        "positive_questions": positive_questions,
        "negative_questions": negative_questions,
    }


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading official CUAD v1 dataset...")
    print(f"Repository : {REPO_ID}")
    print(f"File       : {HF_FILENAME}")

    cached_file = hf_hub_download(
        repo_id=REPO_ID,
        filename=HF_FILENAME,
        repo_type=REPO_TYPE,
    )

    shutil.copy2(cached_file, LOCAL_DATASET_PATH)

    stats = inspect_dataset(LOCAL_DATASET_PATH)
    checksum = sha256sum(LOCAL_DATASET_PATH)

    metadata = {
        "dataset": "Contract Understanding Atticus Dataset (CUAD) v1",
        "source": f"https://huggingface.co/datasets/{REPO_ID}",
        "repository": REPO_ID,
        "source_file": HF_FILENAME,
        "license": "CC BY 4.0",
        "local_file": str(LOCAL_DATASET_PATH.relative_to(PROJECT_ROOT)),
        "sha256": checksum,
        **stats,
    }

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nDownload complete.")
    print(f"Saved to   : {LOCAL_DATASET_PATH}")
    print(f"SHA256     : {checksum}")
    print("\nDataset summary:")

    for key, value in stats.items():
        print(f"  {key:20s}: {value:,}")


if __name__ == "__main__":
    main()