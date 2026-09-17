import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "selected_clause_samples.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "dataset_analysis"
    / "sft_dataset_summary.json"
)

PASSAGE_CHARS = 6000


SYSTEM_PROMPT = """
You are a contract clause verification assistant.

Determine whether the target clause is present in the contract passage.

Return only valid JSON in this format:
{"present": true, "evidence": "exact supporting text"}

If the clause is not present, return:
{"present": false, "evidence": null}

The evidence must be copied exactly from the contract passage.
""".strip()


def load_answers(value):
    if pd.isna(value):
        return []

    return json.loads(value)


def create_passage(context, answer_start=None, answer_text=None):
    """
    Create a manageable passage from the contract.

    For positive examples, center the passage around the
    gold evidence so the answer is guaranteed to be present.
    """

    context = str(context)

    if len(context) <= PASSAGE_CHARS:
        return context

    # Negative/fallback passage
    if answer_start is None:
        return context[:PASSAGE_CHARS]

    answer_end = answer_start + len(answer_text)

    space_for_context = PASSAGE_CHARS - len(answer_text)

    if space_for_context < 0:
        return answer_text

    before = space_for_context // 2
    after = space_for_context - before

    start = max(0, answer_start - before)
    end = min(len(context), answer_end + after)

    # Keep passage length close to PASSAGE_CHARS
    if end - start < PASSAGE_CHARS:
        if start == 0:
            end = min(len(context), PASSAGE_CHARS)
        elif end == len(context):
            start = max(0, end - PASSAGE_CHARS)

    passage = context[start:end]

    # Safety check: positive evidence must remain inside passage
    if answer_text not in passage:
        raise ValueError(
            "Gold evidence was not found in the generated passage."
        )

    return passage


def find_hard_negative_passage(row, contract_rows):
    """
    For a negative target clause, use a passage around another
    positive clause from the same contract when possible.

    This creates a more useful negative than taking unrelated text.
    """

    other_positive_rows = contract_rows[
        (contract_rows["is_positive"])
        & (contract_rows["category"] != row["category"])
    ]

    if not other_positive_rows.empty:
        other_row = other_positive_rows.iloc[0]

        answers = load_answers(other_row["answers"])

        if answers:
            answer = answers[0]

            return create_passage(
                context=other_row["context"],
                answer_start=answer["answer_start"],
                answer_text=answer["text"],
            )

    # Fallback if the contract has no other selected positive clause
    return create_passage(row["context"])


def build_user_prompt(category, question, passage):
    return (
        f"Target clause: {category}\n\n"
        f"Clause guidance: {question}\n\n"
        f"Contract passage:\n{passage}"
    )


def build_example(row, contract_rows):
    answers = load_answers(row["answers"])

    if row["is_positive"]:
        answer = answers[0]

        passage = create_passage(
            context=row["context"],
            answer_start=answer["answer_start"],
            answer_text=answer["text"],
        )

        target = {
            "present": True,
            "evidence": answer["text"],
        }

    else:
        passage = find_hard_negative_passage(
            row,
            contract_rows,
        )

        target = {
            "present": False,
            "evidence": None,
        }

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": build_user_prompt(
                row["category"],
                row["question"],
                passage,
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                target,
                ensure_ascii=False,
            ),
        },
    ]

    return {
        "contract": row["contract"],
        "category": row["category"],
        "is_positive": bool(row["is_positive"]),
        "messages": messages,
    }


def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    df = pd.read_csv(INPUT_PATH)

    # Make sure booleans are handled correctly
    if df["is_positive"].dtype != bool:
        df["is_positive"] = (
            df["is_positive"]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False})
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {}

    for split in ["train", "validation", "test"]:
        split_df = df[df["split"] == split]

        records = []

        for _, row in split_df.iterrows():
            contract_rows = df[
                df["contract"] == row["contract"]
            ]

            example = build_example(
                row,
                contract_rows,
            )

            records.append(example)

        if split == "test":
            output_name = "test_eval.jsonl"
        else:
            output_name = f"{split}_sft.jsonl"

        output_path = OUTPUT_DIR / output_name

        save_jsonl(
            records,
            output_path,
        )

        positives = sum(
            record["is_positive"]
            for record in records
        )

        negatives = len(records) - positives

        summary[split] = {
            "total_examples": len(records),
            "positive_examples": positives,
            "negative_examples": negatives,
        }

        print(f"\n{split.upper()} DATASET")
        print("-" * 40)
        print(f"Total examples    : {len(records)}")
        print(f"Positive examples : {positives}")
        print(f"Negative examples : {negatives}")
        print(f"Saved to          : {output_path}")

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\nSFT dataset summary saved to:")
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()