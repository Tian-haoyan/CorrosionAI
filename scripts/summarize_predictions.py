"""Calculate class proportions and CI* from CorrosionAI predictions."""

import argparse
from pathlib import Path

import pandas as pd


CLASS_NAMES = [
    "Corroded",
    "Etched",
    "Skeletal",
    "Unweathered-A",
    "Unweathered-R",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CorrosionAI predictions.")
    parser.add_argument("--predictions", required=True, help="Prediction CSV")
    parser.add_argument(
        "--output", default="outputs/sample_summary.csv", help="Summary CSV"
    )
    return parser.parse_args()


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_id", "predicted_class"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []
    for sample_id, group in frame.groupby("sample_id", sort=True):
        counts = group["predicted_class"].value_counts()
        total = int(len(group))
        row = {"sample_id": sample_id, "total_grains": total}
        for class_name in CLASS_NAMES:
            count = int(counts.get(class_name, 0))
            row[f"n_{class_name}"] = count
            row[f"percent_{class_name}"] = 100.0 * count / total
        row["CI_star"] = 100.0 * (
            row["n_Skeletal"]
            + 0.75 * row["n_Etched"]
            + 0.5 * row["n_Corroded"]
        ) / total
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    summary = summarize(predictions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Summarized {len(summary)} samples. Results saved to {output}")


if __name__ == "__main__":
    main()
