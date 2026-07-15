"""Command-line entry point for CorrosionAI inference."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corrosionai.inference import predict_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify amphibole corrosion features in an image folder."
    )
    parser.add_argument("--input", required=True, help="Input image folder")
    parser.add_argument("--weights", required=True, help="PyTorch checkpoint")
    parser.add_argument(
        "--output", default="outputs/predictions.csv", help="Output CSV file"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--confidence-threshold", type=float, default=0.60)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = predict_folder(
        input_dir=args.input,
        weights=args.weights,
        output_csv=args.output,
        batch_size=args.batch_size,
        confidence_threshold=args.confidence_threshold,
        device_name=args.device,
    )
    print(f"Processed {len(results)} images. Results saved to {args.output}")


if __name__ == "__main__":
    main()
