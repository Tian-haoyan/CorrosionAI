"""
Run external validation inference for CorrosionAI.

This script predicts corrosion-feature classes for image folders arranged as:

    input_dir/
        sample_1/
            C-image0001.png
            E-image0002.png
            ...
        sample_2/
            ...

For normal user-facing inference, file names are not treated as ground truth.
If --evaluate-ground-truth is used, file-name prefixes are read as labels:
    C = Corroded
    E = Etched
    S = Skeletal
    A = Unweathered-angular
    R = Unweathered-rounded

Example:
    python inference/run_external_test.py ^
        --input examples/external_test ^
        --weights weights/best_acc.pth ^
        --output outputs/external_test ^
        --device auto

Use --save-per-image-figures only when probability plots for every image are
needed, because this can generate many files.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


CLASS_NAMES = [
    "Corroded",       # C
    "Etched",         # E
    "Skeletal",       # S
    "Unweathered-A",  # A
    "Unweathered-R",  # R
]

PREFIX_TO_CLASS = {
    "C": "Corroded",
    "E": "Etched",
    "S": "Skeletal",
    "A": "Unweathered-A",
    "R": "Unweathered-R",
}

SHORT_NAME_MAP = {
    "Corroded": "C",
    "Etched": "E",
    "Skeletal": "S",
    "Unweathered-A": "A",
    "Unweathered-R": "R",
}

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

SUMMARY_OUTPUT_ORDER = [
    "Unweathered-R",
    "Unweathered-A",
    "Corroded",
    "Etched",
    "Skeletal",
]


SAMPLE_GROUPS = {
    "group1_Nile_Delta_core_samples": {
        "samples": [
            "ND3",
            "ND4",
            "ND7",
            "ND9",
            "ND41",
            "ND42",
            "NDM7",
            "NDM16",
            "NDM22",
            "NDM33",
            "NDM34",
        ],
        "title": "Nile Delta core samples",
    },
    "group2_lower_Indus_and_Ganga_Brahmaputra_River_samples": {
        "samples": ["S1481", "S1486", "S1489", "S3559", "S3560", "S3562"],
        "title": "lower Indus River and Ganga-Brahmaputra River samples",
    },
    "group3_Thar_Desert_samples": {
        "samples": ["S5996", "S6135"],
        "title": "Thar Desert samples",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CorrosionAI external validation inference."
    )
    parser.add_argument(
        "--input",
        default="examples/external_test",
        help="Input folder containing one subfolder per sample.",
    )
    parser.add_argument(
        "--weights",
        default="weights/best_acc.pth",
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="outputs/external_test",
        help="Output folder for predictions, summaries, and figures.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=256,
        help="Image size used for model inference.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for inference.",
    )
    parser.add_argument(
        "--save-per-image-figures",
        action="store_true",
        help="Save probability bar plots for every image.",
    )
    parser.add_argument(
        "--no-random-equalize",
        action="store_true",
        help="Disable histogram equalization during preprocessing.",
    )
    parser.add_argument(
        "--evaluate-ground-truth",
        action="store_true",
        help=(
            "Evaluate predictions against labels parsed from file-name prefixes. "
            "Use this for validation datasets only, not for normal user inference."
        ),
    )
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        return torch.device("cuda:0")
    return torch.device("cpu")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def build_transform(img_size: int, use_random_equalize: bool = True) -> transforms.Compose:
    steps: List[object] = [transforms.Resize((img_size, img_size))]
    if use_random_equalize:
        # p=1.0 makes the transform deterministic in whether it is applied.
        steps.append(transforms.RandomEqualize(p=1.0))
    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)


def parse_true_label_from_filename(filename: str) -> Optional[str]:
    """
    Parse true class label from file names such as:
        C-image0030.png -> Corroded
        E-image0257.png -> Etched
        A-image0240.png -> Unweathered-A

    If the file name does not start with a valid class prefix, None is returned.
    """
    prefix = filename.split("-")[0].strip().upper()
    return PREFIX_TO_CLASS.get(prefix)


def load_model(model_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes),
    )

    # PyTorch 2.6 changed torch.load(weights_only=True) to the default.
    # This checkpoint was generated by the authors and may contain metadata
    # objects saved by older PyTorch/Numpy versions, so we explicitly load the
    # trusted checkpoint with weights_only=False for compatibility.
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_one_image(
    model: nn.Module,
    image_path: Path,
    transform: transforms.Compose,
    device: torch.device,
) -> Tuple[Image.Image, np.ndarray, str, float]:
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_name = CLASS_NAMES[pred_idx]
    pred_conf = float(probs[pred_idx])
    return image, probs, pred_name, pred_conf


def init_count_dict() -> Dict[str, int]:
    return {cls: 0 for cls in CLASS_NAMES}


def calculate_ci_star(count_dict: Dict[str, int]) -> float:
    """Calculate the refined corrosion index CI* from class counts."""
    total = sum(count_dict.values())
    if total == 0:
        return 0.0
    weighted_corrosion = (
        count_dict["Skeletal"]
        + 0.75 * count_dict["Etched"]
        + 0.5 * count_dict["Corroded"]
    )
    return weighted_corrosion / total * 100.0


def update_confusion_matrix(cm: np.ndarray, true_name: Optional[str], pred_name: str) -> None:
    if true_name is None:
        return
    cm[CLASS_TO_IDX[true_name], CLASS_TO_IDX[pred_name]] += 1


def safe_divide(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def calculate_metrics_from_cm(cm: np.ndarray) -> Tuple[float, Dict[str, Dict[str, float]]]:
    total = int(cm.sum())
    correct = int(np.trace(cm))
    accuracy = safe_divide(correct, total)

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    for i, cls in enumerate(CLASS_NAMES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(total - tp - fp - fn)

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)

        per_class_metrics[cls] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(cm[i, :].sum()),
        }

    return accuracy, per_class_metrics


def normalize_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    cm_float = cm.astype(np.float64)
    row_sums = cm_float.sum(axis=1, keepdims=True)
    return np.divide(
        cm_float,
        row_sums,
        out=np.zeros_like(cm_float, dtype=np.float64),
        where=row_sums != 0,
    )


def save_count_txt(count_dict: Dict[str, int], total_num: int, save_path: Path, title: str) -> None:
    with save_path.open("w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total images: {total_num}\n\n")
        for cls in SUMMARY_OUTPUT_ORDER:
            f.write(f"{SHORT_NAME_MAP[cls]} ({cls}): {count_dict[cls]}\n")


def save_ci_star_txt(
    pred_count: Dict[str, int],
    save_path: Path,
    title: str,
    true_count: Optional[Dict[str, int]] = None,
) -> None:
    predicted_ci = calculate_ci_star(pred_count)

    with save_path.open("w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write("=" * 60 + "\n")
        f.write("Formula: CI* = (Skeletal + 0.75 * Etched + 0.5 * Corroded) / N * 100\n\n")
        f.write(f"Predicted CI*: {predicted_ci:.6f}\n")
        if true_count is not None:
            true_ci = calculate_ci_star(true_count)
            absolute_difference = abs(predicted_ci - true_ci)
            f.write(f"True CI*: {true_ci:.6f}\n")
            f.write(f"Absolute difference: {absolute_difference:.6f}\n")


def save_confusion_matrix_txt(cm: np.ndarray, save_path: Path, title: str) -> None:
    with save_path.open("w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write("Rows = True label, columns = Predicted label\n\n")
        header = ["True\\Pred"] + [SHORT_NAME_MAP[c] for c in CLASS_NAMES]
        f.write("\t".join(header) + "\n")
        for i, true_cls in enumerate(CLASS_NAMES):
            row_name = SHORT_NAME_MAP[true_cls]
            row_vals = [str(int(x)) for x in cm[i]]
            f.write("\t".join([row_name] + row_vals) + "\n")


def save_normalized_confusion_matrix_txt(norm_cm: np.ndarray, save_path: Path, title: str) -> None:
    with save_path.open("w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write("Rows = True label, columns = Predicted label\n")
        f.write("Values = Row-normalized proportions\n\n")
        header = ["True\\Pred"] + [SHORT_NAME_MAP[c] for c in CLASS_NAMES]
        f.write("\t".join(header) + "\n")
        for i, true_cls in enumerate(CLASS_NAMES):
            row_name = SHORT_NAME_MAP[true_cls]
            row_vals = [f"{x:.4f}" for x in norm_cm[i]]
            f.write("\t".join([row_name] + row_vals) + "\n")


def save_metrics_txt(
    accuracy: float,
    per_class_metrics: Dict[str, Dict[str, float]],
    save_path: Path,
    title: str,
) -> None:
    with save_path.open("w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Overall accuracy: {accuracy:.6f}\n\n")
        f.write("Class\tSupport\tPrecision\tRecall\tF1\n")
        for cls in SUMMARY_OUTPUT_ORDER:
            m = per_class_metrics[cls]
            f.write(
                f"{SHORT_NAME_MAP[cls]} ({cls})\t{int(m['support'])}\t"
                f"{m['precision']:.6f}\t{m['recall']:.6f}\t{m['f1']:.6f}\n"
            )


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Path,
    title: str,
    normalize: bool = False,
) -> None:
    short_names = [SHORT_NAME_MAP[c] for c in CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(8, 7), dpi=200)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=short_names,
        yticklabels=short_names,
        xlabel="Predicted Label",
        ylabel="True Label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_val = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(
                j,
                i,
                text_val,
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
            )

    fig.tight_layout()
    plt.savefig(save_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.savefig(save_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def save_probability_figure(
    image: Image.Image,
    probs: np.ndarray,
    pred_name: str,
    true_name: Optional[str],
    filename: str,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    axes[0].imshow(image)
    if true_name is None:
        axes[0].set_title(f"Predicted: {pred_name}", fontsize=12)
    else:
        title_color = "green" if pred_name == true_name else "red"
        axes[0].set_title(
            f"Predicted: {pred_name}\nTrue: {true_name}",
            fontsize=12,
            color=title_color,
        )
    axes[0].axis("off")

    y_pos = np.arange(len(CLASS_NAMES))
    axes[1].barh(y_pos, probs)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(CLASS_NAMES)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1.05)
    axes[1].set_xlabel("Probability")
    axes[1].set_title("Class probabilities")
    for i, p in enumerate(probs):
        axes[1].text(min(p + 0.02, 1.01), i, f"{p:.3f}", va="center", fontsize=9)

    fig.suptitle(filename, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def collect_sample_folders(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.iterdir() if p.is_dir()], key=lambda p: p.name)


def collect_image_paths(sample_dir: Path) -> List[Path]:
    image_paths: List[Path] = []
    for root, _, files in os.walk(sample_dir):
        root_path = Path(root)
        for filename in files:
            path = root_path / filename
            if is_image_file(path):
                image_paths.append(path)
    return sorted(image_paths, key=lambda p: str(p))


def write_predictions_csv(
    rows: Iterable[Dict[str, object]],
    save_path: Path,
    include_ground_truth: bool = False,
) -> None:
    fieldnames = [
        "sample",
        "filename",
        "predicted_label",
        "confidence",
        "predicted_CI_class_weight",
        "prob_C",
        "prob_E",
        "prob_S",
        "prob_A",
        "prob_R",
    ]
    if include_ground_truth:
        fieldnames.insert(2, "true_label")
    with save_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_sample_summary(
    sample_name: str,
    sample_dir: Path,
    pred_count: Dict[str, int],
    true_count: Dict[str, int],
    cm: np.ndarray,
    evaluate_ground_truth: bool = False,
) -> None:
    ensure_dir(sample_dir)

    total_images = int(sum(pred_count.values()))

    save_count_txt(
        pred_count,
        total_images,
        sample_dir / "prediction_count.txt",
        f"Prediction Count Summary - {sample_name}",
    )
    save_ci_star_txt(
        pred_count,
        sample_dir / "ci_star_summary.txt",
        f"CI* Summary - {sample_name}",
        true_count=true_count if evaluate_ground_truth else None,
    )

    if not evaluate_ground_truth:
        return

    norm_cm = normalize_confusion_matrix(cm)
    accuracy, per_class_metrics = calculate_metrics_from_cm(cm)
    total_labeled = int(sum(true_count.values()))

    save_count_txt(
        true_count,
        total_labeled,
        sample_dir / "true_count.txt",
        f"True Count Summary - {sample_name}",
    )
    save_metrics_txt(
        accuracy,
        per_class_metrics,
        sample_dir / "metrics.txt",
        f"Classification Metrics - {sample_name}",
    )
    save_confusion_matrix_txt(
        cm,
        sample_dir / "confusion_matrix.txt",
        f"Confusion Matrix - {sample_name}",
    )
    save_normalized_confusion_matrix_txt(
        norm_cm,
        sample_dir / "normalized_confusion_matrix.txt",
        f"Normalized Confusion Matrix - {sample_name}",
    )
    plot_confusion_matrix(
        cm,
        sample_dir / "confusion_matrix",
        f"Confusion Matrix - {sample_name}",
        normalize=False,
    )
    plot_confusion_matrix(
        norm_cm,
        sample_dir / "normalized_confusion_matrix",
        f"Normalized Confusion Matrix - {sample_name}",
        normalize=True,
    )


def save_group_confusion_matrices(sample_cm_dict: Dict[str, np.ndarray], output_dir: Path) -> None:
    group_root = output_dir / "group_summary"
    ensure_dir(group_root)

    for group_name, group_info in SAMPLE_GROUPS.items():
        group_cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
        existing_samples = []

        for sample in group_info["samples"]:
            if sample in sample_cm_dict:
                group_cm += sample_cm_dict[sample]
                existing_samples.append(sample)

        if not existing_samples:
            continue

        title = str(group_info["title"])
        group_dir = group_root / group_name
        ensure_dir(group_dir)

        norm_cm = normalize_confusion_matrix(group_cm)
        accuracy, per_class_metrics = calculate_metrics_from_cm(group_cm)
        group_true_count = {
            cls: int(group_cm[CLASS_TO_IDX[cls], :].sum())
            for cls in CLASS_NAMES
        }
        group_pred_count = {
            cls: int(group_cm[:, CLASS_TO_IDX[cls]].sum())
            for cls in CLASS_NAMES
        }

        save_confusion_matrix_txt(
            group_cm,
            group_dir / "group_confusion_matrix.txt",
            f"Confusion Matrix - {title}",
        )
        save_normalized_confusion_matrix_txt(
            norm_cm,
            group_dir / "group_normalized_confusion_matrix.txt",
            f"Normalized Confusion Matrix - {title}",
        )
        save_metrics_txt(
            accuracy,
            per_class_metrics,
            group_dir / "group_metrics.txt",
            f"Classification Metrics - {title}",
        )
        save_ci_star_txt(
            group_pred_count,
            group_dir / "group_ci_star_summary.txt",
            f"Group CI* Summary - {title}",
            true_count=group_true_count,
        )
        plot_confusion_matrix(
            group_cm,
            group_dir / "group_confusion_matrix",
            f"Confusion Matrix - {title}",
            normalize=False,
        )
        plot_confusion_matrix(
            norm_cm,
            group_dir / "group_normalized_confusion_matrix",
            f"Normalized Confusion Matrix - {title}",
            normalize=True,
        )

        with (group_dir / "group_samples.txt").open("w", encoding="utf-8") as f:
            f.write(f"group: {group_name}\n")
            f.write(f"title: {title}\n")
            f.write(f"total_labeled_images: {int(group_cm.sum())}\n")
            f.write(f"samples: {', '.join(existing_samples)}\n")

        print(f"[Group] {title}: {int(group_cm.sum())} labeled images")


def ci_class_weight(class_name: str) -> float:
    if class_name == "Skeletal":
        return 1.0
    if class_name == "Etched":
        return 0.75
    if class_name == "Corroded":
        return 0.5
    return 0.0


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    weights_path = Path(args.weights)
    output_dir = Path(args.output)
    device = choose_device(args.device)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {weights_path}\n"
            "Place the checkpoint at weights/best_acc.pth or pass --weights PATH."
        )

    ensure_dir(output_dir)
    transform = build_transform(args.img_size, use_random_equalize=not args.no_random_equalize)

    print(f"Using device: {device}")
    print(f"Input folder: {input_dir}")
    print(f"Checkpoint: {weights_path}")
    print(f"Output folder: {output_dir}")

    model = load_model(weights_path, len(CLASS_NAMES), device)

    all_rows: List[Dict[str, object]] = []
    sample_cm_dict: Dict[str, np.ndarray] = {}

    overall_pred_count = init_count_dict()
    overall_true_count = init_count_dict()
    overall_cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)

    sample_folders = collect_sample_folders(input_dir)
    if not sample_folders:
        raise RuntimeError(f"No sample folders found under: {input_dir}")

    for sample_folder in sample_folders:
        sample_name = sample_folder.name
        image_paths = collect_image_paths(sample_folder)
        if not image_paths:
            print(f"[Skip] No images found in sample folder: {sample_name}")
            continue

        print(f"\n===== {sample_name} =====")
        print(f"Images: {len(image_paths)}")

        sample_pred_count = init_count_dict()
        sample_true_count = init_count_dict()
        sample_cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)

        figure_dir = output_dir / "per_image_figures" / sample_name
        if args.save_per_image_figures:
            ensure_dir(figure_dir)

        for i, image_path in enumerate(image_paths, start=1):
            filename = image_path.name
            true_name = (
                parse_true_label_from_filename(filename)
                if args.evaluate_ground_truth
                else None
            )
            image, probs, pred_name, pred_conf = predict_one_image(
                model, image_path, transform, device
            )

            sample_pred_count[pred_name] += 1
            overall_pred_count[pred_name] += 1

            if args.evaluate_ground_truth and true_name is not None:
                sample_true_count[true_name] += 1
                overall_true_count[true_name] += 1

            if args.evaluate_ground_truth:
                update_confusion_matrix(sample_cm, true_name, pred_name)
                update_confusion_matrix(overall_cm, true_name, pred_name)

            row = {
                "sample": sample_name,
                "filename": filename,
                "predicted_label": pred_name,
                "confidence": f"{pred_conf:.6f}",
                "predicted_CI_class_weight": f"{ci_class_weight(pred_name):.2f}",
                "prob_C": f"{probs[CLASS_TO_IDX['Corroded']]:.6f}",
                "prob_E": f"{probs[CLASS_TO_IDX['Etched']]:.6f}",
                "prob_S": f"{probs[CLASS_TO_IDX['Skeletal']]:.6f}",
                "prob_A": f"{probs[CLASS_TO_IDX['Unweathered-A']]:.6f}",
                "prob_R": f"{probs[CLASS_TO_IDX['Unweathered-R']]:.6f}",
            }
            if args.evaluate_ground_truth:
                row["true_label"] = true_name if true_name is not None else ""
            all_rows.append(row)

            if args.save_per_image_figures:
                save_probability_figure(
                    image=image,
                    probs=probs,
                    pred_name=pred_name,
                    true_name=true_name if args.evaluate_ground_truth else None,
                    filename=filename,
                    save_path=figure_dir / f"{image_path.stem}_probability.png",
                )

            if i % 25 == 0 or i == len(image_paths):
                print(f"[{sample_name}] {i}/{len(image_paths)} images processed")

        if args.evaluate_ground_truth:
            sample_cm_dict[sample_name] = sample_cm.copy()
        save_sample_summary(
            sample_name,
            output_dir / "sample_summary" / sample_name,
            sample_pred_count,
            sample_true_count,
            sample_cm,
            evaluate_ground_truth=args.evaluate_ground_truth,
        )

    write_predictions_csv(
        all_rows,
        output_dir / "predictions.csv",
        include_ground_truth=args.evaluate_ground_truth,
    )
    save_sample_summary(
        "overall",
        output_dir / "overall_summary",
        overall_pred_count,
        overall_true_count,
        overall_cm,
        evaluate_ground_truth=args.evaluate_ground_truth,
    )
    if args.evaluate_ground_truth:
        save_group_confusion_matrices(sample_cm_dict, output_dir)

    print("\nDone.")
    print(f"Saved predictions to: {output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
