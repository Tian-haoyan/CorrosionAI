import argparse
import os
import csv
import time
import random
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt


# Configuration defaults. These can be overridden from the command line.
DATA_ROOT = "data/dataset"
OUT_DIR = "outputs/model_development"
SEED = 42

NUM_EPOCHS = 100
LR = 1e-4
BATCH_SIZE = 32
IMG_SIZE = 256
NUM_WORKERS = 4
EARLY_STOP_PATIENCE = 15

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TEST_RATIO = 0.2
VAL_RATIO_IN_TRAINVAL = 0.2  # final train/val/test split is approximately 64%/16%/20%


# =========================
# =========================
def set_seed(seed=42):
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Seed each DataLoader worker for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_ckpt(path, epoch, model, optimizer, best, history):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best": best,
        "history": history
    }, path)


def save_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_csv(path: str, rows, header=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if header is not None:
            writer.writerow(header)
        writer.writerows(rows)


def save_dicts_to_csv(path: str, rows: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    headers = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix_png(cm: np.ndarray, class_names, save_path: str, title: str = "Confusion Matrix"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig = plt.figure(figsize=(8, 6), dpi=200)
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    thresh = cm.max() * 0.6 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j, i, f"{val}",
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=8
            )

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_csv(cm: np.ndarray, class_names, save_path: str):
    rows = []
    header = ["True\\Pred"] + list(class_names)
    for i, cname in enumerate(class_names):
        rows.append([cname] + cm[i].tolist())
    save_csv(save_path, rows, header=header)


def save_learning_curves(train_loss, val_loss, train_acc, val_acc, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(train_loss) + 1)
    fig = plt.figure(figsize=(12, 4), dpi=200)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(epochs, train_loss, marker='o', markersize=2, linewidth=1, label='Train loss')
    ax1.plot(epochs, val_loss, marker='s', markersize=2, linewidth=1, label='Val loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(epochs, train_acc, marker='o', markersize=2, linewidth=1, label='Train acc')
    ax2.plot(epochs, val_acc, marker='s', markersize=2, linewidth=1, label='Val acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def split_indices_by_seed(targets, seed):
    """Create one stratified train/validation/test split."""
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=TEST_RATIO, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(np.zeros(len(targets)), targets))

    trainval_targets = targets[trainval_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO_IN_TRAINVAL, random_state=seed)
    tr_rel, va_rel = next(sss2.split(np.zeros(len(trainval_targets)), trainval_targets))

    train_idx = trainval_idx[tr_rel]
    val_idx = trainval_idx[va_rel]
    return train_idx, val_idx, test_idx


def build_transforms():
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=(0, 360)),
        transforms.RandomAdjustSharpness(sharpness_factor=0.2),
        transforms.RandomEqualize(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomEqualize(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    return train_tf, eval_tf


def build_loaders(data_root, train_idx, val_idx, test_idx, seed):
    """Build training, train-evaluation, validation, and test loaders."""
    train_tf, eval_tf = build_transforms()

    ds_train_tf = datasets.ImageFolder(root=data_root, transform=train_tf)
    ds_eval_tf = datasets.ImageFolder(root=data_root, transform=eval_tf)

    train_set = Subset(ds_train_tf, train_idx)
    train_eval_set = Subset(ds_eval_tf, train_idx)
    val_set = Subset(ds_eval_tf, val_idx)
    test_set = Subset(ds_eval_tf, test_idx)

    g = torch.Generator()
    g.manual_seed(seed)

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=g
    )

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    train_eval_loader = DataLoader(train_eval_set, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    return train_loader, train_eval_loader, val_loader, test_loader


def build_model(num_classes):
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    return model.to(DEVICE)


@torch.no_grad()
def evaluate(model, loader, criterion, class_names):
    model.eval()
    losses = []
    all_preds, all_labels = [], []

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.long().to(DEVICE)

        logits = model(x)
        loss = criterion(logits, y)

        losses.append(loss.item() * x.size(0))
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    num_classes = len(class_names)
    all_preds = np.concatenate(all_preds) if len(all_preds) else np.array([])
    all_labels = np.concatenate(all_labels) if len(all_labels) else np.array([])

    avg_loss = float(np.sum(losses) / len(loader.dataset)) if len(loader.dataset) else 0.0

    if len(all_labels):
        report_dict = classification_report(
            all_labels,
            all_preds,
            labels=list(range(num_classes)),
            target_names=class_names,
            digits=4,
            zero_division=0,
            output_dict=True
        )
        report_str = classification_report(
            all_labels,
            all_preds,
            labels=list(range(num_classes)),
            target_names=class_names,
            digits=4,
            zero_division=0
        )
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
        acc = accuracy_score(all_labels, all_preds)
        precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    else:
        report_dict = {}
        report_str = ""
        cm = np.zeros((num_classes, num_classes), dtype=int)
        acc = precision_macro = recall_macro = f1_macro = 0.0

    metrics = {
        "acc": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "cm": cm,
        "report_dict": report_dict,
        "report_str": report_str
    }
    return avg_loss, metrics


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    losses = []
    all_preds, all_labels = [], []

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.long().to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item() * x.size(0))
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    avg_loss = float(np.sum(losses) / len(loader.dataset))

    metrics = {
        "acc": accuracy_score(all_labels, all_preds),
        "precision_macro": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall_macro": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }
    return avg_loss, metrics


def save_epoch_history_csv(history_rows, save_path):
    save_dicts_to_csv(save_path, history_rows)


def save_classification_report_csv(report_dict, class_names, save_path):
    rows = []
    for cname in class_names:
        item = report_dict.get(cname, {})
        rows.append({
            "class": cname,
            "precision": item.get("precision", 0.0),
            "recall": item.get("recall", 0.0),
            "f1-score": item.get("f1-score", 0.0),
            "support": item.get("support", 0),
        })

    for avg_name in ["macro avg", "weighted avg"]:
        item = report_dict.get(avg_name, {})
        rows.append({
            "class": avg_name,
            "precision": item.get("precision", 0.0),
            "recall": item.get("recall", 0.0),
            "f1-score": item.get("f1-score", 0.0),
            "support": item.get("support", 0),
        })

    save_dicts_to_csv(save_path, rows)


def save_split_outputs(split_name, loss_value, metrics, class_names, run_dir):
    split_name = split_name.lower()

    save_confusion_matrix_png(
        metrics["cm"],
        class_names,
        os.path.join(run_dir, f"{split_name}_confusion_matrix.png"),
        title=f"{split_name.capitalize()} Confusion Matrix"
    )
    save_confusion_matrix_csv(
        metrics["cm"],
        class_names,
        os.path.join(run_dir, f"{split_name}_confusion_matrix.csv")
    )

    save_text(
        os.path.join(run_dir, f"{split_name}_classification_report.txt"),
        metrics["report_str"]
    )
    save_classification_report_csv(
        metrics["report_dict"],
        class_names,
        os.path.join(run_dir, f"{split_name}_classification_report.csv")
    )

    summary_text = (
        f"{split_name.upper()} loss: {loss_value:.6f}\n"
        f"{split_name.upper()} acc: {metrics['acc']:.6f}\n"
        f"{split_name.upper()} precision_macro: {metrics['precision_macro']:.6f}\n"
        f"{split_name.upper()} recall_macro: {metrics['recall_macro']:.6f}\n"
        f"{split_name.upper()} f1_macro: {metrics['f1_macro']:.6f}\n"
    )
    save_text(os.path.join(run_dir, f"{split_name}_metrics.txt"), summary_text)


def run_once(seed):
    run_dir = OUT_DIR
    os.makedirs(run_dir, exist_ok=True)

    set_seed(seed)

    print("=" * 80)
    print(f"Single split training | seed={seed}")
    print("Loading full dataset:", DATA_ROOT)

    full_plain = datasets.ImageFolder(root=DATA_ROOT, transform=None)
    classes = full_plain.classes
    targets = np.array(full_plain.targets)
    num_classes = len(classes)

    print("Classes:", classes)
    print("Total images:", len(full_plain))

    train_idx, val_idx, test_idx = split_indices_by_seed(targets, seed)
    print(f"Final split -> train/val/test = {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    np.savez(
        os.path.join(run_dir, "split_indices.npz"),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx
    )
    save_text(
        os.path.join(run_dir, "split_info.txt"),
        f"seed={seed}\n"
        f"train_size={len(train_idx)}\n"
        f"val_size={len(val_idx)}\n"
        f"test_size={len(test_idx)}\n"
        f"classes={classes}\n"
    )

    train_loader, train_eval_loader, val_loader, test_loader = build_loaders(
        DATA_ROOT, train_idx, val_idx, test_idx, seed
    )

    print("Loading pretrained ResNet50 (ImageNet)")
    model = build_model(num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        verbose=True
    )

    best = {
        "best_loss": float("inf"),
        "best_acc": 0.0,
        "best_epoch_loss": -1,
        "best_epoch_acc": -1
    }
    history = {"train": [], "val": []}
    history_rows = []

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    early_stop_counter = 0

    path_latest = os.path.join(run_dir, "latest.pth")
    path_best_loss = os.path.join(run_dir, "best_loss.pth")
    path_best_acc = os.path.join(run_dir, "best_acc.pth")

    # =========================
    # =========================
    for epoch in range(NUM_EPOCHS):
        t0 = time.time()

        train_loss, train_m = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_m = evaluate(model, val_loader, criterion, classes)

        scheduler.step(val_loss)

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_acc_hist.append(train_m["acc"])
        val_acc_hist.append(val_m["acc"])

        history["train"].append({
            "epoch": epoch + 1,
            "loss": train_loss,
            **train_m
        })
        history["val"].append({
            "epoch": epoch + 1,
            "loss": val_loss,
            "acc": val_m["acc"],
            "precision_macro": val_m["precision_macro"],
            "recall_macro": val_m["recall_macro"],
            "f1_macro": val_m["f1_macro"]
        })

        history_rows.append({
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_acc": train_m["acc"],
            "train_precision_macro": train_m["precision_macro"],
            "train_recall_macro": train_m["recall_macro"],
            "train_f1_macro": train_m["f1_macro"],
            "val_loss": val_loss,
            "val_acc": val_m["acc"],
            "val_precision_macro": val_m["precision_macro"],
            "val_recall_macro": val_m["recall_macro"],
            "val_f1_macro": val_m["f1_macro"],
        })

        dt = time.time() - t0
        print(
            f"Epoch {epoch + 1:03d}/{NUM_EPOCHS} | "
            f"time={dt:.1f}s | lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"Train acc={train_m['acc']:.4f} | Val acc={val_m['acc']:.4f} | "
            f"Val loss={val_loss:.4f}"
        )

        save_ckpt(path_latest, epoch + 1, model, optimizer, best, history)

        if val_loss < best["best_loss"]:
            best["best_loss"] = val_loss
            best["best_epoch_loss"] = epoch + 1
            early_stop_counter = 0
            save_ckpt(path_best_loss, epoch + 1, model, optimizer, best, history)
        else:
            early_stop_counter += 1

        if val_m["acc"] > best["best_acc"]:
            best["best_acc"] = val_m["acc"]
            best["best_epoch_acc"] = epoch + 1
            save_ckpt(path_best_acc, epoch + 1, model, optimizer, best, history)

        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    save_learning_curves(
        train_loss_hist,
        val_loss_hist,
        train_acc_hist,
        val_acc_hist,
        os.path.join(run_dir, "learning_curves.png")
    )
    save_epoch_history_csv(
        history_rows,
        os.path.join(run_dir, "epoch_history.csv")
    )

    # =========================
    # =========================
    ckpt = torch.load(path_best_loss, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    train_loss_final, train_m_final = evaluate(model, train_eval_loader, criterion, classes)
    val_loss_final, val_m_final = evaluate(model, val_loader, criterion, classes)
    test_loss_final, test_m_final = evaluate(model, test_loader, criterion, classes)

    save_split_outputs("train", train_loss_final, train_m_final, classes, run_dir)
    save_split_outputs("val", val_loss_final, val_m_final, classes, run_dir)
    save_split_outputs("test", test_loss_final, test_m_final, classes, run_dir)

    run_summary = (
        "===== Single Split Training Finished =====\n"
        f"seed: {seed}\n"
        f"train_size: {len(train_idx)}\n"
        f"val_size  : {len(val_idx)}\n"
        f"test_size : {len(test_idx)}\n\n"
        f"best_epoch_by_val_loss: {best['best_epoch_loss']}\n"
        f"best_epoch_by_val_acc : {best['best_epoch_acc']}\n"
        f"best_val_loss         : {best['best_loss']:.6f}\n"
        f"best_val_acc          : {best['best_acc']:.6f}\n\n"
        f"TRAIN loss: {train_loss_final:.6f}\n"
        f"TRAIN acc : {train_m_final['acc']:.6f}\n"
        f"TRAIN precision_macro: {train_m_final['precision_macro']:.6f}\n"
        f"TRAIN recall_macro   : {train_m_final['recall_macro']:.6f}\n"
        f"TRAIN f1_macro       : {train_m_final['f1_macro']:.6f}\n\n"
        f"VAL loss: {val_loss_final:.6f}\n"
        f"VAL acc : {val_m_final['acc']:.6f}\n"
        f"VAL precision_macro: {val_m_final['precision_macro']:.6f}\n"
        f"VAL recall_macro   : {val_m_final['recall_macro']:.6f}\n"
        f"VAL f1_macro       : {val_m_final['f1_macro']:.6f}\n\n"
        f"TEST loss: {test_loss_final:.6f}\n"
        f"TEST acc : {test_m_final['acc']:.6f}\n"
        f"TEST precision_macro: {test_m_final['precision_macro']:.6f}\n"
        f"TEST recall_macro   : {test_m_final['recall_macro']:.6f}\n"
        f"TEST f1_macro       : {test_m_final['f1_macro']:.6f}\n"
    )
    print(run_summary)
    save_text(os.path.join(run_dir, "training_summary.txt"), run_summary)

    result_row = {
        "seed": seed,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "stopped_epoch": len(history_rows),
        "best_epoch_by_val_loss": best["best_epoch_loss"],
        "best_epoch_by_val_acc": best["best_epoch_acc"],
        "best_val_loss": best["best_loss"],
        "best_val_acc": best["best_acc"],

        "train_loss_final": train_loss_final,
        "train_acc_final": train_m_final["acc"],
        "train_precision_macro_final": train_m_final["precision_macro"],
        "train_recall_macro_final": train_m_final["recall_macro"],
        "train_f1_macro_final": train_m_final["f1_macro"],

        "val_loss_final": val_loss_final,
        "val_acc_final": val_m_final["acc"],
        "val_precision_macro_final": val_m_final["precision_macro"],
        "val_recall_macro_final": val_m_final["recall_macro"],
        "val_f1_macro_final": val_m_final["f1_macro"],

        "test_loss_final": test_loss_final,
        "test_acc_final": test_m_final["acc"],
        "test_precision_macro_final": test_m_final["precision_macro"],
        "test_recall_macro_final": test_m_final["recall_macro"],
        "test_f1_macro_final": test_m_final["f1_macro"],

        "best_loss_ckpt_path": path_best_loss,
        "best_acc_ckpt_path": path_best_acc
    }
    return result_row


def parse_args():
    parser = argparse.ArgumentParser(description="Train CorrosionAI with one stratified train/validation/test split.")
    parser.add_argument("--data-root", default=DATA_ROOT, help="Dataset root in ImageFolder format.")
    parser.add_argument("--out-dir", default=OUT_DIR, help="Output directory for checkpoints and reports.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for the stratified split and training.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="Number of DataLoader workers.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Training device.")
    return parser.parse_args()


def configure_from_args(args):
    global DATA_ROOT, OUT_DIR, SEED, NUM_EPOCHS, BATCH_SIZE, NUM_WORKERS, DEVICE
    DATA_ROOT = args.data_root
    OUT_DIR = args.out_dir
    SEED = args.seed
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers

    if args.device == "auto":
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        DEVICE = torch.device("cuda:0")
    else:
        DEVICE = torch.device("cpu")


def main():
    args = parse_args()
    configure_from_args(args)
    os.makedirs(OUT_DIR, exist_ok=True)

    result_row = run_once(SEED)
    save_dicts_to_csv(os.path.join(OUT_DIR, "single_split_summary.csv"), [result_row])

    summary_text = (
        "===== Single Split Model Development Summary =====\n"
        f"seed: {result_row['seed']}\n"
        f"best_val_loss: {result_row['best_val_loss']:.6f}\n"
        f"best_val_acc : {result_row['best_val_acc']:.6f}\n"
        f"best_epoch_by_val_loss: {result_row['best_epoch_by_val_loss']}\n"
        f"best_epoch_by_val_acc : {result_row['best_epoch_by_val_acc']}\n\n"
        f"test_acc: {result_row['test_acc_final']:.6f}\n"
        f"test_precision_macro: {result_row['test_precision_macro_final']:.6f}\n"
        f"test_recall_macro   : {result_row['test_recall_macro_final']:.6f}\n"
        f"test_f1_macro       : {result_row['test_f1_macro_final']:.6f}\n\n"
        f"best_loss checkpoint: {result_row['best_loss_ckpt_path']}\n"
        f"best_acc checkpoint : {result_row['best_acc_ckpt_path']}\n"
    )
    print(summary_text)
    save_text(os.path.join(OUT_DIR, "single_split_summary.txt"), summary_text)


if __name__ == "__main__":
    main()
