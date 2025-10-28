import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import argparse
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve,
    balanced_accuracy_score
)
import pandas as pd
import os
import yaml

from dataset import SiameseISICDataset
from modules import SiameseClassificationNetwork


def load_config(config_path):
    """
    Load YAML configuration file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def run_inference(model, dataloader, device):
    model.eval()
    all_probs, all_targets = [], []
    for anchor_img, _, anchor_label, _ in dataloader:
        anchor_img = anchor_img.to(device)
        logits = model(anchor_img)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(anchor_label.numpy().astype(int))
    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    return probs, targets

def youden_optimal_threshold(y_true, y_prob):
    # NOTE: https://en.wikipedia.org/wiki/Youden's_J_statistic
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    j_best_idx = int(np.argmax(j))
    return float(thr[j_best_idx])


def compute_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    rocauc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    prauc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")

    metrics = {
        "threshold": thr,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": rocauc,
        "pr_auc": prauc,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    return metrics

def main(config):
    device = torch.device("cuda" if torch.cuda_is_available() else "cpu")
    
    # Get necessary parameters for evaluation
    eval_cfg = config.get("testing", {})
    checkpoint_path = eval_cfg.get("checkpoint", "best_model.pth")
    outdir = eval_cfg.get("output_dir", "eval_out")
    threshold = float(eval_cfg.get("threshold", 0.5))

    
    # Trasform data to match data used for train and val
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['augmentation']['mean'],
            std=config['augmentation']['std']
        )
    ])
    
    # Create dataset
    test_dataset = SiameseISICDataset(
        image_dir=config['data']['test_image_dir'],
        csv_file=config['data']['test_csv'],
        transform=test_transform,
        train=False
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = SiameseClassificationNetwork(
        embedding_dim=config['model']['embedding_dim'],
        pretrained=config['model']['pretrained']
    )
    model = model.to(device)
    
    # Load trained model
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    
    probs, targets = run_inference(model, test_loader, device)
    
    # Determine cutoff point for logits threshold using Youden J's stat
    # NOTE: https://pmc.ncbi.nlm.nih.gov/articles/PMC2515362/
    best_thr = youden_optimal_threshold(targets, probs)
    
    # Compute performance metrics based on given threshold and computed best threshold
    metrics_mannual = compute_metrics(targets, probs, thr=threshold)
    metrics_best = compute_metrics(targets, probs, thr=best_thr)
    
    df_meta = pd.read_csv(config['data']['test_csv'])
    ids = df_meta['isic_id'].astype(str).values
    pd.DataFrame({
        "isic_id": ids,
        "target": targets.astype(int),
        "prob": probs,
        f"pred@{threshold:.2f}": (probs >= threshold).astype(int),
        f"pred@best({best_thr:.3f})": (probs >= best_thr).astype(int)
    }).to_csv(os.path.join(outdir, "predictions.csv"), index=False)
    
    summary = {
        "n_samples": int(len(targets)),
        "pos_rate": float(targets.mean()) if len(targets) else 0.0,
        "default_threshold": threshold,
        "youden_best_threshold": best_thr,
        **{f"default_{k}": v for k, v in metrics_mannual.items()},
        **{f"best_{k}": v for k, v in metrics_best.items()},
        "checkpoint": os.path.abspath(checkpoint_path)
    }
    pd.DataFrame([summary]).to_csv(os.path.join(outdir, "metrics_summary.csv"), index=False)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Siamese Network")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML file")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    main(config)
