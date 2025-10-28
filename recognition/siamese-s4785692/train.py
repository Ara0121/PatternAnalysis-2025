# -*- coding: utf-8 -*-
"""
train_test_combined.py

Unified script for the Siamese Classification Network on the ISIC dataset.
This script supports both training and evaluation modes, depending on the flag.

Evaluation Mode (--test):
    - Loads a trained model checkpoint and evaluates it on a test set.
    - Reports classification metrics and generates evaluation plots.

    Main components:
    - run_inference: Run model predictions and collect probabilities/targets.
    - youden_optimal_threshold: Determine best threshold using Youden’s statistic.
    - compute_metrics: Precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix.
    - save_roc_curve / save_pr_curve: Plot and save ROC and PR curves.
    - save_confusion_heatmap: Generate confusion matrix heatmaps.

    Outputs:
    - Metrics summary CSV
    - ROC and Precision–Recall curves
    - Confusion matrix heatmaps

Training Mode (default):
    - Integrates both contrastive loss and binary cross-entropy (BCE) loss
      to learn image similarity and perform classification.

    Main components:
    - ContrastiveLoss: Custom implementation of contrastive loss.
    - Training & Validation loops with combined losses.
    - Loss visualization (total, contrastive, BCE).
    - Latent space visualization using t-SNE.

    Outputs:
    - Model checkpoints (best and periodic)
    - Loss plots (total, contrastive, BCE)
    - Latent space visualizations

Usage:
    Training:
        python train_test_combined.py --config path/to/config.yaml
    Testing:
        python train_test_combined.py --config path/to/config.yaml --test

Author: Sohtaroh Arakawa
Created: 2025-10-28
"""

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import argparse
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve,
    balanced_accuracy_score, precision_recall_curve
)
import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import argparse
import numpy as np
from sklearn.manifold import TSNE

from dataset import SiameseISICDataset
from modules import SiameseClassificationNetwork


def load_config(config_path):
    """
    Load YAML configuration file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# -------------------- Training --------------------
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    """
    
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, embedding1, embedding2, label):
        """
        Compute Contrastive loss.
        Args:
            embedding1: Embedding of the first image. [B, D]
            embedding2: Embedding of the second image. [B, D]
            label: 1 if same class, 0 otherwise. [B]
        """
        # Euclidean distance between embeddings
        distance = nn.functional.pairwise_distance(embedding1, embedding2)
        
        # Loss calculation
        loss_pos = label * torch.pow(distance, 2)
        loss_neg = (1 -label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        
        loss = torch.mean(loss_pos + loss_neg)
        return loss
    
def compute_pos_weight(meta_csv):
    """Compute positive class weight for imbalanced dataset."""
    df = pd.read_csv(meta_csv)
    y = df['target'].astype(int).values
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    # avoid division by zero
    return float(n_neg / max(n_pos, 1))

def compute_cls_from_logits(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    correct = (preds == (targets >= 0.5).float()).sum().item()
    total = targets.numel()
    return correct, total, probs

    
def train_epoch(model, dataloader, criterion_cont, criterion_bce, optimizer, device, epoch, alpha=1.0, beta=1.0):
    """
    Train for one epoch. 
    Total loss is calculated by combining contrastive loss and classification loss.

        total_loss = alpha * contrastive_loss + beta * bce_loss

    """
    model.train()
    running_total, running_contrastive, running_bce = 0.0, 0.0, 0.0
    cls_correct, cls_total = 0, 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training")
    for batch_idx, (anchor_img, pair_img, anchor_label, pair_label) in enumerate(pbar):
        anchor_img, pair_img = anchor_img.to(device), pair_img.to(device)
        anchor_label, pair_label = anchor_label.to(device).view(-1).float(), pair_label.to(device).view(-1).float()
        optimizer.zero_grad()
        
        # Contrastive loss
        embedding1, embedding2 = model(anchor_img, pair_img)
        loss_cont = criterion_cont(embedding1, embedding2, pair_label)
        
        # BCE loss
        logits = model.classification_head(embedding1)
        loss_bce = criterion_bce(logits, anchor_label)
        
        # Total loss
        total_loss = alpha * loss_cont + beta * loss_bce
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        correct, total, _ = compute_cls_from_logits(logits, anchor_label)
        cls_correct += correct
        cls_total += total

        running_total += float(total_loss.item())
        running_contrastive += float(loss_cont.item())
        running_bce += float(loss_bce.item())

        step = batch_idx + 1
        if step % 10 == 0:
            pbar.set_postfix(
                total=f"{running_total/step:.4f}",
                contr=f"{running_contrastive/step:.4f}",
                bce=f"{running_bce/step:.4f}",
                cls_acc=f"{100.0*cls_correct/max(cls_total,1):.2f}%"
            )


    avg_total = running_total / max(len(dataloader), 1)
    avg_contr = running_contrastive / max(len(dataloader), 1)
    avg_bce   = running_bce / max(len(dataloader), 1)
    cls_acc   = 100.0 * cls_correct / max(cls_total, 1)
    return avg_total, avg_contr, avg_bce, cls_acc

def validate(model, dataloader, criterion_cont, criterion_bce, device, alpha, beta):
    """
    Validate the model on the validation set.
    """
    model.eval()
    running_total = running_contrastive = running_bce = 0.0
    cls_correct = cls_total = 0
    
    pbar = tqdm(dataloader, desc="Validation")
    with torch.no_grad():
        for batch_idx, (anchor_img, pair_img, anchor_label, pair_label) in enumerate(pbar):
            anchor_img, pair_img = anchor_img.to(device), pair_img.to(device)
            anchor_label, pair_label = anchor_label.to(device).view(-1).float(), pair_label.to(device).view(-1).float()
            
            # Contrastive loss
            embedding1, embedding2 = model(anchor_img, pair_img)
            loss_cont = criterion_cont(embedding1, embedding2, pair_label)
            
            # BCE loss
            logits = model.classification_head(embedding1)
            loss_bce = criterion_bce(logits, anchor_label)
            
            # Total loss
            total_loss = alpha * loss_cont + beta * loss_bce
            
            
            correct, total, _ = compute_cls_from_logits(logits, anchor_label)
            cls_correct += correct
            cls_total += total

            running_total += float(total_loss.item())
            running_contrastive += float(loss_cont.item())
            running_bce += float(loss_bce.item())

            step = batch_idx + 1
            if step % 10 == 0:
                pbar.set_postfix(
                    total=f"{running_total/step:.4f}",
                    contr=f"{running_contrastive/step:.4f}",
                    bce=f"{running_bce/step:.4f}",
                    cls_acc=f"{100.0*cls_correct/max(cls_total,1):.2f}%"
                )

        avg_total = running_total / max(len(dataloader), 1)
        avg_contr = running_contrastive / max(len(dataloader), 1)
        avg_bce   = running_bce / max(len(dataloader), 1)
        cls_acc   = 100.0 * cls_correct / max(cls_total, 1)
        return avg_total, avg_contr, avg_bce, cls_acc

    
def plot_loss(train_losses, val_losses, loss_type, output_dir):
    """
    Plot and save training and validation loss curves.
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = os.path.join(output_dir, f'{loss_type}_loss_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f'Loss plot saved to {plot_path}')
    plt.close()


def extract_embeddings(model, dataloader, device, max_samples=1000):
    """Extract embeddings from the model for visualization."""
    model.eval()
    embedding_list = []
    label_list = []
    
    with torch.no_grad():
        for anchor_img, _, anchor_label, _ in tqdm(dataloader, desc="Extracting Embeddings"):
            anchor_img = anchor_img.to(device)
            
            # Get embeddings
            embeddings = model.get_embedding(anchor_img)
            embedding_list.append(embeddings.cpu().numpy())
            label_list.append(anchor_label.numpy())
            
            # Limit number of samples
            if len(embedding_list) * anchor_img.size(0) >= max_samples:
                break
    
    # Concatenate all embeddings and labels
    embeddings = np.vstack(embedding_list)
    labels = np.concatenate(label_list)
    
    embeddings = embeddings[:max_samples]
    labels = labels[:max_samples]
    
    return embeddings, labels

def visualize_latent_space(embeddings, labels, epoch, output_dir):
    """Visualize the latent space using t-SNE"""
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    # Transform embeddings to 2D space
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    colors = ['#FF6B6B', '#4ECDC4']
    markers = ['o', 's']
    
    # Plot each class with different colour and marker
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=colors[int(label)],
            marker=markers[int(label)],
            label=f'Class {int(label)}',
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
    
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.title(f'Latent Space Visualization (Epoch {epoch})', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'latent_space_epoch_{epoch}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f'Latent space visualization saved to {plot_path}')
    plt.close()


def train_main(config):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # NOTE: https://github.com/fastai/fastai2/blob/master/nbs/09_vision.augment.ipynb
    # Rotation may need to be 180 to avoid cutting corners
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(config['augmentation']['rotation']),
        transforms.ColorJitter(
            brightness=config['augmentation']['brightness'],
            contrast=config['augmentation']['contrast'],
            saturation=config['augmentation']['saturation'],
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['augmentation']['mean'],
            std=config['augmentation']['std']
        )
    ])
    
    # No special augmentation for validation set
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['augmentation']['mean'],
            std=config['augmentation']['std']
        )
    ])
    
    # Create datasets
    train_dataset = SiameseISICDataset(
        image_dir=config['data']['train_image_dir'],
        csv_file=config['data']['train_csv'],
        transform=train_transform,
        train=True
    )
    
    val_dataset = SiameseISICDataset(
        image_dir=config['data']['val_image_dir'],
        csv_file=config['data']['val_csv'],
        transform=val_transform,
        train=False
    )
    
    # Create dataloader for training
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # Create dataloader for validation, thus no shuffle
    val_loader = DataLoader(
        val_dataset,
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
    
    # Calculate positive weights for imabalnced data
    pos_weight_value = compute_pos_weight(config['data']['train_csv'])
    
    # Define loss function
    criterion_cont = ContrastiveLoss(margin=config['training']['margin'])
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))

    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['training']['scheduler_factor'],
        patience=config['training']['scheduler_patience'],
    )
    
    # Loss weights
    alpha = float(config['training'].get('alpha', 1.0))  # contrastive weight
    beta  = float(config['training'].get('beta',  1.0))  # BCE weight
    
    # Create output directory
    os.makedirs(config['output']['output_dir'], exist_ok=True)
    
    # Save config to output directory
    config_save_path = os.path.join(config['output']['output_dir'], 'config.yaml')
    with  open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config saved to {config_save_path}")
    
    # Store losses for plotting
    best_val_loss = float('inf')
    train_total_losses = []
    val_total_losses = []
    train_cont_losses = []
    val_cont_losses = []
    train_bce_losses = []
    val_bce_losses = []
    
    # Main training loop
    for epoch in range(1, config['training']['epochs'] + 1):
        train_total, train_cont, train_bce, train_acc = train_epoch(model, train_loader, criterion_cont, criterion_bce, optimizer, device, epoch, alpha, beta)
        val_total, val_cont, val_bce, val_acc = validate(model, val_loader, criterion_cont, criterion_bce, device, alpha, beta)
        
        # Store losses
        train_total_losses.append(train_total)
        val_total_losses.append(val_total)
        train_cont_losses.append(train_cont)
        val_cont_losses.append(val_cont)
        train_bce_losses.append(train_bce)
        val_bce_losses.append(val_bce)
        
        print(f"Epoch {epoch}/{config['training']['epochs']} - Train Loss: {train_total:.4f}, Val Loss: {val_total:.4f}")
        
        # Step scheduler
        scheduler.step(val_total)
        
        # Save best model
        if val_total < best_val_loss:
            best_val_loss = val_total
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_total,
                'val_loss': val_total,
                'config': config
            }, os.path.join(config['output']['output_dir'], 'best_model.pth'))
            print(f"Saved best model (val_loss: {val_total:.4f})")
            
        # Save checkpoint every n epochs
        if epoch % config['output']['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_total,
                'val_loss': val_total,
                'config': config
            }, os.path.join(config['output']['output_dir'], f'checkpoint_epoch_{epoch}.pth'))
            
        # Plot loss curves
        if epoch % config['output']['plot_interval'] == 0 or epoch == config['training']['epochs']:
            plot_loss(train_total_losses, val_total_losses, 'total', config['output']['output_dir'])
            plot_loss(train_cont_losses, val_cont_losses, 'contrastive', config['output']['output_dir'])
            plot_loss(train_bce_losses, val_bce_losses, 'bce', config['output']['output_dir'])
            
        # Visualise latent space
        if epoch % config['visualization']['visualize_interval'] == 0 or epoch == config['training']['epochs']:
            embeddings, labels = extract_embeddings(model, val_loader, device, max_samples=config['visualization']['max_samples'])
            visualize_latent_space(embeddings, labels, epoch, config['output']['output_dir'])
        
    print("Training completed.")
    

# -------------------- Testing --------------------
def run_inference(model, dataloader, device):
    """Run prediction on a given dataloader."""
    model.eval()
    all_probs, all_targets = [], []
    # Loop through test dataset and predict
    for anchor_img, _, anchor_label, _ in dataloader:
        anchor_img = anchor_img.to(device)
        logits = model(anchor_img)
        # Apply sigmoid to obtain probabilities
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_targets.append(anchor_label.numpy().astype(int))
    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    return probs, targets

def youden_optimal_threshold(y_true, y_prob):
    """Determine the best logit threshold based on Youden's statistic."""
    # NOTE: https://en.wikipedia.org/wiki/Youden's_J_statistic
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    # 
    j_best_idx = int(np.argmax(j))
    return float(thr[j_best_idx])


def compute_metrics(y_true, y_prob, thr=0.5):
    """Compute the evaluation metrics based on a given threhold."""
    # Create predictions based on threhold
    y_pred = (y_prob >= thr).astype(int)
    
    # Compute metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    rocauc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    prauc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")

    # Store results as a dictionary
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

def save_roc_curve(y_true, y_prob, out_png, title="ROC Curve"):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", lw=1)
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def save_pr_curve(y_true, y_prob, out_png, title="Precision–Recall Curve"):
    """Plot and save PR curve."""
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(recalls, precisions, lw=2, label=f"AP = {ap:.3f}")
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(title); plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    
def save_confusion_heatmap(cm_counts, out_png, title="Confusion Matrix"):
    """Plot and save heat map with count and percentage."""
    mat = np.array(cm_counts, dtype=int)
    row_sum = mat.sum(axis=1, keepdims=True).clip(min=1)
    # Calculate the percentage rates as well
    pct = (mat / row_sum) * 100.0

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"],
           xlabel="Predicted label", ylabel="True label", title=title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]:,}\n({pct[i, j]:.1f}%)",
                    ha="center", va="center",
                    color="white" if mat[i, j] > mat.max()/2 else "black",
                    fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def test_main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get necessary parameters for evaluation
    eval_cfg = config.get("testing", {})
    checkpoint_path = eval_cfg.get("checkpoint", "best_model.pth")
    outdir = os.path.abspath(os.path.expanduser(eval_cfg.get("output_dir", "eval_out")))
    os.makedirs(outdir, exist_ok=True)
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
    
    # Create a summary of evaluation
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
    
    # Plots
    save_roc_curve(targets, probs, os.path.join(outdir, "roc_curve.png"))
    save_pr_curve(targets, probs, os.path.join(outdir, "pr_curve.png"))

    # Create confusion matrix
    cm_mannual = np.array([[metrics_mannual["tn"], metrics_mannual["fp"]], 
                           [metrics_mannual["fn"], metrics_mannual["tp"]]])
    cm_best = np.array([[metrics_best["tn"], metrics_best["fp"]], 
                        [metrics_best["fn"], metrics_best["tp"]]])
    
    save_confusion_heatmap(cm_mannual, os.path.join(outdir, f"confusion_matrix_thr_{threshold:.2f}.png"),
                           title=f"Confusion Matrix (thr={threshold:.2f})")
    save_confusion_heatmap(cm_best, os.path.join(outdir, f"confusion_matrix_thr_best_{best_thr:.3f}.png"),
                           title=f"Confusion Matrix (thr={best_thr:.3f})")
    
    print(f"\n Finsished testing and evaluation. The results were saved in {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Test Siamese Network")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML file")
    parser.add_argument('--test', action='store_true', help="If set, run testing. Otherwise, training.")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Do testing or training depending on a given argument
    if args.test:
        test_main(config)
    else:
        train_main(config)
