# -*- coding: utf-8 -*-
"""
utils.py

Helper functions for the Siamese Classification Network.
Includes:
- YAML config loading
- Loss curve plotting
- Latent space visualization (t-SNE)
- ROC / PR curve plotting
- Confusion matrix heatmap

Author: Sohtaroh Arakawa
Created: 2025-10-29
"""

import yaml
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    roc_curve, precision_recall_curve
)
import numpy as np
import os

def load_config(config_path):
    """
    Load YAML configuration file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

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