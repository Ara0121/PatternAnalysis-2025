import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import yaml
import argparse
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd

from dataset import SiameseISICDataset
from modules import SiameseClassificationNetwork

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

def load_config(config_path):
    """
    Load YAML configuration file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

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
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
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


def main(config):
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
        transforms.RandomRotation(config['augmentation']['rotation_degree']),
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
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
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
        verbose=True
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
    
    # Training loop
    best_val_loss = float('inf')
    train_total_losses = []
    val_total_losses = []
    train_cont_losses = []
    val_cont_losses = []
    train_bce_losses = []
    val_bce_losses = []
    
    
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
        if epoch % config['visualization']['visualization_interval'] == 0 or epoch == config['training']['epochs']:
            embeddings, labels = extract_embeddings(model, val_loader, device, max_samples=config['visualization']['max_samples'])
            visualize_latent_space(embeddings, labels, epoch, config['output']['output_dir'])
        
    print("Training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Siamese Network")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML file")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    main(config)