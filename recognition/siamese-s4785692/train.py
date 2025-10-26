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
            embedding2: Embedding of the first image. [B, D]
            label: 1 if same class, 0 otherwise. [B]
        """
        # Euclidean distance between embeddings
        distance = nn.functional.pairwise_distance(embedding1, embedding2)
        
        # Loss calculation
        loss_pos = label * torch.pow(distance, 2)
        loss_neg = (1 -label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        
        loss = torch.mean(loss_pos + loss_neg)
        return loss
    
def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train for one epoch.
    """
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training")
    for batch_idx, (anchor_img, pair_img, label) in enumerate(pbar):
        anchor_img, pair_img, label = anchor_img.to(device), pair_img.to(device), label.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        embedding1, embedding2 = model(anchor_img, pair_img)
        
        # Loss calculation
        loss = criterion(embedding1, embedding2, label)
        
        # Backward pass
        loss.bacward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({"Loss": running_loss / (batch_idx + 1)})
    
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """
    Validate the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for anchor_img, pair_img, label in tqdm(dataloader, desc="Validation"):
            anchor_img, pair_img, label = anchor_img.to(device), pair_img.to(device), label.to(device)
            
            # Forward pass
            embedding1, embedding2 = model(anchor_img, pair_img)
            
            # Loss calculation
            loss = criterion(embedding1, embedding2, label)
            running_loss += loss.item()
            
    return running_loss / len(dataloader)


def train_classification_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train classification head for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training")
    for batch_idx, (image, label) in enumerate(pbar):
        image, label = image.to(device), label.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(image)
        
        # Loss calculation
        loss = criterion(logits, label)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        probs = torch.sigmoid(logits)
        predictions = (probs >= 0.5).float()
        correct += (predictions == label).sum().item()
        total += label.size(0)
        
        running_loss += loss.item()
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'accuracy': 100.0 * correct / total
        })
        
        return running_loss / len(dataloader), 100.0 * correct / total
    
def plot_loss(train_losses, val_losses, output_dir):
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
    plot_path = os.path.join(output_dir, 'loss_plot.png')
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
            hue=config['augmentation']['hue']
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
        pin_memoery=True
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
    
    # Define loss function
    criterion = ContrastiveLoss(margin=config['training']['margin'])
    
    # Define optimizer
    optimizer = optim.Adam(
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
    
    # Create output directory
    os.makedirs(config['output']['output_dir'], exist_ok=True)
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Siamese Network")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML file")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    main(config)