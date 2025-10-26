import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm


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
    
