import torch
import torch.nn as nn

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
    

    