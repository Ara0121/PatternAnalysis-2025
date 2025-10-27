import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class EmbeddingNetwork(nn.Module):
    """
    Embedding network that extracts features from input images.
    Uses a pre-trained Vit-B/16 as backbone.
    """
    
    def __init__(self, embedding_dim=256, pretrained=True):
        super(EmbeddingNetwork, self).__init__()
        
        # Load pre-trained Vit-B/16 model
        # NOTE: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        
        self.backbone = vit
        # Remove the classification head
        self.backbone.heads = nn.Identity()
        
        # NOTE: Vit-B/16 has a hidden dimension of 768
        # Ref: https://arxiv.org/abs/2303.08216
        self.feature_dim = vit.hidden_dim
        
        # Embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        # Extract features using Vit-B/16 backbone
        features = self.backbone(x) # [B, 768]
        # Get embeddings
        embeddings = self.embedding_head(features) # [B, embedding_dim]
        # L2 normalize the embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
class SiameseNetwork(nn.Module):
    """
    Siamese Network for similarlity learning.
    """
    def __init__(self, embedding_dim=256, pretrained=True):
        super(SiameseNetwork, self).__init__()
        
        self.embedding_net = EmbeddingNetwork(
            embedding_dim=embedding_dim,
            pretrained=pretrained
        )
        
    def forward(self, x1, x2=None):
        """
        Foward pass through the Siamese Network,
        Args:
            x1: First input image.
            x2: Second input image (optional).
        """
        
        embedding1 = self.embedding_net(x1)
        
        if x2 is not None:
            embedding2 = self.embedding_net(x2)
            return embedding1, embedding2
        
        return embedding1
    
    def get_embedding(self, x):
        """Get embedding for a single input image."""
        return self.embedding_net(x)
    
class ClassificationHead(nn.Module):
    """
    Classification head for binary classification.
    """
    def __init__(self, embedding_dim=256):
        super(ClassificationHead, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128, bias=False),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        return self.classifier(x).squeeze(1)
    
class SiameseClassificationNetwork(nn.Module):
    """
    Siamese Network with classification head.
    """
    def __init__(self, embedding_dim=256, pretrained=True):
        super(SiameseClassificationNetwork, self).__init__()
        
        self.siamese_net = SiameseNetwork(
            embedding_dim=embedding_dim,
            pretrained=pretrained
        )
        
        self.classification_head = ClassificationHead(
            embedding_dim=embedding_dim
        )
        
    def forward(self, x1, x2=None):
        """
        Forward pass through the Siamese Classification Network.
        Args:
            x1: First input image.
            x2: Second input image (optional).
        """
        
        if x2 is not None:
            # Return embeddings
            return self.siamese_net(x1, x2)
        else:
            # Return classification score during inference
            embedding = self.siamese_net(x1)
            logits = self.classification_head(embedding)
            return logits
        
    def get_embedding(self, x):
        """Get embedding for a single input image."""
        return self.siamese_net.get_embedding(x)
    
    def predict(self, x):
        """Get classification score with softmax."""
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        predictions = (probs >= 0.5).float()
        
        return predictions, probs
    