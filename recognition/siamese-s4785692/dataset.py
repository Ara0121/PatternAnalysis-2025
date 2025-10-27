import torch
from torch.utils.data import Dataset

import pandas as pd
import random
from PIL import Image
import os

class SiameseISICDataset(Dataset):
    """Custom Dataset for Siamese Network on ISIC dataset."""
    
    def __init__(self, image_dir, csv_file, transform=None, train=True):
        """
        Args:
            image_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            train (bool): Indicates if the dataset is for training or testing.
        """
        
        self.image_dir = image_dir
        self.transform = transform
        self.train = train
        
        # Read meta data from CSV file
        self.df = pd.read_csv(csv_file)
        
        self.labels = self.df['target'].values
        self.image_names = self.df['image_name'].values
        
        # Create a dictionary to hold indices for each class
        self.label_to_indices = {0: [], 1: []}
        for i, label in enumerate(self.labels):
            self.label_to_indices[label].append(i)
        
        
    def __len__(self):
        return len(self.df)
    
    
    def __getitem__(self, index):
        # Get the anchor image
        anchor_name = self.image_names[index]
        anchor_label = self.labels[index]
        anchor_img = self._load_image(anchor_name)
        
        # Randomly decide whether to get a positive or negative pair
        get_same_class = random.random() > 0.5
        
        if get_same_class:
            # Positive pair (same class)
            
            same_class_indices = self.label_to_indices[anchor_label].copy()
            if index in same_class_indices:
                same_class_indices.remove(index)
                
            if len(same_class_indices) > 0:
                pair_index = random.choice(same_class_indices)
                label = torch.tensor(1.0, dtype=torch.float32)
            else:
                # Fallback to negative pair if no other same class image exists
                diff_label = 1 - anchor_label
                diff_pool = self.label_to_indices.get(diff_label, [])
                if not diff_pool:
                    raise ValueError("No negative examples available.")
                pair_index = random.choice(diff_pool)
                label = torch.tensor(0.0, dtype=torch.float32)
            
        else:
            # Negative pair (different class)
            different_label = 1 - anchor_label
            diff_pool = self.label_to_indices.get(different_label, [])
            if not diff_pool:
                raise RuntimeError("No negative examples available.")
            pair_index = random.choice(diff_pool)
            label = torch.tensor(0.0, dtype=torch.float32)
            
        pair_name = self.image_names[pair_index]
        pair_img = self._load_image(pair_name)
        
        if self.transform:
            anchor_img = self.transform(anchor_img)
            pair_img = self.transform(pair_img)
            
        return anchor_img, pair_img, label
    
    def _load_image(self, image_name):
        """Load an image from the disk."""
        # Try multiple file extensions
        for ext in ['.jpg', '.png']:
            image_path = os.path.join(self.image_dir, image_name + ext)
            if os.path.exists(image_path):
                return Image.open(image_path).convert('RGB')