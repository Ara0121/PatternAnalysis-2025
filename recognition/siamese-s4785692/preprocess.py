import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil

def create_directories(base_dir):
    """Create directory structure for train/val/test splits."""
    splits = ['train', 'val', 'test']
    
    for split in splits:
        # Create image directory
        image_dir = os.path.join(base_dir, split, 'images')
        os.makedirs(image_dir, exist_ok=True)
        
    print(f"Directory strructure created under {base_dir}")
    return splits

def split_data(df, train_ratio=0.7, val_ratio=0.15, random_seed=42, stratify=True):
    """
    Split data ino train, val, test sets.
    Args:
        df: DataFrame containing image metadata and labels.
        train_ratio: Proportion of data for training set.
        val_ratio: Proportion of data for validation set.
        random_seed: Seed for reproducibility.
        stratify: Whether to stratify splits based on labels.
    """
    
    # Get stratify column if needed
    stratify_col = df['target'] if stratify and 'target' in df.columns else None
    
    # First split into train+val and test
    train_val_df, test_df = train_test_split(
        df,
        test_size = 1 - (train_ratio + val_ratio),
        random_state=random_seed,
        stratify=stratify_col
    )
    
    # Second split into train and val
    stratify_col_train_val = train_val_df['target'] if stratify and 'target' in train_val_df.columns else None
    train_df, val_df = train_test_split(
        train_val_df,
        test_size = val_ratio / (train_ratio + val_ratio),
        random_state=random_seed,
        stratify=stratify_col_train_val
    )
    
    return train_df, val_df, test_df

def copy_images(df, src_image_dir, dest_image_dir, image_extension='.jpg'):
    """Copy images from source to destination directoy"""
    print(f"Copying {len(df)} images to {dest_image_dir}")
    
    copied_count = 0
    missing_count = 0
    
    for image_name in tqdm(df['image_name'].values):
        src_path = os.path.join(src_image_dir, image_name + image_extension)
        dest_path = os.path.join(dest_image_dir, image_name + image_extension)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            copied_count += 1
        else:
            missing_count += 1
            print(f"Warning: {src_path} not found.")
        
    print(f"Copied {copied_count} images. {missing_count} images were missing.")
    
