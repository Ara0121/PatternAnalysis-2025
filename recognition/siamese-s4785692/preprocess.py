import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil
import yaml
import argparse

def create_directories(base_dir):
    """Create directory structure for train/val/test splits."""
    splits = ['train', 'val', 'test']
    
    for split in splits:
        # Create image directory
        image_dir = os.path.join(base_dir, split, 'images')
        os.makedirs(image_dir, exist_ok=True)
        
    print(f"\nDirectory strructure created under {base_dir}")
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
    # Check if ratios are valid
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("train_ratio and val_ratio must be in (0,1) and sum to < 1.")
    
    # Get stratify column if needed
    stratify_col = df['target'] if stratify and 'target' in df.columns else None
    
    # Try stratified split, fallback to non-stratified if fails (if class is extremely imbalanced)
    try:
        # First split into train+val and test
        train_val_df, test_df = train_test_split(
            df,
            test_size=1 - (train_ratio + val_ratio),
            random_state=random_seed,
            stratify=stratify_col
        )
        stratify_col_train_val = train_val_df['target'] if stratify and 'target' in train_val_df.columns else None
        # Second split into train and val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio / (train_ratio + val_ratio),
            random_state=random_seed,
            stratify=stratify_col_train_val
        )
    except ValueError as e:
        print(f"\nStratified split failed ({e}). Falling back to non-stratified.")
        train_val_df, test_df = train_test_split(
            df,
            test_size=1 - (train_ratio + val_ratio),
            random_state=random_seed,
            stratify=None
        )
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio / (train_ratio + val_ratio),
            random_state=random_seed,
            stratify=None
        )
    return train_df, val_df, test_df

def copy_images(df, src_image_dir, dest_image_dir, image_extension='.jpg'):
    """Copy images from source to destination directoy"""
    print(f"Copying {len(df)} images to {dest_image_dir}")
    
    copied_count = 0
    missing_count = 0
    
    for image_name in tqdm(df['isic_id'].values):
        src_path = os.path.join(src_image_dir, image_name + image_extension)
        dest_path = os.path.join(dest_image_dir, image_name + image_extension)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            copied_count += 1
        else:
            missing_count += 1
            print(f"Warning: {src_path} not found.")
        
    print(f"\nCopied {copied_count} images. {missing_count} images were missing.")
    
def print_split_stats(train_df, val_df, test_df):
    """Print statistics of each data split."""
    def get_stats(df, split_name):
        total = len(df)
        pos = df['target'].sum()
        neg = total - pos
        pos_ratio = pos / total if total > 0 else 0
        print(f"{split_name} - Total: {total}, Positive: {pos}, Negative: {neg}, Positive Ratio: {pos_ratio:.4f}")
    
    get_stats(train_df, "Train")
    get_stats(val_df, "Validation")
    get_stats(test_df, "Test")

def save_metadata(train_df, val_df, test_df, output_dir):
    """Save metadata CSV files for each data split."""
    train_df.to_csv(os.path.join(output_dir, 'train', 'metadata.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val', 'metadata.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test', 'metadata.csv'), index=False)
    
    print(f"\nMetadata CSV files saved in {output_dir}")
    
def verify_images(df, image_dir, image_extension='.jpg'):
    """Verify that all images in the DataFrame exist in the specified directory."""
    print(f"Verifying image in {image_dir}")
    
    missing_images = []
    for image_name in tqdm(df['isic_id'].values):
        image_path = os.path.join(image_dir, image_name + image_extension)
        if not os.path.exists(image_path):
            missing_images.append(image_name)
    
    if missing_images:
        print(f"Missing {len(missing_images)} images:")
        for img in missing_images:
            print(f"- {img}")
        return False
    return True

def main(config):
    # Load source metadata CSV
    df = pd.read_csv(config['source']['csv_file'])
    
    # Print initial stats
    if 'target' in df.columns:
        print("\nInitial dataset statistics:")
        total = len(df)
        pos = df['target'].sum()
        neg = total - pos
        pos_ratio = pos / total if total > 0 else 0
        print(f"\nTotal: {total}, Positive: {pos}, Negative: {neg}, Positive Ratio: {pos_ratio:.4f}")
    
    # Create directory structure
    create_directories(config['output']['base_dir'])
    
    # Split data
    print("\nSplitting data...")
    train_df, val_df, test_df = split_data(
        df,
        train_ratio=config['splits']['train_ratio'],
        val_ratio=config['splits']['val_ratio'],
        random_seed=config['splits']['random_seed'],
        stratify=config['splits']['stratify']
    )
    
    # Print split stats
    print_split_stats(train_df, val_df, test_df)
    
    # Split images
    splits_data = [('train', train_df), ('val', val_df), ('test', test_df)]
        
    for split_name, split_df in splits_data:
        dest_dir = os.path.join(config['output']['base_dir'], split_name, 'images')
        copy_images(
            split_df,
            config['source']['image_dir'],
            dest_dir,
            config['source']['image_extension']
        )
        
    # Save metadata CSVs
    save_metadata(train_df, val_df, test_df, config['output']['base_dir'])
    
    # Verify images
    print("\nVerifying copied images...")
    all_ok = True
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        image_dir = os.path.join(config['output']['base_dir'], split_name, 'images')
        ok = verify_images(split_df, image_dir, config['source']['image_extension'])
        all_ok = all_ok and ok
    if not all_ok:
        raise SystemExit("Verification failed: some images are missing.")
    else:
        print("\nAll images verified successfully.")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess ISIC dataset - split into train/val/test')
    parser.add_argument('--config', type=str, default='preprocessing_config.yaml',
                        help='Path to YAML configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)
