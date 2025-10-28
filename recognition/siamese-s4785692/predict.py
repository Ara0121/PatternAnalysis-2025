# -*- coding: utf-8 -*-
"""
predict.py

Single image inference for the Siamese Classification Network.

Usage:
  python predict.py --config path/to/predict_config.yaml

Outputs:
  JSON with fields: image, prob, threshold, pred, checkpoint
"""
import argparse
import json
import os

import torch
from torchvision import transforms
from PIL import Image
import yaml

from modules import SiameseClassificationNetwork


def load_config(path):
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def predict_one(model, transform, image_path, device) :
    """Predict a class of a given image and return probability."""
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)  # [1,C,H,W]
    with torch.no_grad():
        # Prediction
        logits = model(x)
        prob = torch.sigmoid(logits).item()
    return float(prob)


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define paths and threold from config
    img_path = config["predict"].get("image_path", "sample.jpg")
    checkpoint_path = config["predict"].get("checkpoint", "best_model.pth")
    thr = float(config["predict"].get("threshold", 0.5))

    # Check if specified paths exit
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    # No special transformation for testing set
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['augmentation']['mean'],
            std=config['augmentation']['std']
        )
    ])
    # Create model
    model = SiameseClassificationNetwork(
        embedding_dim=config['model']['embedding_dim'],
        pretrained=config['model']['pretrained']
    )
    model = model.to(device)
    
    # Load trained model
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    
    prob = predict_one(model, transform, img_path, device)
    pred = int(prob >= thr)

    # Define fields to output
    out = {
        "image": os.path.abspath(img_path),
        "prob": prob,
        "threshold": thr,
        "pred": pred,
        "checkpoint": os.path.abspath(checkpoint_path),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-image predictor")
    parser.add_argument("--config", required=True, type=str, help="Path to predict config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
