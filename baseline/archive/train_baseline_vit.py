#!/usr/bin/env python3
"""
Minimal supervised geo-localization baseline: ViT -> (lat, lon) regression.
Uses GSV-cities style data: per-city CSVs with place_id,year,month,northdeg,city_id,lat,lon,panoid.
Image path: {data_root}/Images/{city_id}/{city_id}_{place_id:07d}_{year}_{month}_{northdeg}_{lat}_{lon}_{panoid}.jpg
"""
import argparse
import math
import os
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm


RANDOM_SEED = 42
IMG_SIZE = 224
# ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def haversine_km(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    """Geodesic distance in km between (lat1, lon1) and (lat2, lon2) in degrees."""
    lat1 = math.radians(lat1_deg)
    lon1 = math.radians(lon1_deg)
    lat2 = math.radians(lat2_deg)
    lon2 = math.radians(lon2_deg)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return 6371.0 * c  # Earth radius km


def build_path(row, data_root):
    """Build image path from a CSV row (place_id, year, month, northdeg, city_id, lat, lon, panoid)."""
    city_id = row["city_id"]
    place_id = int(row["place_id"])
    year = int(row["year"])
    month = int(row["month"])
    northdeg = int(row["northdeg"])
    lat = row["lat"]
    lon = row["lon"]
    panoid = row["panoid"]
    name = f"{city_id}_{place_id:07d}_{year}_{month}_{northdeg}_{lat}_{lon}_{panoid}.jpg"
    return os.path.join(data_root, "Images", city_id, name), float(lat), float(lon)


def load_metadata(metadata_dir, data_root, sample_per_city=100):
    """
    Load per-city CSVs from metadata_dir, build (path, lat, lon) for each row.
    Keep only existing files, then sample sample_per_city per city.
    """
    metadata_path = Path(metadata_dir)
    data_root = Path(data_root)
    rows = []
    for csv_path in sorted(metadata_path.glob("*.csv")):
        df = pd.read_csv(csv_path)
        city_rows = []
        for _, row in df.iterrows():
            path, lat, lon = build_path(row, data_root)
            if os.path.isfile(path):
                city_rows.append((path, lat, lon))
        if len(city_rows) >= sample_per_city:
            chosen = random.Random(RANDOM_SEED).sample(city_rows, sample_per_city)
        else:
            chosen = city_rows
        rows.extend(chosen)
    return rows


class GeoDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples  # list of (path, lat, lon)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, lat, lon = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor([lat, lon], dtype=torch.float32)


def get_transform(train=True):
    t = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    if train:
        t.insert(1, transforms.RandomHorizontalFlip(p=0.5))
    return transforms.Compose(t)


class ViTGeo(nn.Module):
    def __init__(self, backbone_name="vit_base_patch16_224", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="token"
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Linear(feat_dim, 2)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)


def median_geodesic_km(pred_latlon, true_latlon):
    """pred_latlon, true_latlon: (N, 2) tensors [lat, lon] in degrees."""
    pred = pred_latlon.cpu().numpy()
    true = true_latlon.cpu().numpy()
    dists = [
        haversine_km(pred[i, 0], pred[i, 1], true[i, 0], true[i, 1])
        for i in range(len(pred))
    ]
    return float(torch.tensor(dists).median().item()), sum(dists) / len(dists)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default="data/gsv-cities/Dataframes", help="Path to Dataframes dir (per-city CSVs)")
    parser.add_argument("--data_root", type=str, default="data/gsv-cities", help="Path to gsv-cities root (parent of Images/)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--samples_per_city", type=int, default=100)
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print(f"Loading metadata and sampling {args.samples_per_city} images per city...")
    samples = load_metadata(args.metadata, args.data_root, sample_per_city=args.samples_per_city)
    if not samples:
        raise SystemExit("No samples found. Check --metadata and --data_root and that image paths match CSV.")
    print(f"Total samples: {len(samples)}")

    # 80/10/10 split
    n = len(samples)
    random.Random(RANDOM_SEED).shuffle(samples)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_samples = samples[:n_train]
    val_samples = samples[n_train : n_train + n_val]
    test_samples = samples[n_train + n_val :]

    train_ds = GeoDataset(train_samples, transform=get_transform(train=True))
    val_ds = GeoDataset(val_samples, transform=get_transform(train=False))
    test_ds = GeoDataset(test_samples, transform=get_transform(train=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon GPU
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    model = ViTGeo().to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                pred = model(imgs)
                val_preds.append(pred)
                val_targets.append(targets)
        val_pred = torch.cat(val_preds, dim=0)
        val_true = torch.cat(val_targets, dim=0)
        med_km, mean_km = median_geodesic_km(val_pred, val_true)
        print(f"Epoch {epoch + 1}/{args.epochs}  Val median geodesic: {med_km:.4f} km  mean: {mean_km:.4f} km")

    # Final test
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device)
            pred = model(imgs)
            test_preds.append(pred)
            test_targets.append(targets)
    test_pred = torch.cat(test_preds, dim=0)
    test_true = torch.cat(test_targets, dim=0)
    test_med, test_mean = median_geodesic_km(test_pred, test_true)
    print(f"\nFinal test median geodesic distance: {test_med:.4f} km")
    print(f"Final test mean geodesic distance:   {test_mean:.4f} km")


if __name__ == "__main__":
    main()
