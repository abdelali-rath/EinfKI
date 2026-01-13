import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from typing import Tuple
import numpy as np

from src import config


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:

    stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    train_transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),

        # --- Augmentierung
        transforms.RandomHorizontalFlip(p=0.5), # Zufällige Spiegelung der Bilder
        transforms.RandomRotation(degrees=15),  # Leichte Rotierung

        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    return train_transform, test_transform


def get_dataloaders() -> Tuple[DataLoader, DataLoader]:

    # Datenset zweimal laden um verschiedenen Splits unterschiedliche Transforms zuzuweisne
    train_transform, test_transform = get_transforms()
    
    # Basis-Dataset
    full_dataset = datasets.ImageFolder(root=config.DATA_DIR)
    
    # Indizes für Split berechnen 80%/20%
    total_size = len(full_dataset)
    train_size = int(total_size * config.SPLIT_RATIO)
    test_size = total_size - train_size
    
    # Generator für Reproduzierbarkeit des Splits
    generator = torch.Generator().manual_seed(config.SEED)
    
    # Indizes der Aufteilung
    train_indices, test_indices = random_split(
        range(total_size), 
        [train_size, test_size], 
        generator=generator
    )
    
    # Datensets mit korrekten Transformern
    # Subset verknüpft Indizes mit jeweiligen ImageFolder und dessen Transform
    train_dataset = Subset(
        datasets.ImageFolder(root=config.DATA_DIR, transform=train_transform),
        train_indices
    )
    
    test_dataset = Subset(
        datasets.ImageFolder(root=config.DATA_DIR, transform=test_transform),
        test_indices
    )

    print(f"Dataset geladen von: {config.DATA_DIR}")
    print(f"Training Samples: {len(train_dataset)} | Test Samples: {len(test_dataset)}")
    print(f"Klassen: {full_dataset.classes}")

    # DataLoader erstellen
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=2,      # Multiprocessing
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader