import torch
from pathlib import Path

DATA_DIR = Path(r"C:\Users\meist\Downloads\cats_and_dogs_filtered\train")
CHECKPOINT_DIR = Path("checkpoints")
PLOT_DIR = Path("plots")

BATCH_SIZE = 32
LEARNING_RATE = 0.01     # SGD Standard
NUM_EPOCHS = 20
IMG_SIZE = (128, 128)
SPLIT_RATIO = 0.8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42