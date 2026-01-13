import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import numpy as np
from src import config

def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 0:
            tn += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1
            
    # Schutz vor durch 0 teilen
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": [[tn, fp], [fn, tp]] # Zeile, Spalte
    }

def plot_loss_curves(train_losses: List[float], test_losses: List[float]):
    # Trainings und Testloss plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    save_path = config.PLOT_DIR / "loss_curve.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Loss-Plot gespeichert unter: {save_path}")

def plot_confusion_matrix(cm_data: List[List[int]], classes: List[str]):
    # Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    save_path = config.PLOT_DIR / "confusion_matrix.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion Matrix gespeichert unter: {save_path}")


