import torch
import torch.nn as nn
from typing import Tuple, List
from src import config

def train_one_epoch(model: nn.Module, 
                    dataloader: torch.utils.data.DataLoader, 
                    criterion: nn.Module, 
                    optimizer: torch.optim.Optimizer) -> float:

    model.train()
    running_loss = 0.0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate(model: nn.Module, 
             dataloader: torch.utils.data.DataLoader, 
             criterion: nn.Module) -> Tuple[float, List[int], List[int]]:

    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            # Vorhersage ist der Index mit dem höchsten Logit
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss, all_labels, all_preds


