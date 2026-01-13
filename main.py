import torch
import torch.nn as nn
import torch.optim as optim
from src import config, dataset, model, engine, utils


def main():
    # Ordner erstellen
    config.CHECKPOINT_DIR.mkdir(exist_ok=True)
    config.PLOT_DIR.mkdir(exist_ok=True)
    
    print(f"Starte Training auf Device: {config.DEVICE}")

    # Daten laden
    train_loader, test_loader = dataset.get_dataloaders()

    # Modell, Loss und Optimizer initialisieren
    cnn_model = model.SimpleCNN().to(config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()

    # SGD
    optimizer = optim.SGD(cnn_model.parameters(), lr=config.LEARNING_RATE)

    # Liste für Historie für die Plots
    train_losses = []
    test_losses = []
    
    # Variablen für Metriken der letzten Epoche
    last_labels = []
    last_preds = []

    # Training Loop
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        
        # Training
        train_loss = engine.train_one_epoch(cnn_model, train_loader, criterion, optimizer)
        train_losses.append(train_loss)
        
        # Testing
        test_loss, labels, preds = engine.evaluate(cnn_model, test_loader, criterion)
        test_losses.append(test_loss)
        
        # Speichern für spätere Metrik Berechnung
        last_labels = labels
        last_preds = preds
        
        print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        
        # --- Checkpoint speichern
        checkpoint_path = config.CHECKPOINT_DIR / f"model_epoch_{epoch + 1}.pth"
        torch.save(cnn_model.state_dict(), checkpoint_path)
        print(f"Checkpoint gespeichert: {checkpoint_path.name}")

    # Metriken ausgeben
    print("\n--- Finale Evaluation ---")
    metrics = utils.calculate_metrics(last_labels, last_preds)
    
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    
    # Visualisierungen
    utils.plot_loss_curves(train_losses, test_losses)
    # Klassen-Mapping
    utils.plot_confusion_matrix(metrics['confusion_matrix'], classes=['Cat', 'Dog'])

    # Beweis: Checkpoint laden
    print("\n--- Checkpoint-Test (Option 2) ---")
    try:
        # Neues leeres Modell erstellen
        loaded_model = model.SimpleCNN().to(config.DEVICE)
        # Gewichte der letzten Epoche laden
        last_ckpt = config.CHECKPOINT_DIR / f"model_epoch_{config.NUM_EPOCHS}.pth"
        loaded_model.load_state_dict(torch.load(last_ckpt, map_location=config.DEVICE))
        loaded_model.eval()
        print(f"Erfolg: Modell aus '{last_ckpt.name}' geladen und bereit.")
    except Exception as e:
        print(f"Fehler beim Laden des Checkpoints: {e}")

if __name__ == "__main__":
    main()



