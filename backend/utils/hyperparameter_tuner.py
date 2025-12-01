import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import logging
import os
import argparse

# Import your modules (adjust imports based on your structure)
from nn.model import STGCN
from nn.dataset import SkeletonDataset
from utils.utils import set_seed

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "Data/" # or wherever your X_train.npy is
X_PATH = "X_train.npy"
Y_PATH = "y_train.npy"
NUM_CLASSES = 3
IN_CHANNELS = 6 # Assuming Pos + Vel
SEQ_LENGTH = 50
TRAIN_SPLIT = 0.8
SEED = 42

def get_data_loaders(batch_size):
    """Load data and split into train/val loaders."""
    if not os.path.exists(X_PATH):
        raise FileNotFoundError(f"Data not found at {X_PATH}")

    full_dataset = SkeletonDataset(X_PATH, Y_PATH, seq_length=SEQ_LENGTH, augment=True)
    
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    # Disable augmentation for validation
    val_data.dataset.augment = False 
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader

def objective(trial):
    """Optuna objective function."""
    
    # 1. Suggest Hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
    
    # Optimizer choice
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam"])

    # 2. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_data_loaders(batch_size)
    
    # Calculate Class Weights (for imbalance)
    # We'll do a quick approximation or load pre-calculated ones to save time
    # For robust tuning, calculate from full dataset labels
    all_labels = []
    # Quick hack: get labels from dataset directly if possible, or iterate once
    # Here we assume we can access y directly or just skip for speed in this snippet
    # Ideally: weights = calculate_weights(train_loader)
    # Placeholder weights:
    weights = torch.tensor([1.0, 2.0, 2.0]).to(device) # Example weights

    model = STGCN(num_classes=NUM_CLASSES, in_channels=IN_CHANNELS, dropout=dropout).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 3. Training Loop (Short & Fast)
    # We train for fewer epochs than production to speed up search
    n_epochs = 15 
    
    for epoch in range(n_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                _, pred = torch.max(out, 1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        accuracy = correct / total
        
        # Report to Optuna
        trial.report(accuracy, epoch)
        
        # Pruning (Stop bad trials early)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

if __name__ == "__main__":
    set_seed(SEED)
    
    # Create Study
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner() # Efficient pruning
    )
    
    print("Starting Hyperparameter Tuning...")
    # Run 20-50 trials depending on your time/compute
    study.optimize(objective, n_trials=20) 
    
    print("\n--- TUNING COMPLETE ---")
    print("Best Trial:")
    trial = study.best_trial
    print(f"  Value (Accuracy): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Optional: Save best params to a file
    import json
    with open("best_hyperparams.json", "w") as f:
        json.dump(trial.params, f, indent=4)
    print("Saved best parameters to best_hyperparams.json")