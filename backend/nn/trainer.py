import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import time
from utils.utils import AverageMeter

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, logger, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        self.logger = logger
        self.device = device
        
        # Class Imbalance Handling
        self.criterion = self._get_loss_function()
        
        # Optimizer & Scheduler
        self.optimizer = optim.SGD(
            model.parameters(), 
            lr=config.LEARNING_RATE, 
            momentum=0.9, 
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=config.LR_MILESTONES, gamma=config.LR_GAMMA
        )
        
        self.best_acc = 0.0
        self.patience_counter = 0

    def _get_loss_function(self):
        # Calculate weights dynamically from training data
        all_labels = []
        for _, y in self.train_loader:
            all_labels.extend(y.numpy())
        
        counts = torch.bincount(torch.tensor(all_labels))
        weights = 1.0 / (counts.float() + 1e-6)
        weights = weights / weights.sum()
        self.logger.info(f"Computed Class Weights: {weights.tolist()}")
        
        return nn.CrossEntropyLoss(
            weight=weights.to(self.device), 
            label_smoothing=self.cfg.LABEL_SMOOTHING
        )

    def train_epoch(self):
        self.model.train()
        losses = AverageMeter()
        accs = AverageMeter()
        
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            _, pred = torch.max(out, 1)
            acc = (pred == y).float().mean() * 100.0
            losses.update(loss.item(), x.size(0))
            accs.update(acc.item(), x.size(0))
            
        self.scheduler.step()
        return losses.avg, accs.avg

    def validate(self):
        self.model.eval()
        losses = AverageMeter()
        accs = AverageMeter()
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)
                
                _, pred = torch.max(out, 1)
                acc = (pred == y).float().mean() * 100.0
                losses.update(loss.item(), x.size(0))
                accs.update(acc.item(), x.size(0))
                
        return losses.avg, accs.avg

    def fit(self, run_dir):
        csv_path = os.path.join(run_dir, "metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc"])

            for epoch in range(self.cfg.EPOCHS):
                t0 = time.time()
                t_loss, t_acc = self.train_epoch()
                v_loss, v_acc = self.validate()
                dt = time.time() - t0
                
                self.logger.info(f"Epoch {epoch+1:02d} | {dt:.1f}s | "
                                 f"Train: {t_loss:.4f} ({t_acc:.1f}%) | "
                                 f"Val: {v_loss:.4f} ({v_acc:.1f}%)")
                
                # Log to CSV
                writer.writerow([epoch+1, t_loss, t_acc, v_loss, v_acc])
                f.flush()

                # Checkpoint & Early Stopping
                if v_acc > self.best_acc:
                    self.best_acc = v_acc
                    self.patience_counter = 0
                    torch.save(self.model.state_dict(), os.path.join(run_dir, "best_model.pth"))
                    torch.save(self.model.state_dict(), self.cfg.MODEL_SAVE_NAME) # Root copy
                    self.logger.info(f"★ New Best Model Saved! ({v_acc:.2f}%)")
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.cfg.EARLY_STOPPING_PATIENCE:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                    break