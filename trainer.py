from typing import List, Optional, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
from model_types import TrainingMetrics
from data import BinaryEncoder, FizzBuzzLabeler

class FizzBuzzTrainer:
    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[Any] = None,
                 device: str = 'cpu',
                 gradient_clip: float = 1.0,
                 use_wandb: bool = True):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_clip = gradient_clip
        self.use_wandb = use_wandb
        self.model.to(device)
        
        if self.use_wandb:
            wandb.watch(model, log='all')
    
    def evaluate(self, loader: DataLoader) -> TrainingMetrics:
        """Evaluate model on given loader."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return TrainingMetrics(
            epoch=0,
            loss=total_loss / len(loader),
            accuracy=100 * correct / total
        )

    def train_epoch(self, train_loader: DataLoader) -> TrainingMetrics:
        """Train for one epoch and return metrics."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        for inputs, labels in progress_bar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.gradient_clip
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        return TrainingMetrics(
            epoch=0,
            loss=total_loss / len(train_loader),
            accuracy=100 * correct / total
        )
    
    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              epochs: int = 100,
              verbose: bool = True) -> List[TrainingMetrics]:
        """Train the model for specified number of epochs."""
        metrics = []
        best_val_acc = 0
        patience_counter = 0
        
        epoch_iterator = tqdm(range(epochs), desc="Epochs", disable=not verbose)
        
        for epoch in epoch_iterator:
            train_metrics = self.train_epoch(train_loader)
            train_metrics.epoch = epoch + 1
            metrics.append(train_metrics)
            
            log_dict = {
                'epoch': epoch + 1,
                'train_loss': train_metrics.loss,
                'train_accuracy': train_metrics.accuracy
            }
            
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                log_dict.update({
                    'val_loss': val_metrics.loss,
                    'val_accuracy': val_metrics.accuracy
                })
                
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics.accuracy)
                    else:
                        self.scheduler.step()
                
                if val_metrics.accuracy > best_val_acc:
                    best_val_acc = val_metrics.accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 15:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                        break
            
            epoch_iterator.set_postfix({
                'loss': f'{train_metrics.loss:.4f}',
                'acc': f'{train_metrics.accuracy:.2f}%'
            })
            
            if self.use_wandb:
                wandb.log(log_dict)
        
        return metrics

    def test(self, start: int, end: int, encoder: BinaryEncoder) -> List[str]:
        """Test the model on a range of numbers."""
        self.model.eval()
        results = []
        
        for i in tqdm(range(start, end), desc="Testing"):
            x = encoder.encode(torch.tensor([i])).to(self.device)
            with torch.no_grad():
                output = self.model(x)
                prediction = output.argmax().item()
            
            result = FizzBuzzLabeler.decode(prediction)(i)
            results.append(result)
        
        return results