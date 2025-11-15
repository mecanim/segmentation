import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import clearml
from .metrics import DiceScore, IoUScore

class SegmentationTrainer:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Loss functions
        self.loss_fn = self._get_loss_function()
        
        # Metrics
        self.dice_score = DiceScore()
        self.iou_score = IoUScore()
        
        # Optimizer
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        
        # ClearML
        self.task = None
        self._init_clearml()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
    def _get_loss_function(self):
        # ... (остальной код без изменений)
        if self.config.training.loss == "bce":
            return nn.BCEWithLogitsLoss()
        elif self.config.training.loss == "dice":
            # Используем реализацию из segmentation_models_pytorch
            try:
                import segmentation_models_pytorch as smp
                return smp.losses.DiceLoss(mode='binary')
            except ImportError:
                # Fallback implementation
                class DiceLoss(nn.Module):
                    def __init__(self, smooth=1e-6):
                        super().__init__()
                        self.smooth = smooth
                    
                    def forward(self, pred, target):
                        pred = torch.sigmoid(pred)
                        intersection = (pred * target).sum()
                        union = pred.sum() + target.sum()
                        return 1 - (2. * intersection + self.smooth) / (union + self.smooth)
                return DiceLoss()
        elif self.config.training.loss == "combined":
            try:
                import segmentation_models_pytorch as smp
                dice_loss = smp.losses.DiceLoss(mode='binary')
                bce_loss = nn.BCEWithLogitsLoss()
                return lambda pred, target: dice_loss(pred, target) + bce_loss(pred, target)
            except ImportError:
                class CombinedLoss(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.bce = nn.BCEWithLogitsLoss()
                    
                    def forward(self, pred, target):
                        bce = self.bce(pred, target)
                        
                        # Dice component
                        pred_sigmoid = torch.sigmoid(pred)
                        intersection = (pred_sigmoid * target).sum()
                        union = pred_sigmoid.sum() + target.sum()
                        dice = 1 - (2. * intersection + 1e-6) / (union + 1e-6)
                        
                        return bce + dice
                return CombinedLoss()
        else:
            raise ValueError(f"Unknown loss: {self.config.training.loss}")
    
    def _get_optimizer(self):
        # ... (остальной код без изменений)
        if self.config.training.optimizer == "adam":
            return optim.Adam(
                self.model.parameters(), 
                lr=self.config.training.learning_rate
            )
        elif self.config.training.optimizer == "sgd":
            return optim.SGD(
                self.model.parameters(), 
                lr=self.config.training.learning_rate,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
    
    def _get_scheduler(self):
        # ... (остальной код без изменений)
        if self.config.training.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.training.epochs
            )
        elif self.config.training.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=20, 
                gamma=0.1
            )
        else:
            return None
    
    def _init_clearml(self):
        """Initialize ClearML logging"""
        try:
            # Закрываем предыдущую задачу, если она существует
            current_task = clearml.Task.current_task()
            if current_task:
                current_task.close()
                print(f"Closed previous ClearML task: {current_task.name}")
            
            self.task = clearml.Task.init(
                project_name="PetSegmentation",
                task_name=self.config.experiment_name,
                auto_connect_frameworks={'pytorch': False}
            )
            
            # Log parameters
            self.task.connect(vars(self.config))
            print(f"ClearML initialized: {self.config.experiment_name}")
        except Exception as e:
            print(f"ClearML initialization failed: {e}")
            self.task = None
    
    def close(self):
        """Явное закрытие задачи ClearML"""
        if self.task:
            self.task.close()
            print(f"Closed ClearML task: {self.task.name}")
    
    def train_epoch(self, dataloader) -> Tuple[float, float]:
        # ... (остальной код без изменений)
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        batch_count = 0
        
        pbar = tqdm(dataloader, desc=f"Training Epoch {self.current_epoch + 1}")
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            loss = self.loss_fn(outputs, masks)
            
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                dice = self.dice_score(outputs, masks)
            
            total_loss += loss.item()
            total_dice += dice.item()
            batch_count += 1
            
            # Update global step for logging
            self.global_step += 1
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice.item():.4f}'
            })
            
            # Log to ClearML
            if self.task and batch_idx % 10 == 0:
                self.task.logger.report_scalar(
                    title="train_batch",
                    series="loss",
                    value=loss.item(),
                    iteration=self.global_step
                )
                self.task.logger.report_scalar(
                    title="train_batch", 
                    series="dice",
                    value=dice.item(),
                    iteration=self.global_step
                )
        
        avg_loss = total_loss / batch_count
        avg_dice = total_dice / batch_count
        return avg_loss, avg_dice
    
    def validate_epoch(self, dataloader) -> Tuple[float, float, float]:
        # ... (остальной код без изменений)
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for images, masks in tqdm(dataloader, desc="Validation"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)
                
                dice = self.dice_score(outputs, masks)
                iou = self.iou_score(outputs, masks)
                
                total_loss += loss.item()
                total_dice += dice.item()
                total_iou += iou.item()
                batch_count += 1
        
        avg_loss = total_loss / batch_count
        avg_dice = total_dice / batch_count
        avg_iou = total_iou / batch_count
        
        return avg_loss, avg_dice, avg_iou
    
    def train(self, train_loader, val_loader, test_loader) -> Tuple[Dict, float, float]:
        # ... (остальной код без изменений)
        best_dice = 0.0
        history = {
            'train_loss': [], 'train_dice': [],
            'val_loss': [], 'val_dice': [], 'val_iou': []
        }
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch+1}/{self.config.training.epochs}")
            
            # Training
            train_loss, train_dice = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_dice, val_iou = self.validate_epoch(val_loader)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_dice'].append(train_dice)
            history['val_loss'].append(val_loss)
            history['val_dice'].append(val_dice)
            history['val_iou'].append(val_iou)
            
            print(f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
            
            # Log to ClearML
            if self.task:
                self.task.logger.report_scalar(
                    title="epoch_metrics",
                    series="train_loss",
                    value=train_loss,
                    iteration=epoch
                )
                self.task.logger.report_scalar(
                    title="epoch_metrics",
                    series="train_dice", 
                    value=train_dice,
                    iteration=epoch
                )
                self.task.logger.report_scalar(
                    title="epoch_metrics",
                    series="val_loss",
                    value=val_loss,
                    iteration=epoch
                )
                self.task.logger.report_scalar(
                    title="epoch_metrics",
                    series="val_dice",
                    value=val_dice,
                    iteration=epoch
                )
                self.task.logger.report_scalar(
                    title="epoch_metrics", 
                    series="val_iou",
                    value=val_iou,
                    iteration=epoch
                )
            
            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_dice': best_dice,
                    'config': self.config
                }, f'best_model_{self.config.experiment_name}.pth')
                print(f"New best model saved with Dice: {best_dice:.4f}")
        
        # Final test evaluation
        print("\nEvaluating on test set...")
        test_loss, test_dice, test_iou = self.validate_epoch(test_loader)
        print(f"Final Test - Loss: {test_loss:.4f}, Dice: {test_dice:.4f}, IoU: {test_iou:.4f}")
        
        # Log final test results to ClearML
        if self.task:
            self.task.logger.report_scalar(
                title="final_test",
                series="test_dice",
                value=test_dice,
                iteration=0
            )
            self.task.logger.report_scalar(
                title="final_test",
                series="test_iou", 
                value=test_iou,
                iteration=0
            )
            self.task.logger.report_scalar(
                title="final_test",
                series="test_loss",
                value=test_loss,
                iteration=0
            )
        
        return history, test_dice, test_iou