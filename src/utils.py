import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2

def visualize_predictions(model, dataloader, device, num_examples=5):
    """Visualize model predictions"""
    model.eval()
    fig, axes = plt.subplots(num_examples, 3, figsize=(12, 4*num_examples))
    
    with torch.no_grad():
        for idx, (images, masks) in enumerate(dataloader):
            if idx >= num_examples:
                break
                
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds_bin = (preds > 0.5).float()
            
            # Convert to numpy for visualization
            image_np = images[0].cpu().permute(1, 2, 0).numpy()
            image_np = (image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            image_np = np.clip(image_np, 0, 1)
            
            mask_np = masks[0].cpu().squeeze().numpy()
            pred_np = preds_bin[0].cpu().squeeze().numpy()
            
            axes[idx, 0].imshow(image_np)
            axes[idx, 0].set_title('Input Image')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(mask_np, cmap='gray')
            axes[idx, 1].set_title('Ground Truth')
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(pred_np, cmap='gray')
            axes[idx, 2].set_title('Prediction')
            axes[idx, 2].axis('off')
    
    plt.tight_layout()
    return fig

def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Dice score plot
    axes[1].plot(history['train_dice'], label='Train Dice')
    axes[1].plot(history['val_dice'], label='Val Dice')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Training and Validation Dice Score')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    return fig