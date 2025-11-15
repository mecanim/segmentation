import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Optional

class PetSegmentationDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        split: str = 'train',
        img_size: int = 256,
        augmentations: Optional[A.Compose] = None
    ):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.augmentations = augmentations
        
        self.images_dir = os.path.join(data_dir, 'img', split)
        self.masks_dir = os.path.join(data_dir, 'labels', split)
        
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir) 
            if f.endswith('.jpg')
        ])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.image_files[idx]
        mask_name = img_name.replace('.jpg', '.png')
        
        # Load image
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Normalize mask to [0, 1]
        mask = (mask > 0).astype(np.float32)
        
        if self.augmentations:
            augmented = self.augmentations(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Basic resize if no augmentations
            transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(),
                ToTensorV2(),
            ])
            augmented = transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.unsqueeze(0)

def get_augmentations(config, is_training: bool = True) -> A.Compose:
    transforms = []
    
    if is_training:
        if config.augmentation.horizontal_flip:
            transforms.append(A.HorizontalFlip(p=0.5))
        if config.augmentation.vertical_flip:
            transforms.append(A.VerticalFlip(p=0.5))
        if config.augmentation.rotation_range:
            transforms.append(A.Rotate(
                limit=config.augmentation.rotation_range, 
                p=0.5
            ))
        if config.augmentation.brightness_range:
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=config.augmentation.brightness_range,
                contrast_limit=config.augmentation.contrast_range or [0, 0],
                p=0.3
            ))
    
    transforms.extend([
        A.Resize(config.data.img_size, config.data.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return A.Compose(transforms)