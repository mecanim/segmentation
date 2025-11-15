import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def create_model(config):
    """Create segmentation model based on config"""
    if config.model.name.lower() == "unet":
        model = smp.Unet(
            encoder_name=config.model.encoder,
            encoder_weights=config.model.encoder_weights,
            in_channels=3,
            classes=config.model.num_classes,
            activation=None
        )
    elif config.model.name.lower() == "deeplabv3":
        model = smp.DeepLabV3(
            encoder_name=config.model.encoder,
            encoder_weights=config.model.encoder_weights,
            in_channels=3,
            classes=config.model.num_classes,
            activation=None
        )
    else:
        raise ValueError(f"Unknown model: {config.model.name}")
    
    return model