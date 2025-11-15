import yaml
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class DataConfig:
    data_dir: str = "data"
    img_size: int = 256
    batch_size: int = 8
    num_workers: int = 4

@dataclass
class ModelConfig:
    name: str = "unet"
    encoder: str = "resnet34"
    encoder_weights: str = "imagenet"
    num_classes: int = 1

@dataclass
class TrainingConfig:
    epochs: int = 50
    learning_rate: float = 1e-4
    loss: str = "bce"
    optimizer: str = "adam"
    scheduler: str = "cosine"

@dataclass
class AugmentationConfig:
    horizontal_flip: bool = True
    vertical_flip: bool = True
    rotation_range: float = 10.0
    brightness_range: List[float] = None
    contrast_range: List[float] = None

@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    augmentation: AugmentationConfig
    experiment_name: str = "baseline"
    
    @classmethod
    def from_yaml(cls, config_path: str):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            augmentation=AugmentationConfig(**config_dict.get('augmentation', {})),
            experiment_name=config_dict.get('experiment_name', 'baseline')
        )