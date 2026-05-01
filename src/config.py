"""
Multi-modal Action Recognition Configuration

This module contains configuration classes and utilities for the multi-modal action recognition system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from omegaconf import OmegaConf
import torch


@dataclass
class ModelConfig:
    """Configuration for the multi-modal action recognition model."""
    
    # Model architecture
    model_name: str = "multimodal_action_recognition"
    num_classes: int = 10
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    
    # Video encoder
    video_encoder: str = "timesformer"  # timesformer, video_swin, clip
    video_input_size: int = 224
    video_patch_size: int = 16
    video_num_frames: int = 16
    video_temporal_stride: int = 4
    
    # Audio encoder
    audio_encoder: str = "wav2vec2"  # wav2vec2, mel_spectrogram, mfcc
    audio_sample_rate: int = 16000
    audio_max_length: int = 160000  # 10 seconds at 16kHz
    audio_n_mels: int = 80
    audio_n_fft: int = 1024
    audio_hop_length: int = 512
    
    # Fusion strategy
    fusion_strategy: str = "cross_attention"  # early, late, cross_attention
    fusion_dim: int = 512
    
    # Pretrained models
    pretrained_video_model: str = "facebook/timesformer-base-finetuned-k400"
    pretrained_audio_model: str = "facebook/wav2vec2-base-960h"


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Data paths
    data_root: str = "data/raw"
    processed_data_root: str = "data/processed"
    annotations_file: str = "annotations.json"
    
    # Dataset splits
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Data augmentation
    video_augmentation: bool = True
    audio_augmentation: bool = True
    
    # Video preprocessing
    video_resize: tuple = (224, 224)
    video_normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    video_normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Audio preprocessing
    audio_normalize: bool = True
    audio_noise_factor: float = 0.01
    
    # Data loading
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Training parameters
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    
    # Optimization
    optimizer: str = "adamw"  # adam, adamw, sgd
    scheduler: str = "cosine"  # cosine, linear, step
    gradient_clip_val: float = 1.0
    
    # Mixed precision
    use_amp: bool = True
    
    # Checkpointing
    save_every_n_epochs: int = 10
    save_top_k: int = 3
    monitor_metric: str = "val_accuracy"
    mode: str = "max"
    
    # Early stopping
    early_stopping_patience: int = 15
    
    # Logging
    log_every_n_steps: int = 50
    val_check_interval: float = 1.0


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    
    # Metrics
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "top5_accuracy", "precision", "recall", "f1", "confusion_matrix"
    ])
    
    # Temporal metrics
    temporal_iou_thresholds: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7])
    
    # Visualization
    save_predictions: bool = True
    save_attention_maps: bool = True
    num_samples_to_visualize: int = 10


@dataclass
class DemoConfig:
    """Configuration for the demo application."""
    
    # Demo settings
    demo_type: str = "streamlit"  # streamlit, gradio
    host: str = "0.0.0.0"
    port: int = 8501
    
    # Model settings
    model_path: str = "checkpoints/best_model.ckpt"
    device: str = "auto"  # auto, cpu, cuda, mps
    
    # Input settings
    max_video_duration: int = 30  # seconds
    max_audio_duration: int = 30  # seconds
    supported_video_formats: List[str] = field(default_factory=lambda: [".mp4", ".avi", ".mov"])
    supported_audio_formats: List[str] = field(default_factory=lambda: [".wav", ".mp3", ".flac"])
    
    # Safety settings
    enable_safety_filters: bool = True
    max_file_size_mb: int = 100


@dataclass
class Config:
    """Main configuration class."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)
    
    # Global settings
    seed: int = 42
    device: str = "auto"  # auto, cpu, cuda, mps
    num_gpus: int = 1
    precision: int = 16  # 16, 32
    
    # Logging
    log_dir: str = "logs"
    experiment_name: str = "multimodal_action_recognition"
    use_wandb: bool = False
    wandb_project: str = "multimodal-action-recognition"
    
    # Reproducibility
    deterministic: bool = True
    benchmark: bool = False


def get_device() -> torch.device:
    """Get the appropriate device for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    cfg_dict = OmegaConf.load(config_path)
    return OmegaConf.structured(Config(**cfg_dict))


def save_config(config: Config, config_path: str) -> None:
    """Save configuration to YAML file."""
    OmegaConf.save(OmegaConf.structured(config), config_path)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Default action classes for action recognition
DEFAULT_ACTION_CLASSES = [
    "walking",
    "running", 
    "jumping",
    "sitting",
    "standing",
    "clapping",
    "waving",
    "dancing",
    "cooking",
    "reading"
]

# Safety disclaimer
SAFETY_DISCLAIMER = """
DISCLAIMER: This multi-modal action recognition system is for research and educational purposes only.
It should not be used for:
- Surveillance or monitoring without proper consent
- Medical diagnosis or health monitoring
- Biometric identification or authentication
- Any application that could violate privacy or human rights

The model may have biases and limitations. Always ensure proper consent and ethical use.
"""
