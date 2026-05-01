"""
Data loading and preprocessing utilities for multi-modal action recognition.

This module provides data loaders for video and audio data with proper preprocessing,
augmentation, and multi-modal synchronization.
"""

import os
import json
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchaudio.transforms as audio_transforms

import cv2
import numpy as np
import librosa
from PIL import Image
import decord
from decord import VideoReader, cpu

from ..config import DataConfig, DEFAULT_ACTION_CLASSES


class MultiModalActionDataset(Dataset):
    """
    Dataset for multi-modal action recognition with video and audio.
    
    Args:
        data_root: Root directory containing video and audio files
        annotations_file: Path to JSON file with annotations
        config: Data configuration
        split: Dataset split ('train', 'val', 'test')
        action_classes: List of action class names
    """
    
    def __init__(
        self,
        data_root: str,
        annotations_file: str,
        config: DataConfig,
        split: str = "train",
        action_classes: Optional[List[str]] = None
    ):
        self.data_root = Path(data_root)
        self.config = config
        self.split = split
        self.action_classes = action_classes or DEFAULT_ACTION_CLASSES
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Filter annotations by split
        self.samples = [ann for ann in self.annotations if ann['split'] == split]
        
        # Set up video transforms
        self._setup_video_transforms()
        
        # Set up audio transforms
        self._setup_audio_transforms()
        
        # Initialize video reader
        decord.bridge.set_bridge('torch')
    
    def _setup_video_transforms(self) -> None:
        """Set up video preprocessing transforms."""
        if self.split == "train" and self.config.video_augmentation:
            self.video_transform = transforms.Compose([
                transforms.Resize(self.config.video_resize),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.Normalize(
                    mean=self.config.video_normalize_mean,
                    std=self.config.video_normalize_std
                )
            ])
        else:
            self.video_transform = transforms.Compose([
                transforms.Resize(self.config.video_resize),
                transforms.Normalize(
                    mean=self.config.video_normalize_mean,
                    std=self.config.video_normalize_std
                )
            ])
    
    def _setup_audio_transforms(self) -> None:
        """Set up audio preprocessing transforms."""
        self.audio_transform = audio_transforms.MelSpectrogram(
            sample_rate=self.config.audio_sample_rate,
            n_mels=self.config.audio_n_mels,
            n_fft=self.config.audio_n_fft,
            hop_length=self.config.audio_hop_length
        )
        
        if self.split == "train" and self.config.audio_augmentation:
            self.audio_augment = audio_transforms.FrequencyMasking(freq_mask_param=10)
        else:
            self.audio_augment = None
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing video, audio, label, and metadata
        """
        sample = self.samples[idx]
        
        # Load video
        video_path = self.data_root / sample['video_path']
        video_frames = self._load_video(video_path)
        
        # Load audio
        audio_path = self.data_root / sample['audio_path']
        audio_features = self._load_audio(audio_path)
        
        # Get label
        label = self.action_classes.index(sample['action'])
        
        return {
            'video': video_frames,
            'audio': audio_features,
            'label': torch.tensor(label, dtype=torch.long),
            'video_path': str(video_path),
            'audio_path': str(audio_path),
            'action': sample['action'],
            'duration': sample.get('duration', 0.0)
        }
    
    def _load_video(self, video_path: Path) -> torch.Tensor:
        """
        Load and preprocess video frames.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Preprocessed video tensor of shape (C, T, H, W)
        """
        try:
            # Load video with decord
            vr = VideoReader(str(video_path), ctx=cpu(0))
            
            # Sample frames uniformly
            total_frames = len(vr)
            num_frames = self.config.video_num_frames
            
            if total_frames >= num_frames:
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            else:
                indices = np.arange(total_frames)
                # Pad with last frame if needed
                indices = np.pad(indices, (0, num_frames - total_frames), mode='edge')
            
            # Get frames
            frames = vr.get_batch(indices)  # Shape: (T, H, W, C)
            frames = frames.permute(0, 3, 1, 2)  # Shape: (T, C, H, W)
            frames = frames.float() / 255.0  # Normalize to [0, 1]
            
            # Apply transforms
            frames = self.video_transform(frames)
            
            return frames
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # Return dummy video if loading fails
            return torch.zeros(3, self.config.video_num_frames, *self.config.video_resize)
    
    def _load_audio(self, audio_path: Path) -> torch.Tensor:
        """
        Load and preprocess audio features.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio tensor
        """
        try:
            # Load audio
            waveform, sr = librosa.load(str(audio_path), sr=self.config.audio_sample_rate)
            
            # Pad or truncate to fixed length
            max_length = self.config.audio_max_length
            if len(waveform) > max_length:
                waveform = waveform[:max_length]
            else:
                waveform = np.pad(waveform, (0, max_length - len(waveform)))
            
            # Convert to tensor
            waveform = torch.from_numpy(waveform).float()
            
            # Normalize if specified
            if self.config.audio_normalize:
                waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
            
            # Add noise for augmentation
            if self.split == "train" and self.config.audio_augmentation:
                noise = torch.randn_like(waveform) * self.config.audio_noise_factor
                waveform = waveform + noise
            
            # Convert to mel spectrogram
            mel_spec = self.audio_transform(waveform)
            
            # Apply frequency masking if specified
            if self.audio_augment is not None:
                mel_spec = self.audio_augment(mel_spec)
            
            return mel_spec
            
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # Return dummy audio if loading fails
            return torch.zeros(self.config.audio_n_mels, self.config.audio_max_length // self.config.audio_hop_length)


def create_data_loaders(
    config: DataConfig,
    action_classes: Optional[List[str]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        config: Data configuration
        action_classes: List of action class names
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = MultiModalActionDataset(
        data_root=config.data_root,
        annotations_file=config.annotations_file,
        config=config,
        split="train",
        action_classes=action_classes
    )
    
    val_dataset = MultiModalActionDataset(
        data_root=config.data_root,
        annotations_file=config.annotations_file,
        config=config,
        split="val",
        action_classes=action_classes
    )
    
    test_dataset = MultiModalActionDataset(
        data_root=config.data_root,
        annotations_file=config.annotations_file,
        config=config,
        split="test",
        action_classes=action_classes
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


def create_sample_dataset(data_root: str, num_samples: int = 50) -> None:
    """
    Create a sample dataset for testing purposes.
    
    Args:
        data_root: Root directory for the dataset
        num_samples: Number of sample videos to create
    """
    data_path = Path(data_root)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (data_path / "videos").mkdir(exist_ok=True)
    (data_path / "audio").mkdir(exist_ok=True)
    
    annotations = []
    
    for i in range(num_samples):
        # Create dummy video file
        video_path = f"videos/sample_{i:03d}.mp4"
        audio_path = f"audio/sample_{i:03d}.wav"
        
        # Create dummy video (colored frames)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(data_path / video_path),
            fourcc, 30.0, (224, 224)
        )
        
        # Generate random colored frames
        for frame_idx in range(90):  # 3 seconds at 30fps
            frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        
        # Create dummy audio file
        duration = 3.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Generate random audio signal
        audio = np.random.randn(len(t)) * 0.1
        librosa.output.write_wav(str(data_path / audio_path), audio, sample_rate)
        
        # Random action
        action = random.choice(DEFAULT_ACTION_CLASSES)
        
        # Random split
        split = random.choices(
            ["train", "val", "test"],
            weights=[0.7, 0.15, 0.15]
        )[0]
        
        annotations.append({
            "video_path": video_path,
            "audio_path": audio_path,
            "action": action,
            "duration": duration,
            "split": split
        })
    
    # Save annotations
    with open(data_path / "annotations.json", 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Created sample dataset with {num_samples} samples in {data_root}")


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for multi-modal data.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched data dictionary
    """
    videos = torch.stack([item['video'] for item in batch])
    audios = torch.stack([item['audio'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    return {
        'video': videos,
        'audio': audios,
        'label': labels,
        'video_paths': [item['video_path'] for item in batch],
        'audio_paths': [item['audio_path'] for item in batch],
        'actions': [item['action'] for item in batch],
        'durations': [item['duration'] for item in batch]
    }
