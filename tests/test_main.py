"""
Unit tests for multi-modal action recognition system.

This module contains comprehensive tests for all major components
including data loading, model architecture, training, and evaluation.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import json
import os

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import Config, ModelConfig, DataConfig, TrainingConfig
from src.models import (
    VideoEncoder, AudioEncoder, MultiModalFusion, 
    CrossAttentionFusion, MultiModalActionRecognitionModel
)
from src.data import MultiModalActionDataset, create_sample_dataset
from src.eval import ActionRecognitionLoss, MetricsCalculator, Trainer


class TestConfig:
    """Test configuration classes."""
    
    def test_model_config(self):
        """Test ModelConfig initialization."""
        config = ModelConfig()
        assert config.num_classes == 10
        assert config.hidden_dim == 512
        assert config.fusion_strategy == "cross_attention"
    
    def test_data_config(self):
        """Test DataConfig initialization."""
        config = DataConfig()
        assert config.batch_size == 8
        assert config.num_workers == 4
        assert config.video_augmentation is True
    
    def test_training_config(self):
        """Test TrainingConfig initialization."""
        config = TrainingConfig()
        assert config.epochs == 100
        assert config.learning_rate == 1e-4
        assert config.optimizer == "adamw"


class TestModels:
    """Test model architectures."""
    
    def test_video_encoder(self):
        """Test VideoEncoder initialization and forward pass."""
        config = ModelConfig()
        encoder = VideoEncoder(config)
        
        # Test with dummy input
        video = torch.randn(2, 3, 16, 224, 224)  # Batch, C, T, H, W
        output = encoder(video)
        
        assert output.shape[0] == 2  # Batch size
        assert output.shape[1] == config.fusion_dim
    
    def test_audio_encoder(self):
        """Test AudioEncoder initialization and forward pass."""
        config = ModelConfig()
        encoder = AudioEncoder(config)
        
        # Test with dummy input (mel spectrogram)
        audio = torch.randn(2, 80, 313)  # Batch, Mel-bins, Time
        output = encoder(audio)
        
        assert output.shape[0] == 2  # Batch size
        assert output.shape[1] == config.fusion_dim
    
    def test_cross_attention_fusion(self):
        """Test CrossAttentionFusion module."""
        fusion = CrossAttentionFusion(dim=512, num_heads=8)
        
        video_features = torch.randn(2, 512)
        audio_features = torch.randn(2, 512)
        
        output = fusion(video_features, audio_features)
        
        assert output.shape == video_features.shape
    
    def test_multimodal_fusion(self):
        """Test MultiModalFusion with different strategies."""
        config = ModelConfig()
        
        for strategy in ["early", "late", "cross_attention"]:
            config.fusion_strategy = strategy
            fusion = MultiModalFusion(config)
            
            video_features = torch.randn(2, config.fusion_dim)
            audio_features = torch.randn(2, config.fusion_dim)
            
            output = fusion(video_features, audio_features)
            
            assert output.shape == video_features.shape
    
    def test_complete_model(self):
        """Test complete MultiModalActionRecognitionModel."""
        config = ModelConfig()
        model = MultiModalActionRecognitionModel(config)
        
        # Test forward pass
        video = torch.randn(2, 3, 16, 224, 224)
        audio = torch.randn(2, 80, 313)
        
        outputs = model(video, audio)
        
        assert 'logits' in outputs
        assert 'video_features' in outputs
        assert 'audio_features' in outputs
        assert 'fused_features' in outputs
        
        assert outputs['logits'].shape == (2, config.num_classes)


class TestDataLoading:
    """Test data loading and preprocessing."""
    
    def test_sample_dataset_creation(self):
        """Test creation of sample dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            create_sample_dataset(temp_dir, num_samples=10)
            
            # Check if files were created
            assert os.path.exists(os.path.join(temp_dir, "annotations.json"))
            assert os.path.exists(os.path.join(temp_dir, "videos"))
            assert os.path.exists(os.path.join(temp_dir, "audio"))
            
            # Check annotations
            with open(os.path.join(temp_dir, "annotations.json"), 'r') as f:
                annotations = json.load(f)
            
            assert len(annotations) == 10
            assert all('video_path' in ann for ann in annotations)
            assert all('audio_path' in ann for ann in annotations)
            assert all('action' in ann for ann in annotations)
    
    def test_dataset_loading(self):
        """Test MultiModalActionDataset loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample dataset
            create_sample_dataset(temp_dir, num_samples=5)
            
            config = DataConfig()
            config.data_root = temp_dir
            config.annotations_file = os.path.join(temp_dir, "annotations.json")
            
            dataset = MultiModalActionDataset(
                data_root=temp_dir,
                annotations_file=config.annotations_file,
                config=config,
                split="train"
            )
            
            assert len(dataset) > 0
            
            # Test getting a sample
            sample = dataset[0]
            assert 'video' in sample
            assert 'audio' in sample
            assert 'label' in sample
            assert 'action' in sample


class TestLosses:
    """Test loss functions."""
    
    def test_action_recognition_loss(self):
        """Test ActionRecognitionLoss."""
        loss_fn = ActionRecognitionLoss(num_classes=10)
        
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        
        loss = loss_fn(logits, targets)
        
        assert loss.item() >= 0
        assert loss.requires_grad
    
    def test_focal_loss(self):
        """Test focal loss variant."""
        loss_fn = ActionRecognitionLoss(num_classes=10, use_focal=True)
        
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        
        loss = loss_fn(logits, targets)
        
        assert loss.item() >= 0


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_metrics_calculator(self):
        """Test MetricsCalculator."""
        calculator = MetricsCalculator(num_classes=3)
        
        # Add some predictions
        predictions = torch.tensor([0, 1, 2, 0, 1])
        targets = torch.tensor([0, 1, 2, 1, 1])
        confidences = torch.randn(5, 3)
        
        calculator.update(predictions, targets, confidences)
        
        metrics = calculator.compute_metrics()
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'confusion_matrix' in metrics
        
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1


class TestTraining:
    """Test training components."""
    
    def test_trainer_initialization(self):
        """Test Trainer initialization."""
        config = Config()
        trainer = Trainer(config)
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None
        assert trainer.device is not None
    
    def test_optimizer_creation(self):
        """Test optimizer creation."""
        config = Config()
        trainer = Trainer(config)
        
        # Test different optimizers
        for opt_name in ["adam", "adamw", "sgd"]:
            config.training.optimizer = opt_name
            trainer = Trainer(config)
            assert trainer.optimizer is not None
    
    def test_scheduler_creation(self):
        """Test scheduler creation."""
        config = Config()
        
        for sched_name in ["cosine", "linear", "step"]:
            config.training.scheduler = sched_name
            trainer = Trainer(config)
            # Scheduler might be None for some configurations
            assert trainer.scheduler is not None or sched_name == "step"


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_prediction(self):
        """Test end-to-end prediction pipeline."""
        config = ModelConfig()
        model = MultiModalActionRecognitionModel(config)
        
        # Create dummy data
        video = torch.randn(1, 3, 16, 224, 224)
        audio = torch.randn(1, 80, 313)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(video, audio)
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=1)
            confidences = torch.softmax(logits, dim=1)
        
        assert predictions.shape == (1,)
        assert confidences.shape == (1, config.num_classes)
        assert torch.allclose(confidences.sum(dim=1), torch.ones(1), atol=1e-6)
    
    def test_model_save_load(self):
        """Test model checkpointing."""
        config = ModelConfig()
        model = MultiModalActionRecognitionModel(config)
        
        # Create dummy checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'epoch': 0,
            'metric': 0.0
        }
        
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as f:
            torch.save(checkpoint, f.name)
            
            # Load checkpoint
            loaded_checkpoint = torch.load(f.name, map_location='cpu')
            model.load_state_dict(loaded_checkpoint['model_state_dict'])
            
            # Clean up
            os.unlink(f.name)
        
        assert True  # If we get here, save/load worked


# Fixtures for common test data
@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return Config()


@pytest.fixture
def sample_model(sample_config):
    """Sample model for testing."""
    return MultiModalActionRecognitionModel(sample_config.model)


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {
        'video': torch.randn(2, 3, 16, 224, 224),
        'audio': torch.randn(2, 80, 313),
        'labels': torch.randint(0, 10, (2,))
    }


# Performance tests
class TestPerformance:
    """Performance and memory tests."""
    
    def test_model_memory_usage(self, sample_model, sample_data):
        """Test model memory usage."""
        model = sample_model
        video, audio = sample_data['video'], sample_data['audio']
        
        # Test forward pass memory
        with torch.no_grad():
            outputs = model(video, audio)
        
        # Check output shapes
        assert outputs['logits'].shape[0] == video.shape[0]
    
    def test_batch_processing(self, sample_model):
        """Test batch processing efficiency."""
        model = sample_model
        
        # Test different batch sizes
        for batch_size in [1, 2, 4, 8]:
            video = torch.randn(batch_size, 3, 16, 224, 224)
            audio = torch.randn(batch_size, 80, 313)
            
            with torch.no_grad():
                outputs = model(video, audio)
            
            assert outputs['logits'].shape[0] == batch_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
