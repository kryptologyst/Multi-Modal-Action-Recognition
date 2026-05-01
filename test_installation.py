#!/usr/bin/env python3
"""
Quick test script to verify the multi-modal action recognition installation.

This script tests basic functionality without requiring a full dataset.
"""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.config import Config, ModelConfig, DataConfig
        print("✅ Config modules imported successfully")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from src.models import MultiModalActionRecognitionModel, create_model
        print("✅ Model modules imported successfully")
    except ImportError as e:
        print(f"❌ Model import failed: {e}")
        return False
    
    try:
        from src.data import MultiModalActionDataset, create_sample_dataset
        print("✅ Data modules imported successfully")
    except ImportError as e:
        print(f"❌ Data import failed: {e}")
        return False
    
    try:
        from src.eval import Trainer, ActionRecognitionLoss, MetricsCalculator
        print("✅ Evaluation modules imported successfully")
    except ImportError as e:
        print(f"❌ Evaluation import failed: {e}")
        return False
    
    return True


def test_model_creation():
    """Test model creation and forward pass."""
    print("\nTesting model creation...")
    
    try:
        from src.config import ModelConfig
        from src.models import create_model
        
        config = ModelConfig()
        model = create_model(config)
        print(f"✅ Model created successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        video = torch.randn(1, 3, 16, 224, 224)
        audio = torch.randn(1, 80, 313)
        
        with torch.no_grad():
            outputs = model(video, audio)
        
        assert 'logits' in outputs
        assert outputs['logits'].shape == (1, config.num_classes)
        print("✅ Model forward pass successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False


def test_device_detection():
    """Test device detection."""
    print("\nTesting device detection...")
    
    try:
        from src.config import get_device
        
        device = get_device()
        print(f"✅ Device detected: {device}")
        
        # Test tensor operations on device
        x = torch.randn(2, 3).to(device)
        y = torch.randn(3, 4).to(device)
        z = torch.mm(x, y)
        print(f"✅ Tensor operations work on {device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Device detection failed: {e}")
        return False


def test_sample_dataset():
    """Test sample dataset creation."""
    print("\nTesting sample dataset creation...")
    
    try:
        import tempfile
        import os
        from src.data import create_sample_dataset
        
        with tempfile.TemporaryDirectory() as temp_dir:
            create_sample_dataset(temp_dir, num_samples=5)
            
            # Check if files were created
            assert os.path.exists(os.path.join(temp_dir, "annotations.json"))
            assert os.path.exists(os.path.join(temp_dir, "videos"))
            assert os.path.exists(os.path.join(temp_dir, "audio"))
            
            print("✅ Sample dataset created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample dataset creation failed: {e}")
        return False


def test_demo_imports():
    """Test demo application imports."""
    print("\nTesting demo imports...")
    
    try:
        from demo import ActionRecognitionDemo
        from src.config import DemoConfig
        
        config = DemoConfig()
        demo = ActionRecognitionDemo(config)
        print("✅ Demo application imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo import failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Multi-Modal Action Recognition - Installation Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_model_creation,
        test_device_detection,
        test_sample_dataset,
        test_demo_imports
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Installation is working correctly.")
        print("\nNext steps:")
        print("1. Create sample dataset: python scripts/train.py --create_sample_data")
        print("2. Train model: python scripts/train.py")
        print("3. Run demo: streamlit run demo/streamlit_app.py")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
