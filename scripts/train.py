"""
Main training script for multi-modal action recognition.

This script handles the complete training pipeline including data loading,
model training, validation, and checkpointing.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.config import Config, set_seed, get_device
from src.data import create_data_loaders, create_sample_dataset
from src.models import create_model
from src.eval import Trainer, evaluate_model, plot_training_history


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train multi-modal action recognition model")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/train/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/raw",
        help="Root directory for dataset"
    )
    parser.add_argument(
        "--create_sample_data",
        action="store_true",
        help="Create sample dataset for testing"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of sample videos to create"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only evaluate the model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.ckpt",
        help="Path to checkpoint for evaluation"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = OmegaConf.load(args.config)
        config = OmegaConf.structured(Config(**config))
    else:
        print(f"Config file {args.config} not found. Using default configuration.")
        config = Config()
    
    # Override config with command line arguments
    config.data.data_root = args.data_root
    
    # Set random seed
    set_seed(config.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create sample dataset if requested
    if args.create_sample_data:
        print(f"Creating sample dataset with {args.num_samples} samples...")
        create_sample_dataset(args.data_root, args.num_samples)
        print("Sample dataset created successfully!")
    
    # Check if dataset exists
    annotations_file = os.path.join(args.data_root, "annotations.json")
    if not os.path.exists(annotations_file):
        print(f"Dataset not found at {args.data_root}")
        print("Please create a dataset or use --create_sample_data to generate sample data.")
        return
    
    # Create data loaders
    print("Creating data loaders...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(config.data)
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return
    
    # Create model
    print("Creating model...")
    model = create_model(config.model)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Evaluation only mode
    if args.eval_only:
        print("Evaluation mode...")
        
        # Load checkpoint
        if os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            print(f"Loaded checkpoint from {args.checkpoint}")
        else:
            print(f"Checkpoint {args.checkpoint} not found!")
            return
        
        # Evaluate on test set
        print("Evaluating on test set...")
        results = evaluate_model(model, test_loader, config, device)
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
        print(f"Top-5 Accuracy: {results['metrics']['top5_accuracy']:.4f}")
        print(f"Precision: {results['metrics']['precision']:.4f}")
        print(f"Recall: {results['metrics']['recall']:.4f}")
        print(f"F1 Score: {results['metrics']['f1']:.4f}")
        
        # Save results
        results_path = "assets/evaluation_results.txt"
        os.makedirs("assets", exist_ok=True)
        
        with open(results_path, 'w') as f:
            f.write("EVALUATION RESULTS\n")
            f.write("="*50 + "\n")
            f.write(f"Accuracy: {results['metrics']['accuracy']:.4f}\n")
            f.write(f"Top-5 Accuracy: {results['metrics']['top5_accuracy']:.4f}\n")
            f.write(f"Precision: {results['metrics']['precision']:.4f}\n")
            f.write(f"Recall: {results['metrics']['recall']:.4f}\n")
            f.write(f"F1 Score: {results['metrics']['f1']:.4f}\n")
        
        print(f"Results saved to {results_path}")
        return
    
    # Training mode
    print("Starting training...")
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train the model
    trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = evaluate_model(trainer.model, test_loader, config, device)
    
    # Print final results
    print("\n" + "="*50)
    print("FINAL TEST RESULTS")
    print("="*50)
    print(f"Accuracy: {test_results['metrics']['accuracy']:.4f}")
    print(f"Top-5 Accuracy: {test_results['metrics']['top5_accuracy']:.4f}")
    print(f"Precision: {test_results['metrics']['precision']:.4f}")
    print(f"Recall: {test_results['metrics']['recall']:.4f}")
    print(f"F1 Score: {test_results['metrics']['f1']:.4f}")
    
    # Plot training history
    if trainer.train_losses and trainer.val_metrics:
        plot_path = "assets/training_history.png"
        plot_training_history(trainer.train_losses, trainer.val_metrics, plot_path)
    
    # Save final results
    results_path = "assets/final_results.txt"
    os.makedirs("assets", exist_ok=True)
    
    with open(results_path, 'w') as f:
        f.write("FINAL TEST RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"Accuracy: {test_results['metrics']['accuracy']:.4f}\n")
        f.write(f"Top-5 Accuracy: {test_results['metrics']['top5_accuracy']:.4f}\n")
        f.write(f"Precision: {test_results['metrics']['precision']:.4f}\n")
        f.write(f"Recall: {test_results['metrics']['recall']:.4f}\n")
        f.write(f"F1 Score: {test_results['metrics']['f1']:.4f}\n")
    
    print(f"Final results saved to {results_path}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
