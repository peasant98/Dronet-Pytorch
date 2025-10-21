#!/usr/bin/env python3
"""
Dronet PyTorch - Main Entry Point

A PyTorch implementation of Dronet: Learning to Fly by Driving
This script provides a command-line interface for training, testing, and inference.

Usage:
    python main.py train --epochs 150 --batch-size 16
    python main.py test --weights models/dronet_trained.pth
    python main.py infer --weights models/dronet_trained.pth --image path/to/image.jpg
    python main.py export --weights models/dronet_trained.pth --output dronet.onnx
"""

import argparse
import sys
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from dronet_torch import DronetTorch
from dronet_torch_train import getModel, trainModel
from dronet_torch_test import testModel
from dronet_datasets import DronetDataset


def train_command(args):
    """Train the Dronet model."""
    print("=" * 50)
    print("Training Dronet Model")
    print("=" * 50)

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Initialize model
    model = getModel(
        img_dims=(args.img_size, args.img_size),
        img_channels=args.channels,
        output_dim=1,
        weights_path=args.weights if args.weights else None
    )

    print(f"\nTraining Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Hard Mining k: {args.k}")
    print(f"  Image Size: {args.img_size}x{args.img_size}")
    print(f"  Channels: {args.channels}")
    print(f"  Device: {model.device}")
    print()

    # Train the model
    trainModel(model, args.epochs, args.batch_size, args.save_every, args.k)

    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Model saved to: models/dronet_trained.pth")
    print("=" * 50)


def test_command(args):
    """Test the Dronet model."""
    print("=" * 50)
    print("Testing Dronet Model")
    print("=" * 50)

    if not args.weights:
        print("Error: --weights parameter is required for testing")
        sys.exit(1)

    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found at {args.weights}")
        sys.exit(1)

    # Initialize model
    model = DronetTorch(
        img_dims=(args.img_size, args.img_size),
        img_channels=args.channels,
        output_dim=1
    )
    model.to(model.device)

    print(f"\nTesting Configuration:")
    print(f"  Weights: {args.weights}")
    print(f"  Device: {model.device}")
    print()

    # Test the model
    testModel(model, args.weights)


def infer_command(args):
    """Run inference on a single image."""
    print("=" * 50)
    print("Dronet Inference")
    print("=" * 50)

    if not args.weights:
        print("Error: --weights parameter is required for inference")
        sys.exit(1)

    if not args.image:
        print("Error: --image parameter is required for inference")
        sys.exit(1)

    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found at {args.weights}")
        sys.exit(1)

    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        sys.exit(1)

    # Initialize model
    model = DronetTorch(
        img_dims=(args.img_size, args.img_size),
        img_channels=args.channels,
        output_dim=1
    )
    model.to(model.device)
    model.eval()

    # Load weights
    try:
        model.load_state_dict(torch.load(args.weights, map_location=model.device))
        print(f"Loaded weights from: {args.weights}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)

    # Load and preprocess image
    image = Image.open(args.image).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image_tensor = transform(image).unsqueeze(0).to(model.device)

    # Run inference
    with torch.no_grad():
        steering, collision = model(image_tensor)

    print(f"\nInference Results:")
    print(f"  Image: {args.image}")
    print(f"  Steering Angle: {steering.item():.4f} radians ({np.degrees(steering.item()):.2f} degrees)")
    print(f"  Collision Probability: {collision.item():.4f} ({collision.item()*100:.2f}%)")
    print("=" * 50)


def export_command(args):
    """Export model to ONNX format."""
    print("=" * 50)
    print("Exporting Dronet Model to ONNX")
    print("=" * 50)

    if not args.weights:
        print("Warning: No weights specified, exporting untrained model")

    # Initialize model
    model = DronetTorch(
        img_dims=(args.img_size, args.img_size),
        img_channels=args.channels,
        output_dim=1
    )
    model.to(model.device)
    model.eval()

    # Load weights if provided
    if args.weights and os.path.exists(args.weights):
        try:
            model.load_state_dict(torch.load(args.weights, map_location=model.device))
            print(f"Loaded weights from: {args.weights}")
        except Exception as e:
            print(f"Warning: Could not load weights: {e}")

    # Create dummy input
    dummy_input = torch.randn(1, args.channels, args.img_size, args.img_size).to(model.device)

    # Export to ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            args.output,
            verbose=args.verbose,
            input_names=['input'],
            output_names=['steering', 'collision'],
            dynamic_axes={'input': {0: 'batch_size'}}
        )
        print(f"\nModel successfully exported to: {args.output}")
        print("=" * 50)
    except Exception as e:
        print(f"Error exporting model: {e}")
        sys.exit(1)


def info_command(args):
    """Display information about the Dronet model."""
    print("=" * 50)
    print("Dronet Model Information")
    print("=" * 50)

    model = DronetTorch(
        img_dims=(args.img_size, args.img_size),
        img_channels=args.channels,
        output_dim=1
    )

    print(f"\nModel Architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Statistics:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Input Size: {args.channels}x{args.img_size}x{args.img_size}")
    print(f"  Output: Steering angle (1) + Collision probability (1)")
    print(f"  Device: {model.device}")
    print("=" * 50)


def main():
    """Main entry point for the Dronet CLI."""
    parser = argparse.ArgumentParser(
        description='Dronet PyTorch - Learning to Fly by Driving',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model for 150 epochs
  python main.py train --epochs 150 --batch-size 16

  # Test a trained model
  python main.py test --weights models/dronet_trained.pth

  # Run inference on an image
  python main.py infer --weights models/dronet_trained.pth --image test.jpg

  # Export model to ONNX
  python main.py export --weights models/dronet_trained.pth --output dronet.onnx

  # Display model information
  python main.py info
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    subparsers.required = True

    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--img-size', type=int, default=224,
                              help='Input image size (default: 224)')
    common_parser.add_argument('--channels', type=int, default=3,
                              help='Number of input channels (default: 3)')

    # Train command
    train_parser = subparsers.add_parser('train', parents=[common_parser],
                                         help='Train the Dronet model')
    train_parser.add_argument('--epochs', type=int, default=150,
                             help='Number of training epochs (default: 150)')
    train_parser.add_argument('--batch-size', type=int, default=16,
                             help='Batch size for training (default: 16)')
    train_parser.add_argument('--k', type=int, default=8,
                             help='Hard mining parameter (default: 8)')
    train_parser.add_argument('--save-every', type=int, default=5,
                             help='Save weights every N epochs (default: 5)')
    train_parser.add_argument('--weights', type=str, default=None,
                             help='Path to pre-trained weights (optional)')
    train_parser.set_defaults(func=train_command)

    # Test command
    test_parser = subparsers.add_parser('test', parents=[common_parser],
                                        help='Test the Dronet model')
    test_parser.add_argument('--weights', type=str, required=True,
                            help='Path to trained model weights')
    test_parser.set_defaults(func=test_command)

    # Infer command
    infer_parser = subparsers.add_parser('infer', parents=[common_parser],
                                         help='Run inference on an image')
    infer_parser.add_argument('--weights', type=str, required=True,
                             help='Path to trained model weights')
    infer_parser.add_argument('--image', type=str, required=True,
                             help='Path to input image')
    infer_parser.set_defaults(func=infer_command)

    # Export command
    export_parser = subparsers.add_parser('export', parents=[common_parser],
                                          help='Export model to ONNX format')
    export_parser.add_argument('--weights', type=str, default=None,
                              help='Path to trained model weights (optional)')
    export_parser.add_argument('--output', type=str, default='dronet.onnx',
                              help='Output ONNX file path (default: dronet.onnx)')
    export_parser.add_argument('--verbose', action='store_true',
                              help='Verbose ONNX export')
    export_parser.set_defaults(func=export_command)

    # Info command
    info_parser = subparsers.add_parser('info', parents=[common_parser],
                                        help='Display model information')
    info_parser.set_defaults(func=info_command)

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    args.func(args)


if __name__ == '__main__':
    main()
