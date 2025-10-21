# Dronet PyTorch

A PyTorch implementation of **DroNet**: Learning to Fly by Driving. This repository contains a complete training and testing pipeline for autonomous drone navigation using deep learning.

## Overview

DroNet is a convolutional neural network (CNN) that learns to predict steering angles and collision probabilities for autonomous navigation. Originally implemented in TensorFlow/Keras, this repository provides a clean PyTorch implementation with additional features and improvements.

### Features

- Complete PyTorch implementation of the DroNet architecture
- Training pipeline with hard mining for improved learning
- Testing and evaluation with multiple metrics (RMSE, EVA, F1-score)
- Support for both Udacity and ETH Zurich bicycle datasets
- ONNX export for deployment
- TensorRT support for optimized inference (optional)
- Command-line interface for easy usage
- Data augmentation capabilities

## Architecture

DroNet uses a residual CNN architecture with:
- Input: 224x224 RGB images
- Convolutional layers with batch normalization
- Residual connections for better gradient flow
- Two output heads:
  - Steering angle prediction (regression)
  - Collision probability (binary classification)

## Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended)
- PyTorch 1.7+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Dronet-Pytorch.git
cd Dronet-Pytorch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For TensorRT support:
```bash
# Requires CUDA and TensorRT to be installed
pip install onnx-tensorrt
```

## Dataset Setup

### Directory Structure

Create the following directory structure for your datasets:

```
data/
└── collision/
    └── collision_dataset/
        ├── training/
        │   ├── EXPERIMENT_1/
        │   │   ├── images/
        │   │   └── steering_training.txt (or labels.txt)
        │   └── EXPERIMENT_2/
        │       ├── images/
        │       └── steering_training.txt (or labels.txt)
        ├── validation/
        │   └── ... (same structure)
        └── testing/
            └── ... (same structure)
```

### Dataset Format

Each experiment folder should contain:
- `images/`: Directory with image files (JPG/PNG)
- `steering_training.txt` (for steering data): One steering angle per line (in radians)
- `labels.txt` (for collision data): One collision probability per line (0.0 to 1.0)

### Supported Datasets

1. **Udacity Self-Driving Car Dataset**: Use the `move_data.py` script to process and organize
2. **ETH Zurich Bicycle Dataset**: Use `data-utils/time_stamp_matching.py` for timestamp synchronization

## Usage

The repository provides a unified command-line interface through `main.py`:

### Training

Train a model from scratch:

```bash
python main.py train --epochs 150 --batch-size 16
```

Resume training from checkpoint:

```bash
python main.py train --epochs 200 --weights models/weights_050.pth
```

Training options:
- `--epochs`: Number of training epochs (default: 150)
- `--batch-size`: Batch size for training (default: 16)
- `--k`: Hard mining parameter - number of hardest samples to focus on (default: 8)
- `--save-every`: Save checkpoint every N epochs (default: 5)
- `--img-size`: Input image size (default: 224)
- `--channels`: Number of input channels (default: 3)

### Testing

Evaluate a trained model:

```bash
python main.py test --weights models/dronet_trained.pth
```

Metrics computed:
- **RMSE** (Root Mean Square Error): Steering angle prediction accuracy
- **EVA** (Explained Variance): How well the model explains steering variance
- **Accuracy**: Collision prediction accuracy
- **F1 Score**: Harmonic mean of precision and recall for collision detection

### Inference

Run inference on a single image:

```bash
python main.py infer --weights models/dronet_trained.pth --image test_image.jpg
```

Output example:
```
Inference Results:
  Image: test_image.jpg
  Steering Angle: 0.1234 radians (7.07 degrees)
  Collision Probability: 0.0543 (5.43%)
```

### Model Export

Export to ONNX format for deployment:

```bash
python main.py export --weights models/dronet_trained.pth --output dronet.onnx
```

### Model Information

Display model architecture and statistics:

```bash
python main.py info
```

## Advanced Usage

### Custom Training Script

For more control, use the training script directly:

```python
from dronet_torch_train import getModel, trainModel

# Initialize model
model = getModel(
    img_dims=(224, 224),
    img_channels=3,
    output_dim=1,
    weights_path=None  # or path to checkpoint
)

# Train
trainModel(
    model=model,
    epochs=150,
    batch_size=16,
    steps_save=5,  # Save every 5 epochs
    k=8  # Hard mining parameter
)
```

### Custom Testing Script

```python
from dronet_torch_test import testModel
from dronet_torch import DronetTorch

model = DronetTorch(img_dims=(224,224), img_channels=3, output_dim=1)
testModel(model, weights_path='models/dronet_trained.pth')
```

### Data Augmentation

Enable augmentation in the dataset:

```python
from dronet_datasets import DronetDataset

dataset = DronetDataset(
    root_dir='data/collision/collision_dataset',
    data_group='training',
    augmentation=True,  # Enable augmentation
    grayscale=False,
    verbose=False
)
```

Augmentations applied:
- Random resized crop
- Color jitter (brightness, contrast, saturation)
- Random grayscale (10% probability)

### Visualization

Plot training and validation losses:

```bash
python plot.py
```

This will display a graph of losses over epochs using the `models/losses.txt` file.

## Utilities

### Data Processing

**Process Udacity Dataset:**
```python
python move_data.py
```

**Timestamp Matching for ETH Zurich Dataset:**
```python
# Edit data-utils/time_stamp_matching.py to set folder path
python data-utils/time_stamp_matching.py
```

### TensorRT Inference (Optional)

For optimized inference on NVIDIA GPUs:

```python
python dronet_trt.py
```

## Model Architecture Details

### Network Layers

```
Input (3x224x224)
  ↓
Conv2d(32, 5x5, stride=2) + MaxPool
  ↓
[Residual Block 1] → 32 filters
  ↓
[Residual Block 2] → 64 filters
  ↓
[Residual Block 3] → 128 filters
  ↓
Flatten → Dropout(0.5)
  ↓
├─ Linear → Steering Angle (1)
└─ Linear → Sigmoid → Collision Prob (1)
```

### Loss Function

Combined loss with hard mining:
```
Loss = MSE_hard(steering) + β × BCE_hard(collision)
```

Where:
- MSE_hard: Mean Squared Error with hard mining on steering angles
- BCE_hard: Binary Cross Entropy with hard mining on collision probabilities
- β: Dynamic weight that increases during training (β = max(0, 1 - exp(-0.1(epoch-10))))

## Training Tips

1. **Hard Mining Parameter (k)**: Start with k=8. Increase for larger batches.
2. **Learning Rate**: Default is 1e-4 with weight decay 1e-5
3. **Data Balance**: Mix steering and collision datasets for best results
4. **Augmentation**: Enable for smaller datasets to prevent overfitting
5. **Checkpoints**: Monitor validation loss and save best performing models

## File Structure

```
Dronet-Pytorch/
├── main.py                    # Main CLI interface
├── dronet_torch.py           # Model architecture
├── dronet_torch_train.py     # Training pipeline
├── dronet_torch_test.py      # Testing and evaluation
├── dronet_datasets.py        # Dataset loader with augmentation
├── move_data.py              # Udacity data processing
├── plot.py                   # Loss visualization
├── dronet_trt.py            # TensorRT inference (optional)
├── data-utils/
│   └── time_stamp_matching.py  # ETH dataset preprocessing
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch-size 8` or `--batch-size 4`
- Reduce image size: `--img-size 160`

### Poor Training Performance
- Check data labels are correctly formatted (radians for steering, 0-1 for collision)
- Ensure images are properly preprocessed
- Try different hard mining values
- Monitor loss curves with `plot.py`

### Model Not Converging
- Verify dataset diversity
- Enable data augmentation
- Reduce learning rate (modify in `dronet_torch_train.py`)
- Increase training epochs

## Citation

If you use this code, please cite the original DroNet paper:

```bibtex
@article{loquercio2018dronet,
  title={DroNet: Learning to Fly by Driving},
  author={Loquercio, Antonio and Maqueda, Ana I and Del-Blanco, Carlos R and Scaramuzza, Davide},
  journal={IEEE Robotics and Automation Letters},
  volume={3},
  number={2},
  pages={1088--1095},
  year={2018},
  publisher={IEEE}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original DroNet paper and TensorFlow implementation
- Udacity Self-Driving Car Dataset
- ETH Zurich Robotics and Perception Group

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.
