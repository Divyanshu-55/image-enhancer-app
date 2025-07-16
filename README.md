# Image Enhancer App

An AI-powered application for enhancing image quality using deep learning. This project supports training, inference, and deployment of a custom-trained image enhancement model.


## Features

- Deep learning-based image enhancement.
- Train your own model using the provided pipeline.
- Inference with pre-trained models.
- Export to TensorFlow Lite for lightweight deployment.
- Sample inputs and enhanced outputs included.
- Deploy-ready with Heroku `Procfile`.


## Model Structure

- `model.py`: Defines the neural network architecture.
- `loss.py`: Custom loss functions for training.
- `data.py`: Dataset loader and preprocessing.
- `train.py`: Main script for training the model.
- `inference_save.py`: Run inference and save enhanced images.

## CNN Model Architecture Overview

The model is built using custom Convolutional Blocks (ConvBlock) that consist of

 - A 2D convolution layer with:

    - Customizable kernel size and stride
    - ReLU (or other) activation
    - Same padding

 - Trainable weights and bias initialized with:
 
    - tf.random_normal_initializer for weights
    - Zeros for bias
 - Each `ConvBlock` includes:
    - Convolution with a 3x3 kernel (default)
    - ReLU activation
    - SAME padding
 - Trainable parameters are initialized using TensorFlow’s low-level API.

The architecture is designed for efficiency and optimized for TensorFlow Lite deployment on edge devices.This design provides flexibility while staying lightweight, ideal for enhancement tasks.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/image-enhancer-app.git
cd image-enhancer-app

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Training
python train.py

# Inference
python inference_save.py
```
Enhanced images will be saved in the `result/` directory.

## Project Structure

```bash
image-enhancer-app/
├── model.py               # Model definition
├── data.py                # Dataset handling
├── loss.py                # Loss functions
├── train.py               # Training script
├── inference_save.py      # Inference script
├── model_trained/         # Pretrained models
├── samples/               # Example input/output
├── result/                # Inference results
├── requirements.txt       # Dependencies
└── Procfile               # Deployment config
```
## Acknowledgements
Built using TensorFlow and Python. Inspired by common image super-resolution techniques.
