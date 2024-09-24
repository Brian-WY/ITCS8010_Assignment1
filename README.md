# Image Classification Project

This project implements image classification using a pretrained ResNet-50 model from the `transformers` library. The model is fine-tuned and evaluated on a custom dataset.

## Project Structure

- `data/`
  - `train/` - Training images organized by class directories.
- `main.py` - Main script to run the training and evaluation.
- `utils/`
  - `dataloader.py` - Module to load and preprocess data.
- `models.py` - Contains the definition of the pretrained model and any modifications.

## Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- PyTorch
- transformers
- torchvision
- datasets
- PIL
- numpy

You can install the necessary libraries using the following command:

```bash
pip install torch torchvision transformers datasets pillow numpy
