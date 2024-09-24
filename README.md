# Image Classification Assignment

This project implements image classification using a Resnet-50 model pretrained by microsoft from the `transformers` library and a pretrained ResNext-50 model from the `torch` library. The model is fine-tuned and evaluated on the given training dataset.

## Project Structure

- `data/`
  - `train/` - Training images
  - `test/` - Testing images
- `main.py` - Main script to run the training and evaluation.
- `utils/`
  - `dataloader.py` - Module to load and preprocess data.
- `model/`
  - `models.py` - Contains the definition of the pretrained models and corresponding modifications.
- `output/`
  - `resnet50_tuned.pth` - The trained resnet-50 model with best accuracy
  - `resnext_model.pth` - The trained resnext-50 model by efficient training method with a good performence

## Prerequisites
To deploy the code:
```bash
git clone https://github.com/Brian-WY/ITCS8010_Assignment1.git
```
Ensure you have the following installed:
- Python 3.10 or higher
- PyTorch
- transformers
- torchvision
- datasets
- numpy
- random
- tqdm
- timm
- matplotlib

You can install the necessary libraries using the following command:
```bash
pip install torch torchvision transformers datasets numpy random tqdm timm matplotlib
```
Ensure you have put the train and test folder of the images under data directory.

## For Inference
Ensure you're under the same directory as main.py
```
python main.py
```
## Other Settings
For training, set the isTraining in the code to True before
```bash
python main.py
```
For loading finetuned ResNext model to test, set the isResNext in the code to True before
```bash
python main.py
```
