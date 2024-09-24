import numpy as np
import matplotlib.pyplot as plt
import time
import glob
import torch
import random
import tqdm
from torch.autograd import Variable
import torch.nn as nn
from torch.distributions.constraints import boolean
from transformers import AutoImageProcessor
import torch.optim as optim
from utils.dataloader import load_dataset
from model.models import get_resnet_model, get_resnext_model

# show some images
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if len(npimg.shape) > 2:
        npimg = np.transpose(img, [1, 2, 0])
    plt.figure
    plt.imshow(npimg, 'gray')
    plt.show()

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_resnet(model, train_loader, val_loader, num_epochs, device, lr=1e-4, decay=1e-2):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        #validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).logits
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss / len(train_loader):.4f}')
        print(f'Val Loss: {val_loss / len(val_loader):.4f}')
        print(f'Val Accuracy: {correct / total:.4f}')
        print('--------------')

def train_resnext(model, train_loader, val_loader, num_epochs, device, lr=5e-5, decay=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)

    for epoch in tqdm.tqdm(range(num_epochs), desc="Training"):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss / len(train_loader):.4f}')
        print(f'Val Loss: {val_loss / len(val_loader):.4f}')
        print(f'Val Accuracy: {correct / total:.4f}')
        print('--------------')

def eval_model(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f'Test Loss: {test_loss / len(test_loader):.4f}')
    print(f'Test Accuracy: {correct / total:.4f}')
    print('--------------')

def eval_model_2(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f'Test Loss: {test_loss / len(test_loader):.4f}')
    print(f'Test Accuracy: {correct / total:.4f}')
    print('--------------')

def print_trainable_parameters(model):
    """
    Print the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params}"
    )


if __name__ == '__main__':
    isTrain = False
    isInference = True
    isResNext = False
    set_random_seed(0)

    # Directory containing training and test data
    data_path = 'data'
    num_classes = len(glob.glob('./data/train/*'))
    # Load the model and processor
    model, image_processor = get_resnet_model(num_classes=num_classes)

    # Prepare data with validation split
    train_loader, val_loader, test_loader, class_names = load_dataset(data_path, image_processor, 0.1, 32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isTrain:
        model.to(device)
        print_trainable_parameters(model)
        # Train the model
        train_resnet(model, train_loader, val_loader, 10, device, 1e-4, 1e-2)
        # Save the model
        torch.save(model.state_dict(), 'output/resnet50_tuned.pth')

    if isInference:
        model.load_state_dict(torch.load('output/resnet50_tuned.pth'))
        model.to(device)
        # Test the model
        eval_model(model, test_loader, device)

    if isResNext:
        model, transforms = get_resnext_model(num_classes=num_classes)
        train_loader, val_loader, test_loader, class_names = load_dataset(
            data_path, transforms, 0.1, 32  # Use transforms from timm
        )
        model.load_state_dict(torch.load('output/resnext_model.pth'))
        model.to(device)
        eval_model_2(model, test_loader, device)

