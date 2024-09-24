import torch.nn as nn
from transformers import AutoImageProcessor, ResNetForImageClassification, ResNetConfig
import torch.nn as nn
import torch
import torch.hub
import timm
from torchvision.transforms import transforms

class TNet(nn.Module):
    def __init__(self):
        super(TNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(4, stride=4),
        )
        self.classifier = nn.Sequential(
            nn.Linear(3600, 16),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_resnet_model(num_classes, freeze=False):
    '''
    Load the ResNet50 model pretrained on ImageNet
    '''
    # Load the image processor
    image_processor = AutoImageProcessor.from_pretrained('microsoft/resnet-50')
    '''
    image_processor = transforms.Compose([
        transforms.Resize(256),               
        transforms.ToTensor(),            
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    '''
    # Load the ResNet50 model, modify the classifier for the correct number of classes
    model = ResNetForImageClassification.from_pretrained(
        "microsoft/resnet-50",
        config=ResNetConfig(num_labels=num_classes),
        ignore_mismatched_sizes=True
    )
    if freeze:
        # Freeze all layers except the classifier (if desired)
        for param in model.parameters():
            param.requires_grad = False  # Freeze all layers

        # Unfreeze the last layer or classifier to fine-tune
        for param in model.classifier.parameters():
            param.requires_grad = True  # Unfreeze classifier for fine-tuning


    return model, image_processor

def get_resnext_model(num_classes, freeze=False):
    '''
    Load the ResNeXt50_32x4d model pretrained on ImageNet
    '''
    # Load the ResNeXt model from torch.hub
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)

    # Set the model to training mode (enables dropout, batchnorm updates, etc.)
    model.train()
    if freeze:
        # Freeze all layers except the classifier (if desired)
        for param in model.parameters():
            param.requires_grad = False  # Freeze all layers
    # Modify the classifier (final fully connected layer) to match the number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),   # Add a hidden layer with 256 neurons
        nn.ReLU(),                  # ReLU activation
        nn.Dropout(0.5),            # Dropout layer
        nn.Linear(256, num_classes) # Final classification layer
    )
    # Define the image processor
    image_processor = transforms.Compose([
        transforms.Resize(256),               # Resize the smaller side to 256 pixels
        transforms.ToTensor(),                # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])



    return model, image_processor
