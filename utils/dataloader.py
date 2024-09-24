import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms

class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        """
        Initialize the TransformSubset with a subset of data and an optional transform.
        """
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        """
        Retrieve an item from the subset and apply transformation if available.
        """
        data, label = self.subset[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, label

    def __len__(self):
        """
        Get the length of the dataset subset.
        """
        return len(self.subset)


def load_dataset(path, image_processor, val_ratio, batch_size=32):
    img_size = (256,256)
    '''
    transform = transforms.Compose([
        transforms.Resize(img_size),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),  # Convert PIL Image to PyTorch Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(img_size),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    '''
    transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        # transforms.CenterCrop(image_processor.size["shortest_edge"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

    ])

    transform_val = transforms.Compose([
        transforms.Resize(img_size),
        # transforms.CenterCrop(image_processor.size["shortest_edge"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    ])

    full_dataset = datasets.ImageFolder(os.path.join(path, 'train'))
    class_names = full_dataset.classes

    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataset = TransformSubset(train_dataset, transform)
    val_dataset = TransformSubset(val_dataset, transform_val)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size)

    test_path = os.path.join(path, 'test')
    if os.path.exists(test_path):
        test_dataset = datasets.ImageFolder(test_path, transform_val)
        test_loader = DataLoader(test_dataset, batch_size)
    else:
        test_dataset = None
        test_loader = None

    return train_loader, val_loader, test_loader, class_names

