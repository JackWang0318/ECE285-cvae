"""
Data Loader Module, responsible for loading and preprocessing the Typography-MNIST (TMNIST) dataset
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class TMNISTDataset(Dataset):
    """
    Typography-MNIST Dataset Class
    
    This dataset contains 565,292 grayscale images of 1,812 unique glyphs rendered in 1,355 Google Fonts
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Initialization function
        
        Args:
            root_dir (str): Dataset root directory
            split (str): 'train', 'val', or 'test'
            transform: Data preprocessing/augmentation transformations
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # TODO: Load dataset indexes, including image paths and corresponding character/style labels
        self.data_paths = []
        self.char_labels = []
        self.style_labels = []
        
    def __len__(self):
        """Return dataset size"""
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        """
        Get data sample
        
        Args:
            idx (int): Sample index
            
        Returns:
            dict: Dictionary containing image and labels
        """
        # TODO: Implement data loading logic
        image_path = self.data_paths[idx]
        char_label = self.char_labels[idx]
        style_label = self.style_labels[idx]
        
        # Load example image (replace in actual implementation)
        image = np.zeros((28, 28), dtype=np.uint8)
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'char_label': char_label,
            'style_label': style_label
        }

def get_data_loaders(data_dir, batch_size=64, num_workers=4):
    """
    Create data loaders for training, validation, and testing
    
    Args:
        data_dir (str): Data directory
        batch_size (int): Batch size
        num_workers (int): Number of data loading threads
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Create datasets
    train_dataset = TMNISTDataset(data_dir, split='train', transform=transform)
    val_dataset = TMNISTDataset(data_dir, split='val', transform=transform)
    test_dataset = TMNISTDataset(data_dir, split='test', transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader 