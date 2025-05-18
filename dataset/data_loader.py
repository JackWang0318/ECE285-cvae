"""
Data Loader Module, responsible for loading and preprocessing the Typography-MNIST (TMNIST) dataset
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

class TMNISTDataset(Dataset):
    """
    Typography-MNIST Dataset Class
    
    """
    
    def __init__(self, root_dir, split='train', transform=None, create_dummy_data=False):
        """
        Initialization function
        
        Args:
            root_dir (str): Dataset root directory
            split (str): 'train', 'val', or 'test'
            transform: Data preprocessing/augmentation transformations
            create_dummy_data (bool): Whether to create dummy data if real data isn't available
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Try to load real dataset
        self.data_images = []
        self.char_labels = []
        self.style_labels = []
        
        # Check if dataset CSV file exists
        pkl_path = os.path.join(root_dir, 'TMNIST_Alphabet.pkl')
        if os.path.exists(pkl_path):
            try:
                print(f"Found TMNIST pkl data at {pkl_path}, attempting to load...")
                df = pd.read_pickle(pkl_path)
                # Process the pkl file here according to its structure
                self.data_images = df['image'].tolist()
                self.char_labels = df['labels'].tolist()
                self.style_names = df['names'].tolist()
            except Exception as e:
                print(f"Error loading real dataset: {e}")
                if create_dummy_data:
                    self._create_dummy_data()
        elif create_dummy_data:
            print(f"No dataset found at {root_dir}, creating dummy data...")
            self._create_dummy_data()
        
    def _create_dummy_data(self, num_samples=100, num_chars=10, num_styles=5):
        """Create dummy data for visualization purposes"""
        # Generate dummy images (28x28 random noise)
        self.dummy_images = []
        self.data_paths = [f"dummy_image_{i}.png" for i in range(num_samples)]
        
        # Generate random character and style labels
        self.char_labels = torch.randint(0, num_chars, (num_samples,)).tolist()
        self.style_labels = torch.randint(0, num_styles, (num_samples,)).tolist()
        
        # Generate some structured dummy data for better visualization
        # Make sure each character appears with multiple styles
        structured_samples = num_chars * num_styles
        if structured_samples <= num_samples:
            # Override part of the random data with structured data
            for char_idx in range(num_chars):
                for style_idx in range(num_styles):
                    idx = char_idx * num_styles + style_idx
                    # For structured data, use organized indices
                    self.char_labels[idx] = char_idx
                    self.style_labels[idx] = style_idx
        
        print(f"Created {num_samples} dummy samples with {num_chars} characters and {num_styles} styles")
        
    def __len__(self):
        """Return dataset size"""
        return len(self.style_names)
    
    def __getitem__(self, idx):
        """
        Get data sample
        
        Args:
            idx (int): Sample index
            
        Returns:
            dict: Dictionary containing image and labels
        """
        char_label = self.char_labels[idx]
        style_label = self.style_labels[idx]
        
        # Generate a patterned image based on character and style for better visualization
        # Create structured noise that depends on character and style
        image = np.zeros((28, 28), dtype=np.float32)
        
        # Create a simple pattern based on character ID
        for i in range(28):
            for j in range(28):
                # Character affects the base pattern
                if (i + j + char_label) % 4 == 0:
                    image[i, j] = 0.8
                    
                # Style affects details and intensity
                if (i * j + style_label) % 5 == 0:
                    image[i, j] = 0.5
                    
                # Add a frame to differentiate characters
                if i < 2 or i > 25 or j < 2 or j > 25:
                    image[i, j] = 0.3
        
        # Add character-specific feature in the center
        center_size = 8
        start_i = (28 - center_size) // 2
        start_j = (28 - center_size) // 2
        for i in range(start_i, start_i + center_size):
            for j in range(start_j, start_j + center_size):
                if (i + j + char_label * 3) % 5 == 0:
                    image[i, j] = 1.0
        
        # Convert to PIL Image
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
        
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
        pin_memory=True,
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