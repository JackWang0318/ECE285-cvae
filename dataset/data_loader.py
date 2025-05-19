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
import random

class FontTransferDataset(Dataset):
    def __init__(self, data_path, label_list):
        import pickle
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
        self.label_list = sorted(label_list)  # Ensure consistent one-hot encoding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        style_img = torch.tensor(sample["style_image"], dtype=torch.float32).unsqueeze(0) / 255.0
        target_img = torch.tensor(sample["target_image"], dtype=torch.float32).unsqueeze(0) / 255.0
        label = sample["target_label"]
        label_index = self.label_list.index(label)
        label_onehot = torch.nn.functional.one_hot(torch.tensor(label_index), num_classes=len(self.label_list)).float()
        return style_img, label_onehot, target_img


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
        self.samples = []
        
        # Check if dataset CSV file exists
        pkl_path = os.path.join(root_dir, 'TMNIST_Alphabet.pkl')
        if os.path.exists(pkl_path):
            try:
                print(f"Found TMNIST pkl data at {pkl_path}, attempting to load...")
                df = pd.read_pickle(pkl_path)
                # Filter allowed characters: 0-9 + A-Z + a-z
                allowed_chars = set(
                    [chr(i) for i in range(ord('0'), ord('9') + 1)] +
                    [chr(i) for i in range(ord('A'), ord('Z') + 1)] +
                    [chr(i) for i in range(ord('a'), ord('z') + 1)]
                )
                df = df[df['labels'].isin(allowed_chars)]
                
                # Group by font name
                grouped = df.groupby('names')
                self.samples = []
                for font_name, group in grouped:
                    char_to_img = {row['labels']: row['image'] for _, row in group.iterrows()}
                    if len(char_to_img) < len(allowed_chars):
                        continue
                    label_list = list(char_to_img.keys())
                    for _ in range(10):  # N pairs per font
                        style_label, target_label = random.sample(label_list, 2)
                        self.samples.append({
                            'font_name': font_name,
                            'style_image': char_to_img[style_label],
                            'target_label': target_label,
                            'target_image': char_to_img[target_label],
                        })
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
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get data sample
        
        Args:
            idx (int): Sample index
            
        Returns:
            dict: Dictionary containing image and labels
        """
        sample = self.samples[idx]
        style_image = Image.fromarray((sample['style_image'] * 255).astype(np.uint8))
        target_image = Image.fromarray((sample['target_image'] * 255).astype(np.uint8))
        if self.transform:
            style_image = self.transform(style_image)
            target_image = self.transform(target_image)
        return {
            'style_image': style_image,
            'target_label': sample['target_label'],
            'target_image': target_image
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