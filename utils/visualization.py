"""
Visualization tools module, used for displaying generation results and training process
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def show_images(images, title=None, save_path=None, nrow=8, normalize=True):
    """
    Display a set of images
    
    Args:
        images (Tensor): Image tensor [B, C, H, W]
        title (str): Title
        save_path (str): Save path
        nrow (int): Number of images per row
        normalize (bool): Whether to normalize
    """
    # Convert to range [0, 1]
    if normalize:
        images = (images + 1) / 2.0

    # Create grid
    grid = make_grid(images, nrow=nrow, normalize=False)
    
    # Convert to Numpy and adjust axis order
    grid = grid.cpu().detach().numpy().transpose(1, 2, 0)
    
    # Create image
    plt.figure(figsize=(12, 12))
    plt.imshow(grid, cmap='gray' if grid.shape[2] == 1 else None)
    plt.axis('off')
    
    if title:
        plt.title(title)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.close()

def visualize_reconstructions(model, data_loader, device, num_samples=8, save_path=None):
    """
    Visualize model reconstructions
    
    Args:
        model (CVAE): CVAE model
        data_loader (DataLoader): Data loader
        device (torch.device): Compute device
        num_samples (int): Number of samples to display
        save_path (str): Save path
    """
    model.eval()
    
    # Get a batch of data
    batch = next(iter(data_loader))
    images = batch['image'][:num_samples].to(device)
    char_labels = batch['char_label'][:num_samples]
    style_labels = batch['style_label'][:num_samples].to(device)
    
    # Perform reconstruction
    with torch.no_grad():
        outputs = model(images, style_labels)
        reconstructions = outputs['reconstruction']
    
    # Create visualization
    combined = []
    for i in range(num_samples):
        combined.append(images[i])
        combined.append(reconstructions[i])
    
    combined = torch.stack(combined)
    
    # Display original and reconstructed images
    show_images(combined, title="Reconstructions", save_path=save_path, nrow=2)
    
    return combined

def visualize_style_transfer(model, data_loader, device, num_samples=4, num_styles=4, save_path=None):
    """
    Visualize style transfer results
    
    Args:
        model (CVAE): CVAE model
        data_loader (DataLoader): Data loader
        device (torch.device): Compute device
        num_samples (int): Number of samples to display
        num_styles (int): Number of styles to display for each sample
        save_path (str): Save path
    """
    model.eval()
    
    # Get a batch of data
    batch = next(iter(data_loader))
    images = batch['image'][:num_samples].to(device)
    char_labels = batch['char_label'][:num_samples]
    style_labels = batch['style_label'].to(device)
    
    # Get some target styles
    target_styles = []
    for i in range(num_styles):
        idx = np.random.randint(0, len(batch['style_label']))
        target_styles.append(batch['style_label'][idx].to(device))
    
    # Create result grid
    results = []
    
    # First add original images
    for i in range(num_samples):
        results.append(images[i])
    
    # For each style, perform transfer
    with torch.no_grad():
        for style in target_styles:
            style_batch = style.unsqueeze(0).repeat(num_samples, 1)
            outputs = model(images, style_labels[:num_samples], style_batch)
            reconstructions = outputs['reconstruction']
            
            # Add to results
            for i in range(num_samples):
                results.append(reconstructions[i])
    
    # Convert to tensor
    results = torch.stack(results)
    
    # Display results
    show_images(results, title="Style Transfer", save_path=save_path, nrow=num_samples)
    
    return results

def visualize_random_samples(model, condition, device, num_samples=64, save_path=None):
    """
    Visualize samples generated from random latent vectors
    
    Args:
        model (CVAE): CVAE model
        condition (Tensor): Condition label
        device (torch.device): Compute device
        num_samples (int): Number of samples
        save_path (str): Save path
    """
    model.eval()
    
    # Generate random samples
    with torch.no_grad():
        samples = model.sample(num_samples, condition.to(device), device)
    
    # Display samples
    show_images(samples, title="Random Samples", save_path=save_path)
    
    return samples

def plot_loss_curves(losses, save_path=None):
    """
    Plot training loss curves
    
    Args:
        losses (dict): Dictionary containing history of different loss terms
        save_path (str): Save path
    """
    plt.figure(figsize=(12, 8))
    
    for key, values in losses.items():
        plt.plot(values, label=key)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.close()

def visualize_latent_space(model, data_loader, device, num_samples=1000, save_path=None):
    """
    Visualize latent space
    
    Args:
        model (CVAE): CVAE model
        data_loader (DataLoader): Data loader
        device (torch.device): Compute device
        num_samples (int): Number of samples
        save_path (str): Save path
    """
    model.eval()
    
    # Collect latent vectors and labels
    latent_vectors = []
    char_labels = []
    style_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            images = batch['image'].to(device)
            char = batch['char_label']
            style = batch['style_label']
            
            # Get latent vectors
            mu, _ = model.encoder(images, style.to(device))
            
            latent_vectors.append(mu.cpu().numpy())
            char_labels.extend(char.numpy())
            style_labels.extend(style.numpy())
            
            if len(latent_vectors) * images.size(0) >= num_samples:
                break
    
    # Concatenate results from all batches
    latent_vectors = np.concatenate(latent_vectors, axis=0)[:num_samples]
    char_labels = np.array(char_labels)[:num_samples]
    style_labels = np.array(style_labels)[:num_samples]
    
    # Dimensionality reduction to 2D for visualization (if needed)
    if latent_vectors.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        latent_vectors_2d = pca.fit_transform(latent_vectors)
    else:
        latent_vectors_2d = latent_vectors
    
    # Visualization
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(latent_vectors_2d[:, 0], latent_vectors_2d[:, 1], 
                          c=char_labels, cmap='tab20', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Character')
    plt.title('Latent Space Visualization (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.close()
    
    return latent_vectors_2d, char_labels, style_labels 