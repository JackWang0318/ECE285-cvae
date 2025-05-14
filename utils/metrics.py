"""
Evaluation metrics module, used for evaluating model performance
"""

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def compute_ssim(img1, img2):
    """
    Compute Structural Similarity Index (SSIM)
    
    Args:
        img1 (Tensor): First image
        img2 (Tensor): Second image
        
    Returns:
        float: SSIM value
    """
    # Convert to Numpy and adjust range to [0,1]
    if torch.is_tensor(img1):
        img1 = (img1.cpu().detach().numpy() + 1) / 2
    if torch.is_tensor(img2):
        img2 = (img2.cpu().detach().numpy() + 1) / 2
    
    # If batch images, compute average SSIM
    if img1.ndim == 4:
        ssim_values = []
        for i in range(img1.shape[0]):
            ssim_val = ssim(img1[i, 0], img2[i, 0], data_range=1.0)
            ssim_values.append(ssim_val)
        return np.mean(ssim_values)
    else:
        # Single image
        return ssim(img1[0], img2[0], data_range=1.0)

def compute_psnr(img1, img2):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        img1 (Tensor): First image
        img2 (Tensor): Second image
        
    Returns:
        float: PSNR value
    """
    # Convert to Numpy and adjust range to [0,1]
    if torch.is_tensor(img1):
        img1 = (img1.cpu().detach().numpy() + 1) / 2
    if torch.is_tensor(img2):
        img2 = (img2.cpu().detach().numpy() + 1) / 2
    
    # If batch images, compute average PSNR
    if img1.ndim == 4:
        psnr_values = []
        for i in range(img1.shape[0]):
            psnr_val = psnr(img1[i, 0], img2[i, 0], data_range=1.0)
            psnr_values.append(psnr_val)
        return np.mean(psnr_values)
    else:
        # Single image
        return psnr(img1[0], img2[0], data_range=1.0)

def compute_mse(img1, img2):
    """
    Compute Mean Squared Error (MSE)
    
    Args:
        img1 (Tensor): First image
        img2 (Tensor): Second image
        
    Returns:
        float: MSE value
    """
    if torch.is_tensor(img1) and torch.is_tensor(img2):
        return F.mse_loss(img1, img2).item()
    else:
        # Convert to Numpy
        if torch.is_tensor(img1):
            img1 = img1.cpu().detach().numpy()
        if torch.is_tensor(img2):
            img2 = img2.cpu().detach().numpy()
        
        return np.mean((img1 - img2) ** 2)

def compute_l1(img1, img2):
    """
    Compute L1 distance
    
    Args:
        img1 (Tensor): First image
        img2 (Tensor): Second image
        
    Returns:
        float: L1 distance
    """
    if torch.is_tensor(img1) and torch.is_tensor(img2):
        return F.l1_loss(img1, img2).item()
    else:
        # Convert to Numpy
        if torch.is_tensor(img1):
            img1 = img1.cpu().detach().numpy()
        if torch.is_tensor(img2):
            img2 = img2.cpu().detach().numpy()
        
        return np.mean(np.abs(img1 - img2))

def evaluate_model(model, data_loader, device, num_samples=None):
    """
    Evaluate model performance
    
    Args:
        model (CVAE): CVAE model
        data_loader (DataLoader): Data loader
        device (torch.device): Compute device
        num_samples (int): Number of samples, if None uses the entire dataset
        
    Returns:
        dict: Dictionary containing various metrics
    """
    model.eval()
    
    ssim_values = []
    psnr_values = []
    mse_values = []
    l1_values = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if num_samples is not None and batch_idx * batch['image'].size(0) >= num_samples:
                break
                
            images = batch['image'].to(device)
            style_labels = batch['style_label'].to(device)
            
            # Generate reconstructions
            outputs = model(images, style_labels)
            reconstructions = outputs['reconstruction']
            
            # Compute metrics
            ssim_val = compute_ssim(images, reconstructions)
            psnr_val = compute_psnr(images, reconstructions)
            mse_val = compute_mse(images, reconstructions)
            l1_val = compute_l1(images, reconstructions)
            
            ssim_values.append(ssim_val)
            psnr_values.append(psnr_val)
            mse_values.append(mse_val)
            l1_values.append(l1_val)
    
    # Compute averages
    avg_ssim = np.mean(ssim_values)
    avg_psnr = np.mean(psnr_values)
    avg_mse = np.mean(mse_values)
    avg_l1 = np.mean(l1_values)
    
    return {
        'SSIM': avg_ssim,
        'PSNR': avg_psnr,
        'MSE': avg_mse,
        'L1': avg_l1
    }

def compare_models(models, data_loader, device, num_samples=None):
    """
    Compare performance of multiple models
    
    Args:
        models (dict): Dictionary of models, key is model name, value is model
        data_loader (DataLoader): Data loader
        device (torch.device): Compute device
        num_samples (int): Number of samples, if None uses the entire dataset
        
    Returns:
        dict: Dictionary containing performance metrics for each model
    """
    results = {}
    
    for name, model in models.items():
        metrics = evaluate_model(model, data_loader, device, num_samples)
        results[name] = metrics
    
    return results 