"""
Loss functions for CVAE model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss, L1Loss
from skimage.metrics import structural_similarity as ssim

def vae_loss(recon_x, x, mu, log_var, kld_weight=1.0):
    """
    Standard VAE loss function, including reconstruction loss and KL divergence
    
    Args:
        recon_x (Tensor): Reconstructed image [B, C, H, W]
        x (Tensor): Original image [B, C, H, W]
        mu (Tensor): Mean in latent space [B, latent_dim]
        log_var (Tensor): Log variance in latent space [B, latent_dim]
        kld_weight (float): Weight coefficient for KL divergence
        
    Returns:
        tuple: (total_loss, recon_loss, kld_loss)
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Total loss
    total_loss = recon_loss + kld_weight * kld_loss
    
    return total_loss, recon_loss, kld_loss

def ssim_loss(recon_x, x, window_size=11):
    """
    SSIM-based loss function
    
    Args:
        recon_x (Tensor): Reconstructed image [B, C, H, W]
        x (Tensor): Original image [B, C, H, W]
        window_size (int): SSIM window size
        
    Returns:
        Tensor: SSIM loss
    """
    # Convert images to [0,1] range
    recon_x = (recon_x + 1) / 2
    x = (x + 1) / 2
    
    # SSIM loss
    loss = 0
    for i in range(x.size(0)):
        recon_img = recon_x[i, 0].detach().cpu().numpy()
        orig_img = x[i, 0].detach().cpu().numpy()
        ssim_val = ssim(recon_img, orig_img, data_range=1.0, win_size=window_size)
        loss += (1 - ssim_val)
    
    return torch.tensor(loss / x.size(0), device=x.device)

class CVAELoss(nn.Module):
    """
    CVAE loss function class, combining different loss terms
    """
    
    def __init__(self, recon_weight=1.0, kld_weight=1.0, ssim_weight=0.0):
        """
        Initialization function
        
        Args:
            recon_weight (float): Weight for reconstruction loss
            kld_weight (float): Weight for KL divergence
            ssim_weight (float): Weight for SSIM loss
        """
        super(CVAELoss, self).__init__()
        
        self.recon_weight = recon_weight
        self.kld_weight = kld_weight
        self.ssim_weight = ssim_weight
        
        # Use MSE for reconstruction loss
        self.recon_loss_fn = MSELoss(reduction='sum')
        
    def forward(self, recon_x, x, mu, log_var):
        """
        Forward pass to compute loss
        
        Args:
            recon_x (Tensor): Reconstructed image [B, C, H, W]
            x (Tensor): Original image [B, C, H, W]
            mu (Tensor): Mean in latent space [B, latent_dim]
            log_var (Tensor): Log variance in latent space [B, latent_dim]
            
        Returns:
            dict: Dictionary containing various loss terms and total loss
        """
        batch_size = x.size(0)
        
        # Reconstruction loss
        recon_loss = self.recon_loss_fn(recon_x, x) / batch_size
        
        # KL divergence
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
        
        # Total loss
        total_loss = self.recon_weight * recon_loss + self.kld_weight * kld_loss
        
        # If SSIM loss is enabled
        ssim_loss_val = 0
        if self.ssim_weight > 0:
            ssim_loss_val = ssim_loss(recon_x, x)
            total_loss += self.ssim_weight * ssim_loss_val
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kld_loss': kld_loss,
            'ssim_loss': ssim_loss_val
        } 