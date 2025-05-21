"""
Conditional Variational Autoencoder (CVAE) Model Definition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FontTransferCVAE(nn.Module):
    def __init__(self, label_dim=62, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim

        # Style Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # -> 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim + label_dim, 256)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),   # -> 28x28
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, label_onehot):
        z_cat = torch.cat([z, label_onehot], dim=1)
        return self.decoder(self.decoder_input(z_cat))

    def forward(self, style_img, label_onehot):
        mu, logvar = self.encode(style_img)
        z = self.reparameterize(mu, logvar)
        recon_img = self.decode(z, label_onehot)
        return recon_img, mu, logvar

def vae_loss_base(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_loss, recon_loss.item(), kl_loss.item()

class Encoder(nn.Module):
    """
    CVAE encoder network, mapping input image and condition label to the mean and variance of the latent space
    """
    
    def __init__(self, img_channels=1, img_size=28, latent_dim=128, condition_dim=1355, hidden_dims=None):
        """
        Initialization function
        
        Args:
            img_channels (int): Number of input image channels
            img_size (int): Width/height of input image
            latent_dim (int): Dimension of latent space
            condition_dim (int): Dimension of condition variable (number of style labels)
            hidden_dims (list): List of hidden layer dimensions
        """
        super(Encoder, self).__init__()
        
        self.img_size = img_size
        self.latent_dim = latent_dim
        
        # If hidden dimensions are not specified, use default values
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        
        # Build encoder convolutional layers
        modules = []
        in_channels = img_channels
        
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Calculate feature map size
        self.feature_size = img_size // (2 ** len(hidden_dims))
        self.feature_dims = hidden_dims[-1] * self.feature_size * self.feature_size
        
        # Condition embedding layer
        self.condition_embedding = nn.Linear(condition_dim, self.feature_dims)
        
        # Mean and variance projection layers
        self.fc_mu = nn.Linear(self.feature_dims * 2, latent_dim)
        self.fc_var = nn.Linear(self.feature_dims * 2, latent_dim)
        
    def forward(self, x, c):
        """
        Forward propagation
        
        Args:
            x (Tensor): Input image [B, C, H, W]
            c (Tensor): Condition label [B, condition_dim]
            
        Returns:
            tuple: (mu, log_var) Mean and log variance in latent space
        """
        # Encode image
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        
        # Encode condition
        c_embedding = self.condition_embedding(c)
        
        # Merge image features and condition features
        xc = torch.cat([x, c_embedding], dim=1)
        
        # Map to mean and variance in latent space
        mu = self.fc_mu(xc)
        log_var = self.fc_var(xc)
        
        return mu, log_var

class Decoder(nn.Module):
    """
    CVAE decoder network, decoding latent vector and condition label to reconstructed image
    """
    
    def __init__(self, img_channels=1, img_size=28, latent_dim=128, condition_dim=1355, hidden_dims=None):
        """
        Initialization function
        
        Args:
            img_channels (int): Number of output image channels
            img_size (int): Width/height of output image
            latent_dim (int): Dimension of latent space
            condition_dim (int): Dimension of condition variable (number of style labels)
            hidden_dims (list): List of hidden layer dimensions (used in reverse from encoder)
        """
        super(Decoder, self).__init__()
        
        self.img_size = img_size
        
        # If hidden dimensions are not specified, use default values
        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]
        
        # Calculate initial feature map size
        self.feature_size = img_size // (2 ** len(hidden_dims))
        self.feature_dims = hidden_dims[0] * self.feature_size * self.feature_size
        
        # Latent vector and condition projection layer
        self.latent_condition_proj = nn.Linear(latent_dim + condition_dim, self.feature_dims)
        
        # Build decoder transposed convolutional layers
        modules = []
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        
        # Final reconstruction layer
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[-1],
                    hidden_dims[-1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(hidden_dims[-1], img_channels, kernel_size=3, padding=1),
                nn.Tanh()
            )
        )
        
        self.decoder = nn.Sequential(*modules)
        
    def forward(self, z, c):
        """
        Forward propagation
        
        Args:
            z (Tensor): Latent vector [B, latent_dim]
            c (Tensor): Condition label [B, condition_dim]
            
        Returns:
            Tensor: Reconstructed image [B, C, H, W]
        """
        # Merge latent vector and condition
        zc = torch.cat([z, c], dim=1)
        
        # Project to initial feature map
        x = self.latent_condition_proj(zc)
        x = x.view(-1, self.feature_dims // (self.feature_size * self.feature_size), 
                  self.feature_size, self.feature_size)
        
        # Through decoder
        x = self.decoder(x)
        
        return x

class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) Model
    """
    
    def __init__(self, img_channels=1, img_size=28, latent_dim=128, condition_dim=2990, hidden_dims=None):
        """
        Initialization function
        
        Args:
            img_channels (int): Number of input/output image channels
            img_size (int): Width/height of input/output image
            latent_dim (int): Dimension of latent space
            condition_dim (int): Dimension of condition variable (number of style labels)
            hidden_dims (list): List of hidden layer dimensions
        """
        super(CVAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Initialize encoder and decoder
        self.encoder = Encoder(img_channels, img_size, latent_dim, condition_dim, hidden_dims)
        
        # Decoder uses encoder's hidden dimensions but in reverse order
        decoder_hidden_dims = self.encoder.encoder[0][0].out_channels
        if isinstance(hidden_dims, list):
            decoder_hidden_dims = hidden_dims[::-1]
        
        self.decoder = Decoder(img_channels, img_size, latent_dim, condition_dim, decoder_hidden_dims)
        
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick, generates differentiable random samples
        
        Args:
            mu (Tensor): Mean vector
            log_var (Tensor): Log variance vector
            
        Returns:
            Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x, c, target_c=None):
        """
        Forward propagation
        
        Args:
            x (Tensor): Input image [B, C, H, W]
            c (Tensor): Input condition label [B, condition_dim]
            target_c (Tensor, optional): Target condition label for style transfer. Defaults to None, which uses input condition
            
        Returns:
            dict: Contains reconstructed image, mean and variance of latent variables, etc.
        """
        # If target condition is not specified, use input condition
        if target_c is None:
            target_c = c
        
        # Encode
        mu, log_var = self.encoder(x, c)
        
        # Reparameterize sampling
        z = self.reparameterize(mu, log_var)
        
        # Decode
        reconstruction = self.decoder(z, target_c)
        
        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'log_var': log_var
        }
    
    def sample(self, num_samples, condition, device=torch.device("cpu")):
        """
        Generate samples
        
        Args:
            num_samples (int): Number of samples
            condition (Tensor): Condition label [B, condition_dim]
            device (torch.device): Compute device
            
        Returns:
            Tensor: Generated sample images
        """
        # Sample latent vectors from standard normal distribution
        z = torch.randn(num_samples, self.latent_dim).to(device)
        
        # Ensure batch size of condition matches sample count
        if condition.size(0) == 1:
            condition = condition.repeat(num_samples, 1)
        elif condition.size(0) != num_samples:
            raise ValueError(f"Condition batch size ({condition.size(0)}) does not match sample count ({num_samples})")
        
        # Decode sampled latent vectors
        samples = self.decoder(z, condition)
        
        return samples
    
    def reconstruct(self, x, source_c, target_c=None):
        """
        Reconstruct input image, optionally using a different target condition
        
        Args:
            x (Tensor): Input image [B, C, H, W]
            source_c (Tensor): Source condition label [B, condition_dim]
            target_c (Tensor, optional): Target condition label for style transfer. Defaults to None, which uses source condition
            
        Returns:
            Tensor: Reconstructed/style-transferred image
        """
        # If target condition is not specified, use source condition
        if target_c is None:
            target_c = source_c
        
        # Encode
        with torch.no_grad():
            mu, log_var = self.encoder(x, source_c)
            z = self.reparameterize(mu, log_var)
            
            # Decode with target condition
            reconstruction = self.decoder(z, target_c)
        
        return reconstruction 