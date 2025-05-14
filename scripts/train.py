"""
Train CVAE model
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from model import CVAE, CVAELoss
from dataset import get_data_loaders, prepare_dataset
from utils import (
    visualize_reconstructions,
    visualize_style_transfer,
    plot_loss_curves,
    evaluate_model
)

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Train CVAE model')
    
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent space dimension')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[32, 64, 128, 256],
                        help='Hidden layer dimensions')
    parser.add_argument('--kld_weight', type=float, default=0.1,
                        help='KL divergence weight')
    parser.add_argument('--ssim_weight', type=float, default=0.0,
                        help='SSIM loss weight')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU')
    parser.add_argument('--prepare_data', action='store_true',
                        help='Prepare dataset')
    
    return parser.parse_args()

def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, 
                args, output_dir, tensorboard_writer=None):
    """
    Train the model
    
    Args:
        model (CVAE): CVAE model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        optimizer (Optimizer): Optimizer
        loss_fn (Module): Loss function
        device (torch.device): Compute device
        args (Namespace): Command line arguments
        output_dir (str): Output directory
        tensorboard_writer (SummaryWriter): TensorBoard writer
        
    Returns:
        tuple: (trained_model, loss_history)
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Training history
    loss_history = {
        'train_loss': [],
        'val_loss': [],
        'recon_loss': [],
        'kld_loss': [],
        'ssim_loss': []
    }
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        recon_loss = 0
        kld_loss = 0
        ssim_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(device)
            style_labels = batch['style_label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images, style_labels)
            
            # Compute loss
            loss_dict = loss_fn(
                outputs['reconstruction'],
                images,
                outputs['mu'],
                outputs['log_var']
            )
            
            # Backward pass and optimization
            loss_dict['loss'].backward()
            optimizer.step()
            
            # Update progress bar
            train_loss += loss_dict['loss'].item()
            recon_loss += loss_dict['recon_loss'].item()
            kld_loss += loss_dict['kld_loss'].item()
            ssim_loss += loss_dict['ssim_loss'] if isinstance(loss_dict['ssim_loss'], float) else loss_dict['ssim_loss'].item()
            
            avg_loss = train_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': avg_loss})
        
        # Compute average losses
        train_loss /= len(train_loader)
        recon_loss /= len(train_loader)
        kld_loss /= len(train_loader)
        ssim_loss /= len(train_loader)
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images = batch['image'].to(device)
                style_labels = batch['style_label'].to(device)
                
                # Forward pass
                outputs = model(images, style_labels)
                
                # Compute loss
                loss_dict = loss_fn(
                    outputs['reconstruction'],
                    images,
                    outputs['mu'],
                    outputs['log_var']
                )
                
                val_loss += loss_dict['loss'].item()
        
        val_loss /= len(val_loader)
        
        # Record losses
        loss_history['train_loss'].append(train_loss)
        loss_history['val_loss'].append(val_loss)
        loss_history['recon_loss'].append(recon_loss)
        loss_history['kld_loss'].append(kld_loss)
        loss_history['ssim_loss'].append(ssim_loss)
        
        # Log to TensorBoard
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar('Loss/train', train_loss, epoch)
            tensorboard_writer.add_scalar('Loss/val', val_loss, epoch)
            tensorboard_writer.add_scalar('Loss/recon', recon_loss, epoch)
            tensorboard_writer.add_scalar('Loss/kld', kld_loss, epoch)
            tensorboard_writer.add_scalar('Loss/ssim', ssim_loss, epoch)
        
        # Print information
        print(f"Epoch {epoch+1}/{args.epochs}, "
              f"Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"Recon Loss: {recon_loss:.6f}, "
              f"KLD Loss: {kld_loss:.6f}, "
              f"SSIM Loss: {ssim_loss:.6f}")
        
        # Visualize reconstruction results
        if (epoch + 1) % 10 == 0 or epoch == 0:
            visualize_reconstructions(
                model,
                val_loader,
                device,
                save_path=os.path.join(images_dir, f'recon_epoch_{epoch+1}.png')
            )
            
            # Visualize style transfer results
            visualize_style_transfer(
                model,
                val_loader,
                device,
                save_path=os.path.join(images_dir, f'style_transfer_epoch_{epoch+1}.png')
            )
            
            # Save loss curves
            plot_loss_curves(
                {
                    'Train Loss': loss_history['train_loss'],
                    'Val Loss': loss_history['val_loss'],
                    'Recon Loss': loss_history['recon_loss'],
                    'KLD Loss': loss_history['kld_loss'],
                    'SSIM Loss': loss_history['ssim_loss']
                },
                save_path=os.path.join(images_dir, 'loss_curves.png')
            )
        
        # Save model
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args)
            }, os.path.join(models_dir, f'model_epoch_{epoch+1}.pth'))
    
    # Compute final evaluation metrics
    metrics = evaluate_model(model, val_loader, device)
    print("\nFinal evaluation metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.6f}")
    
    return model, loss_history

def main():
    """
    Main function
    """
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set device
    device = torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare dataset
    if args.prepare_data:
        print("Preparing dataset...")
        prepare_dataset(args.data_dir)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = get_data_loaders(
        args.data_dir,
        batch_size=args.batch_size
    )
    
    # Create model
    print("Creating model...")
    model = CVAE(
        img_channels=1,
        img_size=28,
        latent_dim=args.latent_dim,
        condition_dim=1355,  # Number of styles in TMNIST dataset
        hidden_dims=args.hidden_dims
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create loss function
    loss_fn = CVAELoss(
        recon_weight=1.0,
        kld_weight=args.kld_weight,
        ssim_weight=args.ssim_weight
    )
    
    # Create TensorBoard writer
    tensorboard_dir = os.path.join(args.output_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    
    # Train model
    print("Starting training...")
    model, loss_history = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device,
        args,
        args.output_dir,
        writer
    )
    
    # Close TensorBoard writer
    writer.close()
    
    print("Training complete!")

if __name__ == "__main__":
    main() 