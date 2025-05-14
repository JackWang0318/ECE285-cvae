"""
Evaluate CVAE model
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from model import CVAE
from dataset import get_data_loaders
from utils import (
    visualize_reconstructions,
    visualize_style_transfer,
    visualize_random_samples,
    visualize_latent_space,
    evaluate_model,
    compare_models
)

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate CVAE model')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Model weights file path')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./evaluation',
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of evaluation samples')
    
    return parser.parse_args()

def load_model(model_path, device):
    """
    Load model
    
    Args:
        model_path (str): Model weights file path
        device (torch.device): Compute device
        
    Returns:
        CVAE: Loaded model
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration
    args = checkpoint['args']
    
    # Create model
    model = CVAE(
        img_channels=1,
        img_size=28,
        latent_dim=args['latent_dim'],
        condition_dim=1355,  # Number of styles in TMNIST dataset
        hidden_dims=args.get('hidden_dims', [32, 64, 128, 256])
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    return model

def main():
    """
    Main function
    """
    args = parse_args()
    
    # Set device
    device = torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model: {args.model_path}")
    model = load_model(args.model_path, device)
    
    # Create data loaders
    print("Creating data loaders...")
    _, val_loader, test_loader = get_data_loaders(
        args.data_dir,
        batch_size=args.batch_size
    )
    
    # Evaluate model
    print(f"Evaluating model on test set (samples: {args.num_samples})...")
    metrics = evaluate_model(model, test_loader, device, args.num_samples)
    
    # Print metrics
    print("\nEvaluation metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.6f}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(args.output_dir, 'metrics.csv'), index=False)
    
    # Visualize reconstruction results
    print("Visualizing reconstruction results...")
    reconstructions = visualize_reconstructions(
        model,
        test_loader,
        device,
        save_path=os.path.join(args.output_dir, 'reconstructions.png')
    )
    
    # Visualize style transfer results
    print("Visualizing style transfer results...")
    style_transfers = visualize_style_transfer(
        model,
        test_loader,
        device,
        num_samples=4,
        num_styles=6,
        save_path=os.path.join(args.output_dir, 'style_transfers.png')
    )
    
    # Visualize random sampling results
    print("Visualizing random sampling results...")
    # Get a sample style label for sampling
    batch = next(iter(test_loader))
    style_label = batch['style_label'][0].unsqueeze(0).to(device)
    
    random_samples = visualize_random_samples(
        model,
        style_label,
        device,
        num_samples=64,
        save_path=os.path.join(args.output_dir, 'random_samples.png')
    )
    
    # Visualize latent space
    print("Visualizing latent space...")
    latent_vectors, char_labels, style_labels = visualize_latent_space(
        model,
        test_loader,
        device,
        num_samples=args.num_samples,
        save_path=os.path.join(args.output_dir, 'latent_space.png')
    )
    
    print(f"Evaluation results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 