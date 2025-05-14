"""
Generate model architecture summary for reporting
"""

import os
import argparse
import torch
from torchinfo import summary

from model import CVAE

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Generate CVAE model summary')
    
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent space dimension')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[32, 64, 128, 256],
                        help='Hidden layer dimensions')
    parser.add_argument('--output_dir', type=str, default='./results/logs',
                        help='Output directory')
    
    return parser.parse_args()

def main():
    """
    Main function
    """
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model
    model = CVAE(
        img_channels=1,
        img_size=28,
        latent_dim=args.latent_dim,
        condition_dim=1355,  # Number of styles in TMNIST dataset
        hidden_dims=args.hidden_dims
    )
    
    # Create inputs for summary
    batch_size = 8
    x = torch.randn(batch_size, 1, 28, 28)  # Example input images
    c = torch.zeros(batch_size, 1355)  # Example condition labels
    c[:, 0] = 1  # One-hot encoding of first style
    
    # Generate detailed summary
    model_summary = summary(model, input_data=[x, c], depth=4, verbose=2, col_names=[
        "input_size", "output_size", "num_params", "kernel_size", "mult_adds"
    ], col_width=20, row_settings=["var_names"], device=torch.device("cpu"))
    
    # Extract encoder and decoder summaries
    encoder_summary = summary(model.encoder, input_data=[x, c], depth=3, verbose=0)
    decoder_summary = summary(model.decoder, input_data=[
        torch.randn(batch_size, args.latent_dim), c
    ], depth=3, verbose=0)
    
    # Write to file
    with open(os.path.join(args.output_dir, 'model_summary.txt'), 'w') as f:
        f.write("# CVAE Model Architecture Summary\n\n")
        f.write("## Model Configuration\n")
        f.write(f"- Image channels: 1 (grayscale)\n")
        f.write(f"- Image size: 28x28\n")
        f.write(f"- Latent dimension: {args.latent_dim}\n")
        f.write(f"- Condition dimension: 1355 (number of font styles)\n")
        f.write(f"- Hidden dimensions: {args.hidden_dims}\n\n")
        
        f.write("## Complete Model\n")
        f.write(str(model_summary))
        f.write("\n\n")
        
        f.write("## Encoder Network\n")
        f.write(str(encoder_summary))
        f.write("\n\n")
        
        f.write("## Decoder Network\n")
        f.write(str(decoder_summary))
        f.write("\n\n")
        
        f.write("## Model Structure\n")
        f.write("```\n")
        f.write(str(model))
        f.write("\n```\n")
    
    print(f"Model summary saved to {os.path.join(args.output_dir, 'model_summary.txt')}")

if __name__ == "__main__":
    main() 