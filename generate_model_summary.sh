#!/bin/bash

# Create necessary directories
mkdir -p results/logs

# Set model parameters (should match training parameters)
LATENT_DIM=128
HIDDEN_DIMS="32 64 128 256"
OUTPUT_DIR="./results/logs"

echo "Generating CVAE model architecture summary..."

# Install torchinfo if not already installed
pip install -q torchinfo

# Generate model summary
python scripts/model_summary.py \
    --latent_dim $LATENT_DIM \
    --hidden_dims $HIDDEN_DIMS \
    --output_dir $OUTPUT_DIR

echo "Model summary generated at $OUTPUT_DIR/model_summary.txt" 