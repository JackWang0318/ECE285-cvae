#!/bin/bash

# Create a directory for the final report materials
REPORT_DIR="./final_report_materials"
mkdir -p $REPORT_DIR

# Subdirectories for different types of materials
mkdir -p $REPORT_DIR/metrics
mkdir -p $REPORT_DIR/figures
mkdir -p $REPORT_DIR/tables
mkdir -p $REPORT_DIR/models

# Colors for better visualization in terminal
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}Preparing CVAE report materials...${NC}"

# Copy loss curves
echo "Copying learning curves..."
cp results/images/loss_curves.png $REPORT_DIR/figures/

# Copy best model evaluation results
echo "Copying evaluation metrics and visualizations..."
cp results/metrics/*.csv $REPORT_DIR/metrics/
cp results/images/evaluation/*_latent_space.png $REPORT_DIR/figures/
cp results/images/evaluation/*_reconstructions.png $REPORT_DIR/figures/
cp results/images/evaluation/*_style_transfers.png $REPORT_DIR/figures/

# Copy best generation results
echo "Copying generation results..."
cp -r results/images/generated/* $REPORT_DIR/figures/

# Copy model architecture summary (if available)
if [ -f "results/logs/model_summary.txt" ]; then
    cp results/logs/model_summary.txt $REPORT_DIR/models/
fi

# Create a README file with report materials description
cat > $REPORT_DIR/README.md << EOL
# CVAE Font Style Transfer Report Materials

This directory contains supporting materials for the final report on the CVAE font style transfer project.

## Directory Structure

- **figures/**: Contains all visualizations including:
  - Learning curves showing training and validation loss
  - Latent space visualizations
  - Reconstruction examples
  - Style transfer examples
  - Generated samples with different modes

- **metrics/**: Contains CSV files with quantitative evaluation metrics

- **tables/**: Space for formatted tables for the report

- **models/**: Model architecture summary and details

## Key Results

The CVAE model was trained to encode font style and character information in separate parts of the latent space, enabling style transfer between different fonts. Key visualizations show:

1. Character reconstruction quality
2. Style transfer capabilities 
3. Latent space organization
4. Generative capabilities

For more details, see the final report.
EOL

echo -e "${GREEN}Report materials prepared in $REPORT_DIR${NC}"
echo "You can now use these materials in your final report document." 