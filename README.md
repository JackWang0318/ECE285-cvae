# Font Style Transfer with Conditional Variational Autoencoders (CVAE)
2025 Spring ECE285 Project

Hongjie Wang, Kejia Ruan

## Project Introduction
This project aims to re-implement a Conditional Variational Autoencoder (CVAE) architecture for font style transfer, focusing on generating diverse font styles for given characters. We use the Typography-MNIST (TMNIST) dataset, which contains 565,292 grayscale images of 1,812 unique glyphs rendered in 1,355 Google Fonts.

## Innovation Points
- Developing a CVAE model from scratch without using any pre-existing implementation
- Applying the model to the Typography-MNIST dataset, which has not been widely used in generative modeling
- Exploring different conditioning strategies for improved generation quality and conducting ablation studies on model variants

## Project Structure
```
.
├── dataset/             # Dataset related code and processing scripts
├── model/               # CVAE model definition and implementation
├── utils/               # Helper functions and tools
├── scripts/             # Python implementation scripts
│   ├── train.py         # Model training implementation
│   ├── evaluate.py      # Model evaluation implementation
│   ├── generate.py      # Font generation implementation
│   └── model_summary.py # Model architecture summary generator
├── results/             # Results and outputs
│   ├── checkpoints/     # Model checkpoints
│   ├── logs/            # Training logs and model summary
│   ├── images/          # Generated images and visualizations
│   └── metrics/         # Evaluation metrics
├── references/          # References and materials
├── train.sh             # Script for model training
├── evaluate.sh          # Script for model evaluation
├── generate.sh          # Script for font generation
├── generate_model_summary.sh # Script for creating model summary
├── prepare_report.sh    # Script to prepare materials for final report
└── requirements.txt     # Project dependencies
```

## Methodology
1. Data Preparation: Utilize the TMNIST dataset
2. Model Architecture: Implement a CVAE architecture where the encoder maps an image and style label to a latent vector, and the decoder generates glyphs conditioned on the latent vector and target style
3. Training: Train the model by minimizing reconstruction loss and KL divergence
4. Evaluation: Evaluate the generated glyphs using SSIM, pixel-wise loss, and visual quality checks
5. Ablation Study: Explore different conditioning schemes and evaluate their impact through controlled experiments

## Usage Instructions

### Setup
```bash
# Clone the repository
git clone ...
cd ...

# Install dependencies
conda create -n ECE285-cvae python=3.10
conda activate ECE285-cvae
pip install -r requirements.txt
```

### Training
```bash
# Start training with default parameters
./train.sh

# Results will be saved to results/ directory
```

### Evaluation
```bash
# Evaluate trained model
./evaluate.sh

# Results will be saved to results/metrics/ and results/images/evaluation/
```

### Generation
```bash
# Generate font style transfers
./generate.sh

# Results will be saved to results/images/generated/
```

### Model Summary
```bash
# Generate model architecture summary
./generate_model_summary.sh

# Summary will be saved to results/logs/model_summary.txt
```

### Preparing Report Materials
```bash
# Collect and organize materials for the final report
./prepare_report.sh

# Materials will be organized in final_report_materials/
```

## Model Architecture
The CVAE model consists of:
1. **Encoder**: A convolutional network that maps input images and style conditions to a latent distribution
2. **Decoder**: A transposed convolutional network that reconstructs images from latent vectors and target style conditions
3. **Latent Space**: A disentangled representation where character identity and font style are separated

The model is trained to minimize reconstruction loss and KL divergence, enabling both high-quality reconstruction and diverse generation capabilities.

## Results
After training, the model can perform:
- Reconstruction of input glyphs
- Style transfer between different fonts
- Latent space interpolation for smooth style transitions
- Random sampling of new font styles

Detailed visualizations and metrics are available in the results directory after running the evaluation and generation scripts.
