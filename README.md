# Font Style Transfer with Conditional Variational Autoencoders (CVAE)
2025 Spring ECE285 Project

Hongjie Wang, Kejia Ruan

## Project Introduction
This project aims to implement a Conditional Variational Autoencoder (CVAE) architecture for font style transfer, focusing on generating diverse font styles for given characters. We use the Typography-MNIST (TMNIST) Alpha dataset.

## Innovation Points
- Developing a CVAE model from scratch without using any pre-existing implementation
- Applying the model to the Typography-MNIST dataset, which has not been widely used in generative modeling
- Exploring different conditioning strategies for improved generation quality and conducting ablation studies on model variants

## Project Structure
```
.
├── data/                # Raw and processed data storage
├── dataset/             # Dataset related code and processing scripts
├── model/               # CVAE model definition and implementation
├── utils/               # Helper functions and tools
├── results/             # Results and outputs
├── reports/            # Project reports and documentation
├── references/          # References and materials
├── Dataset_EDA.ipynb              # Dataset exploration and analysis notebook
├── Dataset_preprocess.ipynb       # Data preprocessing notebook
├── Dataset_Split_to_Fonts.ipynb   # Font splitting and organization notebook
├── CVAE_pipeline.ipynb           # Main CVAE training and evaluation pipeline
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
