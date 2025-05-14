#!/bin/bash

# Create necessary directories
mkdir -p data
mkdir -p output
mkdir -p output/models
mkdir -p output/images
mkdir -p output/tensorboard
mkdir -p results/checkpoints
mkdir -p results/logs
mkdir -p results/images
mkdir -p results/metrics

# Set training parameters
DATA_DIR="./data"
OUTPUT_DIR="./results"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
LOGS_DIR="$OUTPUT_DIR/logs"
IMAGES_DIR="$OUTPUT_DIR/images"
TENSORBOARD_DIR="$LOGS_DIR/tensorboard"
BATCH_SIZE=128
EPOCHS=100
LEARNING_RATE=1e-3
LATENT_DIM=128
KLD_WEIGHT=0.1
SSIM_WEIGHT=0.05
HIDDEN_DIMS="32 64 128 256"

# Check for GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected, using GPU for training"
    GPU_FLAG="--gpu"
else
    echo "No GPU detected, using CPU for training"
    GPU_FLAG=""
fi

# Download and prepare dataset
echo "Preparing dataset..."
python scripts/train.py --prepare_data --data_dir $DATA_DIR

# Start training
echo "Starting CVAE training..."
python scripts/train.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --latent_dim $LATENT_DIM \
    --kld_weight $KLD_WEIGHT \
    --ssim_weight $SSIM_WEIGHT \
    --hidden_dims $HIDDEN_DIMS \
    $GPU_FLAG

# Copy final results to the results directory for reporting
# echo "Copying results to reporting directory..."
# cp $OUTPUT_DIR/images/loss_curves.png $IMAGES_DIR/
# cp $OUTPUT_DIR/images/recon_epoch_*.png $IMAGES_DIR/
# cp $OUTPUT_DIR/images/style_transfer_epoch_*.png $IMAGES_DIR/

echo "Training complete! Results saved to $OUTPUT_DIR" 