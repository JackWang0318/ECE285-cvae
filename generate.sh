#!/bin/bash

# Create necessary directories
mkdir -p results/images/generated

# Set generation parameters
MODEL_PATH="./results/checkpoints/model_epoch_100.pth" # Path to the best model
DATA_DIR="./data"
OUTPUT_DIR="./results/images/generated"
BATCH_SIZE=16
NUM_SAMPLES=10
MODE="transfer"  # Options: random, interpolation, transfer

# Check for GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected, using GPU for generation"
    GPU_FLAG="--gpu"
else
    echo "No GPU detected, using CPU for generation"
    GPU_FLAG=""
fi

# Start generation
echo "Starting CVAE generation..."
python scripts/generate.py \
    --model_path $MODEL_PATH \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --num_samples $NUM_SAMPLES \
    --mode $MODE \
    $GPU_FLAG

# Extract model name from path for output naming
MODEL_NAME=$(basename $MODEL_PATH .pth)

# Create a timestamp for the generation
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Rename outputs with more descriptive names
echo "Organizing generation results for reporting..."
mkdir -p $OUTPUT_DIR/${MODEL_NAME}_${MODE}_${TIMESTAMP}
mv $OUTPUT_DIR/*.png $OUTPUT_DIR/${MODEL_NAME}_${MODE}_${TIMESTAMP}/

echo "Generation complete! Results saved to $OUTPUT_DIR/${MODEL_NAME}_${MODE}_${TIMESTAMP}" 