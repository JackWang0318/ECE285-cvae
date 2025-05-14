#!/bin/bash

# Create necessary directories
mkdir -p results/metrics
mkdir -p results/images/evaluation

# Set evaluation parameters
MODEL_PATH="./results/checkpoints/model_epoch_100.pth" # Path to the best model
DATA_DIR="./data"
OUTPUT_DIR="./results"
METRICS_DIR="$OUTPUT_DIR/metrics"
IMAGES_DIR="$OUTPUT_DIR/images/evaluation"
BATCH_SIZE=64
NUM_SAMPLES=1000

# Check for GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected, using GPU for evaluation"
    GPU_FLAG="--gpu"
else
    echo "No GPU detected, using CPU for evaluation"
    GPU_FLAG=""
fi

# Start evaluation
echo "Starting CVAE evaluation..."
python scripts/evaluate.py \
    --model_path $MODEL_PATH \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --num_samples $NUM_SAMPLES \
    $GPU_FLAG

# Extract model name from path for report file naming
MODEL_NAME=$(basename $MODEL_PATH .pth)

# Save report data in dedicated formats for the final report
# echo "Preparing evaluation data for reporting..."
# cp $OUTPUT_DIR/metrics.csv $METRICS_DIR/${MODEL_NAME}_metrics.csv
# cp $OUTPUT_DIR/reconstructions.png $IMAGES_DIR/${MODEL_NAME}_reconstructions.png
# cp $OUTPUT_DIR/style_transfers.png $IMAGES_DIR/${MODEL_NAME}_style_transfers.png
# cp $OUTPUT_DIR/random_samples.png $IMAGES_DIR/${MODEL_NAME}_random_samples.png
# cp $OUTPUT_DIR/latent_space.png $IMAGES_DIR/${MODEL_NAME}_latent_space.png

echo "Evaluation complete! Results saved to $OUTPUT_DIR" 