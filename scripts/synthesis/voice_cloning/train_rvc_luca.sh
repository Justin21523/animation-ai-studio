#!/bin/bash
#
# RVC Training Script for Luca Character
# Automated training pipeline for Retrieval-based Voice Conversion
#

set -e  # Exit on error

# Configuration - Optimized for RTX 5080 16GB
export PATH="/home/b0979/.conda/envs/voice_env/bin:$PATH"
RVC_DIR="/mnt/c/AI_LLM_projects/RVC"
EXPERIMENT_NAME="Luca"
DATASET_DIR="$RVC_DIR/datasets/$EXPERIMENT_NAME"
SAMPLE_RATE=40000
F0_METHOD="rmvpe"
BATCH_SIZE=16          # Increased from 8 to 16 for RTX 5080
EPOCHS=500             # Increased for better quality
SAVE_EVERY=50
CACHE_GPU=1            # Enable GPU caching

echo "========================================"
echo "RVC Training Pipeline for $EXPERIMENT_NAME"
echo "========================================"
echo "Dataset: $DATASET_DIR"
echo "Sample Rate: $SAMPLE_RATE Hz"
echo "F0 Method: $F0_METHOD"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo ""

# Change to RVC directory
cd "$RVC_DIR"

# Step 1: Preprocess audio
echo "Step 1/5: Preprocessing audio files..."
python infer/modules/train/preprocess.py \
    "$DATASET_DIR" \
    "$SAMPLE_RATE" \
    4 \
    "$RVC_DIR/logs/$EXPERIMENT_NAME" \
    False \
    3.0

echo "✓ Audio preprocessing complete"

# Step 2: Extract features (HuBERT)
echo ""
echo "Step 2/5: Extracting HuBERT features..."
python infer/modules/train/extract_feature_print.py \
    cuda \
    1 \
    0 \
    "$RVC_DIR/logs/$EXPERIMENT_NAME" \
    v2 \
    True

echo "✓ Feature extraction complete"

# Step 3: Extract F0 (pitch)
echo ""
echo "Step 3/5: Extracting F0 (pitch) using $F0_METHOD..."
python infer/modules/train/extract_f0_print.py \
    "$RVC_DIR/logs/$EXPERIMENT_NAME" \
    4 \
    "$F0_METHOD"

echo "✓ F0 extraction complete"

# Step 4: Train model
echo ""
echo "Step 4/5: Training RVC model..."
echo "This will take approximately 2-4 hours..."
echo "Training log: $RVC_DIR/logs/$EXPERIMENT_NAME/train.log"
echo ""

python infer/modules/train/train.py \
    -e "$EXPERIMENT_NAME" \
    -sr "$SAMPLE_RATE" \
    -f0 1 \
    -bs "$BATCH_SIZE" \
    -g 0 \
    -te "$EPOCHS" \
    -se "$SAVE_EVERY" \
    -pg "$RVC_DIR/assets/pretrained_v2/f0G40k.pth" \
    -pd "$RVC_DIR/assets/pretrained_v2/f0D40k.pth" \
    -l 1 \
    -c 0 \
    -sw 0 \
    -v v2 \
    2>&1 | tee "$RVC_DIR/logs/$EXPERIMENT_NAME/train.log"

echo "✓ Model training complete"

# Step 5: Build FAISS index
echo ""
echo "Step 5/5: Building FAISS index for retrieval..."
python infer/modules/train/train_index.py \
    "$RVC_DIR/logs/$EXPERIMENT_NAME"

echo "✓ FAISS index built"

# Summary
echo ""
echo "========================================"
echo "✅ RVC Training Complete!"
echo "========================================"
echo ""
echo "Model files location:"
echo "  $RVC_DIR/logs/$EXPERIMENT_NAME/"
echo ""
echo "Generated files:"
echo "  - Model checkpoint: *.pth"
echo "  - FAISS index: added_*.index"
echo ""
echo "Next steps:"
echo "1. Test the model with XTTS-generated audio"
echo "2. Create integrated XTTS + RVC pipeline"
echo "3. Adjust parameters if needed"
echo ""
