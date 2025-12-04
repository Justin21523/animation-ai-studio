#!/bin/bash
# GPT-SoVITS Luca Voice Training - Quick Start Script
# Usage: bash scripts/synthesis/tts/train_luca_voice.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}GPT-SoVITS Luca Voice Training Pipeline${NC}"
echo -e "${BLUE}================================================${NC}\n"

# Activate voice_env
echo -e "${YELLOW}[1/6] Activating voice_env environment...${NC}"
export PATH="/home/b0979/.conda/envs/voice_env/bin:$PATH"
which python
python --version
echo -e "${GREEN}✓ Environment activated${NC}\n"

# Check if models are downloaded
echo -e "${YELLOW}[2/6] Checking pretrained models...${NC}"
BERT_MODEL="/mnt/c/AI_LLM_projects/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
HUBERT_MODEL="/mnt/c/AI_LLM_projects/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base"

if [ ! -d "$BERT_MODEL" ] || [ ! -f "$BERT_MODEL/pytorch_model.bin" ]; then
    echo -e "${RED}✗ BERT model not found or incomplete${NC}"
    echo -e "${YELLOW}  Waiting for download to complete...${NC}"
    echo -e "${YELLOW}  Model location: $BERT_MODEL${NC}"
    exit 1
fi

if [ ! -d "$HUBERT_MODEL" ] || [ ! -f "$HUBERT_MODEL/pytorch_model.bin" ]; then
    echo -e "${RED}✗ HuBERT model not found or incomplete${NC}"
    echo -e "${YELLOW}  Waiting for download to complete...${NC}"
    echo -e "${YELLOW}  Model location: $HUBERT_MODEL${NC}"
    exit 1
fi

echo -e "${GREEN}✓ BERT model found${NC}"
echo -e "${GREEN}✓ HuBERT model found${NC}\n"

# Check GPU
echo -e "${YELLOW}[3/6] Checking GPU availability...${NC}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits
echo -e "${GREEN}✓ GPU ready${NC}\n"

# Check training data
echo -e "${YELLOW}[4/6] Checking training data...${NC}"
TRAIN_LIST="/mnt/c/AI_LLM_projects/GPT-SoVITS/logs/Luca/train.list"
VAL_LIST="/mnt/c/AI_LLM_projects/GPT-SoVITS/logs/Luca/val.list"
AUDIO_DIR="/mnt/c/AI_LLM_projects/GPT-SoVITS/logs/Luca/0-audio"

if [ ! -f "$TRAIN_LIST" ]; then
    echo -e "${RED}✗ Training list not found: $TRAIN_LIST${NC}"
    exit 1
fi

if [ ! -f "$VAL_LIST" ]; then
    echo -e "${RED}✗ Validation list not found: $VAL_LIST${NC}"
    exit 1
fi

if [ ! -d "$AUDIO_DIR" ]; then
    echo -e "${RED}✗ Audio directory not found: $AUDIO_DIR${NC}"
    exit 1
fi

TRAIN_COUNT=$(wc -l < "$TRAIN_LIST")
VAL_COUNT=$(wc -l < "$VAL_LIST")
AUDIO_COUNT=$(ls "$AUDIO_DIR"/*.wav 2>/dev/null | wc -l)

echo -e "${GREEN}✓ Training samples: $TRAIN_COUNT${NC}"
echo -e "${GREEN}✓ Validation samples: $VAL_COUNT${NC}"
echo -e "${GREEN}✓ Audio files: $AUDIO_COUNT${NC}\n"

# Start training
echo -e "${YELLOW}[5/6] Starting GPT-SoVITS training pipeline...${NC}"
echo -e "${BLUE}This will take approximately 2.5-4 hours${NC}"
echo -e "${BLUE}Steps: Preprocessing (3 steps) + Training (2 stages)${NC}\n"

LOG_FILE="logs/luca_gpt_sovits_training_$(date +%Y%m%d_%H%M%S).log"

python scripts/synthesis/tts/gpt_sovits_full_pipeline.py \
    --character Luca \
    --samples data/films/luca/voice_samples_auto/by_character/Luca \
    --output models/voices/luca/gpt_sovits \
    --s1-epochs 15 \
    --s2-epochs 10 \
    --batch-size 8 \
    2>&1 | tee "$LOG_FILE"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "\n${GREEN}[6/6] ✓ Training completed successfully!${NC}"
    echo -e "${GREEN}Log file: $LOG_FILE${NC}"
    echo -e "${GREEN}Models saved to: models/voices/luca/gpt_sovits/${NC}\n"
else
    echo -e "\n${RED}[6/6] ✗ Training failed${NC}"
    echo -e "${RED}Check log file: $LOG_FILE${NC}\n"
    exit 1
fi

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}Next steps:${NC}"
echo -e "${BLUE}1. Test the trained model with inference script${NC}"
echo -e "${BLUE}2. Generate sample audio for quality verification${NC}"
echo -e "${BLUE}3. Deploy the model for voice synthesis${NC}"
echo -e "${BLUE}================================================${NC}"
