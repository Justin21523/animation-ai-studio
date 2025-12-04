"""
Batch Processing Modules
Provides large-scale batch processing tools for automation and orchestration.

Batch Processing Components:

1. Orchestration & Generation:
   - synthetic_data_orchestrator: Orchestrate multi-stage synthetic data generation
   - round_robin_generator: Round-robin prompt generation for balanced datasets
   - batch_lora_evaluation: Batch evaluate multiple LoRA checkpoints

2. Data Preparation:
   - generate_character_configs: Auto-generate character configuration files
   - prepare_sdxl_data: Prepare datasets for SDXL training
   - preprocess_images_for_sdxl: Preprocess images for SDXL requirements
   - check_dataset_repeats: Verify dataset repeat calculations

3. Caption Management:
   - expand_all_sdxl_captions: Expand captions for SDXL training (detailed descriptions)
   - simplify_captions_for_training: Simplify captions for better training
   - generate_seamonster_captions: Example caption generation workflow

4. Configuration & Optimization:
   - optimize_all_sdxl_configs: Optimize training configurations for SDXL
   - verify_sdxl_configs: Verify configuration correctness
   - calculate_optimal_epochs: Calculate optimal epoch count for datasets

5. Monitoring:
   - check_training_progress: Monitor training job progress

Usage Examples:

# Orchestrate synthetic data generation
python scripts/batch/synthetic_data_orchestrator.py \
  --config configs/synthetic_generation.yaml \
  --phase 1

# Generate character configs
python scripts/batch/generate_character_configs.py \
  --characters-dir /path/to/characters \
  --output-dir configs/characters/

# Expand captions for SDXL
python scripts/batch/expand_all_sdxl_captions.py \
  --data-dirs /path/to/training_data/* \
  --output-suffix "_expanded"

# Batch evaluate LoRAs
python scripts/batch/batch_lora_evaluation.py \
  --checkpoint-dirs /path/to/checkpoints/* \
  --output-dir evaluation_results/
"""

# Individual scripts are standalone CLI tools

__all__ = [
    # Main modules are used as CLI scripts
]
