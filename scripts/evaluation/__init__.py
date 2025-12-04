"""
Evaluation Modules
Provides comprehensive evaluation tools for LoRA models, quality assessment, and pipeline analysis.

Evaluation Components:

1. LoRA Testing & Evaluation:
   - auto_lora_evaluator: Automated LoRA checkpoint evaluation with comprehensive metrics
   - sdxl_lora_evaluator: SDXL-specific LoRA evaluation and testing
   - quick_lora_test: Quick LoRA testing for rapid validation
   - simple_lora_image_generator: Simple image generation with LoRA
   - auto_evaluate_checkpoints: Automated checkpoint evaluation workflow
   - evaluate_single_checkpoint: Detailed single checkpoint analysis

2. Quality Assessment:
   - lora_quality_metrics: Comprehensive quality metrics (CLIP score, consistency, etc.)
   - ai_quality_assessor: AI-powered quality assessment using VLMs
   - enhanced_quality_assessment: Enhanced quality metrics with advanced analysis
   - compare_lora_models: Compare multiple LoRA models side-by-side

3. Pipeline Analysis:
   - analyze_clustering_results: Analyze clustering quality and statistics
   - compare_segmentation_models: Compare different segmentation approaches
   - compare_inpainting_quality: Evaluate inpainting method quality
   - sdxl_multi_lora_compositor: Compose multiple LoRAs for complex scenes

Usage Examples:

# Quick LoRA test
python scripts/evaluation/quick_lora_test.py \
  --lora-path /path/to/lora.safetensors \
  --prompt "character description" \
  --output-dir outputs/

# Automated checkpoint evaluation
python scripts/evaluation/auto_evaluate_checkpoints.py \
  --checkpoint-dir /path/to/checkpoints \
  --test-prompts prompts.json \
  --output-dir evaluation_results/

# Compare LoRA models
python scripts/evaluation/compare_lora_models.py \
  --lora-paths model1.safetensors model2.safetensors \
  --output-dir comparison/

# Analyze clustering results
python scripts/evaluation/analyze_clustering_results.py \
  --cluster-dir /path/to/clusters \
  --output-report report.json
"""

# Individual scripts are standalone CLI tools

__all__ = [
    # Main modules are used as CLI scripts
]
