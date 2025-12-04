"""
Inpainting Modules
Provides background inpainting tools for character-removed frames.

Inpainting Methods:
- LaMa: Fast, high-quality traditional inpainting (RECOMMENDED)
- PowerPaint: Text-guided diffusion-based inpainting
- OpenCV: Simple fallback methods

Components:
- background_inpainting: General background inpainting
- sam2_background_inpainting: Inpainting for SAM2-segmented backgrounds (OPTIMIZED)
- character_inpainting: Character-aware inpainting with PowerPaint
"""

# Individual scripts are standalone CLI tools

__all__ = [
    # Main modules are used as CLI scripts
]
