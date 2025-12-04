# Data Augmentation Configuration Files

This directory contains configuration files for the Data Augmentation Pipeline. These YAML configurations define augmentation pipelines with different intensity levels and customization options.

## Available Configurations

### 1. Light Augmentation (`light_augmentation_config.yaml`)

**Use Case**: High-quality datasets where images are already clean and well-composed.

**Characteristics**:
- Minimal color adjustments (±10% brightness/contrast)
- Simple geometric transformations (horizontal flip only)
- 50% application probability for most augmentations
- Preserves image quality (95% JPEG quality)

**Best For**:
- Professional photography datasets
- Pre-processed, high-quality training data
- When you want to maintain maximum fidelity

### 2. Medium Augmentation (`medium_augmentation_config.yaml`)

**Use Case**: Standard training datasets requiring balanced augmentation.

**Characteristics**:
- Moderate color adjustments (±20% brightness/contrast/saturation)
- Multiple geometric transformations (flips, rotation ±15°)
- Subtle blur effects (20% probability)
- 70% application probability for color augmentations

**Best For**:
- General-purpose machine learning tasks
- Balanced approach between diversity and quality
- Most common use case

### 3. Strong Augmentation (`strong_augmentation_config.yaml`)

**Use Case**: Limited training data requiring maximum variation.

**Characteristics**:
- Aggressive color adjustments (±30% brightness/contrast/saturation)
- Extensive geometric transformations (rotation ±30°, crops, scaling)
- Blur, noise, and advanced effects (cutout, posterize)
- High application probabilities (up to 80%)
- Slightly lower output quality (90% JPEG quality)

**Best For**:
- Small datasets needing maximum diversity
- Data augmentation for few-shot learning
- When quantity matters more than individual quality

### 4. Custom Configuration Template (`custom_augmentation_config.yaml`)

**Use Case**: Create your own augmentation pipeline tailored to specific needs.

**Characteristics**:
- Fully documented with all available augmentation options
- Comments explaining each parameter
- Many augmentations commented out for easy activation
- Comprehensive usage notes

**Best For**:
- Custom datasets with specific requirements
- Experimentation and fine-tuning
- Learning about available augmentation options

## Configuration Structure

All configuration files follow this structure:

```yaml
metadata:
  name: "Configuration Name"
  description: "Description"
  version: "1.0.0"
  author: "Author Name"

augmentations:
  - name: "augmentation_name"
    type: "augmentation_type"  # color, geometric, noise, blur, advanced
    probability: 0.5  # 0.0 to 1.0
    parameters:
      # Type-specific parameters

output:
  format: "jpg"  # jpg, png, webp
  quality: 95  # 1-100
  preserve_metadata: true

processing:
  seed: null  # null = random, or integer for reproducibility
  num_workers: 4
  batch_size: 16

validation:
  min_image_size: [64, 64]
  max_image_size: [4096, 4096]
  allowed_formats: ["jpg", "jpeg", "png", "webp"]
```

## Augmentation Types

### Color Augmentations
- `random_brightness`: Adjust brightness randomly
- `random_contrast`: Adjust contrast randomly
- `random_saturation`: Adjust saturation randomly
- `random_hue_shift`: Shift hue (color rotation)

### Geometric Augmentations
- `horizontal_flip`: Mirror image horizontally
- `vertical_flip`: Mirror image vertically
- `rotate`: Rotate by random angle
- `random_crop`: Crop random region
- `center_crop`: Crop center region
- `random_scale`: Scale/zoom randomly

### Noise Augmentations
- `gaussian_noise`: Add Gaussian noise
- `salt_and_pepper`: Add salt and pepper noise

### Blur Augmentations
- `random_blur`: Apply random blur effect
  - Types: gaussian, box, motion

### Advanced Augmentations
- `cutout`: Random rectangular masking
- `posterize`: Reduce color depth
- `solarize`: Invert pixels above threshold
- `equalize`: Histogram equalization
- `autocontrast`: Auto adjust contrast

## Usage Examples

### Using Preset Configurations

**Single Image with Light Preset**:
```bash
python scripts/automation/scenarios/data_augmentation.py single \
  --input input.jpg \
  --output output.jpg \
  --preset light \
  --seed 42
```

**Batch Processing with Medium Preset**:
```bash
python scripts/automation/scenarios/data_augmentation.py batch \
  --input-dir ./images \
  --output-dir ./augmented \
  --preset medium \
  --num-per-image 3
```

### Using Custom Configuration Files

**Single Image with Custom Config**:
```bash
python scripts/automation/scenarios/data_augmentation.py single \
  --input input.jpg \
  --output output.jpg \
  --config configs/automation/augmentation/custom_augmentation_config.yaml \
  --seed 42
```

**Batch Processing with Custom Config**:
```bash
python scripts/automation/scenarios/data_augmentation.py batch \
  --input-dir ./images \
  --output-dir ./augmented \
  --config configs/automation/augmentation/strong_augmentation_config.yaml \
  --num-per-image 5
```

## Tips for Choosing Configuration

1. **Start with Medium**: The medium preset works well for most use cases.

2. **Use Light for Quality Data**: If your dataset is already high-quality and diverse, use light augmentation to avoid degrading quality.

3. **Use Strong for Small Datasets**: When you have limited data (< 100 images per class), use strong augmentation to maximize diversity.

4. **Customize for Specific Domains**: Create custom configurations for domain-specific requirements (e.g., medical imaging, satellite imagery).

5. **Test Before Full Processing**: Always test with a small batch first to ensure the augmentation level is appropriate.

6. **Use Seeds for Reproducibility**: Specify a seed value when you need consistent results across runs.

## Creating Custom Configurations

1. Copy `custom_augmentation_config.yaml` to a new file
2. Uncomment desired augmentations
3. Adjust probability and parameter values
4. Test with small batch
5. Iterate based on results

## Parameters Guide

### Probability (0.0 to 1.0)
- `0.0`: Never apply
- `0.3`: Apply to 30% of images
- `0.5`: Apply to 50% of images (balanced)
- `0.7`: Apply to 70% of images
- `1.0`: Always apply

### Color Factor Ranges
- `[0.9, 1.1]`: Subtle (±10%)
- `[0.8, 1.2]`: Moderate (±20%)
- `[0.7, 1.3]`: Strong (±30%)
- `1.0` = no change

### Rotation Angles
- `[-5, 5]`: Very subtle
- `[-15, 15]`: Moderate
- `[-30, 30]`: Strong
- `[-45, 45]`: Very strong

### Crop Fractions
- `0.95`: Keep 95% (minimal crop)
- `0.8-0.9`: Moderate crop
- `0.7-0.8`: Strong crop

## Validation

The pipeline validates:
- Image format compatibility
- Minimum/maximum image sizes
- Parameter ranges
- File existence and permissions

Invalid images are skipped with warnings logged.

## Output Statistics

After batch processing, the pipeline generates `augmentation_stats.json`:

```json
{
  "total_images": 100,
  "total_augmentations": 300,
  "successful": 298,
  "failed": 2,
  "augmentations_per_image": 3,
  "output_dir": "/path/to/output"
}
```

## Troubleshooting

**Issue**: Augmented images look too different from originals
- **Solution**: Reduce probability values or use lighter preset

**Issue**: Not enough variation in augmented images
- **Solution**: Increase probability values or use stronger preset

**Issue**: Output images have artifacts
- **Solution**: Reduce blur/noise intensities, increase output quality

**Issue**: Processing is slow
- **Solution**: Reduce batch_size, decrease num_workers, or use fewer augmentations

## Reference

For complete documentation, see:
- [Data Augmentation Pipeline Documentation](../../../docs/automation/PHASE3_DATA_AUGMENTATION.md)
- [Tool Implementation](../../../scripts/automation/scenarios/data_augmentation.py)
