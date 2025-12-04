# Dataset Quality Inspector - Design Document

## Overview

First complete CPU-only automation scenario that demonstrates integration of:
- Agent Framework (analysis and decision-making)
- RAG System (knowledge retrieval)
- VLM modules (visual analysis)
- Orchestration Layer (workflow execution)
- Safety System (resource management)

## Purpose

Automated inspection of training datasets for LoRA model training, ensuring:
- Image quality meets training requirements
- Dataset distribution is balanced
- Captions are consistent and accurate
- No duplicates or corrupted files
- Proper format and structure

## Architecture

```
Dataset Quality Inspector
│
├── Inspector Core
│   ├── Quality Analyzers
│   │   ├── ImageQualityAnalyzer (blur, noise, resolution)
│   │   ├── DistributionAnalyzer (balance, diversity)
│   │   └── CaptionAnalyzer (consistency, accuracy)
│   │
│   ├── Detection Modules
│   │   ├── DuplicateDetector (perceptual hashing)
│   │   ├── CorruptionDetector (file integrity)
│   │   └── FormatValidator (structure verification)
│   │
│   └── Report Generator
│       ├── IssueCollector
│       ├── RecommendationEngine (Agent-powered)
│       └── ReportFormatter (JSON, Markdown, HTML)
│
├── Integration Layer
│   ├── Agent Integration (for analysis and recommendations)
│   ├── RAG Integration (for best practices lookup)
│   └── VLM Integration (for visual quality assessment)
│
└── Workflow Definition
    ├── YAML configuration
    ├── Task dependencies
    └── Checkpoint/resume support
```

## Components

### 1. Image Quality Analyzer

**Purpose:** Assess technical image quality

**Checks:**
- Blur detection (Laplacian variance)
- Noise level (standard deviation analysis)
- Resolution validation (minimum size requirements)
- Aspect ratio verification
- Color space validation (RGB)
- Bit depth check

**Outputs:**
- Quality scores (0-100)
- Pass/fail status
- Specific issues detected

**CPU-Only:** Uses OpenCV, NumPy (no GPU required)

### 2. Distribution Analyzer

**Purpose:** Ensure balanced and diverse dataset

**Checks:**
- Class distribution (if categorized)
- File count per category
- Image diversity (histogram comparison)
- Pose/view diversity (if applicable)
- Temporal distribution (if from video)

**Outputs:**
- Distribution statistics
- Imbalance warnings
- Diversity scores

**Integration:** Uses RAG for best practice thresholds

### 3. Caption Analyzer

**Purpose:** Validate caption quality and consistency

**Checks:**
- Caption exists for each image
- Length within acceptable range
- Keyword consistency
- Grammar and spelling (basic)
- Format compliance (tokens, special characters)

**Outputs:**
- Caption quality scores
- Consistency metrics
- Suggested improvements

**Integration:**
- Uses Agent for semantic analysis
- Uses RAG for caption best practices

### 4. Duplicate Detector

**Purpose:** Find duplicate or near-duplicate images

**Method:**
- Perceptual hashing (pHash)
- SSIM comparison for close matches
- Filename similarity

**Outputs:**
- Duplicate pairs/groups
- Similarity scores
- Recommendations for removal

**CPU-Only:** Pure Python implementation

### 5. Corruption Detector

**Purpose:** Find corrupted or invalid files

**Checks:**
- File integrity (can be opened)
- Image decoding (no errors)
- Expected format (JPEG, PNG)
- Metadata validation

**Outputs:**
- List of corrupted files
- Specific error messages

### 6. Format Validator

**Purpose:** Verify dataset structure

**Checks:**
- Required directories present
- Naming conventions followed
- metadata.json exists and valid
- Caption files match images
- File permissions

**Outputs:**
- Structure compliance report
- Missing elements
- Format violations

### 7. Recommendation Engine

**Purpose:** AI-powered recommendations for improvements

**Process:**
1. Collect all analysis results
2. Query RAG for best practices
3. Ask Agent to generate recommendations
4. Prioritize by severity

**Outputs:**
- Prioritized action items
- Specific improvement suggestions
- Best practice references

**Integration:**
- Agent: "Given these dataset issues: {issues}, suggest improvements"
- RAG: "What are best practices for 3D character training datasets?"

## Workflow

### YAML Definition

```yaml
workflow_id: dataset_quality_inspection
description: Comprehensive quality inspection for training datasets

parameters:
  dataset_path: ${INPUT_DATASET_PATH}
  output_dir: ${OUTPUT_DIR}
  quality_threshold: 70
  enable_recommendations: true

tasks:
  - task_id: validate_structure
    module: scenario
    task_type: file
    parameters:
      scenario: file_organizer
      operation: validate
      input: ${dataset_path}

  - task_id: analyze_images
    module: inspector
    task_type: quality_analysis
    parameters:
      dataset_path: ${dataset_path}
      checks:
        - image_quality
        - duplicates
        - corruption
    depends_on: [validate_structure]

  - task_id: analyze_captions
    module: inspector
    task_type: caption_analysis
    parameters:
      dataset_path: ${dataset_path}
    depends_on: [validate_structure]

  - task_id: analyze_distribution
    module: inspector
    task_type: distribution_analysis
    parameters:
      dataset_path: ${dataset_path}
    depends_on: [analyze_images]

  - task_id: lookup_best_practices
    module: rag
    task_type: search
    parameters:
      query: "training dataset quality best practices for 3D character LoRA"
      top_k: 5
    depends_on: [analyze_distribution]

  - task_id: generate_recommendations
    module: agent
    task_type: analyze
    parameters:
      user_request: |
        Given dataset analysis results:
        - Images: ${analyze_images.output}
        - Captions: ${analyze_captions.output}
        - Distribution: ${analyze_distribution.output}
        - Best practices: ${lookup_best_practices.output}

        Provide specific recommendations to improve this training dataset.
      enable_rag: true
    depends_on:
      - analyze_images
      - analyze_captions
      - analyze_distribution
      - lookup_best_practices

  - task_id: generate_report
    module: inspector
    task_type: report_generation
    parameters:
      results:
        images: ${analyze_images.output}
        captions: ${analyze_captions.output}
        distribution: ${analyze_distribution.output}
        recommendations: ${generate_recommendations.output}
      output_file: ${output_dir}/quality_report.html
    depends_on: [generate_recommendations]
```

### Execution Flow

1. **Validate Structure** → Ensure dataset is properly formatted
2. **Analyze Images** (parallel) → Quality, duplicates, corruption
3. **Analyze Captions** (parallel) → Caption quality and consistency
4. **Analyze Distribution** → Balance and diversity
5. **Lookup Best Practices** → Query RAG for guidelines
6. **Generate Recommendations** → Agent-powered suggestions
7. **Generate Report** → Comprehensive HTML/JSON report

## Safety Integration

### Resource Management

```python
from scripts.orchestration.safety import SafetyIntegration

safety = SafetyIntegration(
    memory_soft_limit_mb=2048,  # 2GB soft limit for inspector
    memory_hard_limit_mb=3072,  # 3GB hard limit
    enable_auto_degradation=True
)

safety.start()

# Adjust concurrency based on resources
max_concurrent = safety.degradation.get_max_concurrency(original=4)
batch_size = safety.degradation.get_batch_size(original=32)

# Execute analysis
for batch in dataset.batches(batch_size):
    if not safety.is_safe_to_proceed():
        safety.handle_safety_violation(safety.check_safety())
        break

    # Process batch
    process_images(batch)

safety.stop()
```

### Degradation Strategy

- **Light:** Reduce concurrent analysis threads
- **Moderate:** Reduce batch size, disable caching
- **Heavy:** Sequential processing only, minimal memory
- **Emergency:** Checkpoint and pause

## CLI Interface

```bash
# Basic usage
python -m scripts.scenarios.dataset_quality_inspector \
  --dataset /path/to/dataset \
  --output /path/to/report

# Advanced options
python -m scripts.scenarios.dataset_quality_inspector \
  --dataset /path/to/dataset \
  --output /path/to/report \
  --quality-threshold 80 \
  --enable-recommendations \
  --enable-vlm-analysis \
  --memory-limit 4096 \
  --report-format html,json,markdown

# Resume from checkpoint
python -m scripts.scenarios.dataset_quality_inspector \
  --dataset /path/to/dataset \
  --resume /path/to/checkpoint.json
```

## Output

### Report Structure

```json
{
  "dataset": {
    "path": "/path/to/dataset",
    "total_images": 450,
    "total_size_mb": 2340.5,
    "scanned_at": "2025-12-02T10:30:00Z"
  },
  "quality_summary": {
    "overall_score": 82.5,
    "passed": 380,
    "warnings": 50,
    "failed": 20
  },
  "issues": {
    "image_quality": [
      {
        "severity": "high",
        "type": "blur",
        "count": 15,
        "files": ["char_001.png", "char_045.png", ...]
      }
    ],
    "duplicates": {
      "exact": 5,
      "near": 12,
      "groups": [...]
    },
    "captions": {
      "missing": 3,
      "too_short": 8,
      "inconsistent": 15
    }
  },
  "distribution": {
    "categories": {...},
    "balance_score": 75.0,
    "diversity_score": 68.5
  },
  "recommendations": [
    {
      "priority": "high",
      "category": "image_quality",
      "action": "Remove 15 blurry images",
      "details": "Images with Laplacian variance < 100"
    },
    {
      "priority": "medium",
      "category": "captions",
      "action": "Regenerate 8 short captions",
      "details": "Captions < 20 tokens"
    }
  ]
}
```

### HTML Report Features

- Interactive dashboard
- Sortable/filterable issue tables
- Image thumbnails with overlays
- Severity color coding
- Download JSON/CSV exports
- Print-friendly version

## Testing Strategy

### Unit Tests

- Each analyzer independently tested
- Mock dataset fixtures
- Edge case coverage

### Integration Tests

- Full workflow execution
- Agent/RAG integration verified
- Safety system integration
- Checkpoint/resume functionality

### Test Datasets

- `tests/fixtures/dataset_good/` - Clean dataset
- `tests/fixtures/dataset_issues/` - Various problems
- `tests/fixtures/dataset_corrupted/` - Edge cases

## Performance Targets

- **Speed:** 1000 images analyzed in < 5 minutes (CPU-only)
- **Memory:** Peak usage < 4GB
- **Scalability:** Handles datasets up to 10,000 images
- **Reliability:** 100% CPU-only, no GPU interference

## Implementation Phases

### Phase 1: Core Analyzers (Week 3)
- ImageQualityAnalyzer
- DuplicateDetector
- CorruptionDetector
- FormatValidator
- Basic report generation

### Phase 2: Integration (Week 4)
- Agent integration for recommendations
- RAG integration for best practices
- Distribution analysis
- Caption analysis
- HTML report generation

### Phase 3: Testing (End of Week 4)
- Unit tests
- Integration tests
- End-to-end validation
- Performance optimization

## Success Criteria

✅ Analyzes 1000-image dataset in < 5 minutes
✅ Uses < 4GB RAM peak
✅ 100% CPU-only operation (no GPU usage)
✅ Integrates Agent + RAG + VLM successfully
✅ Generates comprehensive HTML report
✅ Provides actionable recommendations
✅ Handles checkpoint/resume
✅ Passes all integration tests
