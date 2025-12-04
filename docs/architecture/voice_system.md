# Complete Voice Analysis & Synthesis System Architecture

**Created**: 2025-11-20
**Status**: Design Phase
**Purpose**: Comprehensive voice processing system for character voice cloning, emotion control, and context-aware synthesis

---

## System Components

### 1. Voice Synthesis (GPT-SoVITS)
**Purpose**: High-quality text-to-speech with character voice cloning
**Features**:
- Few-shot voice cloning (5-10 minutes of data)
- Multi-language support
- Natural prosody and intonation
- Fast inference speed

**Models**:
- GPT Model: Semantic token generation
- SoVITS Model: Acoustic feature to waveform
- Reference encoder: Voice embedding extraction

**Location**: `/mnt/c/AI_LLM_projects/GPT-SoVITS/`

---

### 2. Voice Conversion (RVC - Retrieval-based Voice Conversion)
**Purpose**: Convert any voice to character voice (real-time)
**Features**:
- Real-time voice conversion
- Pitch and formant control
- High quality voice transformation
- Low latency (<100ms)

**Models**:
- Content encoder (HuBERT/ContentVec)
- F0 predictor
- Voice decoder

**Location**: `/mnt/c/AI_LLM_projects/RVC/`
**Repository**: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI

---

### 3. Speech Emotion Recognition (SER)
**Purpose**: Detect emotions from speech audio
**Features**:
- Multi-emotion classification (happy, sad, angry, neutral, etc.)
- Real-time emotion detection
- Confidence scores
- Temporal emotion tracking

**Models**:
- Wav2Vec2-Emotion (Facebook)
- HuBERT-Emotion
- Custom fine-tuned models

**Emotions Supported**:
- Happy
- Sad
- Angry
- Fearful
- Surprised
- Disgusted
- Neutral
- Excited
- Calm

**Location**: `scripts/synthesis/tts/emotion_recognition/`

---

### 4. Emotion-Controlled TTS
**Purpose**: Generate speech with specific emotions
**Features**:
- Emotion intensity control (0.0-1.0)
- Multi-emotion blending
- Prosody modification
- Context-aware emotion selection

**Integration**:
- Works with GPT-SoVITS
- Emotion embeddings
- Prosody transfer

**Location**: `scripts/synthesis/tts/emotion_controller.py`

---

### 5. Context-Aware Speech Synthesis
**Purpose**: Generate contextually appropriate speech
**Features**:
- Dialogue history analysis
- Character personality modeling
- Situational awareness
- Appropriate prosody and tone

**Components**:
- LLM-based context analyzer
- Dialogue state tracker
- Prosody predictor
- Style controller

**Location**: `scripts/synthesis/tts/context_aware/`

---

### 6. Voice Analysis Tools
**Purpose**: Comprehensive voice feature extraction
**Features**:
- Pitch (F0) extraction
- Formant analysis
- Speaking rate
- Energy/volume
- Spectral features
- Voice quality metrics

**Location**: `scripts/analysis/audio/voice_analyzer.py`

---

## System Architecture

```
Input Audio/Text
     |
     v
[Voice Analysis Layer]
     |-- Emotion Recognition
     |-- Pitch Extraction
     |-- Speaking Rate
     |-- Voice Quality
     |
     v
[Context Analysis Layer]
     |-- Dialogue State
     |-- Character Personality
     |-- Situational Context
     |
     v
[Synthesis Control Layer]
     |-- Emotion Selection
     |-- Prosody Control
     |-- Style Parameters
     |
     v
[Voice Generation Layer]
     |-- GPT-SoVITS (TTS)
     |-- RVC (Voice Conversion)
     |
     v
Output Speech
```

---

## Environment Setup

### Main Environment: `ai_env`
**Purpose**: Video/audio analysis, frame extraction, Whisper, Pyannote
**PyTorch**: 2.7.1+cu128
**Lightning**: 1.9.0

### Voice Training Environment: `voice_training`
**Purpose**: GPT-SoVITS, RVC training
**PyTorch**: 2.4+
**Lightning**: 2.4+

### Inference Environment: Shared with `ai_env`
**Purpose**: Real-time voice synthesis and conversion
**Requirements**: Compatible with both environments

---

## Directory Structure

```
/mnt/c/AI_LLM_projects/
├── GPT-SoVITS/              # GPT-SoVITS project (independent)
├── RVC/                      # RVC project (independent)
├── ai_warehouse/             # Shared model storage
│   └── models/audio/
│       ├── gpt_sovits/pretrained/
│       ├── rvc/pretrained/
│       └── emotion/
│
└── animation-ai-studio/
    ├── scripts/synthesis/tts/
│   ├── gpt_sovits_wrapper.py        # GPT-SoVITS interface
│   ├── rvc_wrapper.py                # RVC interface
│   ├── emotion_controller.py        # Emotion control
│   ├── context_aware_tts.py         # Context-aware synthesis
│   ├── voice_model_trainer.py       # Training pipeline
│   │
│   ├── emotion_recognition/
│   │   ├── ser_model.py             # Emotion recognition
│   │   └── emotion_features.py      # Feature extraction
│   │
│   └── context_aware/
│       ├── dialogue_analyzer.py     # Dialogue context
│       └── prosody_predictor.py     # Prosody control
│
├── scripts/analysis/audio/
│   ├── voice_analyzer.py            # Voice analysis
│   └── prosody_extractor.py         # Prosody features
│
└── models/voices/                    # Trained models
    ├── luca/
    │   ├── gpt_sovits/              # GPT-SoVITS models
    │   ├── rvc/                      # RVC models
    │   └── emotion_profiles/         # Emotion templates
    ├── alberto/
    └── giulia/
```

---

## Workflow

### Training Phase

1. **Extract Voice Samples** ✓ (Already done)
   - Input: Film audio
   - Output: Segmented voice samples per character
   - Tools: Whisper + Pyannote

2. **Train GPT-SoVITS Models**
   - Input: Voice samples + transcripts
   - Output: Character voice models
   - Time: ~2-4 hours per character

3. **Train RVC Models**
   - Input: Voice samples
   - Output: Voice conversion models
   - Time: ~4-8 hours per character

4. **Extract Emotion Profiles**
   - Input: Voice samples with emotions
   - Output: Emotion embeddings per character
   - Tools: Emotion recognition + clustering

### Inference Phase

1. **Text Input** → Context Analyzer
2. **Context** → Emotion Selector
3. **Text + Emotion** → GPT-SoVITS
4. **Raw Speech** → RVC (optional refinement)
5. **Final Speech** → Output

---

## Models to Download

### GPT-SoVITS Pretrained Models
- `GPT_SoVITS-e15.ckpt` (~1.5 GB)
- `s2G488k.pth` (~500 MB)
- Chinese ASR model (~1 GB) - optional
- Location: `/mnt/c/AI_LLM_projects/ai_warehouse/models/audio/gpt_sovits/pretrained/`

### RVC Pretrained Models
- HuBERT Base (~200 MB)
- RMVPEv2 F0 predictor (~50 MB)
- Location: `/mnt/c/AI_LLM_projects/ai_warehouse/models/audio/rvc/pretrained/`

### Emotion Recognition Models
- Wav2Vec2-Emotion (~400 MB)
- HuBERT-Emotion (~400 MB)
- Location: `/mnt/c/AI_LLM_projects/ai_warehouse/models/audio/emotion/`

---

## Implementation Priority

**Phase 1: Core Voice Synthesis** (Week 3-4)
1. ✓ Voice sample extraction (DONE)
2. GPT-SoVITS setup and training
3. Basic TTS inference

**Phase 2: Voice Conversion** (Week 5)
4. RVC setup and training
5. Voice conversion inference
6. Integration with GPT-SoVITS

**Phase 3: Emotion & Context** (Week 6)
7. Emotion recognition implementation
8. Emotion-controlled TTS
9. Context-aware synthesis
10. Complete pipeline integration

**Phase 4: Testing & Refinement** (Week 7)
11. Quality testing
12. Performance optimization
13. Documentation

---

## Technical Requirements

### Hardware
- GPU: RTX 5080 16GB
- VRAM Usage:
  - Training: ~12-14 GB
  - Inference: ~4-6 GB
  - Real-time conversion: ~2-3 GB

### Software
- CUDA 12.8
- PyTorch 2.4+ (voice_training env)
- PyTorch 2.7.1 (ai_env for analysis)
- FFmpeg
- Various audio processing libraries

---

## Integration Points

### With Video Generation
- Lip-sync generation (Wav2Lip)
- Emotion-driven facial animation
- Timeline synchronization

### With Character Animation
- Emotion → facial expression mapping
- Prosody → gesture mapping
- Context → body language

### With LLM Agent
- Dialogue generation
- Emotion prediction
- Context understanding
- Character personality modeling

---

## Quality Metrics

### Voice Quality
- MOS (Mean Opinion Score): Target > 4.0/5.0
- Speaker similarity: Target > 85%
- Naturalness: Target > 90%

### Emotion Accuracy
- Emotion classification F1: Target > 0.85
- Emotion intensity RMSE: Target < 0.15

### Context Appropriateness
- Human evaluation: Target > 80% appropriate
- Prosody naturalness: Target > 85%

---

## Next Steps

1. Create `voice_training` conda environment
2. Install GPT-SoVITS + dependencies
3. Download pretrained models
4. Install RVC
5. Implement training scripts
6. Train Luca voice model (test)
7. Implement emotion recognition
8. Implement context-aware synthesis
9. Build unified inference API
10. Complete testing and documentation

---

**Status**: Ready to begin implementation
**Estimated Time**: 3-4 weeks for complete system
**Dependencies**: Already have voice samples ready ✓
