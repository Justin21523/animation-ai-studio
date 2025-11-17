# Voice Synthesis Module

**Purpose:** GPT-SoVITS-based character voice synthesis and cloning
**Status:** 📋 Planned (0% Complete)
**Hardware:** RTX 5080 16GB VRAM
**VRAM Usage:** 3-4GB (can run with SDXL stopped or alongside LLM)

---

## 📊 Module Overview

### Core Capabilities

```
User Request (text + character + emotion)
              ↓
    LLM Intent Analysis
              ↓
  ┌───────────────────────┐
  │  GPT-SoVITS Engine    │
  │  - Voice model selection│
  │  - Emotion control    │
  │  - Language handling  │
  └───────────────────────┘
              ↓
    Audio Generation
              ↓
     Quality Check
              ↓
    Output (WAV file)
```

### Technologies

- **Base Engine**: GPT-SoVITS
- **Voice Cloning**: Custom character voice models
- **Emotion Control**: Temperature and style adjustment
- **Languages**: English, Italian
- **Audio Quality**: 44.1kHz, 16-bit WAV

---

## 🎯 Functional Requirements

### 1. Text-to-Speech
- Multi-character voice synthesis
- High-quality audio output
- Natural prosody and intonation
- Speed control (0.5x - 2.0x)
- Batch text synthesis

### 2. Voice Cloning
- Train character voices from film audio
- Extract clean voice samples (speaker diarization)
- Voice model training pipeline
- Quality validation

### 3. Emotion Control
- Emotional speech synthesis (happy, sad, excited, neutral, etc.)
- Temperature-based control
- Style mixing
- Emotion transitions

### 4. Language Support
- English (primary)
- Italian (for character authenticity)
- Multilingual mixing
- Accent preservation

### 5. Lip-sync Integration
- Generate lip-sync metadata
- Viseme extraction
- Timing alignment
- Integration with video generation (future)

---

## 🔧 Technical Architecture

### Component Structure

```
scripts/synthesis/tts/
├── __init__.py
├── gpt_sovits_wrapper.py      # GPT-SoVITS Python wrapper
├── voice_model_trainer.py     # Train character voices
├── emotion_controller.py      # Emotion-aware synthesis
├── voice_dataset_builder.py   # Extract voice samples from films
└── lip_sync_generator.py      # Lip-sync metadata (future)

configs/generation/
├── tts_config.yaml             # Voice synthesis settings
└── character_voices.yaml       # Character voice models registry

requirements/
└── voice_synthesis.txt         # TTS dependencies
```

### Core Classes

#### GPTSoVITSWrapper
```python
class GPTSoVITSWrapper:
    """Interface to GPT-SoVITS inference engine"""

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        dtype: str = "float16"
    )

    def load_voice_model(
        self,
        character_name: str,
        sovits_path: str,
        gpt_path: str,
        reference_audio: str = None
    ):
        """Load character-specific voice model"""

    def synthesize(
        self,
        text: str,
        character: str,
        emotion: str = "neutral",
        language: str = "en",
        speed: float = 1.0,
        temperature: float = 1.0,
        top_k: int = 15,
        top_p: float = 1.0,
        output_path: str = None
    ) -> str:
        """
        Synthesize speech from text

        Args:
            text: Input text to synthesize
            character: Character name (from character_voices.yaml)
            emotion: Emotion preset (neutral, happy, sad, excited)
            language: Language code (en, it)
            speed: Speech speed multiplier
            temperature: Sampling temperature for emotion control
            output_path: Path to save WAV file

        Returns:
            Path to generated audio file
        """

    def synthesize_batch(
        self,
        texts: List[str],
        character: str,
        emotions: List[str] = None,
        output_dir: str = "outputs/tts/batch"
    ) -> List[str]:
        """Batch synthesis for multiple texts"""

    def get_available_voices(self) -> List[str]:
        """List all loaded character voices"""

    def unload_voice_model(self, character: str = None):
        """Unload voice model to free VRAM"""
```

#### VoiceModelTrainer
```python
class VoiceModelTrainer:
    """Train GPT-SoVITS voice models from audio samples"""

    def __init__(
        self,
        gpt_sovits_repo: str,
        output_dir: str,
        device: str = "cuda"
    )

    def prepare_dataset(
        self,
        audio_files: List[str],
        transcripts: List[str],
        output_dir: str,
        sample_rate: int = 44100
    ):
        """Prepare training dataset from audio files"""

    def train(
        self,
        dataset_dir: str,
        character_name: str,
        base_gpt_model: str,
        base_sovits_model: str,
        epochs: int = 100,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        save_interval: int = 10
    ) -> Dict[str, str]:
        """
        Train character voice model

        Returns:
            {
                "gpt_model": "path/to/gpt_model.ckpt",
                "sovits_model": "path/to/sovits_model.pth",
                "reference_audio": "path/to/reference.wav"
            }
        """

    def validate_quality(
        self,
        model_dir: str,
        test_texts: List[str]
    ) -> Dict[str, float]:
        """Validate trained model quality"""
```

#### VoiceDatasetBuilder
```python
class VoiceDatasetBuilder:
    """Extract and prepare voice samples from film audio"""

    def __init__(
        self,
        whisper_model: str = "large-v3",
        diarization_model: str = "pyannote/speaker-diarization"
    )

    def extract_from_film(
        self,
        video_path: str,
        character_name: str,
        output_dir: str,
        min_duration: float = 1.0,
        max_duration: float = 10.0,
        min_silence: float = 0.3
    ) -> List[str]:
        """
        Extract character voice samples from film

        Process:
        1. Extract audio from video
        2. Run speaker diarization
        3. Transcribe with Whisper
        4. Match character name to speaker
        5. Extract clean segments
        6. Filter by quality (low noise, clear speech)

        Returns:
            List of extracted audio file paths
        """

    def clean_audio_sample(
        self,
        audio_path: str,
        output_path: str,
        reduce_noise: bool = True,
        normalize: bool = True
    ) -> str:
        """Clean and enhance audio sample"""

    def validate_sample_quality(
        self,
        audio_path: str,
        min_snr: float = 15.0  # Signal-to-noise ratio
    ) -> Dict[str, Any]:
        """
        Validate audio sample quality

        Returns:
            {
                "snr": float,
                "duration": float,
                "sample_rate": int,
                "is_clean": bool
            }
        """
```

#### EmotionController
```python
class EmotionController:
    """Control emotion in synthesized speech"""

    def __init__(self, base_wrapper: GPTSoVITSWrapper)

    def synthesize_with_emotion(
        self,
        text: str,
        character: str,
        emotion: str,
        intensity: float = 1.0  # 0.0-2.0
    ) -> str:
        """
        Synthesize speech with specific emotion

        Emotion mapping:
        - neutral: temperature=1.0
        - happy: temperature=1.2
        - excited: temperature=1.3
        - sad: temperature=0.8
        - angry: temperature=1.4
        """

    def blend_emotions(
        self,
        text: str,
        character: str,
        emotion_mix: Dict[str, float]  # {"happy": 0.7, "excited": 0.3}
    ) -> str:
        """Blend multiple emotions"""

    def create_emotion_transition(
        self,
        texts: List[str],
        character: str,
        emotions: List[str],
        transition_smoothness: float = 0.5
    ) -> List[str]:
        """Create smooth emotion transitions across multiple lines"""
```

---

## ⚙️ Configuration System

### tts_config.yaml

```yaml
gpt_sovits:
  repo_path: "/mnt/c/AI_LLM_projects/ai_warehouse/tools/GPT-SoVITS"
  models_dir: "/mnt/c/AI_LLM_projects/ai_warehouse/models/tts"

  base_models:
    gpt: "pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
    sovits: "pretrained_models/s2G488k.pth"

  inference:
    device: "cuda"
    dtype: "float16"
    max_length: 1024
    top_k: 15
    top_p: 1.0
    temperature: 1.0
    seed: null  # null for random

  audio:
    sample_rate: 44100
    format: "wav"
    bit_depth: 16

  languages:
    supported: ["en", "it", "zh"]
    default: "en"

training:
  default_epochs: 100
  batch_size: 4
  learning_rate: 0.0001
  gradient_accumulation_steps: 1
  save_interval: 10
  validation_interval: 5

  preprocessing:
    sample_rate: 44100
    hop_length: 512
    win_length: 2048
    n_fft: 2048

voice_extraction:
  whisper:
    model: "large-v3"
    device: "cuda"
    language: "en"

  diarization:
    model: "pyannote/speaker-diarization"
    min_speakers: 1
    max_speakers: 10

  quality_filters:
    min_duration: 1.0
    max_duration: 10.0
    min_snr: 15.0
    min_silence: 0.3
```

### character_voices.yaml

```yaml
luca:
  character_name: "Luca Paguro"
  sovits_model: "luca_sovits.pth"
  gpt_model: "luca_gpt.ckpt"
  reference_audio: "luca_reference.wav"
  language: "en"
  accent: "italian-english"

  emotion_presets:
    neutral:
      temperature: 1.0
      description: "Normal speaking voice"
    happy:
      temperature: 1.2
      description: "Joyful, upbeat"
    excited:
      temperature: 1.3
      description: "Very energetic, enthusiastic"
    sad:
      temperature: 0.8
      description: "Melancholic, subdued"
    scared:
      temperature: 1.1
      description: "Nervous, anxious"

  voice_characteristics:
    age_range: "13-14"
    pitch: "medium-high"
    energy: "moderate-high"
    speaking_rate: "moderate"

alberto:
  character_name: "Alberto Scorfano"
  sovits_model: "alberto_sovits.pth"
  gpt_model: "alberto_gpt.ckpt"
  reference_audio: "alberto_reference.wav"
  language: "en"
  accent: "italian-english"

  emotion_presets:
    neutral:
      temperature: 1.0
    happy:
      temperature: 1.2
    excited:
      temperature: 1.4
    sad:
      temperature: 0.8
    confident:
      temperature: 1.1

  voice_characteristics:
    age_range: "14-15"
    pitch: "medium"
    energy: "high"
    speaking_rate: "fast"

giulia:
  character_name: "Giulia Marcovaldo"
  sovits_model: "giulia_sovits.pth"
  gpt_model: "giulia_gpt.ckpt"
  reference_audio: "giulia_reference.wav"
  language: "en"
  accent: "italian"

  emotion_presets:
    neutral:
      temperature: 1.0
    happy:
      temperature: 1.3
    excited:
      temperature: 1.4
    sad:
      temperature: 0.7
    determined:
      temperature: 1.1

  voice_characteristics:
    age_range: "13-14"
    pitch: "medium-high"
    energy: "very-high"
    speaking_rate: "very-fast"
```

---

## 📦 Dependencies

### requirements/voice_synthesis.txt

```
# GPT-SoVITS (clone from GitHub)
# https://github.com/RVC-Boss/GPT-SoVITS

# Audio processing
librosa>=0.10.0
soundfile>=0.12.0
audioread>=3.0.0
pydub>=0.25.0

# Speech recognition
openai-whisper>=20231117
faster-whisper>=0.10.0  # Faster inference

# Speaker diarization
pyannote.audio>=3.1.0
pyannote.core>=5.0.0

# Audio enhancement
noisereduce>=3.0.0
pedalboard>=0.7.0  # Spotify's audio effects

# General
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
```

---

## 🚀 Usage Examples

### Basic Voice Synthesis

```python
from scripts.synthesis.tts import GPTSoVITSWrapper

# Initialize wrapper
tts = GPTSoVITSWrapper(
    model_dir="/mnt/c/AI_LLM_projects/ai_warehouse/models/tts",
    device="cuda"
)

# Load character voice
tts.load_voice_model(
    character_name="luca",
    sovits_path="luca_sovits.pth",
    gpt_path="luca_gpt.ckpt"
)

# Synthesize speech
audio_path = tts.synthesize(
    text="Silenzio, Bruno!",
    character="luca",
    emotion="excited",
    language="en",
    output_path="outputs/tts/luca_silenzio.wav"
)

print(f"Generated: {audio_path}")
```

### Batch Synthesis with Different Emotions

```python
from scripts.synthesis.tts import GPTSoVITSWrapper

tts = GPTSoVITSWrapper(model_dir="...")
tts.load_voice_model(character_name="luca", ...)

# Batch synthesis
texts = [
    "We can do this!",
    "I'm scared...",
    "That was amazing!"
]
emotions = ["excited", "sad", "happy"]

audio_files = tts.synthesize_batch(
    texts=texts,
    character="luca",
    emotions=emotions,
    output_dir="outputs/tts/luca_batch"
)

for text, audio, emotion in zip(texts, audio_files, emotions):
    print(f"{emotion}: {text} → {audio}")
```

### Voice Model Training

```python
from scripts.synthesis.tts import VoiceDatasetBuilder, VoiceModelTrainer

# Step 1: Extract voice samples from film
extractor = VoiceDatasetBuilder(
    whisper_model="large-v3",
    diarization_model="pyannote/speaker-diarization"
)

samples = extractor.extract_from_film(
    video_path="/mnt/data/ai_data/datasets/3d-anime/luca/film.mp4",
    character_name="Luca",
    output_dir="data/films/luca/voice_samples",
    min_duration=1.0,
    max_duration=10.0
)

print(f"Extracted {len(samples)} voice samples")

# Step 2: Train voice model
trainer = VoiceModelTrainer(
    gpt_sovits_repo="/mnt/c/AI_LLM_projects/ai_warehouse/tools/GPT-SoVITS",
    output_dir="/mnt/c/AI_LLM_projects/ai_warehouse/models/tts/luca",
    device="cuda"
)

# Prepare dataset
trainer.prepare_dataset(
    audio_files=samples,
    transcripts=[...],  # From Whisper
    output_dir="data/films/luca/training_dataset"
)

# Train
result = trainer.train(
    dataset_dir="data/films/luca/training_dataset",
    character_name="Luca",
    base_gpt_model="pretrained_models/s1bert25hz-2kh-longer.ckpt",
    base_sovits_model="pretrained_models/s2G488k.pth",
    epochs=100,
    batch_size=4
)

print(f"Training complete!")
print(f"GPT model: {result['gpt_model']}")
print(f"SoVITS model: {result['sovits_model']}")
```

### Emotion Control

```python
from scripts.synthesis.tts import EmotionController, GPTSoVITSWrapper

# Initialize
tts = GPTSoVITSWrapper(...)
emotion_ctrl = EmotionController(tts)

# Synthesize with specific emotion and intensity
audio = emotion_ctrl.synthesize_with_emotion(
    text="I can't believe it!",
    character="luca",
    emotion="excited",
    intensity=1.5  # Extra excited
)

# Blend emotions
audio = emotion_ctrl.blend_emotions(
    text="I don't know...",
    character="luca",
    emotion_mix={"sad": 0.6, "scared": 0.4}
)

# Emotion transition across dialogue
dialogue = [
    "This is great!",
    "Wait, what's that?",
    "Oh no!"
]
emotions = ["happy", "neutral", "scared"]

audio_files = emotion_ctrl.create_emotion_transition(
    texts=dialogue,
    character="luca",
    emotions=emotions,
    transition_smoothness=0.7
)
```

### Voice Dataset Quality Validation

```python
from scripts.synthesis.tts import VoiceDatasetBuilder

builder = VoiceDatasetBuilder()

# Validate single sample
quality = builder.validate_sample_quality(
    audio_path="data/films/luca/voice_samples/sample_001.wav",
    min_snr=15.0
)

print(f"SNR: {quality['snr']:.2f} dB")
print(f"Duration: {quality['duration']:.2f}s")
print(f"Clean: {quality['is_clean']}")

# Clean audio sample
if not quality['is_clean']:
    builder.clean_audio_sample(
        audio_path="sample_001.wav",
        output_path="sample_001_clean.wav",
        reduce_noise=True,
        normalize=True
    )
```

---

## 📈 Performance Targets

### Latency

```yaml
Short sentence (1-2s audio): < 3 seconds generation
Medium sentence (3-5s audio): < 5 seconds generation
Long sentence (10s audio): < 10 seconds generation
Model loading: ~2-3 seconds
Voice sample extraction: ~1-2 minutes per minute of film
Voice model training: ~2-4 hours (100 epochs, RTX 5080)
```

### Quality

```yaml
Voice similarity: > 85% (subjective evaluation)
Naturalness: > 4.0/5.0 MOS score
Emotion accuracy: Clear distinction between emotions
Intelligibility: > 95% word recognition rate
```

### VRAM

```yaml
GPT-SoVITS small model: ~3-4GB
Training: ~8-10GB
Can run with LLM stopped: Yes
Can run with SDXL stopped: Yes
Can run alongside LLM: Possible (tight but manageable)
```

---

## 🚨 Known Challenges

### 1. Voice Sample Quality
**Issue**: Need clean voice samples from films for training

**Solution**:
- Use Whisper + Pyannote for accurate speaker separation
- Filter by signal-to-noise ratio (SNR > 15 dB)
- Manually curate 1-5 minutes of best samples per character
- Use audio enhancement (noise reduction, normalization)

### 2. Emotion Control Accuracy
**Issue**: Temperature-based emotion control is indirect

**Solution**:
- Create emotion presets per character
- Fine-tune temperature values empirically
- Consider training separate models per emotion (if needed)
- Use reference audio examples for each emotion

### 3. Multi-language Support
**Issue**: GPT-SoVITS quality varies by language

**Solution**:
- Train primarily on English samples
- Add Italian samples for authenticity
- Use language-specific base models if available
- Validate quality for each language

### 4. Lip-sync Alignment
**Issue**: Generated audio must align with video

**Solution**:
- Generate viseme metadata alongside audio
- Use forced alignment tools (Montreal Forced Aligner)
- Implement timing adjustment
- Integration with future video generation module

---

## ✅ Implementation Checklist

### Phase 1: GPT-SoVITS Setup (Estimated: 2 days)
- [ ] Clone GPT-SoVITS repository
- [ ] Install dependencies
- [ ] Download pretrained base models
- [ ] Create GPTSoVITSWrapper class
- [ ] Create tts_config.yaml
- [ ] Test basic inference

### Phase 2: Voice Dataset Extraction (Estimated: 3 days)
- [ ] Create VoiceDatasetBuilder class
- [ ] Implement Whisper transcription
- [ ] Implement Pyannote diarization
- [ ] Implement quality filtering
- [ ] Extract Luca voice samples
- [ ] Validate sample quality

### Phase 3: Voice Model Training (Estimated: 4 days)
- [ ] Create VoiceModelTrainer class
- [ ] Implement dataset preparation
- [ ] Implement training pipeline
- [ ] Train Luca voice model
- [ ] Validate model quality
- [ ] Create character_voices.yaml

### Phase 4: Emotion Control (Estimated: 2 days)
- [ ] Create EmotionController class
- [ ] Implement emotion presets
- [ ] Test emotion synthesis
- [ ] Validate emotion accuracy
- [ ] Create emotion blending

### Phase 5: Batch Processing (Estimated: 1 day)
- [ ] Implement batch synthesis
- [ ] Add progress tracking
- [ ] Test batch workflows

### Phase 6: Testing & Documentation (Estimated: 2 days)
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Create usage examples
- [ ] API documentation
- [ ] Quality benchmarking

---

## 🔄 Integration Points

### With LLM Backend Module
- LLM analyzes dialogue intent
- LLM suggests appropriate emotions
- LLM validates synthesis quality

### With Image Generation Module
- Coordinated character scene creation
- Dialogue + character images
- Emotion consistency (image expression + voice emotion)

### With Model Manager Module
- VRAM management (can run with LLM or standalone)
- Dynamic loading/unloading
- Resource monitoring

### With Agent Framework Module (Future)
- Agent decides character emotions
- Agent generates dialogue
- Agent coordinates multi-character scenes

---

## 📚 References

- **GPT-SoVITS**: https://github.com/RVC-Boss/GPT-SoVITS
- **Whisper**: https://github.com/openai/whisper
- **Pyannote**: https://github.com/pyannote/pyannote-audio
- **Coqui TTS**: https://github.com/coqui-ai/TTS (alternative)
- **Montreal Forced Aligner**: https://montreal-forced-aligner.readthedocs.io/

---

**Last Updated:** 2025-11-17
**Status:** 📋 Planned (Architecture Complete, Implementation Pending)
