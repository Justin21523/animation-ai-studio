# File Organization Scenario - Design Document

## Overview

The **File Organization Scenario** provides intelligent file system analysis and automated organization capabilities for the Animation AI Studio project. It analyzes directory structures, detects duplicates, identifies organizational issues, and provides AI-powered recommendations for improving file organization.

## Architecture

### Component Hierarchy

```
file_organization/
├── common.py                    # Data structures and enums
├── __init__.py                  # Package exports
├── __main__.py                  # CLI entry point
├── organizer.py                 # Main orchestrator
│
├── analyzers/                   # Analysis components
│   ├── __init__.py
│   ├── file_classifier.py       # File type classification
│   ├── structure_analyzer.py    # Directory structure analysis
│   └── duplicate_detector.py    # Duplicate file detection
│
├── processors/                  # Organization processors
│   ├── __init__.py
│   └── smart_organizer.py       # AI-powered organization logic
│
├── integration/                 # External system integration
│   ├── __init__.py
│   ├── agent_integration.py     # Agent Framework integration
│   └── rag_integration.py       # RAG System integration
│
└── tests/                       # Unit tests
    ├── __init__.py
    ├── test_analyzers.py
    └── test_organizer.py
```

## Core Components

### 1. Data Structures (common.py)

**Enums:**
- `FileType`: Classification of files (IMAGE, VIDEO, AUDIO, DOCUMENT, CODE, ARCHIVE, MODEL, DATASET, CONFIG, LOG, OTHER)
- `OrganizationIssue`: Types of issues detected (DUPLICATE, MISPLACED, NAMING_INCONSISTENT, ORPHANED, NESTED_EXCESSIVE, SIZE_ANOMALY, PERMISSION_ISSUE, SYMLINK_BROKEN)
- `IssueSeverity`: Severity levels (LOW, MEDIUM, HIGH, CRITICAL)
- `OrganizationStrategy`: Organization methods (BY_TYPE, BY_DATE, BY_PROJECT, BY_SIZE, CUSTOM, SMART)

**Dataclasses:**
- `FileMetadata`: Complete metadata for a single file (path, type, size, timestamps, hashes, permissions)
- `DuplicateGroup`: Group of duplicate files with deduplication recommendations
- `Issue`: Single organizational issue with severity and recommendations
- `StructureAnalysis`: Directory structure analysis results
- `OrganizationReport`: Comprehensive analysis report

### 2. Analyzers

#### FileClassifier (`analyzers/file_classifier.py`)
Classifies files into categories using multiple detection methods:
- **Magic bytes detection**: Read file headers to identify true file type
- **Extension mapping**: Fallback classification by extension
- **MIME type detection**: Cross-validation using MIME types
- **Special handling**: AI models (.safetensors, .pt), datasets, configs

**Key Methods:**
```python
class FileClassifier:
    def classify_file(self, path: Path) -> FileType
    def detect_mime_type(self, path: Path) -> str
    def is_ai_model(self, path: Path) -> bool
    def is_dataset(self, path: Path) -> bool
```

#### DuplicateDetector (`analyzers/duplicate_detector.py`)
Identifies duplicate files using multiple hash methods:
- **SHA256 content hash**: Exact duplicates
- **Perceptual hash**: Near-duplicates for images/videos
- **Size-based pre-filtering**: Optimize performance
- **Smart grouping**: Group duplicates with deduplication strategies

**Key Methods:**
```python
class DuplicateDetector:
    def detect_duplicates(self, files: List[FileMetadata]) -> List[DuplicateGroup]
    def compute_sha256(self, path: Path) -> str
    def compute_perceptual_hash(self, path: Path) -> Optional[str]
    def group_by_hash(self, files: List[FileMetadata]) -> Dict[str, List[Path]]
```

#### StructureAnalyzer (`analyzers/structure_analyzer.py`)
Analyzes directory structure and identifies organizational issues:
- **Depth analysis**: Calculate nesting levels
- **Pattern detection**: Identify project structures
- **Issue detection**: Find misplaced files, orphaned directories
- **Naming consistency**: Check naming conventions

**Key Methods:**
```python
class StructureAnalyzer:
    def analyze_structure(self, root: Path) -> StructureAnalysis
    def calculate_depth(self, path: Path) -> int
    def detect_project_structure(self, root: Path) -> Optional[str]
    def find_orphaned_directories(self, root: Path) -> List[Path]
```

### 3. Processors

#### SmartOrganizer (`processors/smart_organizer.py`)
AI-powered file organization logic:
- **Strategy selection**: Choose optimal organization strategy
- **Rule generation**: Create organization rules based on patterns
- **Safe operations**: Dry-run mode, backup creation
- **Batch processing**: Efficient bulk operations

**Key Methods:**
```python
class SmartOrganizer:
    async def organize(
        self,
        files: List[FileMetadata],
        strategy: OrganizationStrategy,
        dry_run: bool = True
    ) -> OrganizationResult

    async def generate_rules(
        self,
        files: List[FileMetadata]
    ) -> List[OrganizationRule]

    def preview_changes(
        self,
        files: List[FileMetadata],
        rules: List[OrganizationRule]
    ) -> Dict[Path, Path]
```

### 4. Main Orchestrator

#### FileOrganizer (`organizer.py`)
Main orchestrator coordinating all components:
- **Workflow coordination**: Scan → Analyze → Detect → Report
- **Safety integration**: Memory budgets, checkpoints, emergency handling
- **Progress tracking**: Real-time progress via EventBus
- **Report generation**: Comprehensive JSON/HTML/Markdown reports

**Key Methods:**
```python
class FileOrganizer:
    def __init__(
        self,
        root_path: str,
        config: Dict[str, Any],
        safety_integration: Optional[SafetyIntegration] = None
    )

    async def analyze(
        self,
        enable_recommendations: bool = False
    ) -> OrganizationReport

    async def organize(
        self,
        strategy: OrganizationStrategy,
        dry_run: bool = True
    ) -> OrganizationReport
```

### 5. Integration Layer

#### AgentIntegration (`integration/agent_integration.py`)
AI-powered recommendations via Agent Framework:
- **Organization recommendations**: Suggest optimal file structure
- **Rule generation**: Create custom organization rules
- **Issue prioritization**: Rank issues by impact
- **Actionable suggestions**: Provide step-by-step fixes

#### RAGIntegration (`integration/rag_integration.py`)
Knowledge base integration for best practices:
- **Best practices lookup**: File organization standards
- **Pattern recognition**: Identify common project structures
- **Naming conventions**: Retrieve naming guidelines
- **Tool recommendations**: Suggest cleanup tools

## Workflow

### 1. Analysis Workflow

```
Input: Root Directory
  ↓
Scan Files (Parallel, Memory-Budgeted)
  ↓
Classify Files (FileClassifier)
  ↓
Analyze Structure (StructureAnalyzer)
  ↓
Detect Duplicates (DuplicateDetector)
  ↓
Detect Issues (All Analyzers)
  ↓
[Optional] Generate Recommendations (Agent + RAG)
  ↓
Output: OrganizationReport
```

### 2. Organization Workflow

```
Input: OrganizationReport + Strategy
  ↓
Generate Organization Rules (SmartOrganizer)
  ↓
[Optional] Get AI Recommendations (Agent)
  ↓
Preview Changes (Dry-Run)
  ↓
[User Approval]
  ↓
Execute Organization (SmartOrganizer)
  ↓
Create Backup/Checkpoint
  ↓
Output: Updated OrganizationReport
```

## Safety Integration

### Memory Management
- **Budget tracking**: Monitor memory usage during file scanning
- **Batch processing**: Process files in memory-safe batches
- **Graceful degradation**: Reduce scan depth under memory pressure

### Checkpointing
- **Progress saving**: Save scan progress every N files
- **Resume capability**: Resume interrupted scans
- **Rollback support**: Undo organization operations

### Emergency Handling
- **Disk space monitoring**: Stop operations if disk space low
- **Permission errors**: Gracefully handle access errors
- **Corruption detection**: Skip corrupted files safely

## Configuration

### Default Configuration
```yaml
file_organization:
  # Scanning
  max_depth: 10
  follow_symlinks: false
  skip_hidden: false
  exclude_patterns:
    - "*.tmp"
    - "*.cache"
    - "__pycache__"
    - ".git"

  # Classification
  enable_magic_bytes: true
  enable_perceptual_hash: true

  # Duplicates
  min_file_size: 1024  # 1KB minimum for duplicate detection
  perceptual_hash_threshold: 5  # Hamming distance for near-duplicates

  # Structure Analysis
  max_nesting_depth: 5
  min_files_per_directory: 3

  # Safety
  memory_budget_mb: 2048
  checkpoint_interval: 1000  # Save progress every 1000 files

  # Organization
  default_strategy: "smart"
  dry_run_by_default: true
  create_backups: true
```

## Performance Characteristics

### Expected Performance
- **Scanning**: ~10,000 files/second (SSD, no hashing)
- **SHA256 hashing**: ~5,000 files/second (depends on file size)
- **Perceptual hashing**: ~500 images/second (CPU-only)
- **Structure analysis**: O(n) where n = number of files

### Optimization Strategies
- **Parallel scanning**: Multi-threaded directory traversal
- **Lazy hashing**: Only hash when needed for duplicate detection
- **Size pre-filtering**: Skip hash computation for unique file sizes
- **Batch processing**: Process files in configurable batch sizes

## Output Formats

### JSON Report
```json
{
  "root_path": "/path/to/directory",
  "timestamp": "2025-12-03T10:00:00",
  "total_files": 15000,
  "total_size_bytes": 5368709120,
  "file_type_counts": {
    "image": 5000,
    "video": 100,
    "document": 200,
    "code": 9500,
    "other": 200
  },
  "duplicate_groups": [...],
  "issues": [...],
  "structure_analysis": {...},
  "organization_score": 72.5,
  "recommendations": [...],
  "potential_savings_bytes": 1073741824
}
```

### HTML Report
- **Interactive dashboard**: File type pie charts, duplicate tree view
- **Issue summary**: Critical/High/Medium/Low counts with drill-down
- **Recommendations**: Actionable improvement steps
- **File browser**: Navigate directory structure visually

### Markdown Report
- **Executive summary**: Key metrics and scores
- **Issues breakdown**: Categorized by severity
- **Duplicate analysis**: Groups with savings estimates
- **Action items**: Prioritized recommendations

## Testing Strategy

### Unit Tests
- `test_file_classifier.py`: Test file type detection accuracy
- `test_duplicate_detector.py`: Test hash computation and grouping
- `test_structure_analyzer.py`: Test depth calculation and issue detection
- `test_smart_organizer.py`: Test organization rule generation

### Integration Tests
- End-to-end workflow tests
- Safety integration tests
- Agent/RAG integration tests

### Test Coverage Goal
- **Minimum**: 80% line coverage
- **Target**: 90% line coverage

## Future Enhancements

### Phase 2 (Post-MVP)
- **File content analysis**: Detect sensitive data, secrets
- **Smart compression**: Auto-compress old/unused files
- **Cloud sync integration**: Organize cloud storage
- **Visual file browser**: Web UI for interactive organization

### Phase 3 (Advanced)
- **Machine learning**: Learn user organization preferences
- **Predictive organization**: Suggest organization before asked
- **Auto-tagging**: Add metadata tags automatically
- **Version control**: Track file organization history

## Dependencies

### Core Dependencies
- **pathlib**: Path manipulation
- **hashlib**: SHA256 hashing
- **magic**: Magic bytes detection
- **imagehash**: Perceptual hashing (Pillow-based)

### Integration Dependencies
- **Safety System**: Memory budgets, emergency handling
- **Agent Framework**: AI recommendations
- **RAG System**: Best practices lookup
- **EventBus**: Progress tracking

## Success Criteria

1. **Accuracy**: >95% file type classification accuracy
2. **Performance**: Scan 10,000 files in <60 seconds
3. **Safety**: Zero data loss, all operations reversible
4. **Usability**: Clear reports, actionable recommendations
5. **Reliability**: Graceful handling of all error conditions

## References

- Dataset Quality Inspector: `/scripts/scenarios/dataset_quality_inspector/`
- Safety System: `/scripts/orchestration/safety/`
- Agent Framework: `/scripts/orchestration/agents/`
- RAG System: `/scripts/orchestration/rag/`
