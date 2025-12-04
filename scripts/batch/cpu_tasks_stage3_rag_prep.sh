#!/bin/bash
################################################################################
# CPU Tasks Stage 3: RAG Knowledge Base Preparation
#
# Pure CPU-only RAG preparation tasks (NO GPU usage):
#   - Process character description documents
#   - Process film metadata
#   - Generate embeddings using CPU-only models (sentence-transformers)
#   - Build FAISS/ChromaDB knowledge base
#
# Features:
#   - CPU-only embedding models (all-MiniLM-L6-v2)
#   - Memory-efficient batch processing
#   - Document chunking and preprocessing
#   - FAISS index creation (CPU)
#
# Hardware Requirements:
#   - CPU: 4+ cores
#   - RAM: 8GB+ (embedding models need ~2GB)
#   - Disk: Minimal (~1GB for index)
#
# Usage:
#   bash scripts/batch/cpu_tasks_stage3_rag_prep.sh \
#     FILM_NAME \
#     OUTPUT_DIR \
#     [--embedding-model MODEL]
#
# Author: Animation AI Studio
# Date: 2025-12-04
################################################################################

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

# Default CPU-only embedding model (lightweight, fast, no GPU needed)
DEFAULT_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
MEMORY_THRESHOLD_PCT=90

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# RAG scripts
DOCUMENT_PROCESSOR="${PROJECT_ROOT}/scripts/rag/documents/document_processor.py"
KNOWLEDGE_INGESTER="${PROJECT_ROOT}/scripts/rag/ingest_film_knowledge.py"

# ============================================================================
# Color Output
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================================================
# Resource Monitoring
# ============================================================================

check_memory_usage() {
    if command -v free &> /dev/null; then
        free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}'
    else
        echo "0"
    fi
}

monitor_resources() {
    local mem_pct=$(check_memory_usage)
    log_info "Resources: RAM=${mem_pct}%"

    if [ "$mem_pct" -gt "$MEMORY_THRESHOLD_PCT" ]; then
        log_error "Memory usage too high (${mem_pct}% > ${MEMORY_THRESHOLD_PCT}%)"
        return 1
    fi
    return 0
}

# ============================================================================
# Document Processing
# ============================================================================

process_documents() {
    local film_name="$1"
    local output_dir="$2"

    log_info "Processing documents for: $film_name"

    local film_data_dir="${PROJECT_ROOT}/data/films/${film_name}"
    local docs_output_dir="${output_dir}/rag/documents"

    mkdir -p "$docs_output_dir"

    # Check if film data exists
    if [ ! -d "$film_data_dir" ]; then
        log_warning "Film data directory not found: $film_data_dir"
        log_warning "Creating minimal placeholder data"

        mkdir -p "$film_data_dir/characters"
        cat > "${film_data_dir}/README.md" <<EOF
# ${film_name^} Film Data

This is a placeholder for ${film_name} film metadata.
Add character descriptions, scene information, and other data here.
EOF
    fi

    # Process character documents
    if [ -d "${film_data_dir}/characters" ]; then
        log_info "Processing character documents..."

        # Check if document processor exists
        if [ ! -f "$DOCUMENT_PROCESSOR" ]; then
            log_warning "Document processor not found, using basic copy"
            cp -r "${film_data_dir}/characters" "${docs_output_dir}/"
        else
            # Run document processor (CPU-only)
            python "$DOCUMENT_PROCESSOR" \
                --input-dir "${film_data_dir}/characters" \
                --output-dir "${docs_output_dir}/characters" \
                --format markdown 2>&1 | tee "${output_dir}/rag/logs/document_processing.log" || {
                    log_warning "Document processor failed, using fallback"
                    mkdir -p "${docs_output_dir}/characters"
                    cp -r "${film_data_dir}/characters/"* "${docs_output_dir}/characters/" 2>/dev/null || true
                }
        fi

        local char_docs=$(find "${docs_output_dir}/characters" -type f 2>/dev/null | wc -l)
        log_success "Processed $char_docs character documents"
    else
        log_warning "No character documents found in: ${film_data_dir}/characters"
    fi

    # Process film metadata
    if [ -f "${film_data_dir}/README.md" ]; then
        log_info "Processing film metadata..."
        cp "${film_data_dir}/README.md" "${docs_output_dir}/film_info.md"
        log_success "Film metadata copied"
    fi

    # Create document manifest
    local total_docs=$(find "$docs_output_dir" -type f 2>/dev/null | wc -l)

    cat > "${docs_output_dir}/manifest.json" <<EOF
{
  "film_name": "$film_name",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "total_documents": $total_docs,
  "output_dir": "$docs_output_dir"
}
EOF

    log_success "Document processing complete: $total_docs documents"
}

# ============================================================================
# Knowledge Base Ingestion (CPU-only embeddings)
# ============================================================================

ingest_knowledge_base() {
    local film_name="$1"
    local output_dir="$2"
    local embedding_model="$3"

    log_info "Building knowledge base for: $film_name"
    log_info "Using CPU-only embedding model: $embedding_model"

    local docs_dir="${output_dir}/rag/documents"
    local kb_output_dir="${output_dir}/rag/knowledge_base"

    mkdir -p "$kb_output_dir"

    # Check if knowledge ingester exists
    if [ ! -f "$KNOWLEDGE_INGESTER" ]; then
        log_error "Knowledge ingester script not found: $KNOWLEDGE_INGESTER"
        log_warning "Creating placeholder knowledge base"

        cat > "${kb_output_dir}/index_metadata.json" <<EOF
{
  "film_name": "$film_name",
  "embedding_model": "$embedding_model",
  "device": "cpu",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "status": "placeholder",
  "note": "Knowledge ingester script not available"
}
EOF
        return 0
    fi

    # Ensure we're using CPU device
    export CUDA_VISIBLE_DEVICES=""  # Force CPU-only

    log_info "Starting knowledge base ingestion (this may take 5-15 minutes)..."

    # Run knowledge ingester with CPU-only settings
    if python "$KNOWLEDGE_INGESTER" \
        --film-name "$film_name" \
        --embedding-model "$embedding_model" \
        --device cpu \
        --batch-size 32 \
        --chunk-size 512 \
        --output-dir "$kb_output_dir" 2>&1 | tee "${output_dir}/rag/logs/knowledge_ingestion.log"
    then
        log_success "Knowledge base created successfully"

        # Check index size
        if [ -f "${kb_output_dir}/faiss.index" ]; then
            local index_size=$(du -h "${kb_output_dir}/faiss.index" | cut -f1)
            log_info "FAISS index size: $index_size"
        fi

        return 0
    else
        log_error "Knowledge base ingestion failed"

        # Create error placeholder
        cat > "${kb_output_dir}/index_metadata.json" <<EOF
{
  "film_name": "$film_name",
  "embedding_model": "$embedding_model",
  "device": "cpu",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "status": "failed",
  "note": "Knowledge ingestion encountered errors"
}
EOF
        return 1
    fi
}

# ============================================================================
# Main Processing Function
# ============================================================================

process_stage3() {
    local film_name="$1"
    local output_dir="$2"
    local embedding_model="$3"

    log_info "========================================="
    log_info "CPU Stage 3: RAG Preparation"
    log_info "========================================="
    log_info "Film: $film_name"
    log_info "Output: $output_dir"
    log_info "Embedding Model: $embedding_model"
    log_info "Device: CPU (forced)"
    log_info ""

    # Create output directories
    mkdir -p "${output_dir}/rag/documents"
    mkdir -p "${output_dir}/rag/knowledge_base"
    mkdir -p "${output_dir}/rag/logs"

    # Check initial resources
    if ! monitor_resources; then
        log_error "Resource check failed"
        exit 1
    fi

    # ========================================================================
    # Phase 1: Document Processing
    # ========================================================================

    log_info ""
    log_info "Phase 1/2: Document Processing"
    log_info "========================================="

    process_documents "$film_name" "$output_dir"

    # ========================================================================
    # Phase 2: Knowledge Base Ingestion
    # ========================================================================

    log_info ""
    log_info "Phase 2/2: Knowledge Base Ingestion (CPU-only embeddings)"
    log_info "========================================="

    ingest_knowledge_base "$film_name" "$output_dir" "$embedding_model"

    # ========================================================================
    # Generate Summary
    # ========================================================================

    log_info ""
    log_info "Generating RAG preparation summary..."

    local summary_file="${output_dir}/rag/rag_summary.json"
    local doc_count=$(find "${output_dir}/rag/documents" -type f 2>/dev/null | wc -l)
    local kb_status="unknown"

    if [ -f "${output_dir}/rag/knowledge_base/index_metadata.json" ]; then
        kb_status=$(cat "${output_dir}/rag/knowledge_base/index_metadata.json" | \
                    python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))" 2>/dev/null || echo "unknown")
    fi

    cat > "$summary_file" <<EOF
{
  "stage": "cpu_stage3_rag_prep",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "film_name": "$film_name",
  "output_dir": "$output_dir",
  "document_processing": {
    "total_documents": $doc_count,
    "output_dir": "${output_dir}/rag/documents"
  },
  "knowledge_base": {
    "status": "$kb_status",
    "embedding_model": "$embedding_model",
    "device": "cpu",
    "output_dir": "${output_dir}/rag/knowledge_base"
  },
  "completed": true
}
EOF

    log_success "RAG summary saved: $summary_file"

    # ========================================================================
    # Final Summary
    # ========================================================================

    log_info ""
    log_info "========================================="
    log_info "CPU Stage 3 COMPLETED"
    log_info "========================================="
    log_info "Documents processed: $doc_count"
    log_info "Knowledge base status: $kb_status"
    log_info "Embedding model: $embedding_model (CPU)"
    log_info "Output: ${output_dir}/rag/"
    log_info ""

    monitor_resources || true
}

# ============================================================================
# Argument Parsing
# ============================================================================

usage() {
    echo "Usage: $0 FILM_NAME OUTPUT_DIR [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  FILM_NAME    Name of the film (e.g., luca, coco)"
    echo "  OUTPUT_DIR   Output directory for RAG data"
    echo ""
    echo "Options:"
    echo "  --embedding-model MODEL  CPU-only embedding model"
    echo "                          (default: $DEFAULT_EMBEDDING_MODEL)"
    echo "  --help                  Show this help"
    echo ""
    echo "Recommended CPU-only embedding models:"
    echo "  - sentence-transformers/all-MiniLM-L6-v2 (fastest, 80MB)"
    echo "  - sentence-transformers/all-mpnet-base-v2 (better quality, 420MB)"
    echo "  - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (multilingual)"
    echo ""
    echo "Example:"
    echo "  $0 luca /mnt/data/ai_data/datasets/3d-anime/luca"
    exit 1
}

FILM_NAME=""
OUTPUT_DIR=""
EMBEDDING_MODEL="$DEFAULT_EMBEDDING_MODEL"

while [[ $# -gt 0 ]]; do
    case $1 in
        --embedding-model)
            EMBEDDING_MODEL="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            if [ -z "$FILM_NAME" ]; then
                FILM_NAME="$1"
            elif [ -z "$OUTPUT_DIR" ]; then
                OUTPUT_DIR="$1"
            else
                log_error "Unknown argument: $1"
                usage
            fi
            shift
            ;;
    esac
done

if [ -z "$FILM_NAME" ] || [ -z "$OUTPUT_DIR" ]; then
    log_error "Missing required arguments"
    usage
fi

# Validate film name (alphanumeric and underscore only)
if [[ ! "$FILM_NAME" =~ ^[a-zA-Z0-9_]+$ ]]; then
    log_error "Invalid film name: $FILM_NAME (use only letters, numbers, underscore)"
    exit 1
fi

# Export variables
export DOCUMENT_PROCESSOR KNOWLEDGE_INGESTER
export RED GREEN YELLOW BLUE NC
export CUDA_VISIBLE_DEVICES=""  # Force CPU-only

# Run processing
process_stage3 "$FILM_NAME" "$OUTPUT_DIR" "$EMBEDDING_MODEL"

log_success "All CPU Stage 3 tasks completed successfully"
