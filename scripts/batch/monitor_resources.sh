#!/bin/bash
################################################################################
# Resource Monitoring Script
#
# Real-time monitoring of system resources:
#   - CPU usage (per-core and overall)
#   - RAM usage (used/total/percentage)
#   - GPU VRAM usage (if NVIDIA GPU available)
#   - GPU temperature and utilization
#   - Disk space (multiple mount points)
#   - Process tracking (top consumers)
#
# Features:
#   - Continuous monitoring with configurable interval
#   - CSV log output for analysis
#   - Threshold-based warnings
#   - Background daemon mode
#   - Graceful shutdown (SIGTERM/SIGINT)
#
# Usage:
#   # Foreground mode (print to console)
#   bash scripts/batch/monitor_resources.sh --interval 10
#
#   # Background daemon mode (log to file)
#   bash scripts/batch/monitor_resources.sh --daemon --log-dir /tmp/monitoring
#
#   # Stop daemon
#   pkill -f monitor_resources.sh
#
# Author: Animation AI Studio
# Date: 2025-12-04
################################################################################

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_INTERVAL=10        # Monitoring interval in seconds
DEFAULT_LOG_DIR="/tmp/resource_monitoring"

# Warning thresholds
CPU_WARNING_THRESHOLD=85   # CPU usage % (warning level)
CPU_CRITICAL_THRESHOLD=95  # CPU usage % (critical level)
RAM_WARNING_THRESHOLD=80   # RAM usage %
RAM_CRITICAL_THRESHOLD=90  # RAM usage %
VRAM_WARNING_THRESHOLD=80  # GPU VRAM usage %
VRAM_CRITICAL_THRESHOLD=90 # GPU VRAM usage %
DISK_WARNING_THRESHOLD=90  # Disk usage %
DISK_CRITICAL_THRESHOLD=95 # Disk usage %
GPU_TEMP_WARNING=80        # GPU temperature (°C)
GPU_TEMP_CRITICAL=85       # GPU temperature (°C)

# ============================================================================
# Color Output
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_critical() { echo -e "${RED}[CRIT]${NC} $1"; }

# ============================================================================
# System Information
# ============================================================================

get_timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

get_cpu_count() {
    if command -v nproc &> /dev/null; then
        nproc
    else
        echo "4"
    fi
}

get_cpu_usage() {
    # Get overall CPU usage percentage
    if command -v mpstat &> /dev/null; then
        mpstat 1 1 | awk '/Average/ {print 100 - $NF}'
    else
        # Fallback: use top
        top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//'
    fi
}

get_cpu_per_core() {
    # Get per-core CPU usage (if mpstat available)
    if command -v mpstat &> /dev/null; then
        mpstat -P ALL 1 1 | awk '/Average.*[0-9]/ {printf "Core%s:%.1f%% ", $2, 100-$NF}'
    else
        echo "N/A"
    fi
}

get_ram_usage() {
    # Returns: used_gb total_gb percentage
    if command -v free &> /dev/null; then
        free -g | awk '/^Mem:/ {printf "%.1f %.1f %.1f", $3, $2, ($3/$2)*100}'
    else
        echo "0 0 0"
    fi
}

get_disk_usage() {
    # Get disk usage for key mount points
    local mount_point="$1"
    df -h "$mount_point" 2>/dev/null | awk 'NR==2 {printf "%s %s %s", $3, $2, $5}' || echo "N/A N/A N/A"
}

get_gpu_info() {
    # Get GPU VRAM, utilization, and temperature (NVIDIA only)
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
                   --format=csv,noheader,nounits 2>/dev/null || echo "N/A,N/A,N/A,N/A"
    else
        echo "N/A,N/A,N/A,N/A"
    fi
}

get_top_processes() {
    # Get top 3 CPU-consuming processes
    if command -v ps &> /dev/null; then
        ps aux --sort=-%cpu | head -n 4 | tail -n 3 | \
            awk '{printf "%s(%.1f%%) ", $11, $3}' || echo "N/A"
    else
        echo "N/A"
    fi
}

# ============================================================================
# Monitoring Functions
# ============================================================================

check_thresholds() {
    local metric_name="$1"
    local value="$2"
    local warning_threshold="$3"
    local critical_threshold="$4"
    local unit="${5:-%}"

    # Convert to integer for comparison
    local value_int=$(printf "%.0f" "$value" 2>/dev/null || echo "0")

    if [ "$value_int" -ge "$critical_threshold" ]; then
        log_critical "${metric_name}: ${value}${unit} (>= ${critical_threshold}${unit})"
        return 2
    elif [ "$value_int" -ge "$warning_threshold" ]; then
        log_warning "${metric_name}: ${value}${unit} (>= ${warning_threshold}${unit})"
        return 1
    fi

    return 0
}

monitor_once() {
    local log_file="$1"
    local daemon_mode="$2"

    local timestamp=$(get_timestamp)
    local cpu_cores=$(get_cpu_count)
    local cpu_usage=$(get_cpu_usage)
    local cpu_per_core=$(get_cpu_per_core)

    # RAM info
    read -r ram_used ram_total ram_pct <<< $(get_ram_usage)

    # GPU info
    IFS=',' read -r vram_used vram_total gpu_util gpu_temp <<< $(get_gpu_info)

    # Calculate VRAM percentage
    if [ "$vram_used" != "N/A" ] && [ "$vram_total" != "N/A" ]; then
        vram_pct=$(awk "BEGIN {printf \"%.1f\", ($vram_used/$vram_total)*100}")
    else
        vram_pct="N/A"
    fi

    # Disk usage (check multiple mount points)
    read -r disk_used disk_total disk_pct <<< $(get_disk_usage "/")
    read -r mnt_used mnt_total mnt_pct <<< $(get_disk_usage "/mnt")

    # Top processes
    local top_procs=$(get_top_processes)

    # ========================================================================
    # Console Output (if not daemon mode)
    # ========================================================================

    if [ "$daemon_mode" = "false" ]; then
        echo ""
        echo -e "${CYAN}=========================================${NC}"
        echo -e "${CYAN}Resource Monitor - $(date)${NC}"
        echo -e "${CYAN}=========================================${NC}"

        # CPU
        echo -e "${BLUE}CPU:${NC}"
        echo "  Cores: $cpu_cores"
        echo "  Overall: ${cpu_usage}%"
        if [ "$cpu_per_core" != "N/A" ]; then
            echo "  Per-core: $cpu_per_core"
        fi
        check_thresholds "CPU" "$cpu_usage" "$CPU_WARNING_THRESHOLD" "$CPU_CRITICAL_THRESHOLD"

        # RAM
        echo -e "${BLUE}RAM:${NC}"
        echo "  Used: ${ram_used}GB / ${ram_total}GB (${ram_pct}%)"
        check_thresholds "RAM" "$ram_pct" "$RAM_WARNING_THRESHOLD" "$RAM_CRITICAL_THRESHOLD"

        # GPU
        if [ "$vram_used" != "N/A" ]; then
            echo -e "${BLUE}GPU:${NC}"
            echo "  VRAM: ${vram_used}MB / ${vram_total}MB (${vram_pct}%)"
            echo "  Utilization: ${gpu_util}%"
            echo "  Temperature: ${gpu_temp}°C"

            if [ "$vram_pct" != "N/A" ]; then
                check_thresholds "GPU VRAM" "$vram_pct" "$VRAM_WARNING_THRESHOLD" "$VRAM_CRITICAL_THRESHOLD"
            fi

            if [ "$gpu_temp" != "N/A" ]; then
                check_thresholds "GPU Temp" "$gpu_temp" "$GPU_TEMP_WARNING" "$GPU_TEMP_CRITICAL" "°C"
            fi
        else
            echo -e "${BLUE}GPU:${NC} Not available (no NVIDIA GPU detected)"
        fi

        # Disk
        echo -e "${BLUE}Disk:${NC}"
        echo "  /     : ${disk_used} / ${disk_total} (${disk_pct})"
        if [ "$mnt_used" != "N/A" ]; then
            echo "  /mnt  : ${mnt_used} / ${mnt_total} (${mnt_pct})"
        fi

        # Top processes
        echo -e "${BLUE}Top CPU Consumers:${NC}"
        echo "  $top_procs"

        echo -e "${CYAN}=========================================${NC}"
    fi

    # ========================================================================
    # CSV Log Output
    # ========================================================================

    if [ -n "$log_file" ]; then
        # Create CSV header if file doesn't exist
        if [ ! -f "$log_file" ]; then
            echo "timestamp,cpu_cores,cpu_usage_pct,ram_used_gb,ram_total_gb,ram_pct,vram_used_mb,vram_total_mb,vram_pct,gpu_util_pct,gpu_temp_c,disk_root_used,disk_root_total,disk_root_pct,disk_mnt_pct,top_processes" > "$log_file"
        fi

        # Strip percentage signs for CSV
        disk_pct_num=$(echo "$disk_pct" | sed 's/%//')
        mnt_pct_num=$(echo "$mnt_pct" | sed 's/%//')

        # Append data row
        echo "${timestamp},${cpu_cores},${cpu_usage},${ram_used},${ram_total},${ram_pct},${vram_used},${vram_total},${vram_pct},${gpu_util},${gpu_temp},${disk_used},${disk_total},${disk_pct_num},${mnt_pct_num},\"${top_procs}\"" >> "$log_file"
    fi
}

# ============================================================================
# Continuous Monitoring Loop
# ============================================================================

KEEP_RUNNING=true

trap_handler() {
    log_info "Received shutdown signal, stopping monitor..."
    KEEP_RUNNING=false
}

trap trap_handler SIGTERM SIGINT

monitor_loop() {
    local interval="$1"
    local log_file="$2"
    local daemon_mode="$3"

    log_info "Starting resource monitor (interval: ${interval}s, daemon: ${daemon_mode})"

    if [ -n "$log_file" ]; then
        log_info "Logging to: $log_file"
    fi

    while $KEEP_RUNNING; do
        monitor_once "$log_file" "$daemon_mode"
        sleep "$interval"
    done

    log_info "Resource monitor stopped"
}

# ============================================================================
# Argument Parsing
# ============================================================================

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --interval N       Monitoring interval in seconds (default: $DEFAULT_INTERVAL)"
    echo "  --daemon           Run in background daemon mode"
    echo "  --log-dir DIR      Directory for log files (default: $DEFAULT_LOG_DIR)"
    echo "  --once             Run once and exit (useful for scripts)"
    echo "  --help             Show this help"
    echo ""
    echo "Examples:"
    echo "  # Foreground mode with 5-second intervals"
    echo "  $0 --interval 5"
    echo ""
    echo "  # Background daemon mode"
    echo "  $0 --daemon --log-dir /var/log/monitoring"
    echo ""
    echo "  # Single snapshot"
    echo "  $0 --once"
    echo ""
    echo "  # Stop daemon"
    echo "  pkill -f monitor_resources.sh"
    exit 1
}

INTERVAL=$DEFAULT_INTERVAL
LOG_DIR="$DEFAULT_LOG_DIR"
DAEMON_MODE="false"
RUN_ONCE="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        --daemon)
            DAEMON_MODE="true"
            shift
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --once)
            RUN_ONCE="true"
            shift
            ;;
        --help)
            usage
            ;;
        *)
            log_critical "Unknown argument: $1"
            usage
            ;;
    esac
done

# Validate interval
if [ "$INTERVAL" -lt 1 ]; then
    log_critical "Interval must be >= 1 second"
    exit 1
fi

# Setup log file
LOG_FILE=""
if [ "$DAEMON_MODE" = "true" ] || [ "$RUN_ONCE" = "true" ]; then
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/resources_$(date +%Y%m%d_%H%M%S).csv"
fi

# ============================================================================
# Main Execution
# ============================================================================

if [ "$RUN_ONCE" = "true" ]; then
    # Single snapshot mode
    monitor_once "$LOG_FILE" "false"
    if [ -n "$LOG_FILE" ]; then
        log_success "Snapshot saved to: $LOG_FILE"
    fi
else
    # Continuous monitoring mode
    if [ "$DAEMON_MODE" = "true" ]; then
        log_info "Starting daemon mode (PID: $$)"
        log_info "To stop: pkill -f monitor_resources.sh"
        # Redirect output to log
        exec &>> "${LOG_DIR}/monitor_daemon.log"
    fi

    monitor_loop "$INTERVAL" "$LOG_FILE" "$DAEMON_MODE"
fi
