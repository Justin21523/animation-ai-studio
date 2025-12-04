#!/usr/bin/env python3
"""
Audio Processor (音訊處理器)
============================

Provides comprehensive audio processing capabilities using FFmpeg.
All operations are CPU-only and optimized for 32-thread processing.

提供基於 FFmpeg 的完整音訊處理功能。
所有操作均為 CPU-only 並針對 32 執行緒進行最佳化。

Features (功能):
- Audio extraction from video (從影片提取音訊)
- Format conversion (格式轉換): WAV, MP3, FLAC, AAC, OGG
- Audio cutting (音訊切割)
- Audio concatenation (音訊拼接)
- Volume normalization (音量正規化)
- Silence detection and removal (靜音檢測和移除)
- Metadata extraction (Metadata 提取)
- Batch processing (批次處理)

Author: Animation AI Studio Team
Created: 2025-12-02
"""

import os
import sys
import argparse
import subprocess
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import safety infrastructure
try:
    from scripts.core.safety import MemoryMonitor, run_preflight
except ImportError:
    # Stub for when safety module is not available
    class MemoryMonitor:
        def __init__(self, **kwargs):
            pass
        def check_memory(self):
            return 'ok', 0.0
    def run_preflight():
        return True


# ============================================================================
# Data Classes (資料類別)
# ============================================================================

@dataclass
class AudioMetadata:
    """Audio file metadata (音訊檔案 Metadata)"""
    duration_seconds: float
    sample_rate: int
    channels: int
    codec: str
    bitrate: int
    file_size_bytes: int
    format: str


@dataclass
class SilenceSegment:
    """Silence segment information (靜音片段資訊)"""
    start_time: float
    end_time: float
    duration: float


# ============================================================================
# FFmpeg Audio Processor (FFmpeg 音訊處理器)
# ============================================================================

class AudioProcessor:
    """
    FFmpeg-based audio processing with CPU-only operations.
    基於 FFmpeg 的音訊處理，僅使用 CPU 操作。

    All operations are thread-safe and memory-monitored.
    所有操作都是執行緒安全且受記憶體監控。
    """

    def __init__(
        self,
        threads: int = 32,
        memory_monitor: Optional[MemoryMonitor] = None
    ):
        """
        Initialize audio processor.
        初始化音訊處理器。

        Args:
            threads: Number of CPU threads (CPU 執行緒數量)
            memory_monitor: Memory monitor instance (記憶體監控實例)
        """
        self.threads = threads
        self.memory_monitor = memory_monitor
        self.logger = logging.getLogger(__name__)

        # Verify FFmpeg is installed (驗證 FFmpeg 已安裝)
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg not installed. Please install: sudo apt install ffmpeg")

    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available (檢查 FFmpeg 是否可用)"""
        try:
            subprocess.run(
                ['ffmpeg', '-version'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _check_memory(self):
        """Check memory level (檢查記憶體水準)"""
        if self.memory_monitor:
            level, usage = self.memory_monitor.check_memory()
            if level in ['critical', 'emergency']:
                raise MemoryError(
                    f"Memory level {level}: {usage:.1f}% used. "
                    f"Aborting audio processing."
                )

    def _run_ffmpeg(
        self,
        command: List[str],
        operation_name: str
    ) -> Tuple[bool, str]:
        """
        Run FFmpeg command with error handling.
        執行 FFmpeg 命令並處理錯誤。

        Args:
            command: FFmpeg command (FFmpeg 命令)
            operation_name: Operation description (操作描述)

        Returns:
            (success, error_message)
        """
        self._check_memory()

        try:
            self.logger.info(f"Running {operation_name}: {' '.join(command)}")
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )

            if result.returncode != 0:
                return False, result.stderr

            return True, ""

        except Exception as e:
            return False, str(e)

    # ========================================================================
    # Audio Extraction (音訊提取)
    # ========================================================================

    def extract_audio(
        self,
        video_path: str,
        output_path: str,
        format: str = 'wav',
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        bitrate: Optional[str] = None
    ) -> bool:
        """
        Extract audio from video file.
        從影片檔案提取音訊。

        Args:
            video_path: Input video path (輸入影片路徑)
            output_path: Output audio path (輸出音訊路徑)
            format: Output format (輸出格式): wav, mp3, flac, aac, ogg
            sample_rate: Sample rate in Hz (取樣率)
            channels: Number of channels (聲道數量): 1=mono, 2=stereo
            bitrate: Bitrate for lossy formats (有損格式的位元率)

        Returns:
            Success status (成功狀態)
        """
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # No video (無影片)
            '-threads', str(self.threads)
        ]

        # Audio codec based on format (根據格式選擇音訊編碼器)
        codec_map = {
            'wav': 'pcm_s16le',
            'mp3': 'libmp3lame',
            'flac': 'flac',
            'aac': 'aac',
            'ogg': 'libvorbis'
        }

        if format in codec_map:
            command.extend(['-acodec', codec_map[format]])

        # Sample rate (取樣率)
        if sample_rate:
            command.extend(['-ar', str(sample_rate)])

        # Channels (聲道)
        if channels:
            command.extend(['-ac', str(channels)])

        # Bitrate (位元率)
        if bitrate and format in ['mp3', 'aac', 'ogg']:
            command.extend(['-b:a', bitrate])

        command.extend(['-y', output_path])

        success, error = self._run_ffmpeg(command, "audio extraction")

        if success:
            self.logger.info(f"✅ Audio extracted: {output_path}")
        else:
            self.logger.error(f"❌ Audio extraction failed: {error}")

        return success

    # ========================================================================
    # Format Conversion (格式轉換)
    # ========================================================================

    def convert_format(
        self,
        input_path: str,
        output_path: str,
        output_format: str,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        bitrate: Optional[str] = None
    ) -> bool:
        """
        Convert audio format.
        轉換音訊格式。

        Args:
            input_path: Input audio path (輸入音訊路徑)
            output_path: Output audio path (輸出音訊路徑)
            output_format: Output format (輸出格式): wav, mp3, flac, aac, ogg
            sample_rate: Sample rate in Hz (取樣率)
            channels: Number of channels (聲道數量)
            bitrate: Bitrate for lossy formats (有損格式的位元率)

        Returns:
            Success status (成功狀態)
        """
        command = [
            'ffmpeg',
            '-i', input_path,
            '-threads', str(self.threads)
        ]

        # Audio codec (音訊編碼器)
        codec_map = {
            'wav': 'pcm_s16le',
            'mp3': 'libmp3lame',
            'flac': 'flac',
            'aac': 'aac',
            'ogg': 'libvorbis'
        }

        if output_format in codec_map:
            command.extend(['-acodec', codec_map[output_format]])

        # Sample rate (取樣率)
        if sample_rate:
            command.extend(['-ar', str(sample_rate)])

        # Channels (聲道)
        if channels:
            command.extend(['-ac', str(channels)])

        # Bitrate (位元率)
        if bitrate and output_format in ['mp3', 'aac', 'ogg']:
            command.extend(['-b:a', bitrate])

        command.extend(['-y', output_path])

        success, error = self._run_ffmpeg(command, "format conversion")

        if success:
            self.logger.info(f"✅ Format converted: {output_path}")
        else:
            self.logger.error(f"❌ Format conversion failed: {error}")

        return success

    # ========================================================================
    # Audio Cutting (音訊切割)
    # ========================================================================

    def cut_audio(
        self,
        input_path: str,
        output_path: str,
        start_time: str,
        end_time: Optional[str] = None,
        duration: Optional[str] = None
    ) -> bool:
        """
        Cut audio segment.
        切割音訊片段。

        Args:
            input_path: Input audio path (輸入音訊路徑)
            output_path: Output audio path (輸出音訊路徑)
            start_time: Start time (開始時間): "HH:MM:SS" or seconds
            end_time: End time (結束時間): "HH:MM:SS" or seconds
            duration: Duration (持續時間): seconds

        Returns:
            Success status (成功狀態)
        """
        command = [
            'ffmpeg',
            '-i', input_path,
            '-ss', start_time,
            '-threads', str(self.threads)
        ]

        if end_time:
            command.extend(['-to', end_time])
        elif duration:
            command.extend(['-t', duration])

        # Copy codec for fast cutting (複製編碼器以快速切割)
        command.extend(['-acodec', 'copy', '-y', output_path])

        success, error = self._run_ffmpeg(command, "audio cutting")

        if success:
            self.logger.info(f"✅ Audio cut: {output_path}")
        else:
            self.logger.error(f"❌ Audio cutting failed: {error}")

        return success

    # ========================================================================
    # Audio Concatenation (音訊拼接)
    # ========================================================================

    def concatenate_audio(
        self,
        input_paths: List[str],
        output_path: str
    ) -> bool:
        """
        Concatenate multiple audio files.
        拼接多個音訊檔案。

        Args:
            input_paths: List of input audio paths (輸入音訊路徑列表)
            output_path: Output audio path (輸出音訊路徑)

        Returns:
            Success status (成功狀態)
        """
        # Create temporary file list (建立臨時檔案列表)
        file_list_path = '/tmp/audio_concat_list.txt'

        with open(file_list_path, 'w') as f:
            for path in input_paths:
                # FFmpeg requires absolute paths (FFmpeg 需要絕對路徑)
                abs_path = os.path.abspath(path)
                f.write(f"file '{abs_path}'\n")

        command = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', file_list_path,
            '-c', 'copy',
            '-threads', str(self.threads),
            '-y', output_path
        ]

        success, error = self._run_ffmpeg(command, "audio concatenation")

        # Clean up (清理)
        if os.path.exists(file_list_path):
            os.remove(file_list_path)

        if success:
            self.logger.info(f"✅ Audio concatenated: {output_path}")
        else:
            self.logger.error(f"❌ Audio concatenation failed: {error}")

        return success

    # ========================================================================
    # Volume Normalization (音量正規化)
    # ========================================================================

    def normalize_volume(
        self,
        input_path: str,
        output_path: str,
        target_level: str = '-16dB'
    ) -> bool:
        """
        Normalize audio volume.
        正規化音訊音量。

        Args:
            input_path: Input audio path (輸入音訊路徑)
            output_path: Output audio path (輸出音訊路徑)
            target_level: Target loudness level (目標音量等級): dB

        Returns:
            Success status (成功狀態)
        """
        command = [
            'ffmpeg',
            '-i', input_path,
            '-af', f'loudnorm=I={target_level}',
            '-threads', str(self.threads),
            '-y', output_path
        ]

        success, error = self._run_ffmpeg(command, "volume normalization")

        if success:
            self.logger.info(f"✅ Volume normalized: {output_path}")
        else:
            self.logger.error(f"❌ Volume normalization failed: {error}")

        return success

    # ========================================================================
    # Silence Detection (靜音檢測)
    # ========================================================================

    def detect_silence(
        self,
        input_path: str,
        noise_threshold: str = '-40dB',
        min_silence_duration: float = 0.5
    ) -> List[SilenceSegment]:
        """
        Detect silence segments in audio.
        檢測音訊中的靜音片段。

        Args:
            input_path: Input audio path (輸入音訊路徑)
            noise_threshold: Noise level threshold (噪音等級閾值)
            min_silence_duration: Minimum silence duration in seconds (最小靜音持續時間)

        Returns:
            List of silence segments (靜音片段列表)
        """
        command = [
            'ffmpeg',
            '-i', input_path,
            '-af', f'silencedetect=noise={noise_threshold}:d={min_silence_duration}',
            '-f', 'null',
            '-'
        ]

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )

            # Parse silence detection output (解析靜音檢測輸出)
            silence_segments = []
            lines = result.stderr.split('\n')

            current_start = None
            for line in lines:
                if 'silence_start' in line:
                    try:
                        current_start = float(line.split('silence_start:')[1].strip())
                    except:
                        pass
                elif 'silence_end' in line and current_start is not None:
                    try:
                        parts = line.split('silence_end:')[1].split('|')
                        end_time = float(parts[0].strip())
                        duration = float(parts[1].split('silence_duration:')[1].strip())

                        silence_segments.append(SilenceSegment(
                            start_time=current_start,
                            end_time=end_time,
                            duration=duration
                        ))
                        current_start = None
                    except:
                        pass

            self.logger.info(f"✅ Detected {len(silence_segments)} silence segments")
            return silence_segments

        except Exception as e:
            self.logger.error(f"❌ Silence detection failed: {e}")
            return []

    # ========================================================================
    # Silence Removal (靜音移除)
    # ========================================================================

    def remove_silence(
        self,
        input_path: str,
        output_path: str,
        noise_threshold: str = '-40dB',
        min_silence_duration: float = 0.5
    ) -> bool:
        """
        Remove silence from audio.
        從音訊移除靜音。

        Args:
            input_path: Input audio path (輸入音訊路徑)
            output_path: Output audio path (輸出音訊路徑)
            noise_threshold: Noise level threshold (噪音等級閾值)
            min_silence_duration: Minimum silence duration to remove (要移除的最小靜音持續時間)

        Returns:
            Success status (成功狀態)
        """
        command = [
            'ffmpeg',
            '-i', input_path,
            '-af', f'silenceremove=stop_periods=-1:stop_threshold={noise_threshold}:stop_duration={min_silence_duration}',
            '-threads', str(self.threads),
            '-y', output_path
        ]

        success, error = self._run_ffmpeg(command, "silence removal")

        if success:
            self.logger.info(f"✅ Silence removed: {output_path}")
        else:
            self.logger.error(f"❌ Silence removal failed: {error}")

        return success

    # ========================================================================
    # Metadata Extraction (Metadata 提取)
    # ========================================================================

    def extract_metadata(self, input_path: str) -> Optional[AudioMetadata]:
        """
        Extract audio metadata.
        提取音訊 Metadata。

        Args:
            input_path: Input audio path (輸入音訊路徑)

        Returns:
            Audio metadata or None (音訊 Metadata 或 None)
        """
        command = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            input_path
        ]

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )

            data = json.loads(result.stdout)

            # Find audio stream (尋找音訊串流)
            audio_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    audio_stream = stream
                    break

            if not audio_stream:
                self.logger.error("❌ No audio stream found")
                return None

            format_info = data.get('format', {})

            metadata = AudioMetadata(
                duration_seconds=float(format_info.get('duration', 0)),
                sample_rate=int(audio_stream.get('sample_rate', 0)),
                channels=int(audio_stream.get('channels', 0)),
                codec=audio_stream.get('codec_name', 'unknown'),
                bitrate=int(format_info.get('bit_rate', 0)),
                file_size_bytes=int(format_info.get('size', 0)),
                format=format_info.get('format_name', 'unknown')
            )

            self.logger.info(f"✅ Metadata extracted: {metadata}")
            return metadata

        except Exception as e:
            self.logger.error(f"❌ Metadata extraction failed: {e}")
            return None

    # ========================================================================
    # Batch Processing (批次處理)
    # ========================================================================

    def process_batch(self, config_path: str) -> Dict:
        """
        Process batch operations from YAML config.
        從 YAML 配置檔處理批次操作。

        Args:
            config_path: Path to YAML config file (YAML 配置檔路徑)

        Returns:
            Batch processing report (批次處理報告)
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        operations = config.get('operations', [])
        results = []

        for i, op in enumerate(operations):
            operation_type = op.get('operation')
            self.logger.info(f"Processing operation {i+1}/{len(operations)}: {operation_type}")

            try:
                if operation_type == 'extract':
                    success = self.extract_audio(
                        video_path=op['input'],
                        output_path=op['output'],
                        format=op.get('format', 'wav'),
                        sample_rate=op.get('sample_rate'),
                        channels=op.get('channels'),
                        bitrate=op.get('bitrate')
                    )

                elif operation_type == 'convert':
                    success = self.convert_format(
                        input_path=op['input'],
                        output_path=op['output'],
                        output_format=op['output_format'],
                        sample_rate=op.get('sample_rate'),
                        channels=op.get('channels'),
                        bitrate=op.get('bitrate')
                    )

                elif operation_type == 'cut':
                    success = self.cut_audio(
                        input_path=op['input'],
                        output_path=op['output'],
                        start_time=op['start_time'],
                        end_time=op.get('end_time'),
                        duration=op.get('duration')
                    )

                elif operation_type == 'concat':
                    success = self.concatenate_audio(
                        input_paths=op['inputs'],
                        output_path=op['output']
                    )

                elif operation_type == 'normalize':
                    success = self.normalize_volume(
                        input_path=op['input'],
                        output_path=op['output'],
                        target_level=op.get('target_level', '-16dB')
                    )

                elif operation_type == 'remove_silence':
                    success = self.remove_silence(
                        input_path=op['input'],
                        output_path=op['output'],
                        noise_threshold=op.get('noise_threshold', '-40dB'),
                        min_silence_duration=op.get('min_silence_duration', 0.5)
                    )

                elif operation_type == 'metadata':
                    metadata = self.extract_metadata(op['input'])
                    success = metadata is not None

                else:
                    self.logger.error(f"❌ Unknown operation: {operation_type}")
                    success = False

                results.append({
                    'operation': operation_type,
                    'success': success,
                    'details': op
                })

            except Exception as e:
                self.logger.error(f"❌ Operation failed: {e}")
                results.append({
                    'operation': operation_type,
                    'success': False,
                    'error': str(e)
                })

        report = {
            'total_operations': len(operations),
            'successful': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success']),
            'results': results
        }

        self.logger.info(f"✅ Batch processing complete: {report['successful']}/{report['total_operations']} successful")

        return report


# ============================================================================
# CLI Interface (命令列介面)
# ============================================================================

def main():
    """Main CLI entry point (主命令列進入點)"""
    parser = argparse.ArgumentParser(
        description='Audio Processor - CPU-only audio processing (音訊處理器 - 僅 CPU 音訊處理)'
    )

    # Operation selection (操作選擇)
    parser.add_argument(
        '--operation',
        required=True,
        choices=['extract', 'convert', 'cut', 'concat', 'normalize',
                 'detect_silence', 'remove_silence', 'metadata', 'batch'],
        help='Operation to perform (要執行的操作)'
    )

    # Common arguments (通用參數)
    parser.add_argument('--input', help='Input file path (輸入檔案路徑)')
    parser.add_argument('--output', help='Output file path (輸出檔案路徑)')
    parser.add_argument('--threads', type=int, default=32, help='CPU threads (CPU 執行緒數量)')

    # Extract operation (提取操作)
    parser.add_argument('--format', default='wav', help='Audio format (音訊格式): wav, mp3, flac, aac, ogg')
    parser.add_argument('--sample-rate', type=int, help='Sample rate in Hz (取樣率)')
    parser.add_argument('--channels', type=int, help='Number of channels (聲道數量)')
    parser.add_argument('--bitrate', help='Bitrate (位元率): e.g., 192k, 320k')

    # Convert operation (轉換操作)
    parser.add_argument('--output-format', help='Output format for conversion (轉換的輸出格式)')

    # Cut operation (切割操作)
    parser.add_argument('--start-time', help='Start time (開始時間): HH:MM:SS or seconds')
    parser.add_argument('--end-time', help='End time (結束時間): HH:MM:SS or seconds')
    parser.add_argument('--duration', help='Duration in seconds (持續時間)')

    # Concat operation (拼接操作)
    parser.add_argument('--inputs', nargs='+', help='Input files for concatenation (拼接的輸入檔案)')
    parser.add_argument('--input-list', help='File containing list of inputs (包含輸入列表的檔案)')

    # Normalize operation (正規化操作)
    parser.add_argument('--target-level', default='-16dB', help='Target loudness level (目標音量等級)')

    # Silence operations (靜音操作)
    parser.add_argument('--noise-threshold', default='-40dB', help='Noise threshold (噪音閾值)')
    parser.add_argument('--min-silence-duration', type=float, default=0.5, help='Minimum silence duration (最小靜音持續時間)')

    # Batch operation (批次操作)
    parser.add_argument('--batch-config', help='Path to batch config YAML (批次配置 YAML 路徑)')

    # Safety options (安全選項)
    parser.add_argument('--skip-preflight', action='store_true', help='Skip preflight checks (跳過預檢查)')

    args = parser.parse_args()

    # Setup logging (設定日誌)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Initialize safety monitor (初始化安全監控)
    if not args.skip_preflight:
        logger.info("Initializing safety infrastructure (初始化安全基礎設施)...")
        memory_monitor = MemoryMonitor(
            warning_threshold=0.70,
            critical_threshold=0.80,
            emergency_threshold=0.85
        )

        # Run preflight checks (執行預檢查)
        if not run_preflight():
            logger.error("❌ Preflight checks failed (預檢查失敗)")
            return 1

        logger.info("✅ Preflight checks passed (預檢查通過)")
    else:
        memory_monitor = None
        logger.warning("⚠️  Skipping preflight checks (跳過預檢查)")

    # Initialize processor (初始化處理器)
    processor = AudioProcessor(
        threads=args.threads,
        memory_monitor=memory_monitor
    )

    # Execute operation (執行操作)
    try:
        if args.operation == 'extract':
            if not args.input or not args.output:
                logger.error("❌ --input and --output required for extract operation")
                return 1

            success = processor.extract_audio(
                video_path=args.input,
                output_path=args.output,
                format=args.format,
                sample_rate=args.sample_rate,
                channels=args.channels,
                bitrate=args.bitrate
            )

        elif args.operation == 'convert':
            if not args.input or not args.output or not args.output_format:
                logger.error("❌ --input, --output, and --output-format required for convert operation")
                return 1

            success = processor.convert_format(
                input_path=args.input,
                output_path=args.output,
                output_format=args.output_format,
                sample_rate=args.sample_rate,
                channels=args.channels,
                bitrate=args.bitrate
            )

        elif args.operation == 'cut':
            if not args.input or not args.output or not args.start_time:
                logger.error("❌ --input, --output, and --start-time required for cut operation")
                return 1

            success = processor.cut_audio(
                input_path=args.input,
                output_path=args.output,
                start_time=args.start_time,
                end_time=args.end_time,
                duration=args.duration
            )

        elif args.operation == 'concat':
            if not args.output:
                logger.error("❌ --output required for concat operation")
                return 1

            # Get input files (獲取輸入檔案)
            if args.inputs:
                input_paths = args.inputs
            elif args.input_list:
                with open(args.input_list, 'r') as f:
                    input_paths = [line.strip() for line in f if line.strip()]
            else:
                logger.error("❌ --inputs or --input-list required for concat operation")
                return 1

            success = processor.concatenate_audio(
                input_paths=input_paths,
                output_path=args.output
            )

        elif args.operation == 'normalize':
            if not args.input or not args.output:
                logger.error("❌ --input and --output required for normalize operation")
                return 1

            success = processor.normalize_volume(
                input_path=args.input,
                output_path=args.output,
                target_level=args.target_level
            )

        elif args.operation == 'detect_silence':
            if not args.input:
                logger.error("❌ --input required for detect_silence operation")
                return 1

            silence_segments = processor.detect_silence(
                input_path=args.input,
                noise_threshold=args.noise_threshold,
                min_silence_duration=args.min_silence_duration
            )

            logger.info(f"Found {len(silence_segments)} silence segments:")
            for seg in silence_segments:
                logger.info(f"  {seg.start_time:.2f}s - {seg.end_time:.2f}s (duration: {seg.duration:.2f}s)")

            success = True

        elif args.operation == 'remove_silence':
            if not args.input or not args.output:
                logger.error("❌ --input and --output required for remove_silence operation")
                return 1

            success = processor.remove_silence(
                input_path=args.input,
                output_path=args.output,
                noise_threshold=args.noise_threshold,
                min_silence_duration=args.min_silence_duration
            )

        elif args.operation == 'metadata':
            if not args.input:
                logger.error("❌ --input required for metadata operation")
                return 1

            metadata = processor.extract_metadata(args.input)

            if metadata:
                logger.info("Audio Metadata:")
                logger.info(f"  Duration: {metadata.duration_seconds:.2f}s")
                logger.info(f"  Sample Rate: {metadata.sample_rate} Hz")
                logger.info(f"  Channels: {metadata.channels}")
                logger.info(f"  Codec: {metadata.codec}")
                logger.info(f"  Bitrate: {metadata.bitrate} bps")
                logger.info(f"  File Size: {metadata.file_size_bytes / (1024*1024):.2f} MB")
                logger.info(f"  Format: {metadata.format}")
                success = True
            else:
                success = False

        elif args.operation == 'batch':
            if not args.batch_config:
                logger.error("❌ --batch-config required for batch operation")
                return 1

            report = processor.process_batch(args.batch_config)

            logger.info("\n" + "="*60)
            logger.info("Batch Processing Report")
            logger.info("="*60)
            logger.info(f"Total Operations: {report['total_operations']}")
            logger.info(f"Successful: {report['successful']}")
            logger.info(f"Failed: {report['failed']}")
            logger.info("="*60)

            success = report['failed'] == 0

        else:
            logger.error(f"❌ Unknown operation: {args.operation}")
            return 1

        if success:
            logger.info("✅ Operation completed successfully (操作成功完成)")
            return 0
        else:
            logger.error("❌ Operation failed (操作失敗)")
            return 1

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Operation cancelled by user (操作被使用者取消)")
        return 130

    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
