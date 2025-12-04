"""
Quality Analyzer

Assesses media quality and generates improvement recommendations.
CPU-only analysis with no processing.

Author: Animation AI Studio
Date: 2025-12-03
"""

import logging
from typing import List

from ..common import (
    MediaMetadata,
    VideoStreamInfo,
    AudioStreamInfo,
    QualityMetrics
)

logger = logging.getLogger(__name__)


class QualityAnalyzer:
    """
    Assess media quality and generate recommendations

    Features:
    - Resolution scoring (720p=70, 1080p=85, 4K=100)
    - Bitrate adequacy assessment
    - Framerate scoring (24/30/60 fps standards)
    - Codec efficiency evaluation
    - Audio quality scoring
    - Overall quality rating
    - Issue detection and recommendations
    - CPU-only analysis (no processing)

    Example:
        analyzer = QualityAnalyzer()
        metrics = analyzer.analyze_quality(metadata)

        print(f"Overall quality: {metrics.overall_score}/100")
        print(f"Rating: {metrics.quality_rating}")
        for issue in metrics.detected_issues:
            print(f"Issue: {issue}")
        for rec in metrics.recommendations:
            print(f"Recommendation: {rec}")
    """

    # Quality thresholds
    RESOLUTION_SCORES = {
        (7680, 4320): 100,  # 8K
        (3840, 2160): 100,  # 4K
        (2560, 1440): 95,   # 1440p
        (1920, 1080): 85,   # 1080p
        (1280, 720): 70,    # 720p
        (854, 480): 50,     # 480p
        (640, 360): 30,     # 360p
    }

    FRAMERATE_SCORES = {
        120: 100,
        60: 95,
        50: 90,
        30: 85,
        29.97: 85,
        25: 80,
        24: 80,
        23.976: 80
    }

    # Recommended bitrates (kbps) for 1080p
    RECOMMENDED_VIDEO_BITRATE_1080P = 5000  # 5 Mbps
    RECOMMENDED_AUDIO_BITRATE = 192  # 192 kbps

    def __init__(self):
        """Initialize quality analyzer"""
        logger.info("QualityAnalyzer initialized")

    def analyze_quality(self, metadata: MediaMetadata) -> QualityMetrics:
        """
        Analyze media quality and generate metrics

        Args:
            metadata: MediaMetadata from MetadataExtractor

        Returns:
            QualityMetrics with scores and recommendations
        """
        logger.info(f"Analyzing quality for: {metadata.path}")

        # Initialize metrics
        metrics = QualityMetrics()

        # Analyze video quality (if has video)
        if metadata.has_video:
            video_metrics = self._analyze_video_quality(metadata.primary_video_stream)
            metrics.resolution_score = video_metrics["resolution"]
            metrics.bitrate_score = video_metrics["bitrate"]
            metrics.framerate_score = video_metrics["framerate"]
            metrics.codec_efficiency = video_metrics["codec"]

        # Analyze audio quality (if has audio)
        if metadata.has_audio:
            audio_metrics = self._analyze_audio_quality(metadata.primary_audio_stream)
            metrics.audio_bitrate_score = audio_metrics["bitrate"]
            metrics.audio_sample_rate_score = audio_metrics["sample_rate"]

        # Calculate overall score
        metrics.overall_score = self._calculate_overall_score(metrics, metadata)

        # Determine quality rating
        metrics.quality_rating = self._determine_quality_rating(metrics.overall_score)

        # Detect issues
        metrics.detected_issues = self._detect_issues(metadata, metrics)

        # Generate recommendations
        metrics.recommendations = self._generate_recommendations(metadata, metrics)

        logger.info(
            f"Quality analysis complete: score={metrics.overall_score:.1f}, "
            f"rating={metrics.quality_rating}, issues={len(metrics.detected_issues)}"
        )

        return metrics

    def _analyze_video_quality(self, video: VideoStreamInfo) -> dict:
        """
        Analyze video stream quality

        Args:
            video: VideoStreamInfo from metadata

        Returns:
            Dict with quality scores
        """
        scores = {
            "resolution": 0.0,
            "bitrate": 0.0,
            "framerate": 0.0,
            "codec": 0.0
        }

        # Resolution score
        scores["resolution"] = self._score_resolution(video.width, video.height)

        # Bitrate score (compared to recommended for resolution)
        if video.bitrate:
            scores["bitrate"] = self._score_bitrate(
                video.bitrate,
                video.width,
                video.height
            )
        else:
            scores["bitrate"] = 50.0  # Unknown bitrate = medium score

        # Framerate score
        scores["framerate"] = self._score_framerate(video.fps)

        # Codec efficiency score
        scores["codec"] = self._score_video_codec(video.codec)

        return scores

    def _analyze_audio_quality(self, audio: AudioStreamInfo) -> dict:
        """
        Analyze audio stream quality

        Args:
            audio: AudioStreamInfo from metadata

        Returns:
            Dict with quality scores
        """
        scores = {
            "bitrate": 0.0,
            "sample_rate": 0.0
        }

        # Bitrate score
        if audio.bitrate:
            bitrate_kbps = audio.bitrate / 1000
            if bitrate_kbps >= 320:
                scores["bitrate"] = 100
            elif bitrate_kbps >= 256:
                scores["bitrate"] = 95
            elif bitrate_kbps >= 192:
                scores["bitrate"] = 90
            elif bitrate_kbps >= 128:
                scores["bitrate"] = 80
            elif bitrate_kbps >= 96:
                scores["bitrate"] = 70
            else:
                scores["bitrate"] = 50
        else:
            scores["bitrate"] = 50

        # Sample rate score
        if audio.sample_rate >= 96000:
            scores["sample_rate"] = 100
        elif audio.sample_rate >= 48000:
            scores["sample_rate"] = 95
        elif audio.sample_rate >= 44100:
            scores["sample_rate"] = 85
        else:
            scores["sample_rate"] = 70

        return scores

    def _score_resolution(self, width: int, height: int) -> float:
        """Score resolution quality"""
        # Find closest matching resolution
        for (res_width, res_height), score in self.RESOLUTION_SCORES.items():
            if width >= res_width and height >= res_height:
                return score

        # Below 360p
        return 20.0

    def _score_bitrate(self, bitrate: int, width: int, height: int) -> float:
        """
        Score bitrate adequacy for resolution

        Args:
            bitrate: Bitrate in bits/sec
            width: Video width
            height: Video height

        Returns:
            Score 0-100
        """
        bitrate_kbps = bitrate / 1000

        # Calculate recommended bitrate based on resolution
        pixel_count = width * height
        base_1080p = 1920 * 1080

        # Scale recommended bitrate by pixel count
        recommended_kbps = self.RECOMMENDED_VIDEO_BITRATE_1080P * (pixel_count / base_1080p)

        # Compare actual to recommended
        ratio = bitrate_kbps / recommended_kbps

        if ratio >= 1.5:
            return 100  # Higher than needed (good)
        elif ratio >= 1.0:
            return 95   # Recommended
        elif ratio >= 0.75:
            return 85   # Acceptable
        elif ratio >= 0.5:
            return 70   # Below recommended
        else:
            return 50   # Too low

    def _score_framerate(self, fps: float) -> float:
        """Score framerate"""
        # Find closest standard framerate
        min_diff = float('inf')
        best_score = 50.0

        for standard_fps, score in self.FRAMERATE_SCORES.items():
            diff = abs(fps - standard_fps)
            if diff < min_diff:
                min_diff = diff
                best_score = score

        # Penalize if too far from standards
        if min_diff > 2.0:
            best_score -= 20

        return max(best_score, 0.0)

    def _score_video_codec(self, codec) -> float:
        """Score video codec efficiency"""
        from ..common import VideoCodec

        codec_scores = {
            VideoCodec.AV1: 100,      # Best compression
            VideoCodec.H265: 95,      # Excellent compression
            VideoCodec.VP9: 90,       # Good compression
            VideoCodec.H264: 85,      # Standard, widely compatible
            VideoCodec.PRORES: 80,    # Production quality but large
            VideoCodec.MPEG4: 60,     # Legacy
            VideoCodec.MPEG2: 50,     # Old standard
            VideoCodec.THEORA: 70,    # Open source
            VideoCodec.UNKNOWN: 50
        }

        return codec_scores.get(codec, 50.0)

    def _calculate_overall_score(
        self,
        metrics: QualityMetrics,
        metadata: MediaMetadata
    ) -> float:
        """
        Calculate overall quality score

        Args:
            metrics: Partially filled QualityMetrics
            metadata: MediaMetadata

        Returns:
            Overall score 0-100
        """
        scores = []

        # Video scores (if has video)
        if metadata.has_video:
            scores.extend([
                metrics.resolution_score * 0.3,   # 30% weight
                metrics.bitrate_score * 0.25,     # 25% weight
                metrics.framerate_score * 0.2,    # 20% weight
                metrics.codec_efficiency * 0.15   # 15% weight
            ])

        # Audio scores (if has audio)
        if metadata.has_audio:
            scores.extend([
                metrics.audio_bitrate_score * 0.07,      # 7% weight
                metrics.audio_sample_rate_score * 0.03   # 3% weight
            ])

        # Calculate weighted average
        if scores:
            overall = sum(scores)
        else:
            overall = 0.0

        return round(overall, 1)

    def _determine_quality_rating(self, overall_score: float) -> str:
        """Determine quality rating from score"""
        if overall_score >= 90:
            return "ultra"
        elif overall_score >= 75:
            return "high"
        elif overall_score >= 60:
            return "medium"
        elif overall_score >= 40:
            return "low"
        else:
            return "very_low"

    def _detect_issues(
        self,
        metadata: MediaMetadata,
        metrics: QualityMetrics
    ) -> List[str]:
        """
        Detect quality issues

        Args:
            metadata: MediaMetadata
            metrics: QualityMetrics

        Returns:
            List of detected issues
        """
        issues = []

        # Check video issues
        if metadata.has_video:
            video = metadata.primary_video_stream

            # Low resolution
            if video.width < 1280 or video.height < 720:
                issues.append(f"Low resolution: {video.resolution} (recommend 1280x720 or higher)")

            # Low bitrate
            if video.bitrate and video.bitrate < 2000000:  # < 2 Mbps
                issues.append(f"Low video bitrate: {video.bitrate/1000:.0f} kbps")

            # Unusual framerate
            if video.fps < 20 or video.fps > 65:
                issues.append(f"Unusual framerate: {video.fps:.2f} fps")

        # Check audio issues
        if metadata.has_audio:
            audio = metadata.primary_audio_stream

            # Low bitrate
            if audio.bitrate and audio.bitrate < 128000:  # < 128 kbps
                issues.append(f"Low audio bitrate: {audio.bitrate/1000:.0f} kbps (recommend 192 kbps+)")

            # Low sample rate
            if audio.sample_rate < 44100:
                issues.append(f"Low sample rate: {audio.sample_rate} Hz (recommend 44100 Hz+)")

        return issues

    def _generate_recommendations(
        self,
        metadata: MediaMetadata,
        metrics: QualityMetrics
    ) -> List[str]:
        """
        Generate improvement recommendations

        Args:
            metadata: MediaMetadata
            metrics: QualityMetrics

        Returns:
            List of recommendations
        """
        recommendations = []

        # Resolution recommendations
        if metadata.has_video:
            video = metadata.primary_video_stream

            if metrics.resolution_score < 70:
                recommendations.append(
                    "Consider upscaling to 1080p or higher for better quality"
                )

            # Bitrate recommendations
            if metrics.bitrate_score < 80:
                recommended_kbps = self.RECOMMENDED_VIDEO_BITRATE_1080P * (
                    (video.width * video.height) / (1920 * 1080)
                )
                recommendations.append(
                    f"Increase video bitrate to {recommended_kbps:.0f} kbps for better quality"
                )

            # Codec recommendations
            if metrics.codec_efficiency < 80:
                recommendations.append(
                    "Consider re-encoding with H.265 or AV1 for better compression"
                )

        # Audio recommendations
        if metadata.has_audio:
            if metrics.audio_bitrate_score < 90:
                recommendations.append(
                    "Increase audio bitrate to 192 kbps or higher"
                )

            if metrics.audio_sample_rate_score < 90:
                recommendations.append(
                    "Increase audio sample rate to 48 kHz for professional quality"
                )

        # General recommendations
        if metrics.overall_score < 60:
            recommendations.append(
                "Overall quality is below recommended standards. "
                "Consider re-encoding with higher quality settings."
            )

        return recommendations
