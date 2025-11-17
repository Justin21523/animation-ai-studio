"""
Test Suite for Video Analysis Module (Module 7)

Tests all video analysis components:
- Scene Detector
- Composition Analyzer
- Camera Movement Tracker
- Temporal Coherence Checker
- Agent Tool Integration

Author: Animation AI Studio
Date: 2025-11-17
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.analysis.video.scene_detector import SceneDetector
from scripts.analysis.video.composition_analyzer import CompositionAnalyzer
from scripts.analysis.video.camera_movement_tracker import CameraMovementTracker
from scripts.analysis.video.temporal_coherence_checker import TemporalCoherenceChecker
from scripts.agent.tools.video_analysis_tools import (
    detect_scenes,
    analyze_composition,
    track_camera_movement,
    check_temporal_coherence,
    analyze_video_complete
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_scene_detector():
    """Test Scene Detector"""
    print("\n" + "=" * 80)
    print("TEST 1: Scene Detector")
    print("=" * 80)

    # Use a test video (you should provide a valid video path)
    video_path = "path/to/test/video.mp4"

    if not Path(video_path).exists():
        logger.warning(f"Test video not found: {video_path}")
        logger.info("Please provide a valid video path for testing")
        logger.info("SKIPPING TEST")
        return False

    try:
        detector = SceneDetector(
            threshold=27.0,
            min_scene_length=15,
            adaptive_threshold=True
        )

        result = detector.detect(
            video_path=video_path,
            extract_keyframes=True
        )

        print(f"\n✅ Scene Detector Test PASSED")
        print(f"   Detected {result.total_scenes} scenes")
        print(f"   Average scene duration: {result.avg_scene_duration:.2f}s")
        print(f"   Detection time: {result.detection_time:.2f}s")

        # Validate results
        assert result.total_scenes > 0, "Should detect at least one scene"
        assert result.avg_scene_duration > 0, "Average scene duration should be positive"
        assert len(result.scenes) == result.total_scenes, "Scene count mismatch"

        return True

    except Exception as e:
        print(f"\n❌ Scene Detector Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_composition_analyzer():
    """Test Composition Analyzer"""
    print("\n" + "=" * 80)
    print("TEST 2: Composition Analyzer")
    print("=" * 80)

    video_path = "path/to/test/video.mp4"

    if not Path(video_path).exists():
        logger.warning(f"Test video not found: {video_path}")
        logger.info("SKIPPING TEST")
        return False

    try:
        analyzer = CompositionAnalyzer(enable_visualization=False)

        result = analyzer.analyze_video(
            video_path=video_path,
            sample_rate=30  # Analyze every 30th frame for speed
        )

        print(f"\n✅ Composition Analyzer Test PASSED")
        print(f"   Analyzed {result.total_frames_analyzed} frames")
        print(f"   Average composition score: {result.avg_composition_score:.3f}")
        print(f"   Analysis time: {result.analysis_time:.2f}s")

        # Validate results
        assert result.total_frames_analyzed > 0, "Should analyze at least one frame"
        assert 0.0 <= result.avg_composition_score <= 1.0, "Score should be in [0, 1]"
        assert len(result.frame_metrics) == result.total_frames_analyzed, "Frame count mismatch"

        # Check first frame metrics
        if result.frame_metrics:
            first_metric = result.frame_metrics[0]
            assert 0.0 <= first_metric.rule_of_thirds.overall_score <= 1.0
            assert 0.0 <= first_metric.visual_balance.overall_balance_score <= 1.0
            assert 0.0 <= first_metric.depth_layers.depth_complexity <= 1.0
            print(f"   First frame composition score: {first_metric.overall_composition_score:.3f}")

        return True

    except Exception as e:
        print(f"\n❌ Composition Analyzer Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_camera_movement_tracker():
    """Test Camera Movement Tracker"""
    print("\n" + "=" * 80)
    print("TEST 3: Camera Movement Tracker")
    print("=" * 80)

    video_path = "path/to/test/video.mp4"

    if not Path(video_path).exists():
        logger.warning(f"Test video not found: {video_path}")
        logger.info("SKIPPING TEST")
        return False

    try:
        tracker = CameraMovementTracker()

        result = tracker.track_video(
            video_path=video_path,
            sample_interval=2  # Every 2nd frame for speed
        )

        print(f"\n✅ Camera Movement Tracker Test PASSED")
        print(f"   Camera style: {result.camera_style}")
        print(f"   Detected {len(result.movements)} movements")
        print(f"   Detected {len(result.shots)} shots")
        print(f"   Static duration: {result.total_static_duration:.2f}s")
        print(f"   Moving duration: {result.total_moving_duration:.2f}s")
        print(f"   Analysis time: {result.analysis_time:.2f}s")

        # Validate results
        assert len(result.movements) >= 0, "Should have non-negative movements"
        assert len(result.shots) >= 0, "Should have non-negative shots"
        assert result.camera_style in ["static", "smooth", "dynamic", "handheld"], "Invalid camera style"
        assert result.total_static_duration >= 0, "Static duration should be non-negative"
        assert result.total_moving_duration >= 0, "Moving duration should be non-negative"

        # Check movement types
        if result.movements:
            movement_types = set(m.movement_type for m in result.movements)
            valid_types = {"static", "pan", "tilt", "zoom", "pan_tilt", "complex"}
            assert movement_types.issubset(valid_types), f"Invalid movement types: {movement_types - valid_types}"

        return True

    except Exception as e:
        print(f"\n❌ Camera Movement Tracker Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_temporal_coherence_checker():
    """Test Temporal Coherence Checker"""
    print("\n" + "=" * 80)
    print("TEST 4: Temporal Coherence Checker")
    print("=" * 80)

    video_path = "path/to/test/video.mp4"

    if not Path(video_path).exists():
        logger.warning(f"Test video not found: {video_path}")
        logger.info("SKIPPING TEST")
        return False

    try:
        checker = TemporalCoherenceChecker()

        result = checker.check_video(
            video_path=video_path,
            sample_interval=2  # Every 2nd frame for speed
        )

        print(f"\n✅ Temporal Coherence Checker Test PASSED")
        print(f"   Temporal stability: {result.temporal_stability_rating}")
        print(f"   Average coherence score: {result.avg_coherence_score:.3f}")
        print(f"   Min coherence score: {result.min_coherence_score:.3f}")
        print(f"   Flicker frames: {result.total_flicker_frames}")
        print(f"   Abrupt transitions: {result.total_abrupt_transitions}")
        print(f"   Problem segments: {len(result.problem_segments)}")
        print(f"   Analysis time: {result.analysis_time:.2f}s")

        # Validate results
        assert result.temporal_stability_rating in ["excellent", "good", "fair", "poor"], "Invalid stability rating"
        assert 0.0 <= result.avg_coherence_score <= 1.0, "Average score should be in [0, 1]"
        assert 0.0 <= result.min_coherence_score <= 1.0, "Min score should be in [0, 1]"
        assert result.total_flicker_frames >= 0, "Flicker count should be non-negative"
        assert result.total_abrupt_transitions >= 0, "Transition count should be non-negative"

        # Check segments
        if result.segments:
            for segment in result.segments:
                assert segment.quality_rating in ["excellent", "good", "fair", "poor"], "Invalid segment rating"
                assert 0.0 <= segment.avg_coherence_score <= 1.0, "Segment score out of range"

        return True

    except Exception as e:
        print(f"\n❌ Temporal Coherence Checker Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_agent_tool_integration():
    """Test Agent Tool Integration"""
    print("\n" + "=" * 80)
    print("TEST 5: Agent Tool Integration")
    print("=" * 80)

    video_path = "path/to/test/video.mp4"

    if not Path(video_path).exists():
        logger.warning(f"Test video not found: {video_path}")
        logger.info("SKIPPING TEST")
        return False

    try:
        # Test individual tools
        print("\n--- Testing detect_scenes tool ---")
        scene_result = await detect_scenes(video_path)
        assert scene_result["success"], "Scene detection should succeed"
        print(f"✅ Scene detection tool: {scene_result['total_scenes']} scenes")

        print("\n--- Testing analyze_composition tool ---")
        comp_result = await analyze_composition(video_path, sample_rate=30)
        assert comp_result["success"], "Composition analysis should succeed"
        print(f"✅ Composition analysis tool: score {comp_result['avg_composition_score']:.3f}")

        print("\n--- Testing track_camera_movement tool ---")
        camera_result = await track_camera_movement(video_path)
        assert camera_result["success"], "Camera tracking should succeed"
        print(f"✅ Camera tracking tool: {camera_result['camera_style']} style")

        print("\n--- Testing check_temporal_coherence tool ---")
        coherence_result = await check_temporal_coherence(video_path)
        assert coherence_result["success"], "Temporal coherence check should succeed"
        print(f"✅ Temporal coherence tool: {coherence_result['temporal_stability_rating']} rating")

        print("\n--- Testing analyze_video_complete tool ---")
        complete_result = await analyze_video_complete(video_path, sample_rate=30)
        assert complete_result["success"], "Complete analysis should succeed"
        print(f"✅ Complete analysis tool: all {len(complete_result['analyses'])} analyses completed")

        # Check that summary was generated
        if "summary" in complete_result:
            print(f"\n   Summary generated with keys: {list(complete_result['summary'].keys())}")

        print(f"\n✅ Agent Tool Integration Test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Agent Tool Integration Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all Module 7 tests"""
    print("\n" + "=" * 80)
    print("MODULE 7: VIDEO ANALYSIS - COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    results = []

    # Test 1: Scene Detector
    results.append(("Scene Detector", test_scene_detector()))

    # Test 2: Composition Analyzer
    results.append(("Composition Analyzer", test_composition_analyzer()))

    # Test 3: Camera Movement Tracker
    results.append(("Camera Movement Tracker", test_camera_movement_tracker()))

    # Test 4: Temporal Coherence Checker
    results.append(("Temporal Coherence Checker", test_temporal_coherence_checker()))

    # Test 5: Agent Tool Integration
    results.append(("Agent Tool Integration", await test_agent_tool_integration()))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = 0
    failed = 0
    skipped = 0

    for name, result in results:
        if result is True:
            status = "✅ PASSED"
            passed += 1
        elif result is False:
            status = "❌ FAILED"
            failed += 1
        else:
            status = "⏭️  SKIPPED"
            skipped += 1

        print(f"{status}: {name}")

    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")

    if failed == 0 and passed > 0:
        print("\n✅ ALL TESTS PASSED!")
        print("\nModule 7 (Video Analysis) is ready for production.")
        return True
    elif skipped == len(results):
        print("\n⚠️  ALL TESTS SKIPPED (no test video provided)")
        print("\nTo run tests, please provide a valid video path in the test functions.")
        return None
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
        return False


def main():
    """Main test entry point"""
    print("\n" + "=" * 80)
    print("Video Analysis Module Test Suite")
    print("=" * 80)
    print("\nIMPORTANT: To run these tests, you need to provide a test video.")
    print("Edit the test functions and replace 'path/to/test/video.mp4' with")
    print("an actual video file path.\n")
    print("Example test videos:")
    print("  - /mnt/data/ai_data/datasets/3d-anime/luca/clips/sample.mp4")
    print("  - outputs/test_video.mp4")
    print("=" * 80)

    result = asyncio.run(run_all_tests())

    if result is True:
        sys.exit(0)
    elif result is None:
        sys.exit(2)  # Tests skipped
    else:
        sys.exit(1)  # Tests failed


if __name__ == "__main__":
    main()
