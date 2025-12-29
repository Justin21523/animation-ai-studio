#!/usr/bin/env python3
"""
Creative Studio CLI - Command Line Interface

User-friendly CLI for all Creative Studio capabilities:
- Parody video generation
- Multimodal analysis
- Creative workflows
- Module testing

Usage:
    # Parody video
    python cli.py parody input.mp4 output.mp4 --style dramatic

    # Analysis
    python cli.py analyze input.mp4 --visual --audio

    # Workflow
    python cli.py workflow parody input.mp4 --style chaotic --duration 30

    # List capabilities
    python cli.py list

Author: Animation AI Studio
Date: 2025-11-17
"""

import os
import sys
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.applications.creative_studio.parody_video_generator import ParodyVideoGenerator
from scripts.applications.creative_studio.multimodal_analysis_pipeline import MultimodalAnalysisPipeline
from scripts.applications.creative_studio.creative_workflows import CreativeWorkflows


logger = logging.getLogger(__name__)


class CreativeStudioCLI:
    """Creative Studio Command Line Interface"""

    def __init__(self):
        """Initialize CLI"""
        self.parody_generator = ParodyVideoGenerator()
        self.analysis_pipeline = MultimodalAnalysisPipeline()
        self.workflows = CreativeWorkflows()

    async def parody_command(self, args):
        """Generate parody video"""
        print(f"\n{'='*70}")
        print("PARODY VIDEO GENERATION")
        print(f"{'='*70}")
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"Style: {args.style}")
        print(f"Duration: {args.duration}s" if args.duration else "Duration: automatic")
        print(f"{'='*70}\n")

        result = await self.parody_generator.generate_parody(
            input_video=args.input,
            output_video=args.output,
            style=args.style,
            target_duration=args.duration,
            effects=args.effects.split(",") if args.effects else None
        )

        print(f"\n{'='*70}")
        print("RESULT")
        print(f"{'='*70}")
        print(f"Success: {'✅' if result.success else '❌'}")
        print(f"Quality Score: {result.quality_score:.3f}")
        print(f"Generation Time: {result.generation_time:.1f}s")
        print(f"Iterations: {result.iterations}")
        print(f"\nAnalysis Summary:")
        print(f"  Scenes: {result.scenes_detected}")
        print(f"  Composition: {result.avg_composition_score:.3f}")
        print(f"  Camera: {result.camera_style}")
        print(f"\nFeedback: {result.feedback}")

        if args.save_result:
            result_path = args.output.replace(".mp4", "_result.json")
            self.parody_generator.save_result(result, result_path)
            print(f"\nResult saved to: {result_path}")

    async def analyze_command(self, args):
        """Analyze video"""
        print(f"\n{'='*70}")
        print("MULTIMODAL ANALYSIS")
        print(f"{'='*70}")
        print(f"Input: {args.input}")
        print(f"Visual: {args.visual}")
        print(f"Audio: {args.audio}")
        print(f"Context: {args.context}")
        print(f"{'='*70}\n")

        result = await self.analysis_pipeline.analyze(
            video_path=args.input,
            include_visual=args.visual,
            include_audio=args.audio,
            include_context=args.context,
            sample_rate=args.sample_rate
        )

        print(result.summary())

        if args.output:
            self.analysis_pipeline.save_result(result, args.output)
            print(f"\nAnalysis saved to: {args.output}")

    async def workflow_command(self, args):
        """Execute workflow"""
        print(f"\n{'='*70}")
        print(f"WORKFLOW: {args.workflow_type.upper()}")
        print(f"{'='*70}")

        if args.workflow_type == "parody":
            result = await self.workflows.create_parody_video(
                input_video=args.input,
                output_video=args.output,
                style=args.style,
                target_duration=args.duration
            )
        elif args.workflow_type == "analyze":
            result = await self.workflows.analyze_and_report(
                video_path=args.input,
                output_report=args.output,
                include_visual=True,
                include_audio=args.audio,
                include_context=args.context
            )
        else:
            print(f"Unknown workflow type: {args.workflow_type}")
            return

        print(f"\n{'='*70}")
        print("WORKFLOW RESULT")
        print(f"{'='*70}")
        print(f"Success: {'✅' if result.success else '❌'}")
        print(f"Workflow: {result.workflow_name}")
        print(f"Execution Time: {result.execution_time:.1f}s")
        print(f"\nOutputs:")
        for output_type, path in result.outputs.items():
            print(f"  {output_type}: {path}")

    async def voice_command(self, args):
        """Synthesize character voice line (TTS)"""
        print(f"\n{'='*70}")
        print("VOICE SYNTHESIS (TTS)")
        print(f"{'='*70}")
        print(f"Character: {args.character}")
        print(f"Emotion: {args.emotion} (intensity={args.intensity})")
        print(f"Output: {args.output if args.output else '(auto)'}")
        print(f"{'='*70}\n")

        result = await self.workflows.synthesize_character_voice(
            character=args.character,
            text=args.text,
            emotion=args.emotion,
            intensity=float(args.intensity),
            output_audio=args.output,
        )

        print(f"\n{'='*70}")
        print("RESULT")
        print(f"{'='*70}")
        print(f"Success: {'✅' if result.success else '❌'}")
        if result.success:
            print(f"Audio: {result.outputs.get('audio')}")
        else:
            print(f"Error: {result.metadata.get('error')}")

    def list_command(self, args):
        """List available capabilities"""
        print(f"\n{'='*70}")
        print("CREATIVE STUDIO CAPABILITIES")
        print(f"{'='*70}")

        print("\n📹 PARODY VIDEO GENERATION")
        print("  Create funny/parody videos with automatic editing")
        print("  Styles: dramatic, chaotic, wholesome")
        print("  Command: cli.py parody INPUT OUTPUT --style STYLE")

        print("\n🔍 MULTIMODAL ANALYSIS")
        print("  Comprehensive video analysis")
        print("  Modalities: visual, audio, context")
        print("  Command: cli.py analyze INPUT --visual --audio")

        print("\n🎨 CREATIVE WORKFLOWS")
        workflows = self.workflows.list_workflows()
        for wf in workflows:
            print(f"\n  {wf['name']}:")
            print(f"    {wf['description']}")
            print(f"    Inputs: {wf['inputs']}")
            print(f"    Outputs: {wf['outputs']}")

        print("\n📦 INTEGRATED MODULES")
        print("  ✅ Module 1: LLM Backend")
        print("  ✅ Module 2: Image Generation")
        print("  ✅ Module 3: Voice Synthesis")
        print("  ✅ Module 4: Model Manager")
        print("  ✅ Module 5: RAG System")
        print("  ✅ Module 6: Agent Framework")
        print("  ✅ Module 7: Video Analysis")
        print("  ✅ Module 8: Video Editing")
        print("  ✅ Module 9: Creative Studio")

        print(f"\n{'='*70}\n")

    async def run(self, args):
        """Run CLI command"""
        if args.command == "parody":
            await self.parody_command(args)
        elif args.command == "analyze":
            await self.analyze_command(args)
        elif args.command == "voice":
            await self.voice_command(args)
        elif args.command == "workflow":
            await self.workflow_command(args)
        elif args.command == "list":
            self.list_command(args)
        else:
            print(f"Unknown command: {args.command}")
            print("Available commands: parody, analyze, voice, workflow, list")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Creative Studio CLI - AI-powered creative video generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate dramatic parody video
  %(prog)s parody luca.mp4 luca_funny.mp4 --style dramatic --duration 30

  # Analyze video
  %(prog)s analyze luca.mp4 --visual --audio --output analysis.json

  # Execute workflow
  %(prog)s workflow parody luca.mp4 --output luca_parody.mp4 --style chaotic

  # List capabilities
  %(prog)s list
        """
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Parody command
    parody_parser = subparsers.add_parser("parody", help="Generate parody video")
    parody_parser.add_argument("input", help="Input video path")
    parody_parser.add_argument("output", help="Output video path")
    parody_parser.add_argument(
        "--style", "-s",
        default="dramatic",
        choices=["dramatic", "chaotic", "wholesome"],
        help="Parody style (default: dramatic)"
    )
    parody_parser.add_argument(
        "--duration", "-d",
        type=float,
        help="Target duration in seconds"
    )
    parody_parser.add_argument(
        "--effects", "-e",
        help="Comma-separated list of effects (e.g., zoom_punch,speed_ramp)"
    )
    parody_parser.add_argument(
        "--save-result",
        action="store_true",
        help="Save result JSON"
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze video")
    analyze_parser.add_argument("input", help="Input video path")
    analyze_parser.add_argument(
        "--visual",
        action="store_true",
        default=True,
        help="Include visual analysis (default: True)"
    )
    analyze_parser.add_argument(
        "--audio",
        action="store_true",
        help="Include audio analysis"
    )
    analyze_parser.add_argument(
        "--context",
        action="store_true",
        help="Include context retrieval"
    )
    analyze_parser.add_argument(
        "--sample-rate",
        type=int,
        default=30,
        help="Frame sampling rate (default: 30)"
    )
    analyze_parser.add_argument(
        "--output", "-o",
        help="Output JSON path"
    )

    # Workflow command
    workflow_parser = subparsers.add_parser("workflow", help="Execute workflow")
    workflow_parser.add_argument(
        "workflow_type",
        choices=["parody", "analyze"],
        help="Workflow type"
    )
    workflow_parser.add_argument("input", help="Input video path")
    workflow_parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output path"
    )
    workflow_parser.add_argument(
        "--style", "-s",
        default="dramatic",
        help="Parody style (for parody workflow)"
    )
    workflow_parser.add_argument(
        "--duration", "-d",
        type=float,
        help="Target duration (for parody workflow)"
    )
    workflow_parser.add_argument(
        "--audio",
        action="store_true",
        help="Include audio analysis (for analyze workflow)"
    )
    workflow_parser.add_argument(
        "--context",
        action="store_true",
        help="Include context retrieval (for analyze workflow)"
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List capabilities")

    # Voice command
    voice_parser = subparsers.add_parser("voice", help="Synthesize character voice (TTS)")
    voice_parser.add_argument("--character", "-c", required=True, help="Character name")
    voice_parser.add_argument("--text", "-t", required=True, help="Text to synthesize")
    voice_parser.add_argument("--emotion", "-e", default="neutral", help="Emotion preset")
    voice_parser.add_argument("--intensity", type=float, default=0.8, help="Emotion intensity (default: 0.8)")
    voice_parser.add_argument("--output", "-o", help="Output WAV path (default: auto under outputs/creative_studio/voice)")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Print header
    print("\n" + "="*70)
    print("🎬 ANIMATION AI STUDIO - CREATIVE STUDIO")
    print("="*70)

    if not args.command:
        parser.print_help()
        return

    # Run CLI
    cli = CreativeStudioCLI()
    asyncio.run(cli.run(args))


if __name__ == "__main__":
    main()
