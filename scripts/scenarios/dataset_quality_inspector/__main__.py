"""
Dataset Quality Inspector - CLI Entry Point

Command-line interface for dataset quality inspection.

Usage:
    python -m scripts.scenarios.dataset_quality_inspector \\
        --dataset /path/to/dataset \\
        --output /path/to/report.json \\
        --format json

Author: Animation AI Studio
Date: 2025-12-02
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any

from .inspector import DatasetInspector
from .common import InspectionReport


def setup_logging(verbose: bool = False):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Dataset Quality Inspector - Automated quality analysis for training datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inspection
  python -m scripts.scenarios.dataset_quality_inspector \\
      --dataset /path/to/dataset \\
      --output report.json

  # With custom thresholds
  python -m scripts.scenarios.dataset_quality_inspector \\
      --dataset /path/to/dataset \\
      --output report.json \\
      --min-resolution 512 512 \\
      --blur-threshold 100 \\
      --quality-threshold 70

  # Generate HTML report
  python -m scripts.scenarios.dataset_quality_inspector \\
      --dataset /path/to/dataset \\
      --output report.html \\
      --format html

  # With recommendations
  python -m scripts.scenarios.dataset_quality_inspector \\
      --dataset /path/to/dataset \\
      --output report.json \\
      --enable-recommendations
        """
    )

    # Required arguments
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to dataset directory'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output file path for report'
    )

    # Output format
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'html', 'markdown'],
        default='json',
        help='Report output format (default: json)'
    )

    # Quality thresholds
    parser.add_argument(
        '--min-resolution',
        type=int,
        nargs=2,
        metavar=('WIDTH', 'HEIGHT'),
        default=[512, 512],
        help='Minimum image resolution (default: 512 512)'
    )

    parser.add_argument(
        '--blur-threshold',
        type=float,
        default=100.0,
        help='Blur detection threshold (default: 100.0)'
    )

    parser.add_argument(
        '--noise-threshold',
        type=float,
        default=50.0,
        help='Noise detection threshold (default: 50.0)'
    )

    parser.add_argument(
        '--quality-threshold',
        type=float,
        default=70.0,
        help='Overall quality threshold (default: 70.0)'
    )

    # Feature flags
    parser.add_argument(
        '--enable-recommendations',
        action='store_true',
        help='Enable AI-powered recommendations (requires Agent Framework)'
    )

    parser.add_argument(
        '--skip-duplicates',
        action='store_true',
        help='Skip duplicate detection'
    )

    parser.add_argument(
        '--skip-format-validation',
        action='store_true',
        help='Skip format and structure validation'
    )

    # Caption settings
    parser.add_argument(
        '--min-caption-length',
        type=int,
        default=10,
        help='Minimum caption length in tokens (default: 10)'
    )

    parser.add_argument(
        '--max-caption-length',
        type=int,
        default=77,
        help='Maximum caption length in tokens (default: 77)'
    )

    # Distribution settings
    parser.add_argument(
        '--min-category-size',
        type=int,
        default=50,
        help='Minimum images per category (default: 50)'
    )

    parser.add_argument(
        '--imbalance-ratio',
        type=float,
        default=3.0,
        help='Maximum category size ratio (default: 3.0)'
    )

    # Other options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress output'
    )

    return parser.parse_args()


def build_config(args) -> Dict[str, Any]:
    """Build configuration from command-line arguments"""
    return {
        # Image quality
        "min_resolution": tuple(args.min_resolution),
        "blur_threshold": args.blur_threshold,
        "noise_threshold": args.noise_threshold,

        # Captions
        "min_caption_length": args.min_caption_length,
        "max_caption_length": args.max_caption_length,

        # Distribution
        "min_category_size": args.min_category_size,
        "imbalance_ratio": args.imbalance_ratio,

        # Feature flags
        "enable_duplicates": not args.skip_duplicates,
        "enable_format_validation": not args.skip_format_validation,

        # Metadata
        "metadata_required": not args.skip_format_validation,
        "captions_required": True
    }


def format_report_json(report: InspectionReport) -> str:
    """Format report as JSON"""
    return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)


def format_report_markdown(report: InspectionReport) -> str:
    """Format report as Markdown"""
    md = []

    # Header
    md.append("# Dataset Quality Inspection Report")
    md.append("")
    md.append(f"**Dataset:** {report.dataset_summary.dataset_path}")
    md.append(f"**Total Images:** {report.dataset_summary.total_images}")
    md.append(f"**Overall Score:** {report.overall_score:.1f}/100")
    md.append("")

    # Summary
    md.append("## Summary")
    md.append("")
    md.append(f"- **Total Issues:** {report.total_issues}")
    md.append(f"- **Critical:** {report.critical_issues}")
    md.append(f"- **High:** {report.high_issues}")
    md.append(f"- **Medium:** {report.medium_issues}")
    md.append(f"- **Low:** {report.low_issues}")
    md.append("")

    # Category Scores
    md.append("## Category Scores")
    md.append("")
    md.append("| Category | Score |")
    md.append("|----------|-------|")
    for category, score in report.category_scores.items():
        md.append(f"| {category.replace('_', ' ').title()} | {score:.1f}/100 |")
    md.append("")

    # Issues
    if report.issues:
        md.append("## Issues Detected")
        md.append("")
        for issue in report.issues[:20]:  # First 20 issues
            md.append(f"### [{issue.severity.value.upper()}] {issue.category.value}")
            md.append(f"{issue.description}")
            if issue.recommendation:
                md.append(f"**Recommendation:** {issue.recommendation}")
            md.append("")

    # Recommendations
    if report.recommendations:
        md.append("## Recommendations")
        md.append("")
        for idx, rec in enumerate(report.recommendations, 1):
            md.append(f"{idx}. **[{rec['priority'].upper()}]** {rec['action']}")
            md.append("")

    return "\n".join(md)


def format_report_html(report: InspectionReport) -> str:
    """Format report as HTML"""
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Dataset Quality Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 20px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .score {{
            font-size: 48px;
            font-weight: bold;
        }}
        .card {{
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .severity-critical {{ color: #dc3545; }}
        .severity-high {{ color: #fd7e14; }}
        .severity-medium {{ color: #ffc107; }}
        .severity-low {{ color: #28a745; }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        .progress-bar {{
            height: 30px;
            background-color: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Dataset Quality Inspection Report</h1>
        <p><strong>Dataset:</strong> {report.dataset_summary.dataset_path}</p>
        <p><strong>Total Images:</strong> {report.dataset_summary.total_images}</p>
        <div class="score">Overall Score: {report.overall_score:.1f}/100</div>
    </div>

    <div class="card">
        <h2>Issue Summary</h2>
        <p><strong>Total Issues:</strong> {report.total_issues}</p>
        <ul>
            <li><span class="severity-critical">Critical: {report.critical_issues}</span></li>
            <li><span class="severity-high">High: {report.high_issues}</span></li>
            <li><span class="severity-medium">Medium: {report.medium_issues}</span></li>
            <li><span class="severity-low">Low: {report.low_issues}</span></li>
        </ul>
    </div>

    <div class="card">
        <h2>Category Scores</h2>
        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Score</th>
                    <th>Progress</th>
                </tr>
            </thead>
            <tbody>
"""

    for category, score in report.category_scores.items():
        html += f"""
                <tr>
                    <td>{category.replace('_', ' ').title()}</td>
                    <td>{score:.1f}/100</td>
                    <td>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {score}%">{score:.0f}%</div>
                        </div>
                    </td>
                </tr>
"""

    html += """
            </tbody>
        </table>
    </div>
"""

    # Issues
    if report.issues:
        html += """
    <div class="card">
        <h2>Issues Detected</h2>
"""
        for issue in report.issues[:20]:
            severity_class = f"severity-{issue.severity.value}"
            html += f"""
        <div style="margin-bottom: 15px; padding: 10px; border-left: 4px solid;" class="{severity_class}">
            <strong>[{issue.severity.value.upper()}] {issue.category.value}</strong><br>
            {issue.description}
            {f'<br><em>Recommendation: {issue.recommendation}</em>' if issue.recommendation else ''}
        </div>
"""
        html += """
    </div>
"""

    # Recommendations
    if report.recommendations:
        html += """
    <div class="card">
        <h2>Recommendations</h2>
        <ol>
"""
        for rec in report.recommendations:
            html += f"""
            <li><strong>[{rec['priority'].upper()}]</strong> {rec['action']}</li>
"""
        html += """
        </ol>
    </div>
"""

    html += """
</body>
</html>
"""
    return html


def main():
    """Main entry point"""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("Starting Dataset Quality Inspector")

    # Validate dataset path
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    # Build configuration
    config = build_config(args)

    # Create inspector
    logger.info(f"Inspecting dataset: {dataset_path}")
    inspector = DatasetInspector(
        dataset_path=str(dataset_path),
        config=config
    )

    # Run inspection
    try:
        report = inspector.inspect(enable_recommendations=args.enable_recommendations)
    except Exception as e:
        logger.error(f"Inspection failed: {e}", exc_info=True)
        sys.exit(1)

    # Format output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == 'json':
        output_content = format_report_json(report)
    elif args.format == 'markdown':
        output_content = format_report_markdown(report)
    elif args.format == 'html':
        output_content = format_report_html(report)
    else:
        logger.error(f"Unsupported format: {args.format}")
        sys.exit(1)

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output_content)

    logger.info(f"Report saved to: {output_path}")

    # Print summary
    print("\n" + "="*70)
    print("DATASET QUALITY INSPECTION SUMMARY")
    print("="*70)
    print(f"Dataset: {report.dataset_summary.dataset_path}")
    print(f"Total Images: {report.dataset_summary.total_images}")
    print(f"Overall Score: {report.overall_score:.1f}/100")
    print(f"\nIssues Found: {report.total_issues}")
    print(f"  Critical: {report.critical_issues}")
    print(f"  High: {report.high_issues}")
    print(f"  Medium: {report.medium_issues}")
    print(f"  Low: {report.low_issues}")
    print("="*70)

    # Exit with appropriate code
    if report.critical_issues > 0:
        sys.exit(2)  # Critical issues
    elif report.high_issues > 0:
        sys.exit(1)  # High issues
    else:
        sys.exit(0)  # Success


if __name__ == '__main__':
    main()
