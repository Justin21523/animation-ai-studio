"""
File Organization Scenario - CLI Entry Point

Command-line interface for intelligent file organization and analysis.

Usage:
    # Analyze directory
    python -m scripts.scenarios.file_organization analyze \\
        --directory /path/to/organize \\
        --output analysis.json

    # Plan organization
    python -m scripts.scenarios.file_organization plan \\
        --directory /path/to/organize \\
        --strategy SMART \\
        --output plan.json

    # Execute organization
    python -m scripts.scenarios.file_organization organize \\
        --directory /path/to/organize \\
        --strategy SMART \\
        --dry-run

Author: Animation AI Studio
Date: 2025-12-03
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from .organizer import FileOrganizer
from .common import OrganizationStrategy, OrganizationReport
from .processors.smart_organizer import OrganizationRule


def setup_logging(verbose: bool = False):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description='File Organization - AI-powered file system analysis and organization',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Common arguments
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    # Subcommands
    subparsers = parser.add_subparsers(
        dest='command',
        required=True,
        help='Available commands'
    )

    # Analyze command
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze directory structure and detect issues'
    )
    add_common_args(analyze_parser)
    analyze_parser.add_argument(
        '--enable-recommendations',
        action='store_true',
        help='Generate AI-powered recommendations'
    )

    # Plan command
    plan_parser = subparsers.add_parser(
        'plan',
        help='Plan file organization without executing'
    )
    add_common_args(plan_parser)
    add_strategy_args(plan_parser)

    # Organize command
    organize_parser = subparsers.add_parser(
        'organize',
        help='Execute file organization'
    )
    add_common_args(organize_parser)
    add_strategy_args(organize_parser)
    organize_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without actually moving files'
    )
    organize_parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Disable backup creation (use with caution!)'
    )

    return parser


def add_common_args(parser):
    """Add common arguments to parser"""
    parser.add_argument(
        '--directory', '-d',
        type=str,
        required=True,
        help='Path to directory to analyze/organize'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output file path for report/plan'
    )
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['json', 'html', 'markdown'],
        default='json',
        help='Report output format (default: json)'
    )


def add_strategy_args(parser):
    """Add organization strategy arguments"""
    parser.add_argument(
        '--strategy', '-s',
        type=str,
        choices=['BY_TYPE', 'BY_DATE', 'BY_PROJECT', 'BY_SIZE', 'CUSTOM', 'SMART'],
        default='SMART',
        help='Organization strategy (default: SMART)'
    )
    parser.add_argument(
        '--custom-rules',
        type=str,
        help='Path to JSON file with custom organization rules'
    )


def command_analyze(args):
    """Execute analyze command"""
    logger = logging.getLogger(__name__)
    logger.info(f"Analyzing directory: {args.directory}")

    # Validate directory
    directory = Path(args.directory)
    if not directory.exists():
        logger.error(f"Directory does not exist: {directory}")
        return 1
    if not directory.is_dir():
        logger.error(f"Path is not a directory: {directory}")
        return 1

    # Create organizer
    organizer = FileOrganizer(
        root_path=str(directory),
        config={
            "min_file_size": 1024,
            "enable_perceptual_hashing": True,
            "create_backup": True
        }
    )

    # Run analysis
    try:
        report = organizer.analyze(
            enable_recommendations=args.enable_recommendations
        )

        # Generate output
        output_path = Path(args.output)
        if args.format == 'json':
            save_json_report(report, output_path)
        elif args.format == 'html':
            save_html_report(report, output_path, "Analysis Report")
        elif args.format == 'markdown':
            save_markdown_report(report, output_path, "Analysis Report")

        # Print summary
        print_analysis_summary(report)

        logger.info(f"Analysis report saved to: {output_path}")
        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1


def command_plan(args):
    """Execute plan command"""
    logger = logging.getLogger(__name__)
    logger.info(f"Planning organization for: {args.directory}")

    # Validate directory
    directory = Path(args.directory)
    if not directory.exists():
        logger.error(f"Directory does not exist: {directory}")
        return 1

    # Load custom rules if provided
    custom_rules = None
    if args.custom_rules:
        try:
            with open(args.custom_rules, 'r') as f:
                rules_data = json.load(f)
                custom_rules = [
                    OrganizationRule(**rule) for rule in rules_data
                ]
        except Exception as e:
            logger.error(f"Failed to load custom rules: {e}")
            return 1

    # Create organizer
    organizer = FileOrganizer(
        root_path=str(directory),
        config={"create_backup": not args.get('no_backup', False)}
    )

    # Plan organization
    try:
        strategy = OrganizationStrategy[args.strategy]
        plan = organizer.plan_organization(
            strategy=strategy,
            custom_rules=custom_rules
        )

        if not plan:
            logger.error("No files to organize")
            return 1

        # Generate output
        output_path = Path(args.output)
        if args.format == 'json':
            save_json_plan(plan, output_path)
        elif args.format == 'markdown':
            save_markdown_plan(plan, output_path)

        # Print summary
        print_plan_summary(plan)

        logger.info(f"Organization plan saved to: {output_path}")
        return 0

    except Exception as e:
        logger.error(f"Planning failed: {e}", exc_info=True)
        return 1


def command_organize(args):
    """Execute organize command"""
    logger = logging.getLogger(__name__)
    mode = "DRY-RUN" if args.dry_run else "EXECUTE"
    logger.info(f"Organizing directory ({mode}): {args.directory}")

    # Validate directory
    directory = Path(args.directory)
    if not directory.exists():
        logger.error(f"Directory does not exist: {directory}")
        return 1

    # Load custom rules if provided
    custom_rules = None
    if args.custom_rules:
        try:
            with open(args.custom_rules, 'r') as f:
                rules_data = json.load(f)
                custom_rules = [
                    OrganizationRule(**rule) for rule in rules_data
                ]
        except Exception as e:
            logger.error(f"Failed to load custom rules: {e}")
            return 1

    # Create organizer
    organizer = FileOrganizer(
        root_path=str(directory),
        config={"create_backup": not args.no_backup}
    )

    # Plan and execute organization
    try:
        strategy = OrganizationStrategy[args.strategy]

        # Step 1: Plan
        logger.info("Step 1/2: Planning organization...")
        plan = organizer.plan_organization(
            strategy=strategy,
            custom_rules=custom_rules
        )

        if not plan:
            logger.error("No files to organize")
            return 1

        print_plan_summary(plan)

        # Step 2: Execute
        logger.info(f"Step 2/2: Executing organization ({mode})...")
        result = organizer.organize(plan, dry_run=args.dry_run)

        # Generate output
        output_path = Path(args.output)
        if args.format == 'json':
            save_json_result(result, output_path)
        elif args.format == 'markdown':
            save_markdown_result(result, output_path)

        # Print summary
        print_result_summary(result, args.dry_run)

        logger.info(f"Organization result saved to: {output_path}")

        if result.success:
            return 0
        else:
            logger.error("Organization completed with errors")
            return 1

    except Exception as e:
        logger.error(f"Organization failed: {e}", exc_info=True)
        return 1


def save_json_report(report: OrganizationReport, output_path: Path):
    """Save report as JSON"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report.to_dict(), f, indent=2, default=str)


def save_json_plan(plan, output_path: Path):
    """Save plan as JSON"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(plan.to_dict(), f, indent=2, default=str)


def save_json_result(result, output_path: Path):
    """Save result as JSON"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)


def save_html_report(report: OrganizationReport, output_path: Path, title: str):
    """Save report as HTML"""
    html = generate_html_report(report, title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)


def save_markdown_report(report: OrganizationReport, output_path: Path, title: str):
    """Save report as Markdown"""
    markdown = generate_markdown_report(report, title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(markdown)


def save_markdown_plan(plan, output_path: Path):
    """Save plan as Markdown"""
    markdown = generate_markdown_plan(plan)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(markdown)


def save_markdown_result(result, output_path: Path):
    """Save result as Markdown"""
    markdown = generate_markdown_result(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(markdown)


def generate_html_report(report: OrganizationReport, title: str) -> str:
    """Generate HTML report"""
    # Simplified HTML template
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .score {{ font-size: 48px; font-weight: bold; color: #2ecc71; }}
        .issue {{ border-left: 4px solid #e74c3c; padding: 10px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="summary">
        <h2>Overall Score</h2>
        <div class="score">{report.organization_score:.1f}/100</div>
        <p>Total Files: {report.total_files}</p>
        <p>Total Size: {report.total_size_bytes / 1024 / 1024:.1f} MB</p>
        <p>Issues Found: {len(report.issues)}</p>
        <p>Duplicate Groups: {len(report.duplicate_groups)}</p>
    </div>
    <h2>Issues</h2>
    {''.join(f'<div class="issue"><strong>{issue.category}</strong>: {issue.description}</div>' for issue in report.issues[:10])}
</body>
</html>"""


def generate_markdown_report(report: OrganizationReport, title: str) -> str:
    """Generate Markdown report"""
    lines = [
        f"# {title}",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- **Overall Score**: {report.organization_score:.1f}/100",
        f"- **Total Files**: {report.total_files}",
        f"- **Total Size**: {report.total_size_bytes / 1024 / 1024:.1f} MB",
        f"- **Issues Found**: {len(report.issues)}",
        f"- **Duplicate Groups**: {len(report.duplicate_groups)}",
        ""
    ]

    if report.issues:
        lines.append("## Issues")
        lines.append("")
        for issue in report.issues[:20]:
            lines.append(f"- **[{issue.severity.value}] {issue.category}**: {issue.description}")
        lines.append("")

    return "\n".join(lines)


def generate_markdown_plan(plan) -> str:
    """Generate Markdown plan"""
    lines = [
        "# Organization Plan",
        "",
        f"**Strategy**: {plan.strategy.value}",
        f"**Files to Move**: {plan.total_files}",
        f"**Total Size**: {plan.total_size_bytes / 1024 / 1024:.1f} MB",
        f"**Estimated Time**: {plan.estimated_time:.1f}s",
        "",
        "## Planned Moves",
        ""
    ]

    for source, dest in list(plan.moves.items())[:50]:
        lines.append(f"- `{source}` → `{dest}`")

    if len(plan.moves) > 50:
        lines.append(f"- ... and {len(plan.moves) - 50} more files")

    return "\n".join(lines)


def generate_markdown_result(result) -> str:
    """Generate Markdown result"""
    lines = [
        "# Organization Result",
        "",
        f"**Success**: {'✓' if result.success else '✗'}",
        f"**Files Moved**: {result.moved_files}",
        f"**Failed Moves**: {result.failed_moves}",
        ""
    ]

    if result.backup_path:
        lines.append(f"**Backup Created**: `{result.backup_path}`")
        lines.append("")

    if result.errors:
        lines.append("## Errors")
        lines.append("")
        for error in result.errors[:20]:
            lines.append(f"- {error}")
        lines.append("")

    return "\n".join(lines)


def print_analysis_summary(report: OrganizationReport):
    """Print analysis summary to console"""
    print("\n" + "=" * 70)
    print(" " * 20 + "ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\nOverall Score: {report.organization_score:.1f}/100")
    print(f"Total Files: {report.total_files}")
    print(f"Total Size: {report.total_size_bytes / 1024 / 1024:.1f} MB")
    print(f"Issues Found: {len(report.issues)}")
    print(f"Duplicate Groups: {len(report.duplicate_groups)}")
    print("\n" + "=" * 70 + "\n")


def print_plan_summary(plan):
    """Print plan summary to console"""
    print("\n" + "=" * 70)
    print(" " * 20 + "ORGANIZATION PLAN")
    print("=" * 70)
    print(f"\nStrategy: {plan.strategy.value}")
    print(f"Files to Move: {plan.total_files}")
    print(f"Total Size: {plan.total_size_bytes / 1024 / 1024:.1f} MB")
    print(f"Estimated Time: {plan.estimated_time:.1f}s")
    print("\n" + "=" * 70 + "\n")


def print_result_summary(result, dry_run: bool):
    """Print result summary to console"""
    mode = "DRY-RUN PREVIEW" if dry_run else "EXECUTION RESULT"
    print("\n" + "=" * 70)
    print(" " * 20 + mode)
    print("=" * 70)
    print(f"\nSuccess: {'✓' if result.success else '✗'}")
    print(f"Files {'Would Be ' if dry_run else ''}Moved: {result.moved_files}")
    print(f"Failed Moves: {result.failed_moves}")

    if result.backup_path:
        print(f"Backup Created: {result.backup_path}")

    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for error in result.errors[:5]:
            print(f"  - {error}")
        if len(result.errors) > 5:
            print(f"  ... and {len(result.errors) - 5} more errors")

    print("\n" + "=" * 70 + "\n")


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Route to appropriate command
    if args.command == 'analyze':
        return command_analyze(args)
    elif args.command == 'plan':
        return command_plan(args)
    elif args.command == 'organize':
        return command_organize(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
