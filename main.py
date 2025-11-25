#!/usr/bin/env python3
"""
Investment Analysis Agent - Main Entry Point

This script analyzes YouTube videos from investment channels and generates
a comprehensive stock recommendation report.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from src.utils import load_config
from src.agents import create_investment_agent, AgentState


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Investment Analysis Agent - Analyze YouTube investment videos"
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--output',
        default='investment_report.md',
        help='Output file for report (default: investment_report.md)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    try:
        # Load configuration
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)

        # Create initial state
        initial_state: AgentState = {
            "config": config,
            "analyzer": None,
            "channels": [],
            "video_urls": [],
            "video_analyses": [],
            "aggregated_stocks": [],
            "critic_result": {},
            "investigation_results": [],
            "final_report": "",
            "errors": [],
            "total_cost": 0.0
        }

        # Create and run agent
        agent = create_investment_agent()

        print()
        result = agent.invoke(initial_state)

        # Display results
        print("=" * 60)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 60)
        print()

        # Print report summary
        print(f"📊 Videos analyzed: {len(result['video_analyses'])}")
        print(f"🎯 Stocks found: {len(result['aggregated_stocks'])}")
        print(f"🔥 Multi-channel stocks: {sum(1 for s in result['aggregated_stocks'] if s['num_channels'] > 1)}")
        print(f"💰 Total cost: ${result['total_cost']:.2f}")

        if result['errors']:
            print(f"⚠️ Errors: {len(result['errors'])}")

        # Save report
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result['final_report'])

        print(f"\n📄 Report saved to: {output_path.absolute()}")

        # Print report to console if verbose
        if args.verbose:
            print("\n" + "=" * 60)
            print(result['final_report'])

        # Print errors if any
        if result['errors']:
            print("\n⚠️ ERRORS ENCOUNTERED:")
            for error in result['errors']:
                print(f"  - {error}")

        return 0

    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        print("\nTip: Create a config.yaml file using config.example.yaml as a template")
        return 1

    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)

        if args.verbose:
            import traceback
            print("\nTraceback:")
            traceback.print_exc()

        return 1


if __name__ == "__main__":
    sys.exit(main())
